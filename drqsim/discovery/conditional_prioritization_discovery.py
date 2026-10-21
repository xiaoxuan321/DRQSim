# drq/discovery/conditional_prioritization_discovery.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


LEFT_ACTIVITY_PREFIX = "left_activity = "
RIGHT_ACTIVITY_PREFIX = "right_activity = "


def discover_conditional_prioritization_model(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        activities: list,
        max_depths_cv: Optional[List[int]] = None,
        label_data_attributes: Optional[List[str]] = None,
        label_data_attributes_categorical: Optional[List[str]] = None,
        values_categorical: Optional[Dict[str, list]] = None,
        min_samples_leaf: Optional[int] = None,
        min_training_pairs: int = 80,
        min_overtaking_rate_for_tree: float = 0.02,
        max_competitors_per_winner: int = 8,
        max_training_winners: Optional[int] = 12000,
        random_state: int = 72,
        max_leaf_nodes: int = 64,
) -> Dict[str, Any]:
    """Discover a white-box pairwise Conditional Prioritization Tree (CPT).

    A training decision is reconstructed at the *actual start time* of an event.
    For the actual resource r and winner j starting at t, competitors are events k
    satisfying enabled_k <= t < start_k and r is a candidate resource for k.
    The pair (j, k) is a positive sample and its reversed pair is a negative sample.

    The model therefore learns: P(left task should be dispatched before right task
    | queue state, task context, resource context).  Runtime aggregation converts
    pairwise probabilities into a queue ranking.  FCFS is always kept as fallback.
    """
    max_depths_cv = [2, 3, 4] if max_depths_cv is None else [max(1, int(v)) for v in max_depths_cv]
    label_data_attributes = label_data_attributes or []
    label_data_attributes_categorical = label_data_attributes_categorical or []
    values_categorical = values_categorical or {}
    resources = [str(r) for r in (resources or [])]
    activities = [str(a) for a in (activities or [])]

    train_df, metadata = build_conditional_prioritization_training_df(
        df_features=df_features,
        act_to_resources=act_to_resources,
        resources=resources,
        activities=activities,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
        min_training_pairs=min_training_pairs,
        max_competitors_per_winner=max_competitors_per_winner,
        max_training_winners=max_training_winners,
        random_state=random_state,
    )

    if train_df.empty or int(metadata.get("n_competition_pairs", 0)) < int(min_training_pairs):
        return _fcfs_fallback_params(
            metadata.get("reason", "Insufficient resource-queue competition samples."),
            metadata=metadata,
        )

    y = train_df["class"].astype(int)
    feature_columns = [c for c in train_df.columns if c != "class"]
    X = _coerce_numeric(train_df[feature_columns])

    overtaking_rate = float(metadata.get("overtaking_rate", 0.0))
    if overtaking_rate < max(0.0, float(min_overtaking_rate_for_tree)):
        return {
            "model_type": "conditional_prioritization_tree",
            "model_kind": "fcfs_fallback",
            "description": (
                "A real ready/resource queue is enabled, but the training log shows "
                "no material overtaking; dispatch therefore uses explicit FCFS."
            ),
            "feature_columns": [],
            "rules": None,
            "n_samples": int(len(train_df)),
            "n_competition_pairs": int(metadata.get("n_competition_pairs", 0)),
            "max_depth": 0,
            "min_samples_leaf": 0,
            "train_pair_accuracy": float(metadata.get("fcfs_pair_accuracy", 1.0)),
            "overtaking_rate": overtaking_rate,
            "fcfs_pair_accuracy": float(metadata.get("fcfs_pair_accuracy", 1.0)),
            "fallback_policy": "fcfs",
            "tie_breaker": "enabled_time_then_case_id",
            "min_queue_size": 2,
            "min_overtaking_rate_for_tree": float(min_overtaking_rate_for_tree),
            "metadata": metadata,
        }

    if min_samples_leaf is None:
        min_samples_leaf = max(20, int(len(train_df) * 0.01))

    clf = _fit_tree(
        X=X,
        y=y,
        max_depths_cv=max_depths_cv,
        min_samples_leaf=int(min_samples_leaf),
        random_state=random_state,
        max_leaf_nodes=max_leaf_nodes,
    )

    if isinstance(clf, DummyClassifier):
        rules = None
        model_kind = "fcfs_fallback"
        depth = 0
        train_accuracy = float(metadata.get("fcfs_pair_accuracy", 0.5))
    else:
        rules = _decision_tree_to_score_rules(clf, feature_columns)
        model_kind = "pairwise_decision_tree"
        depth = int(clf.get_depth())
        train_accuracy = float(clf.score(X, y))

    return {
        "model_type": "conditional_prioritization_tree",
        "model_kind": model_kind,
        "description": (
            "Pairwise white-box queue-dispatch model. At a resource dispatch point, "
            "it ranks simultaneously waiting eligible tasks and selects the next task."
        ),
        "feature_columns": feature_columns,
        "rules": rules,
        "n_samples": int(len(train_df)),
        "n_competition_pairs": int(metadata.get("n_competition_pairs", 0)),
        "max_depth": depth,
        "min_samples_leaf": int(min_samples_leaf),
        "train_pair_accuracy": train_accuracy,
        "overtaking_rate": float(metadata.get("overtaking_rate", 0.0)),
        "fcfs_pair_accuracy": float(metadata.get("fcfs_pair_accuracy", 0.5)),
        "min_overtaking_rate_for_tree": float(min_overtaking_rate_for_tree),
        "fallback_policy": "fcfs",
        "tie_breaker": "enabled_time_then_case_id",
        "min_queue_size": 2,
        "metadata": metadata,
    }


def build_conditional_prioritization_training_df(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        activities: list,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        min_training_pairs: int = 80,
        max_competitors_per_winner: int = 8,
        max_training_winners: Optional[int] = 12000,
        random_state: int = 72,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Build pairwise CPT samples using one global sweep over the event log.

    The implementation is deliberately array/index based.  Large BPI logs can
    contain hundreds of thousands of events; rebuilding a Pandas mask for every
    winner would make prioritization discovery prohibitively expensive.
    """
    required = ["transition_label", "resource", "enabled_t", "start_t", "end_t"]
    missing = [c for c in required if c not in df_features.columns]
    if missing:
        return pd.DataFrame(), {"reason": f"df_features misses columns: {missing}"}

    df = df_features.copy()
    df = df[(~df["transition_label"].isna()) & (~df["resource"].isna())].copy()
    for col in ("enabled_t", "start_t", "end_t"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df.dropna(subset=required, inplace=True)
    if df.empty:
        return pd.DataFrame(), {"reason": "No valid synchronized events for prioritization discovery."}

    df["transition_label"] = df["transition_label"].astype(str)
    df["resource"] = df["resource"].astype(str)
    df.reset_index(drop=True, inplace=True)

    activities = [str(a) for a in (activities or [])]
    resources = [str(r) for r in (resources or [])]
    act_to_resources = {
        str(a): [str(r) for r in (rs or [])]
        for a, rs in (act_to_resources or {}).items()
    }
    rng = np.random.default_rng(random_state)

    n = len(df)
    activity_arr = df["transition_label"].astype(str).to_numpy()
    resource_arr = df["resource"].astype(str).to_numpy()
    enabled_ns = df["enabled_t"].astype("int64").to_numpy(dtype=np.int64, copy=False)
    start_ns = df["start_t"].astype("int64").to_numpy(dtype=np.int64, copy=False)
    end_ns = df["end_t"].astype("int64").to_numpy(dtype=np.int64, copy=False)

    # Candidate-resource list per event.  This is created once and reused by the
    # global sweep, instead of duplicating whole event tables per resource.
    candidate_resources_by_event: list[tuple[str, ...]] = []
    all_resource_set = set(resources)
    for i in range(n):
        activity = str(activity_arr[i])
        candidates = [str(r) for r in act_to_resources.get(activity, resources)]
        actual = str(resource_arr[i])
        if actual not in candidates:
            candidates.append(actual)
        # Stable deduplication limits active-set update cost.
        seen = set()
        clean = []
        for r in candidates:
            if r in seen:
                continue
            seen.add(r)
            clean.append(r)
            all_resource_set.add(r)
        candidate_resources_by_event.append(tuple(clean))

    # Global winner cap.  The old per-resource cap could accidentally generate
    # 12k winners *per resource* on BPI 2017, defeating the purpose of sampling.
    winner_indices = np.arange(n, dtype=int)
    if max_training_winners is not None:
        try:
            max_training_winners = int(max_training_winners)
        except Exception:
            max_training_winners = None
    if max_training_winners is not None and max_training_winners > 0 and n > max_training_winners:
        winner_indices = rng.choice(
            winner_indices,
            size=int(max_training_winners),
            replace=False,
        ).astype(int)
    winner_indices = winner_indices[np.argsort(start_ns[winner_indices], kind="mergesort")]

    enabled_order = np.argsort(enabled_ns, kind="mergesort")
    start_order = np.argsort(start_ns, kind="mergesort")
    active_by_resource: Dict[str, set[int]] = {str(r): set() for r in all_resource_set}
    add_pos = 0
    remove_pos = 0

    # Fast resource workload index from actual execution intervals.
    resource_start_ns: Dict[str, np.ndarray] = {}
    resource_end_ns: Dict[str, np.ndarray] = {}
    for resource, sub_idx in df.groupby("resource", sort=False).groups.items():
        idx = np.asarray(list(sub_idx), dtype=int)
        resource_start_ns[str(resource)] = np.sort(start_ns[idx])
        resource_end_ns[str(resource)] = np.sort(end_ns[idx])

    # Pre-extract optional case-history and case-attribute columns used by the
    # white-box pair features. This avoids thousands of Series allocations.
    history_arrays: Dict[str, np.ndarray] = {}
    for activity in activities:
        if activity in df.columns:
            history_arrays[activity] = pd.to_numeric(df[activity], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    attribute_arrays: Dict[str, np.ndarray] = {}
    for attr in label_data_attributes:
        if attr in df.columns:
            if attr in label_data_attributes_categorical:
                attribute_arrays[attr] = df[attr].astype(object).to_numpy()
            else:
                attribute_arrays[attr] = pd.to_numeric(df[attr], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    resource_history_arrays: Dict[str, np.ndarray] = {}
    # Resource-history columns are optional and only materialized for resources
    # that actually appear as a winner resource in the sampled decisions.
    sampled_winner_resources = {str(resource_arr[i]) for i in winner_indices}
    for resource in sampled_winner_resources:
        if resource in df.columns:
            resource_history_arrays[resource] = pd.to_numeric(df[resource], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    context = {
        "activity_arr": activity_arr,
        "enabled_ns": enabled_ns,
        "history_arrays": history_arrays,
        "attribute_arrays": attribute_arrays,
        "resource_history_arrays": resource_history_arrays,
    }

    rows = []
    original_pairs = 0
    overtakes = 0
    fcfs_correct = 0
    competition_decisions = 0
    queue_sizes = []

    for winner_idx in winner_indices.tolist():
        t_ns = int(start_ns[winner_idx])

        # Add all tasks enabled by t to each resource queue for which they are
        # eligible. Then remove tasks whose start <= t; simultaneous starts are
        # treated as parallel capacity, not as priority losers.
        while add_pos < n and int(enabled_ns[enabled_order[add_pos]]) <= t_ns:
            idx = int(enabled_order[add_pos])
            for resource in candidate_resources_by_event[idx]:
                active_by_resource.setdefault(resource, set()).add(idx)
            add_pos += 1

        while remove_pos < n and int(start_ns[start_order[remove_pos]]) <= t_ns:
            idx = int(start_order[remove_pos])
            for resource in candidate_resources_by_event[idx]:
                active_by_resource.setdefault(resource, set()).discard(idx)
            remove_pos += 1

        resource = str(resource_arr[winner_idx])
        active = active_by_resource.get(resource, set())
        if not active:
            continue

        competitors = list(active)
        if not competitors:
            continue

        competition_decisions += 1
        queue_sizes.append(1 + len(competitors))
        if max_competitors_per_winner and len(competitors) > int(max_competitors_per_winner):
            competitors = rng.choice(
                np.asarray(competitors, dtype=int),
                size=int(max_competitors_per_winner),
                replace=False,
            ).astype(int).tolist()

        workload = _resource_workload_at_ns(
            resource=resource,
            current_time_ns=t_ns,
            resource_start_ns=resource_start_ns,
            resource_end_ns=resource_end_ns,
        )
        queue_length = 1 + len(active)

        for competitor_idx in competitors:
            positive = _build_pair_features_by_index(
                left_idx=int(winner_idx),
                right_idx=int(competitor_idx),
                current_time_ns=t_ns,
                resource=resource,
                queue_length=queue_length,
                resource_workload=workload,
                activities=activities,
                label_data_attributes=label_data_attributes,
                label_data_attributes_categorical=label_data_attributes_categorical,
                values_categorical=values_categorical,
                context=context,
            )
            positive["class"] = 1
            rows.append(positive)

            negative = _build_pair_features_by_index(
                left_idx=int(competitor_idx),
                right_idx=int(winner_idx),
                current_time_ns=t_ns,
                resource=resource,
                queue_length=queue_length,
                resource_workload=workload,
                activities=activities,
                label_data_attributes=label_data_attributes,
                label_data_attributes_categorical=label_data_attributes_categorical,
                values_categorical=values_categorical,
                context=context,
            )
            negative["class"] = 0
            rows.append(negative)

            original_pairs += 1
            if int(enabled_ns[winner_idx]) > int(enabled_ns[competitor_idx]):
                overtakes += 1
            if int(enabled_ns[winner_idx]) <= int(enabled_ns[competitor_idx]):
                fcfs_correct += 1

    train_df = pd.DataFrame(rows)
    if not train_df.empty:
        for col in train_df.columns:
            if col == "class":
                train_df[col] = train_df[col].astype(int)
            else:
                train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)

    metadata = {
        "n_events": int(n),
        "n_sampled_winners": int(len(winner_indices)),
        "n_competition_decisions": int(competition_decisions),
        "n_competition_pairs": int(original_pairs),
        "n_pairwise_training_rows": int(len(train_df)),
        "overtaking_rate": float(overtakes / max(original_pairs, 1)),
        "fcfs_pair_accuracy": float(fcfs_correct / max(original_pairs, 1)),
        "mean_competition_queue_size": float(np.mean(queue_sizes)) if queue_sizes else 0.0,
        "p95_competition_queue_size": float(np.quantile(queue_sizes, 0.95)) if queue_sizes else 0.0,
        "min_training_pairs": int(min_training_pairs),
        "max_competitors_per_winner": int(max_competitors_per_winner),
        "max_training_winners": None if max_training_winners is None else int(max_training_winners),
        "training_target": "pairwise actual dispatch order conditioned on assigned resource and ready-queue state",
        "fallback_policy": "fcfs",
        "feature_builder": "global_event_sweep_plus_numpy_arrays",
    }
    if original_pairs < int(min_training_pairs):
        metadata["reason"] = "Insufficient resource-conditioned queue competition pairs."
    return train_df, metadata


def _build_pair_features_by_index(
        left_idx: int,
        right_idx: int,
        current_time_ns: int,
        resource: str,
        queue_length: int,
        resource_workload: int,
        activities: list[str],
        label_data_attributes: list[str],
        label_data_attributes_categorical: list[str],
        values_categorical: dict,
        context: dict,
) -> Dict[str, float]:
    activity_arr = context["activity_arr"]
    enabled_ns = context["enabled_ns"]
    history_arrays = context["history_arrays"]
    attribute_arrays = context["attribute_arrays"]
    resource_history_arrays = context["resource_history_arrays"]

    left_activity = str(activity_arr[left_idx])
    right_activity = str(activity_arr[right_idx])
    minute_ns = 60.0 * 1_000_000_000.0
    left_wait = max(0.0, float(current_time_ns - int(enabled_ns[left_idx])) / minute_ns)
    right_wait = max(0.0, float(current_time_ns - int(enabled_ns[right_idx])) / minute_ns)
    now = pd.Timestamp(int(current_time_ns), unit="ns", tz="UTC")

    rec: Dict[str, float] = {
        "left_waiting_age_min": float(left_wait),
        "right_waiting_age_min": float(right_wait),
        "waiting_age_diff_min": float(left_wait - right_wait),
        "left_waiting_age_log": _log_scale_minutes(left_wait),
        "right_waiting_age_log": _log_scale_minutes(right_wait),
        "waiting_age_log_diff": _log_scale_minutes(left_wait) - _log_scale_minutes(right_wait),
        "same_activity": float(left_activity == right_activity),
        "resource_queue_length": float(max(1, int(queue_length))),
        "resource_queue_length_log": _log_scale_count(queue_length),
        "resource_workload": float(max(0, int(resource_workload))),
        "hour": float(now.hour),
        "weekday": float(now.weekday()),
    }

    for activity in activities:
        rec[f"{LEFT_ACTIVITY_PREFIX}{activity}"] = float(left_activity == str(activity))
        rec[f"{RIGHT_ACTIVITY_PREFIX}{activity}"] = float(right_activity == str(activity))
        arr = history_arrays.get(activity)
        if arr is not None:
            rec[f"history_diff::{activity}"] = float(arr[left_idx] - arr[right_idx])

    for attr in label_data_attributes:
        arr = attribute_arrays.get(attr)
        if arr is None:
            continue
        if attr in label_data_attributes_categorical:
            lv = arr[left_idx]
            rv = arr[right_idx]
            for value in values_categorical.get(attr, []):
                rec[f"left::{attr} = {value}"] = float(lv == value)
                rec[f"right::{attr} = {value}"] = float(rv == value)
        else:
            lv = _safe_float(arr[left_idx])
            rv = _safe_float(arr[right_idx])
            rec[f"left::{attr}"] = lv
            rec[f"right::{attr}"] = rv
            rec[f"diff::{attr}"] = float(lv - rv)

    res_arr = resource_history_arrays.get(str(resource))
    if res_arr is not None:
        rec["resource_history_diff"] = float(res_arr[left_idx] - res_arr[right_idx])
    return rec


def _resource_workload_at_ns(resource, current_time_ns: int, resource_start_ns, resource_end_ns) -> int:
    starts = resource_start_ns.get(str(resource))
    ends = resource_end_ns.get(str(resource))
    if starts is None or ends is None:
        return 0
    t = np.int64(current_time_ns)
    return int(max(0, np.searchsorted(starts, t, side="right") - np.searchsorted(ends, t, side="right")))


def _resource_workload_at(resource, current_time, resource_start_ns, resource_end_ns) -> int:
    starts = resource_start_ns.get(str(resource))
    ends = resource_end_ns.get(str(resource))
    if starts is None or ends is None:
        return 0
    t = np.int64(pd.Timestamp(current_time).value)
    return int(max(0, np.searchsorted(starts, t, side="right") - np.searchsorted(ends, t, side="right")))


def _fit_tree(X, y, max_depths_cv, min_samples_leaf, random_state, max_leaf_nodes):
    if y.nunique() < 2:
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X, y)
        return dummy
    depths = sorted({max(1, int(d)) for d in (max_depths_cv or [3])})
    if len(depths) == 1:
        clf = DecisionTreeClassifier(
            max_depth=depths[0],
            min_samples_leaf=max(1, int(min_samples_leaf)),
            max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes else None,
            class_weight="balanced",
            random_state=random_state,
        )
        clf.fit(X, y)
        return clf
    cv = min(3, max(2, int(len(y) // max(2 * min_samples_leaf, 1))))
    base = DecisionTreeClassifier(
        min_samples_leaf=max(1, int(min_samples_leaf)),
        max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes else None,
        class_weight="balanced",
        random_state=random_state,
    )
    search = GridSearchCV(base, {"max_depth": depths}, scoring="accuracy", cv=cv, n_jobs=1)
    search.fit(X, y)
    return search.best_estimator_


def _decision_tree_to_score_rules(clf: DecisionTreeClassifier, feature_columns: list[str]):
    tree = clf.tree_
    classes = [int(c) for c in clf.classes_]
    positive_index = classes.index(1) if 1 in classes else None

    def build(node_id: int):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right:
            counts = tree.value[node_id][0].astype(float)
            total = float(counts.sum())
            score = float(counts[positive_index] / total) if total > 0 and positive_index is not None else 0.5
            return {"type": "leaf", "score": float(np.clip(score, 0.0, 1.0)), "n_node_samples": int(tree.n_node_samples[node_id])}
        return {
            "type": "node",
            "feature": feature_columns[tree.feature[node_id]],
            "threshold": float(tree.threshold[node_id]),
            "children": {"true": build(left), "false": build(right)},
        }
    return build(0)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _log_scale_minutes(value: float, cap_min: float = 7 * 24 * 60.0) -> float:
    value = max(0.0, min(_safe_float(value), float(cap_min)))
    return float(np.log1p(value) / np.log1p(max(1.0, float(cap_min))))


def _log_scale_count(value: float, cap: float = 50.0) -> float:
    value = max(0.0, min(_safe_float(value), float(cap)))
    return float(np.log1p(value) / np.log1p(max(1.0, float(cap))))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def _fcfs_fallback_params(reason: str, metadata: Optional[dict] = None) -> Dict[str, Any]:
    """Return the data-driven FCFS policy when CPT evidence is insufficient.

    The queue module is always active. ``model_kind`` alone decides whether
    the dataset supports a learned CPT or requires deterministic FCFS.
    """
    return {
        "model_type": "conditional_prioritization_tree",
        "model_kind": "fcfs_fallback",
        "reason": str(reason),
        "feature_columns": [],
        "rules": None,
        "n_samples": 0,
        "n_competition_pairs": int((metadata or {}).get("n_competition_pairs", 0)),
        "overtaking_rate": float((metadata or {}).get("overtaking_rate", 0.0)),
        "fcfs_pair_accuracy": float((metadata or {}).get("fcfs_pair_accuracy", 0.5)),
        "fallback_policy": "fcfs",
        "tie_breaker": "enabled_time_then_case_id",
        "min_queue_size": 2,
        "metadata": metadata or {},
    }
