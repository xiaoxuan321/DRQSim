# drq/discovery/queue_scheduling_discovery.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# =========================================================
# 五模块队列调度特征名。
# 这些特征用于刻画 ready task 与候选资源在当前时刻的局部排队压力。
# =========================================================
QUEUE_FEATURE_NAMES = [
    "global_queue_length",
    "activity_queue_length",
    "resource_queue_length",
    "resource_activity_queue_length",
    "queue_pressure",
    "system_wip",
]

ACTIVITY_PREFIX = "activity = "

RESOURCE_POLICY_CANDIDATES = {
    "frequency_oriented": {"tree": 0.0, "frequency": 0.80, "availability": 0.20},
    "balanced": {"tree": 0.35, "frequency": 0.40, "availability": 0.25},
    "queue_oriented": {"tree": 0.45, "frequency": 0.25, "availability": 0.30},
}



def discover_queue_scheduling_model(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        activities: list,
        max_depths_cv: Optional[List[int]] = None,
        label_data_attributes: Optional[List[str]] = None,
        label_data_attributes_categorical: Optional[List[str]] = None,
        values_categorical: Optional[Dict[str, list]] = None,
        min_samples_leaf: Optional[int] = None,
        max_negative_per_positive: int = 3,
        random_state: int = 72,
        # 加速参数
        force_empirical: bool = False,
        max_training_events: Optional[int] = 15000,
        use_fast_queue_features: bool = True,
        max_leaf_nodes: int = 64,
        resource_capacity_profile: Optional[Dict[str, dict]] = None,
) -> Dict[str, Any]:
    """
    发现五模块模型中的全局资源感知队列调度树。

    关键加速点：
    1. 当运行时策略不使用 tree_score 时，可设置 force_empirical=True，
       只输出活动-资源频率模型，跳过候选对特征构造和决策树训练。
    2. 队列特征使用 searchsorted 索引构造，避免每个候选对全表扫描。
    3. 外层 Optuna 已经选择 queue_scheduling_max_depth 时，max_depths_cv
       通常只有一个值，此时 _fit_queue_tree 会直接训练一棵树，不再 GridSearchCV。
    4. 可通过 max_training_events 对训练事件下采样，避免大日志拖慢每个 trial。
    """

    if max_depths_cv is None:
        max_depths_cv = [2]

    label_data_attributes = label_data_attributes or []
    label_data_attributes_categorical = label_data_attributes_categorical or []
    values_categorical = values_categorical or {}
    resources = [str(r) for r in (resources or [])]
    activities = [str(a) for a in (activities or [])]

    valid_df = _prepare_valid_event_df(df_features)
    activity_resource_frequency = _activity_resource_frequency(valid_df) if not valid_df.empty else {}

    if force_empirical:
        params = _empirical_params(
            reason="force_empirical=True; tree_score is not required by the selected queue strategy.",
            df=valid_df,
            resources=resources,
            activities=activities,
            activity_resource_frequency=activity_resource_frequency,
            max_negative_per_positive=max_negative_per_positive,
        )
        params.setdefault("metadata", {})["resource_capacity_profile"] = dict(resource_capacity_profile or {})
        return params

    train_df, metadata = build_queue_scheduling_training_df(
        df_features=df_features,
        act_to_resources=act_to_resources,
        resources=resources,
        activities=activities,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
        max_negative_per_positive=max_negative_per_positive,
        random_state=random_state,
        max_training_events=max_training_events,
        use_fast_queue_features=use_fast_queue_features,
        activity_resource_frequency=activity_resource_frequency,
        resource_capacity_profile=resource_capacity_profile,
    )

    if train_df.empty:
        if activity_resource_frequency:
            return _empirical_params(
                reason=metadata.get("reason", "No queue-scheduling training samples are available."),
                df=valid_df,
                resources=resources,
                activities=activities,
                activity_resource_frequency=activity_resource_frequency,
                max_negative_per_positive=max_negative_per_positive,
            )
        return _fallback_params("No queue-scheduling training samples are available.")

    y = train_df["class"].astype(int)
    feature_columns = [c for c in train_df.columns if c not in ["class"]]
    X = _coerce_numeric(train_df[feature_columns].copy())

    if y.nunique() < 2:
        positive_rate = float(y.mean()) if len(y) else 0.0
        return {
                "model_type": "global_resource_aware_queue_scheduling_tree",
            "model_role": "queue_aware_resource_allocation",
            "model_kind": "empirical",
            "feature_columns": [],
            "rules": None,
            "positive_rate": positive_rate,
            "n_samples": int(len(train_df)),
            "metadata": metadata,
            "fallback": _fallback_config(),
            "runtime_wait_penalty": 0.08,
            "resource_policy_candidates": RESOURCE_POLICY_CANDIDATES,
            "resource_policy_mode": "balanced",
            "selected_strategy": "balanced",
            "score_mixing_weights": RESOURCE_POLICY_CANDIDATES["balanced"],
            "hybrid_score_weights": RESOURCE_POLICY_CANDIDATES["balanced"],
            "wait_penalty_mode": "log_scaled",
            "wait_penalty_cap_min": 1440.0,
            "first_activity_can_queue": True,
            "enforce_selected_resource_capacity": True,
            "availability_score_mode": "ranked_earliest_start",
        }

    if min_samples_leaf is None:
        min_samples_leaf = max(20, int(len(train_df) * 0.01))

    clf = _fit_queue_tree(
        X=X,
        y=y,
        max_depths_cv=max_depths_cv,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        max_leaf_nodes=max_leaf_nodes,
    )

    if isinstance(clf, DummyClassifier):
        positive_rate = float(y.mean())
        rules = None
        model_kind = "empirical"
        depth = 0
    else:
        rules = _decision_tree_to_score_rules(clf=clf, feature_columns=feature_columns)
        positive_rate = float(y.mean())
        model_kind = "decision_tree"
        depth = int(clf.get_depth())

    return {
        "model_type": "global_resource_aware_queue_scheduling_tree",
        "model_role": "queue_aware_resource_allocation",
        "description": (
            "A global white-box resource-allocation tree over task-resource pairs. "
            "Queue-state features make allocation queue-aware; actual task dispatch "
            "order is modeled separately by the Conditional Prioritization Tree."
        ),
        "model_kind": model_kind,
        "feature_columns": feature_columns,
        "rules": rules,
        "positive_rate": positive_rate,
        "n_samples": int(len(train_df)),
        "max_depth": depth,
        "min_samples_leaf": int(min_samples_leaf),
        "metadata": metadata,
        "fallback": _fallback_config(),

        # Runtime defaults: queue scheduling should explain resource-competition waiting.
        "runtime_wait_penalty": 0.08,
        "resource_policy_candidates": RESOURCE_POLICY_CANDIDATES,
        "resource_policy_mode": "balanced",
        "selected_strategy": "balanced",
        "score_mixing_weights": RESOURCE_POLICY_CANDIDATES["balanced"],
        "hybrid_score_weights": RESOURCE_POLICY_CANDIDATES["balanced"],
        "wait_penalty_mode": "log_scaled",
        "wait_penalty_cap_min": 1440.0,
        "first_activity_can_queue": True,
        "enforce_selected_resource_capacity": True,
        "availability_score_mode": "ranked_earliest_start",
    }


def build_queue_scheduling_training_df(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        activities: list,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_negative_per_positive: int = 3,
        random_state: int = 72,
        max_training_events: Optional[int] = 15000,
        use_fast_queue_features: bool = True,
        activity_resource_frequency: Optional[Dict[tuple, float]] = None,
        resource_capacity_profile: Optional[Dict[str, dict]] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    构造队列调度树训练样本。

    与旧版相比，当前版本不再为每个候选任务-资源对扫描整个 df。
    use_fast_queue_features=True 时会构建 QueueFeatureIndex，通过 searchsorted
    查询某一时刻的活动等待数、资源可服务等待数和资源 workload。
    """

    required = ["transition_label", "resource", "enabled_t", "start_t", "end_t"]
    missing = [c for c in required if c not in df_features.columns]
    if missing:
        return pd.DataFrame(), {"reason": f"df_features misses columns: {missing}"}

    rng = np.random.default_rng(random_state)

    full_df = _prepare_valid_event_df(df_features)
    if full_df.empty:
        return pd.DataFrame(), {"reason": "No valid labeled events."}

    resources = [str(r) for r in resources]
    activities = [str(a) for a in activities]
    activity_resource_frequency = activity_resource_frequency or _activity_resource_frequency(full_df)
    resource_capacity_profile = resource_capacity_profile or {}

    if max_training_events is not None:
        try:
            max_training_events = int(max_training_events)
        except Exception:
            max_training_events = None

    if max_training_events is not None and max_training_events > 0 and len(full_df) > max_training_events:
        # 用事件级下采样控制候选对数量。队列上下文仍然基于 full_df 计算，
        # 因而不会把等待队列统计也限制在采样子集内。
        sampled_df = (
            full_df.sample(n=max_training_events, random_state=random_state)
            .sort_values(["enabled_t", "start_t", "end_t"])
            .reset_index(drop=False)
            .rename(columns={"index": "_original_index"})
        )
    else:
        sampled_df = full_df.reset_index(drop=False).rename(columns={"index": "_original_index"})

    if use_fast_queue_features:
        feature_index = _QueueFeatureIndex(
            df=full_df,
            act_to_resources=act_to_resources,
            resources=resources,
        )
    else:
        feature_index = None

    # Used to estimate the waiting time a candidate resource would impose at enabled_time.
    # The estimate is deliberately lightweight and mirrors the runtime capacity guard.
    intervals_by_resource = _build_resource_intervals(full_df)

    rows = []

    for _, row in sampled_df.iterrows():
        original_index = int(row.get("_original_index", row.name))
        activity = str(row["transition_label"])
        actual_resource = str(row["resource"])
        enabled_time = row["enabled_t"]

        candidate_resources = [
            str(r) for r in act_to_resources.get(activity, resources)
        ]

        if actual_resource not in candidate_resources:
            candidate_resources.append(actual_resource)

        positive_candidates = [actual_resource]
        negative_candidates = [r for r in candidate_resources if r != actual_resource]

        if max_negative_per_positive and len(negative_candidates) > max_negative_per_positive:
            negative_candidates = list(
                rng.choice(
                    negative_candidates,
                    size=max_negative_per_positive,
                    replace=False,
                )
            )

        sampled_candidates = positive_candidates + negative_candidates

        for candidate_resource in sampled_candidates:
            candidate_resource = str(candidate_resource)

            if feature_index is not None:
                candidate_workload = feature_index.count_active_resource(
                    candidate_resource,
                    enabled_time,
                )
                queue_features = feature_index.compute_queue_features(
                    current_original_index=original_index,
                    current_time=enabled_time,
                    current_start_time=row["start_t"],
                    activity=activity,
                    candidate_resource=candidate_resource,
                )
            else:
                candidate_workload = _count_active_intervals(
                    intervals_by_resource.get(candidate_resource, []),
                    enabled_time,
                )
                queue_features = _compute_queue_features_for_candidate(
                    df=full_df,
                    current_index=original_index,
                    current_time=enabled_time,
                    activity=activity,
                    candidate_resource=candidate_resource,
                    act_to_resources=act_to_resources,
                )

            capacity_cfg = resource_capacity_profile.get(candidate_resource, {})
            try:
                candidate_capacity = max(1, int(capacity_cfg.get("capacity", 1))) \
                    if isinstance(capacity_cfg, dict) else 1
            except Exception:
                candidate_capacity = 1

            candidate_waiting_time_min = _estimate_candidate_resource_wait_min(
                intervals=intervals_by_resource.get(candidate_resource, []),
                current_time=enabled_time,
                current_start_time=row["start_t"],
                current_index=original_index,
                capacity=candidate_capacity,
            )
            candidate_waiting_time_log = _log_scale_minutes(candidate_waiting_time_min)

            ar_freq = float(activity_resource_frequency.get((activity, candidate_resource), 0.0))
            skill_state = _skill_state_from_frequency(ar_freq)
            fatigue_state = _fatigue_state_from_workload(candidate_workload)

            record = {
                "class": int(candidate_resource == actual_resource),
                "candidate_workload": float(candidate_workload),
                "candidate_capacity": float(candidate_capacity),
                "candidate_available_now": float(candidate_workload < candidate_capacity),
                "candidate_resource_wait_min": float(candidate_waiting_time_min),
                "candidate_waiting_time_min": float(candidate_waiting_time_min),
                "candidate_waiting_time_log": float(candidate_waiting_time_log),
                "candidate_queue_blocked": float(candidate_waiting_time_min > 0.0),
                "activity_resource_frequency": ar_freq,
                "candidate_resource_history_count": float(row.get(candidate_resource, 0.0)),

                "resource_fatigue_low": int(fatigue_state == "low"),
                "resource_fatigue_medium": int(fatigue_state == "medium"),
                "resource_fatigue_high": int(fatigue_state == "high"),
                "resource_skill_low": int(skill_state == "low"),
                "resource_skill_medium": int(skill_state == "medium"),
                "resource_skill_high": int(skill_state == "high"),
                "resource_efficiency_low": int(skill_state == "low"),
                "resource_efficiency_medium": int(skill_state == "medium"),
                "resource_efficiency_high": int(skill_state == "high"),
                "resource_available_state": int(candidate_workload < candidate_capacity),
            }

            for activity_name in activities:
                record[f"{ACTIVITY_PREFIX}{activity_name}"] = int(activity == activity_name)

            for feature_name in QUEUE_FEATURE_NAMES:
                record[feature_name] = float(queue_features.get(feature_name, 0.0))

            # case history / activity history
            for col in activities:
                if col in sampled_df.columns:
                    record[col] = _safe_float(row.get(col, 0.0))

            # case attributes
            for attr in label_data_attributes:
                if attr in label_data_attributes_categorical:
                    raw_value = row.get(attr, None)
                    for value in values_categorical.get(attr, []):
                        record[f"{attr} = {value}"] = int(raw_value == value)
                else:
                    if attr in sampled_df.columns:
                        record[attr] = _safe_float(row.get(attr, 0.0))

            rows.append(record)

    train_df = pd.DataFrame(rows)

    if train_df.empty:
        return train_df, {"reason": "No candidate pairs were generated."}

    for col in train_df.columns:
        if col == "class":
            train_df[col] = train_df[col].astype(int)
        else:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)

    metadata = _build_metadata(
        df=full_df,
        train_df=train_df,
        resources=resources,
        activities=activities,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
        max_negative_per_positive=max_negative_per_positive,
        activity_resource_frequency=activity_resource_frequency,
    )
    metadata.update({
        "resource_policy_candidates": RESOURCE_POLICY_CANDIDATES,
        "resource_calibration_target": "validation CWD with complete runtime metrics as final tie-breaker",
        "feature_builder": "searchsorted_index" if use_fast_queue_features else "legacy_scan",
        "max_training_events": int(max_training_events) if max_training_events else None,
        "n_training_events_used": int(len(sampled_df)),
        "n_context_events": int(len(full_df)),
        "resource_capacity_profile": dict(resource_capacity_profile or {}),
        "capacity_aware_training": True,
    })

    return train_df, metadata


class _QueueFeatureIndex:
    """基于排序数组的队列特征索引，避免候选对级别的 O(N) 全表扫描。"""

    def __init__(self, df: pd.DataFrame, act_to_resources: dict, resources: list[str]):
        self.df = df
        self.act_to_resources = {
            str(a): [str(r) for r in (rs or [])]
            for a, rs in (act_to_resources or {}).items()
        }
        self.resources = [str(r) for r in (resources or [])]

        self.enabled_ns_all = np.sort(_datetime_series_to_ns(df["enabled_t"]))
        self.start_ns_all = np.sort(_datetime_series_to_ns(df["start_t"]))
        self.end_ns_all = np.sort(_datetime_series_to_ns(df["end_t"]))
        self.activities = df["transition_label"].astype(str).to_numpy()
        self.resource_values = df["resource"].astype(str).to_numpy()

        self.activity_enabled = {}
        self.activity_start = {}
        for activity, sub in df.groupby("transition_label", sort=False):
            key = str(activity)
            self.activity_enabled[key] = np.sort(_datetime_series_to_ns(sub["enabled_t"]))
            self.activity_start[key] = np.sort(_datetime_series_to_ns(sub["start_t"]))

        self.resource_start = {}
        self.resource_end = {}
        for resource, sub in df.groupby("resource", sort=False):
            key = str(resource)
            self.resource_start[key] = np.sort(_datetime_series_to_ns(sub["start_t"]))
            self.resource_end[key] = np.sort(_datetime_series_to_ns(sub["end_t"]))

        resource_to_activities = {str(r): set() for r in self.resources}
        for activity, rs in self.act_to_resources.items():
            for r in rs:
                resource_to_activities.setdefault(str(r), set()).add(str(activity))
        self.resource_to_activities = {
            r: sorted(list(acts)) for r, acts in resource_to_activities.items()
        }
        self._activity_wait_cache: dict[tuple[str, int], int] = {}
        self._resource_queue_cache: dict[tuple[str, int], int] = {}

    def count_activity_waiting(self, activity: str, current_time) -> int:
        activity = str(activity)
        t = _to_datetime_ns(current_time)
        key = (activity, int(t))
        if key in self._activity_wait_cache:
            return self._activity_wait_cache[key]

        enabled = self.activity_enabled.get(activity)
        started = self.activity_start.get(activity)
        if enabled is None or started is None:
            value = 0
        else:
            value = int(
                np.searchsorted(enabled, t, side="right")
                - np.searchsorted(started, t, side="right")
            )
            value = max(0, value)

        self._activity_wait_cache[key] = value
        return value

    def count_resource_queue(self, resource: str, current_time) -> int:
        resource = str(resource)
        t = _to_datetime_ns(current_time)
        key = (resource, int(t))
        if key in self._resource_queue_cache:
            return self._resource_queue_cache[key]

        total = 0
        for activity in self.resource_to_activities.get(resource, []):
            total += self.count_activity_waiting(activity, current_time)

        self._resource_queue_cache[key] = int(max(0, total))
        return int(max(0, total))

    def count_active_resource(self, resource: str, current_time) -> int:
        resource = str(resource)
        t = _to_datetime_ns(current_time)
        starts = self.resource_start.get(resource)
        ends = self.resource_end.get(resource)
        if starts is None or ends is None:
            return 0
        return int(max(0, np.searchsorted(starts, t, side="right") - np.searchsorted(ends, t, side="right")))

    def compute_queue_features(
            self,
            current_original_index: int,
            current_time,
            current_start_time,
            activity: str,
            candidate_resource: str,
    ) -> Dict[str, float]:
        activity = str(activity)
        candidate_resource = str(candidate_resource)

        # activity waiting count 中可能包含当前事件本身，需要剔除。
        current_is_waiting = _to_datetime_ns(current_start_time) > _to_datetime_ns(current_time)
        current_can_resource = activity in self.resource_to_activities.get(candidate_resource, [])

        activity_q = self.count_activity_waiting(activity, current_time)
        if current_is_waiting:
            activity_q = max(0, activity_q - 1)

        resource_q = self.count_resource_queue(candidate_resource, current_time)
        if current_is_waiting and current_can_resource:
            resource_q = max(0, resource_q - 1)

        resource_activity_q = activity_q if current_can_resource else 0

        t = _to_datetime_ns(current_time)
        global_q = int(
            np.searchsorted(self.enabled_ns_all, t, side="right")
            - np.searchsorted(self.start_ns_all, t, side="right")
        )
        system_wip = int(
            np.searchsorted(self.enabled_ns_all, t, side="right")
            - np.searchsorted(self.end_ns_all, t, side="right")
        )
        if current_is_waiting:
            global_q = max(0, global_q - 1)
            system_wip = max(0, system_wip - 1)

        raw = {
            "global_queue_length": max(0, global_q),
            "activity_queue_length": activity_q,
            "resource_queue_length": resource_q,
            "resource_activity_queue_length": resource_activity_q,
            "queue_pressure": max(activity_q, resource_q, resource_activity_q),
            "system_wip": max(0, system_wip),
        }
        return _post_process_queue_features(raw)


def apply_queue_scheduling_score(
        queue_scheduling_params: Dict[str, Any],
        features: Dict[str, Any],
) -> float:
    """根据队列调度树计算一个候选任务-资源对的调度分数。"""

    if not isinstance(queue_scheduling_params, dict):
        return 1.0

    model_kind = queue_scheduling_params.get("model_kind", "empirical")

    if model_kind != "decision_tree":
        return float(queue_scheduling_params.get("positive_rate", 1.0))

    rules = queue_scheduling_params.get("rules")
    feature_columns = queue_scheduling_params.get("feature_columns", [])

    prepared = {
        col: _safe_float(features.get(col, 0.0))
        for col in feature_columns
    }

    score = _apply_score_rules(rules, prepared)

    if not np.isfinite(score):
        score = float(queue_scheduling_params.get("positive_rate", 1.0))

    return float(np.clip(score, 0.0, 1.0))


def _fit_queue_tree(X, y, max_depths_cv, min_samples_leaf, random_state, max_leaf_nodes: int = 64):
    if y.nunique() < 2:
        clf = DummyClassifier(strategy="prior", random_state=random_state)
        clf.fit(X, y)
        return clf

    depths = sorted({
        max(1, int(d))
        for d in (max_depths_cv or [2])
    })

    # 外层 Optuna 已经选定深度时，直接训练一棵树，不再做内部 GridSearchCV。
    if len(depths) == 1:
        clf = DecisionTreeClassifier(
            max_depth=depths[0],
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=int(min_samples_leaf),
            max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes else None,
        )
        clf.fit(X, y)
        return clf

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 1

    base = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
        min_samples_leaf=int(min_samples_leaf),
        max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes else None,
    )

    if min_class_count >= 3 and len(y) >= 20:
        try:
            grid = GridSearchCV(
                estimator=base,
                param_grid={"max_depth": depths},
                cv=min(3, min_class_count),
                scoring="neg_log_loss",
                n_jobs=-1,
            )
            grid.fit(X, y)
            return grid.best_estimator_
        except Exception:
            pass

    clf = DecisionTreeClassifier(
        max_depth=max(depths),
        random_state=random_state,
        class_weight="balanced",
        min_samples_leaf=int(min_samples_leaf),
        max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes else None,
    )
    clf.fit(X, y)
    return clf


def _decision_tree_to_score_rules(clf: DecisionTreeClassifier, feature_columns: list[str]):
    tree = clf.tree_
    classes = [int(c) for c in clf.classes_]
    positive_index = classes.index(1) if 1 in classes else None

    def build_node(node_id: int):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left == right:
            counts = tree.value[node_id][0].astype(float)
            total = float(counts.sum())
            if total <= 0 or positive_index is None:
                score = 0.0
            else:
                score = float(counts[positive_index] / total)
            return {
                "type": "leaf",
                "score": float(np.clip(score, 0.0, 1.0)),
                "n_node_samples": int(tree.n_node_samples[node_id]),
            }

        feature_index = tree.feature[node_id]
        return {
            "type": "node",
            "feature": feature_columns[feature_index],
            "threshold": float(tree.threshold[node_id]),
            "children": {
                "true": build_node(left),
                "false": build_node(right),
            },
        }

    return build_node(0)


def _apply_score_rules(rules: Dict[str, Any], features: Dict[str, float]) -> float:
    node = rules

    while isinstance(node, dict) and node.get("type") != "leaf":
        feature = node.get("feature")
        threshold = float(node.get("threshold", 0.0))
        value = _safe_float(features.get(feature, 0.0))
        if value <= threshold:
            node = node.get("children", {}).get("true")
        else:
            node = node.get("children", {}).get("false")
        if node is None:
            return 0.0

    if not isinstance(node, dict):
        return 0.0

    return float(node.get("score", 0.0))


def _compute_queue_features_for_candidate(
        df: pd.DataFrame,
        current_index: int,
        current_time,
        activity: str,
        candidate_resource: str,
        act_to_resources: dict,
) -> Dict[str, float]:
    """保留旧版精确扫描逻辑作为兜底，不作为默认路径。"""
    waiting = df[(df["enabled_t"] <= current_time) & (df["start_t"] > current_time)].copy()
    if current_index in waiting.index:
        waiting = waiting.drop(index=current_index)

    global_q = int(len(waiting))
    activity_q = int((waiting["transition_label"].astype(str) == str(activity)).sum()) if not waiting.empty else 0

    resource_q = 0
    resource_activity_q = 0
    for _, wrow in waiting.iterrows():
        w_activity = str(wrow.get("transition_label"))
        possible_resources = [str(r) for r in act_to_resources.get(w_activity, [])]
        if str(candidate_resource) in possible_resources:
            resource_q += 1
            if w_activity == str(activity):
                resource_activity_q += 1

    system_wip = int(((df["enabled_t"] <= current_time) & (df["end_t"] > current_time)).sum())

    raw = {
        "global_queue_length": global_q,
        "activity_queue_length": activity_q,
        "resource_queue_length": resource_q,
        "resource_activity_queue_length": resource_activity_q,
        "queue_pressure": max(activity_q, resource_q, resource_activity_q),
        "system_wip": system_wip,
    }

    return _post_process_queue_features(raw)


def _post_process_queue_features(
        raw_features: dict,
        max_queue_count: float = 8.0,
        max_wip_count: float = 30.0,
        disable_global_queue_features: bool = False,
) -> Dict[str, float]:
    def compress(value, max_count):
        value = max(0.0, min(float(value), float(max_count)))
        if max_count <= 0:
            return 0.0
        return float(np.log1p(value) / np.log1p(float(max_count)))

    if disable_global_queue_features:
        global_queue_value = 0.0
        system_wip_value = 0.0
    else:
        global_queue_value = compress(raw_features.get("global_queue_length", 0.0), max_queue_count)
        system_wip_value = compress(raw_features.get("system_wip", 0.0), max_wip_count)

    activity_q = compress(raw_features.get("activity_queue_length", 0.0), max_queue_count)
    resource_q = compress(raw_features.get("resource_queue_length", 0.0), max_queue_count)
    resource_activity_q = compress(raw_features.get("resource_activity_queue_length", 0.0), max_queue_count)

    return {
        "global_queue_length": global_queue_value,
        "activity_queue_length": activity_q,
        "resource_queue_length": resource_q,
        "resource_activity_queue_length": resource_activity_q,
        "queue_pressure": max(activity_q, resource_q, resource_activity_q),
        "system_wip": system_wip_value,
    }


def _prepare_valid_event_df(df_features: pd.DataFrame) -> pd.DataFrame:
    required = ["transition_label", "resource", "enabled_t", "start_t", "end_t"]
    if df_features is None or any(c not in df_features.columns for c in required):
        return pd.DataFrame(columns=required)

    df = df_features.copy()
    df = df[(~df["transition_label"].isna()) & (~df["resource"].isna())].copy()
    df["transition_label"] = df["transition_label"].astype(str)
    df["resource"] = df["resource"].astype(str)
    df["enabled_t"] = pd.to_datetime(df["enabled_t"], errors="coerce")
    df["start_t"] = pd.to_datetime(df["start_t"], errors="coerce")
    df["end_t"] = pd.to_datetime(df["end_t"], errors="coerce")
    df.dropna(subset=["enabled_t", "start_t", "end_t"], inplace=True)
    df.sort_values(["enabled_t", "start_t", "end_t"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _datetime_series_to_ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, errors="coerce").astype("int64").to_numpy(dtype=np.int64)


def _to_datetime_ns(value) -> np.int64:
    return np.int64(pd.Timestamp(value).value)


def _build_resource_intervals(df: pd.DataFrame) -> Dict[str, list]:
    intervals = {}
    for _, row in df.iterrows():
        r = str(row.get("resource"))
        intervals.setdefault(r, []).append((row.get("start_t"), row.get("end_t")))
    for r in intervals:
        intervals[r].sort(key=lambda x: x[0])
    return intervals


def _count_active_intervals(intervals: list, current_time) -> int:
    count = 0
    for start, end in intervals:
        if start <= current_time < end:
            count += 1
    return int(count)


def _estimate_candidate_resource_wait_min(
        intervals: list,
        current_time,
        current_start_time=None,
        current_index: int | None = None,
        capacity: int = 1,
) -> float:
    """
    Estimate the capacity-based waiting time a candidate resource would impose.

    The estimate mirrors the runtime capacity guard with capacity=1 by default.
    It is used as a feature, not as a training target, so it should stay simple
    and robust across logs.
    """
    try:
        current_time = pd.Timestamp(current_time)
    except Exception:
        return 0.0

    active_ends = []
    for item in intervals or []:
        try:
            if len(item) == 2:
                start, end = item
                idx = None
            else:
                idx, start, end = item[:3]
            if current_index is not None and idx == current_index:
                continue
            if start <= current_time < end:
                active_ends.append(end)
        except Exception:
            continue

    capacity = max(1, int(capacity))
    if len(active_ends) < capacity:
        return 0.0

    try:
        # Need enough completions for active workload to become < capacity.
        active_ends = sorted(active_ends)
        completions_needed = len(active_ends) - capacity + 1
        candidate_start = active_ends[completions_needed - 1]
        return max(0.0, float((candidate_start - current_time).total_seconds() / 60.0))
    except Exception:
        return 0.0


def _log_scale_minutes(value: float, cap_min: float = 1440.0) -> float:
    try:
        value = max(0.0, min(float(value), float(cap_min)))
    except Exception:
        value = 0.0
    if cap_min <= 0:
        return 0.0
    return float(np.log1p(value) / np.log1p(float(cap_min)))


def _activity_resource_frequency(df: pd.DataFrame) -> Dict[tuple, float]:
    if df is None or df.empty:
        return {}
    counts = df.groupby(["transition_label", "resource"]).size().to_dict()
    activity_totals = df.groupby("transition_label").size().to_dict()
    freq = {}
    for (activity, resource), count in counts.items():
        total = max(1, int(activity_totals.get(activity, 1)))
        freq[(str(activity), str(resource))] = float(count) / float(total)
    return freq


def _build_metadata(
        df: pd.DataFrame,
        train_df: Optional[pd.DataFrame],
        resources: list[str],
        activities: list[str],
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_negative_per_positive: int,
        activity_resource_frequency: dict,
) -> dict:
    n_candidate_pairs = int(len(train_df)) if train_df is not None else 0
    positive_rate = float(train_df["class"].mean()) if train_df is not None and not train_df.empty else 0.0
    return {
        "n_events": int(len(df)) if df is not None else 0,
        "n_candidate_pairs": n_candidate_pairs,
        "positive_rate": positive_rate,
        "resources": resources,
        "activities": activities,
        "queue_feature_names": QUEUE_FEATURE_NAMES,
        "label_data_attributes": label_data_attributes,
        "label_data_attributes_categorical": label_data_attributes_categorical,
        "values_categorical": values_categorical,
        "max_negative_per_positive": int(max_negative_per_positive),
        "legacy_dbps_models_removed": True,
        "modeling_note": (
            "This tree learns queue-aware resource allocation from task-resource "
            "candidate pairs. It does not determine which waiting task is dispatched "
            "next; that responsibility belongs to conditional_prioritization."
        ),
        "queue_explanation_target": (
            "resource-competition waiting should be represented by candidate_resource_wait_min "
            "and runtime capacity enforcement; external delay should only model residual waiting."
        ),
        "activity_resource_frequency": {
            f"{activity}|||{resource}": float(value)
            for (activity, resource), value in activity_resource_frequency.items()
        },
    }


def _empirical_params(
        reason: str,
        df: pd.DataFrame,
        resources: list[str],
        activities: list[str],
        activity_resource_frequency: dict,
        max_negative_per_positive: int,
) -> Dict[str, Any]:
    metadata = _build_metadata(
        df=df,
        train_df=None,
        resources=resources,
        activities=activities,
        label_data_attributes=[],
        label_data_attributes_categorical=[],
        values_categorical={},
        max_negative_per_positive=max_negative_per_positive,
        activity_resource_frequency=activity_resource_frequency,
    )
    metadata["empirical_reason"] = reason
    return {
        "model_type": "global_resource_aware_queue_scheduling_tree",
        "model_role": "queue_aware_resource_allocation",
        "model_kind": "empirical",
        "feature_columns": [],
        "rules": None,
        "positive_rate": 1.0,
        "n_samples": 0,
        "metadata": metadata,
        "fallback": _fallback_config(),
        "runtime_wait_penalty": 0.08,
        "resource_policy_candidates": RESOURCE_POLICY_CANDIDATES,
        "resource_policy_mode": "balanced",
        "selected_strategy": "balanced",
        "score_mixing_weights": RESOURCE_POLICY_CANDIDATES["balanced"],
        "hybrid_score_weights": RESOURCE_POLICY_CANDIDATES["balanced"],
        "wait_penalty_mode": "log_scaled",
        "wait_penalty_cap_min": 1440.0,
        "first_activity_can_queue": True,
        "enforce_selected_resource_capacity": True,
        "availability_score_mode": "ranked_earliest_start",
    }


def _skill_state_from_frequency(value: float) -> str:
    value = _safe_float(value)
    if value >= 0.34:
        return "high"
    if value > 0.0:
        return "medium"
    return "low"


def _fatigue_state_from_workload(value: float) -> str:
    value = _safe_float(value)
    if value <= 0:
        return "low"
    if value <= 1:
        return "medium"
    return "high"


def _coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X


def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def _fallback_config() -> dict:
    return {
        "missing_model": "activity_resource_frequency",
        "empty_candidate_resources": "first_candidate",
        "all_zero_scores": "uniform_random",
    }


def _fallback_params(reason: str) -> Dict[str, Any]:
    """Keep queue-aware allocation active using its deterministic fallback path."""
    return {
        "model_type": "global_resource_aware_queue_scheduling_tree",
        "model_role": "queue_aware_resource_allocation",
        "model_kind": "empirical",
        "reason": reason,
        "feature_columns": [],
        "rules": None,
        "positive_rate": 0.0,
        "n_samples": 0,
        "metadata": {},
        "fallback": _fallback_config(),
        "runtime_wait_penalty": 0.08,
        "resource_policy_candidates": RESOURCE_POLICY_CANDIDATES,
        "resource_policy_mode": "balanced",
        "selected_strategy": "balanced",
        "score_mixing_weights": RESOURCE_POLICY_CANDIDATES["balanced"],
        "hybrid_score_weights": RESOURCE_POLICY_CANDIDATES["balanced"],
        "wait_penalty_mode": "log_scaled",
        "wait_penalty_cap_min": 1440.0,
        "first_activity_can_queue": True,
        "enforce_selected_resource_capacity": True,
        "availability_score_mode": "ranked_earliest_start",
    }
