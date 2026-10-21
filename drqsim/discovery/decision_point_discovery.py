# drq/discovery/decision_point_discovery.py

from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from .tau_abstraction import visible_successors


# ---------------------------------------------------------------------------
# Basic transition / decision-point utilities
# ---------------------------------------------------------------------------

def transition_key(t) -> str:
    """
    返回 Petri 网变迁的稳定 key。

    优先使用 transition.name，因为 label 可能重复或为 None。
    """
    if t is None:
        return ""
    name = getattr(t, "name", None)
    if name is not None:
        return str(name)
    return str(t)


def transition_display_name(t) -> str:
    """
    返回用于展示的变迁名称。
    """
    if t is None:
        return ""
    label = getattr(t, "label", None)
    if label is not None:
        return str(label)
    name = getattr(t, "name", None)
    if name is not None:
        return str(name)
    return str(t)


def decision_point_signature(enabled_transitions) -> str:
    """
    将当前可使能变迁集合转换为稳定决策点签名。

    该函数保留原 Petri-net enabled-set 版本，作为兼容 fallback 使用。
    新版流程树引导模型会使用 pt::<node_path>::<operator> 形式的签名。
    """
    keys = sorted(
        [
            transition_key(t)
            for t in enabled_transitions
            if transition_key(t) != ""
        ]
    )
    return json.dumps(keys, ensure_ascii=False)


def parse_decision_point_signature(signature: str) -> List[str]:
    try:
        value = json.loads(signature)
        if isinstance(value, list):
            return [str(v) for v in value]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Petri-net structural decision-point discovery (preferred)
# ---------------------------------------------------------------------------

def place_key(place) -> str:
    """Return a stable identifier for a Petri-net place."""
    if place is None:
        return ""
    name = getattr(place, "name", None)
    if name is not None:
        return str(name)
    return str(place)


def petri_structural_decision_signature(place, branches) -> str:
    """Stable signature for one structural conflict/choice group.

    A decision point is anchored at the shared input place rather than at the
    complete runtime enabled-set.  This is important because concurrently
    enabled transitions that do not consume the same token are *not* branches
    of the same decision.
    """
    branch_keys = sorted({transition_key(t) for t in branches if transition_key(t)})
    return "petri_place::" + json.dumps(
        {"place": place_key(place), "branches": branch_keys},
        sort_keys=True,
        ensure_ascii=False,
    )


def _transition_preset_keys(transition) -> List[str]:
    keys = []
    for arc in getattr(transition, "in_arcs", []) or []:
        source = getattr(arc, "source", None)
        key = place_key(source)
        if key:
            keys.append(key)
    return sorted(set(keys))


def _structurally_reachable_visible_labels_from_transition(transition, max_tau_depth: int = 8) -> set[str]:
    """Resolve a Petri transition to observable activity futures.

    Version 2 uses a dedicated tau abstraction layer.  The returned labels are
    only used for learning/display; the original Petri transitions are kept
    for runtime firing.
    """
    try:
        return set(
            visible_successors(
                transition,
                max_depth=max_tau_depth,
            )
        )
    except Exception:
        return set()


def _visible_activity_branch_id(labels) -> Optional[str]:
    """Return a stable *observable* branch id from the first visible activities.

    For the common case there is exactly one first visible activity and the
    activity label itself is used as the decision-tree class.  When a tau
    branch can reach several first visible activities before any visible event
    is observed, the class is an activity-set id rather than an internal Petri
    transition id.  Empty sets are left unresolved and are not used as
    supervised activity-level classes.
    """
    normalized = sorted({str(x) for x in (labels or []) if str(x)})
    if not normalized:
        return None
    if len(normalized) == 1:
        return normalized[0]
    return "visible_activity_set::" + json.dumps(normalized, ensure_ascii=False)


def _visible_activity_branch_display(branch_id: str, labels) -> str:
    normalized = sorted({str(x) for x in (labels or []) if str(x)})
    if len(normalized) == 1:
        return normalized[0]
    if normalized:
        return " | ".join(normalized)
    return str(branch_id)


def extract_petri_structural_decision_points(net, max_tau_depth: int = 8) -> Dict[str, Dict[str, Any]]:
    """Extract Petri XOR/conflict points and resolve them to visible branches.

    The structural source of a decision remains the Petri net: a place with at
    least two outgoing transitions is a local conflict point.  The supervised
    class, however, is *not* the internal transition key.  Each structural
    branch is followed through a tau-closure until the first visible activity
    (or first-visible activity set) is reached.  Consequently one classifier is
    learned per XOR/conflict place and its outputs have activity semantics.

    ``structural_branches`` keeps the exact Petri transitions needed at runtime;
    ``branches`` contains the observable activity-branch classes used by the
    decision tree.  This separation is what prevents tau transition ids from
    leaking into the learned control-flow classes.
    """
    decision_points: Dict[str, Dict[str, Any]] = {}
    if net is None:
        return decision_points

    for place in sorted(list(getattr(net, "places", []) or []), key=lambda p: place_key(p)):
        outgoing = []
        seen = set()
        for arc in getattr(place, "out_arcs", []) or []:
            target = getattr(arc, "target", None)
            if target is None:
                continue
            key = transition_key(target)
            if not key or key in seen:
                continue
            seen.add(key)
            outgoing.append(target)

        if len(outgoing) < 2:
            continue

        outgoing = sorted(outgoing, key=transition_key)
        signature = petri_structural_decision_signature(place, outgoing)
        structural_branches = [transition_key(t) for t in outgoing]
        presets = {transition_key(t): _transition_preset_keys(t) for t in outgoing}
        preset_values = list(presets.values())
        is_free_choice = bool(preset_values) and all(p == preset_values[0] for p in preset_values[1:])
        simple_free_choice = is_free_choice and len(preset_values[0]) == 1

        structural_display_names = {
            transition_key(t): transition_display_name(t)
            for t in outgoing
        }
        structural_label_sets = {
            transition_key(t): sorted(
                _structurally_reachable_visible_labels_from_transition(
                    t, max_tau_depth=max_tau_depth
                )
            )
            for t in outgoing
        }
        branch_is_invisible = {
            transition_key(t): getattr(t, "label", None) is None
            for t in outgoing
        }

        transition_to_activity_branch: Dict[str, str] = {}
        activity_branch_to_transitions: Dict[str, List[str]] = {}
        activity_branch_label_sets: Dict[str, List[str]] = {}
        activity_branch_display_names: Dict[str, str] = {}
        unresolved_structural_branches: List[str] = []

        for t_key in structural_branches:
            labels = structural_label_sets.get(t_key, []) or []
            activity_branch = _visible_activity_branch_id(labels)
            if activity_branch is None:
                unresolved_structural_branches.append(t_key)
                continue
            transition_to_activity_branch[t_key] = activity_branch
            activity_branch_to_transitions.setdefault(activity_branch, []).append(t_key)
            activity_branch_label_sets.setdefault(activity_branch, sorted({str(x) for x in labels}))
            activity_branch_display_names[activity_branch] = _visible_activity_branch_display(
                activity_branch, labels
            )

        for key in list(activity_branch_to_transitions.keys()):
            activity_branch_to_transitions[key] = sorted(set(activity_branch_to_transitions[key]))

        activity_branches = sorted(activity_branch_to_transitions.keys())
        # Observable gateway abstraction:
        # structural conflict branches are kept for execution, while only
        # visible activity branches are exposed to the classifier.
        observable_gateway = {
            "gateway_id": signature,
            "visible_branches": activity_branches,
            "hidden_branches": list(unresolved_structural_branches),
            "transition_mapping": copy.deepcopy(transition_to_activity_branch),
        }

        is_observable_decision = (
            len(activity_branches) >= 2
            and len(unresolved_structural_branches) == 0
        )

        decision_points[signature] = {
            "signature": signature,
            "operator": "PETRI_STRUCTURAL_CONFLICT",
            "source_place": place_key(place),
            # Decision-tree classes: visible activity branches.
            "branches": activity_branches,
            "branch_display_names": activity_branch_display_names,
            "branch_label_sets": activity_branch_label_sets,
            # Petri execution branches: exact transitions enabled at the place.
            "structural_branches": structural_branches,
            "structural_branch_display_names": structural_display_names,
            "structural_branch_label_sets": structural_label_sets,
            "transition_to_activity_branch": transition_to_activity_branch,
            "activity_branch_to_transitions": activity_branch_to_transitions,
            "unresolved_structural_branches": unresolved_structural_branches,
            "is_observable_activity_decision": bool(is_observable_decision),
            "branch_is_invisible": branch_is_invisible,
            "branch_presets": presets,
            "is_free_choice": bool(is_free_choice),
            "is_simple_free_choice": bool(simple_free_choice),
            "has_invisible_branch": any(getattr(t, "label", None) is None for t in outgoing),
            "observable_gateway": observable_gateway,
        }

    return decision_points


def _petri_structural_feature_columns(
        df_features: pd.DataFrame,
        net_transition_labels: List[str],
        label_data_attributes: List[str],
) -> List[str]:
    feature_cols: List[str] = []

    for col in label_data_attributes or []:
        if col in df_features.columns and col not in feature_cols:
            feature_cols.append(col)

    for col in net_transition_labels or []:
        if col in df_features.columns and col not in feature_cols:
            feature_cols.append(col)

    optional_context_cols = [
        "prefix_length",
        "case_elapsed_time",
        "hour",
        "weekday",
        "loop_count_at_current_node",
        "gateway_visit_count",
        "loop_state",
        "previous_activity",
        "is_revisit",
    ]
    for col in optional_context_cols:
        if col in df_features.columns and col not in feature_cols:
            feature_cols.append(col)

    for col in df_features.columns:
        col_str = str(col)
        if (
            col_str.startswith("last_activity = ")
            or col_str.startswith("last_2_activities = ")
        ) and col_str not in feature_cols:
            feature_cols.append(col_str)

    return feature_cols


def build_petri_structural_training_df(
        df_features: pd.DataFrame,
        structural_decision_points: Dict[str, Dict[str, Any]],
        net_transition_labels: List[str],
        label_data_attributes: List[str],
        label_data_attributes_categorical: List[str],
        values_categorical: Dict[str, list],
) -> tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Build one supervised activity-branch sample per observable XOR choice.

    Petri replay still tells us which *structural transition* was taken.  That
    transition is immediately mapped to the corresponding first-visible
    activity branch before it becomes the target label.  Therefore the tree
    never learns tau ids; it learns business-visible branch outcomes.
    """
    feature_cols = _petri_structural_feature_columns(
        df_features=df_features,
        net_transition_labels=net_transition_labels,
        label_data_attributes=label_data_attributes,
    )
    # Third version: loop-aware observable gateway context features.
    # They are generated below and added dynamically when fitting the tree.
    for _col in [
        "gateway_visit_count",
        "previous_activity",
        "loop_state",
        "is_revisit",
    ]:
        if _col not in feature_cols:
            feature_cols.append(_col)

    metadata = copy.deepcopy(structural_decision_points or {})
    rows = []
    diagnostics = {
        "feature_rows": int(len(df_features)),
        "candidate_structural_decision_points": int(len(metadata)),
        "observable_activity_decision_points": int(sum(
            1 for node in metadata.values()
            if node.get("is_observable_activity_decision", False)
        )),
        "training_samples": 0,
        "rows_with_multiple_enabled_transitions": 0,
        "rows_with_structural_conflict_active": 0,
        "rows_with_observable_activity_conflict_active": 0,
        "rows_with_only_concurrency_or_unrelated_enabled": 0,
        "fired_transition_not_in_enabled_set": 0,
        "fired_transition_outside_active_conflict": 0,
        "activity_proxy_training_samples": 0,
        "ambiguous_activity_proxy_rows": 0,
        "unresolved_activity_branch_rows": 0,
        "active_subset_counts": {},
    }

    gateway_visit_counter = {}

    for _, row in df_features.iterrows():
        fired_transition = row.get("transition")
        enabled_transitions = row.get("prev_enabled_transitions")
        if fired_transition is None or enabled_transitions is None:
            continue

        try:
            enabled_list = list(enabled_transitions)
        except Exception:
            continue

        if len(enabled_list) >= 2:
            diagnostics["rows_with_multiple_enabled_transitions"] += 1

        fired_key = transition_key(fired_transition)
        enabled_keys = {transition_key(t) for t in enabled_list if transition_key(t)}
        if fired_key not in enabled_keys:
            diagnostics["fired_transition_not_in_enabled_set"] += 1

        active_conflicts = []
        for signature, node in metadata.items():
            structural_branches = [
                str(b) for b in (
                    node.get("structural_branches")
                    or node.get("branches")
                    or []
                )
            ]
            active_structural = [b for b in structural_branches if b in enabled_keys]
            if len(active_structural) < 2:
                continue

            diagnostics["rows_with_structural_conflict_active"] += 1
            transition_to_activity = node.get("transition_to_activity_branch", {}) or {}
            active_activity = sorted({
                str(transition_to_activity[b])
                for b in active_structural
                if b in transition_to_activity and transition_to_activity[b] is not None
            })
            if len(active_activity) < 2:
                continue

            diagnostics["rows_with_observable_activity_conflict_active"] += 1
            active_conflicts.append((
                str(signature), node, active_structural, active_activity
            ))

        if not active_conflicts and len(enabled_list) >= 2:
            diagnostics["rows_with_only_concurrency_or_unrelated_enabled"] += 1

        fired_visible_label = getattr(fired_transition, "label", None)
        if fired_visible_label is None and isinstance(fired_transition, str):
            fired_visible_label = str(fired_transition)
        if fired_visible_label is not None:
            fired_visible_label = str(fired_visible_label)

        matched = False
        for signature, node, active_structural, active_activity in active_conflicts:
            transition_to_activity = node.get("transition_to_activity_branch", {}) or {}
            target_activity = None
            used_activity_proxy = False

            # Preferred path: replay reports the exact structural transition.
            if fired_key in active_structural:
                target_activity = transition_to_activity.get(fired_key)

            # Compatibility path: some feature builders expose the visible event
            # while prev_enabled_transitions contains the preceding tau branches.
            if target_activity is None and fired_visible_label:
                activity_label_sets = node.get("branch_label_sets", {}) or {}
                proxy_matches = [
                    activity_branch
                    for activity_branch in active_activity
                    if fired_visible_label in {
                        str(x) for x in activity_label_sets.get(activity_branch, []) or []
                    }
                ]
                if len(proxy_matches) == 1:
                    target_activity = proxy_matches[0]
                    used_activity_proxy = True
                elif len(proxy_matches) > 1:
                    diagnostics["ambiguous_activity_proxy_rows"] += 1

            if target_activity is None or str(target_activity) not in set(active_activity):
                diagnostics["unresolved_activity_branch_rows"] += 1
                continue

            target_activity = str(target_activity)
            matched = True
            if used_activity_proxy:
                diagnostics["activity_proxy_training_samples"] += 1

            active_signature = json.dumps(sorted(active_activity), ensure_ascii=False)
            diagnostics["active_subset_counts"][active_signature] = (
                int(diagnostics["active_subset_counts"].get(active_signature, 0)) + 1
            )

            case_id = str(row.get("case_id", ""))
            visit_key = (case_id, str(signature))
            gateway_visit_counter[visit_key] = int(gateway_visit_counter.get(visit_key, 0)) + 1
            gateway_visit_count = gateway_visit_counter[visit_key]
            previous_activity = row.get("activity", row.get("transition_label", ""))
            if gateway_visit_count <= 1:
                loop_state = "first_visit"
            elif gateway_visit_count <= 3:
                loop_state = "revisit"
            else:
                loop_state = "deep_loop"

            is_revisit = 1.0 if gateway_visit_count > 1 else 0.0

            record = {
                "decision_point_signature": signature,
                "branch": target_activity,
                "cf_active_branch_count": float(len(active_activity)),
                "gateway_visit_count": float(gateway_visit_count),
                "previous_activity": previous_activity,
                "loop_state": loop_state,
                "is_revisit": is_revisit,
            }
            for activity_branch in node.get("branches", []) or []:
                record[f"cf_enabled::{activity_branch}"] = (
                    1.0 if str(activity_branch) in set(active_activity) else 0.0
                )
            for col in feature_cols:
                record[col] = row.get(col, 0.0)
            rows.append(record)

        if active_conflicts and not matched:
            diagnostics["fired_transition_outside_active_conflict"] += 1

    df = pd.DataFrame(rows)
    diagnostics["training_samples"] = int(len(df))

    if df.empty:
        return df, metadata, diagnostics

    # V4 small patch: make loop state usable by the numeric decision tree.
    # Previous versions stored loop_state as a string, which was later coerced
    # to zero by _coerce_features_to_numeric(). Keep the simple design but
    # expose interpretable binary context columns.
    if "loop_state" in df.columns:
        for state in ["first_visit", "revisit", "deep_loop"]:
            df[f"loop_state::{state}"] = (df["loop_state"] == state).astype(float)

    for attr in label_data_attributes_categorical or []:
        if attr not in df.columns:
            continue
        for value in (values_categorical or {}).get(attr, []):
            new_col = f"{attr} = {value}"
            df[new_col] = (df[attr] == value).astype(int)
        df.drop(columns=[attr], inplace=True)

    df = df.fillna(0.0)
    for col in df.columns:
        if col in ["decision_point_signature", "branch"]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df, metadata, diagnostics


def _uniform_structural_decision_model(signature: str, node: Dict[str, Any]) -> Dict[str, Any]:
    activity_branches = [str(b) for b in node.get("branches", []) or []]
    probs = _uniform_probs(activity_branches)
    return {
        "enabled": bool(node.get("is_observable_activity_decision", False)),
        "signature": str(signature),
        "operator": "PETRI_STRUCTURAL_CONFLICT",
        "source_place": node.get("source_place"),
        "path": None,
        # Observable classes used by the classifier.
        "branches": activity_branches,
        "branch_display_names": copy.deepcopy(node.get("branch_display_names", {})),
        "branch_label_sets": copy.deepcopy(node.get("branch_label_sets", {})),
        # Exact Petri transitions used to execute the chosen activity branch.
        "structural_branches": copy.deepcopy(node.get("structural_branches", [])),
        "structural_branch_display_names": copy.deepcopy(node.get("structural_branch_display_names", {})),
        "structural_branch_label_sets": copy.deepcopy(node.get("structural_branch_label_sets", {})),
        "transition_to_activity_branch": copy.deepcopy(node.get("transition_to_activity_branch", {})),
        "activity_branch_to_transitions": copy.deepcopy(node.get("activity_branch_to_transitions", {})),
        "unresolved_structural_branches": copy.deepcopy(node.get("unresolved_structural_branches", [])),
        "is_observable_activity_decision": bool(node.get("is_observable_activity_decision", False)),
        "branch_is_invisible": copy.deepcopy(node.get("branch_is_invisible", {})),
        "branch_presets": copy.deepcopy(node.get("branch_presets", {})),
        "is_free_choice": bool(node.get("is_free_choice", False)),
        "is_simple_free_choice": bool(node.get("is_simple_free_choice", False)),
        "has_invisible_branch": bool(node.get("has_invisible_branch", False)),
        "n_samples": 0,
        "empirical_probs": probs,
        "dominant_prob": max(probs.values()) if probs else 1.0,
        "model_kind": "empirical",
        "rules": None,
        "feature_columns": [],
        "max_depth": None,
        "unobserved_in_replay": True,
    }



def build_petri_structural_replay_audit(
        df_features: pd.DataFrame,
        structural_decision_points: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a JSON-safe per-decision-point baseline from aligned log replay.

    This is diagnostic only: it does not change training samples or routing.
    A visit is counted only when a structural conflict is active and replay
    actually fires one of that conflict's structural branches.  Unresolved
    silent branches are deliberately retained under ``__UNRESOLVED__`` labels.
    """
    if df_features is None or df_features.empty:
        return {
            "total_cases": 0,
            "decision_points": {},
            "rows": 0,
        }

    total_cases = int(df_features.get("case_id", pd.Series(dtype=object)).nunique())
    visit_counts: Counter = Counter()
    per_dp: Dict[str, Dict[str, Any]] = {}

    def _bucket(idx: int) -> str:
        if idx <= 1:
            return "1"
        if idx == 2:
            return "2"
        if idx == 3:
            return "3"
        return "4+"

    for signature, node in (structural_decision_points or {}).items():
        per_dp[str(signature)] = {
            "signature": str(signature),
            "source_place": node.get("source_place"),
            "structural_branches": [str(x) for x in node.get("structural_branches", []) or []],
            "structural_branch_display_names": copy.deepcopy(node.get("structural_branch_display_names", {})),
            "structural_branch_label_sets": copy.deepcopy(node.get("structural_branch_label_sets", {})),
            "transition_to_activity_branch": copy.deepcopy(node.get("transition_to_activity_branch", {})),
            "unresolved_structural_branches": [str(x) for x in node.get("unresolved_structural_branches", []) or []],
            "is_observable_activity_decision": bool(node.get("is_observable_activity_decision", False)),
            "is_free_choice": bool(node.get("is_free_choice", False)),
            "has_invisible_branch": bool(node.get("has_invisible_branch", False)),
            "active_conflict_rows": 0,
            "fired_inside_conflict_rows": 0,
            "fired_outside_conflict_rows": 0,
            "unresolved_fired_rows": 0,
            "cases_reached": 0,
            "reach_rate": 0.0,
            "total_visits": 0,
            "visits_per_case": 0.0,
            "mean_visits_when_reached": 0.0,
            "max_visits_in_case": 0,
            "structural_branch_counts": {},
            "activity_branch_counts": {},
            "activity_branch_freq": {},
            "branch_by_visit": {},
        }

    reached_cases: Dict[str, set] = {str(sig): set() for sig in (structural_decision_points or {})}
    per_case_visits: Counter = Counter()

    for _, row in df_features.iterrows():
        fired_transition = row.get("transition")
        enabled_transitions = row.get("prev_enabled_transitions")
        if fired_transition is None or enabled_transitions is None:
            continue
        try:
            enabled_list = list(enabled_transitions)
        except Exception:
            continue

        fired_key = transition_key(fired_transition)
        enabled_keys = {transition_key(t) for t in enabled_list if transition_key(t)}
        case_id = str(row.get("case_id", ""))

        for signature, node in (structural_decision_points or {}).items():
            signature = str(signature)
            dp = per_dp[signature]
            structural = [str(x) for x in node.get("structural_branches", []) or []]
            active = [b for b in structural if b in enabled_keys]
            if len(active) < 2:
                continue

            dp["active_conflict_rows"] += 1
            if fired_key not in active:
                dp["fired_outside_conflict_rows"] += 1
                continue

            dp["fired_inside_conflict_rows"] += 1
            key = (case_id, signature)
            visit_counts[key] += 1
            visit_index = int(visit_counts[key])
            per_case_visits[key] = visit_index
            reached_cases[signature].add(case_id)

            transition_to_activity = node.get("transition_to_activity_branch", {}) or {}
            activity_branch = transition_to_activity.get(fired_key)
            if activity_branch is None:
                dp["unresolved_fired_rows"] += 1
                activity_branch = f"__UNRESOLVED__::{fired_key}"
            activity_branch = str(activity_branch)

            sbc = dp["structural_branch_counts"]
            sbc[fired_key] = int(sbc.get(fired_key, 0)) + 1
            abc = dp["activity_branch_counts"]
            abc[activity_branch] = int(abc.get(activity_branch, 0)) + 1

            bucket = _bucket(visit_index)
            by_visit = dp["branch_by_visit"].setdefault(bucket, {})
            by_visit[activity_branch] = int(by_visit.get(activity_branch, 0)) + 1

    for signature, dp in per_dp.items():
        case_counts = [
            int(v) for (case_id, sig), v in per_case_visits.items()
            if sig == signature
        ]
        total_visits = int(sum(case_counts))
        cases_reached = int(len(reached_cases.get(signature, set())))
        dp["cases_reached"] = cases_reached
        dp["reach_rate"] = float(cases_reached / total_cases) if total_cases else 0.0
        dp["total_visits"] = total_visits
        dp["visits_per_case"] = float(total_visits / total_cases) if total_cases else 0.0
        dp["mean_visits_when_reached"] = (
            float(total_visits / cases_reached) if cases_reached else 0.0
        )
        dp["max_visits_in_case"] = int(max(case_counts)) if case_counts else 0

        total_branch = float(sum(dp["activity_branch_counts"].values()))
        dp["activity_branch_freq"] = {
            str(branch): float(count / total_branch) if total_branch else 0.0
            for branch, count in dp["activity_branch_counts"].items()
        }
        for bucket, counts in list(dp["branch_by_visit"].items()):
            denom = float(sum(counts.values()))
            dp["branch_by_visit"][bucket] = {
                "counts": {str(k): int(v) for k, v in counts.items()},
                "freq": {
                    str(k): float(v / denom) if denom else 0.0
                    for k, v in counts.items()
                },
                "n": int(denom),
            }

    return {
        "total_cases": total_cases,
        "rows": int(len(df_features)),
        "decision_points": per_dp,
    }


def explain_petri_structural_decisions(
        decision_point_params: Dict[str, Any],
        enabled_transitions,
        features: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return runtime explanations for currently active structural conflicts.

    The function is side-effect free and consumes no random numbers.  It is
    intended for simulator debugging and therefore reports disabled/unresolved
    decision points instead of silently skipping them.
    """
    enabled_list = list(enabled_transitions or [])
    enabled_keys = {transition_key(t) for t in enabled_list if transition_key(t)}
    if not isinstance(decision_point_params, dict):
        return []

    rows: List[Dict[str, Any]] = []
    decision_points = decision_point_params.get("decision_points", {}) or {}
    for signature, model in decision_points.items():
        if not isinstance(model, dict):
            continue
        if str(model.get("operator", "")).upper() != "PETRI_STRUCTURAL_CONFLICT":
            continue
        structural = [str(x) for x in model.get("structural_branches", []) or []]
        active_structural = [b for b in structural if b in enabled_keys]
        if len(active_structural) < 2:
            continue

        transition_to_activity = {
            str(k): str(v) for k, v in (model.get("transition_to_activity_branch", {}) or {}).items()
            if v is not None
        }
        unresolved_active = [b for b in active_structural if b not in transition_to_activity]
        activity_to_transitions: Dict[str, List[str]] = {}
        for t_key in active_structural:
            branch = transition_to_activity.get(t_key)
            if branch is not None:
                activity_to_transitions.setdefault(str(branch), []).append(t_key)
        active_activity = sorted(activity_to_transitions.keys())

        local_features = dict(features or {})
        local_features["cf_active_branch_count"] = float(len(active_activity))
        for branch in [str(x) for x in model.get("branches", []) or []]:
            local_features[f"cf_enabled::{branch}"] = 1.0 if branch in active_activity else 0.0

        model_enabled = bool(model.get("enabled", True))
        raw_probs: Dict[str, float] = {}
        active_probs: Dict[str, float] = {}
        runtime_policy = "learned_or_empirical"
        if not model_enabled:
            runtime_policy = "model_disabled_neutral_weights"
        elif len(active_activity) < 2:
            runtime_policy = "no_observable_activity_choice"
        else:
            raw_probs = _model_branch_probs(model=model, features=local_features)
            if not raw_probs:
                raw_probs = {str(k): float(v) for k, v in (model.get("empirical_probs", {}) or {}).items()}
            active_probs = _normalize_probs({
                branch: max(0.0, float(raw_probs.get(branch, 0.0)))
                for branch in active_activity
            })
            if not active_probs:
                active_probs = _uniform_probs(active_activity)
                runtime_policy = "uniform_activity_fallback"

        rows.append({
            "decision_point_signature": str(signature),
            "source_place": model.get("source_place"),
            "model_enabled": model_enabled,
            "model_kind": model.get("model_kind"),
            "n_samples": int(model.get("n_samples", 0) or 0),
            "feature_columns": [str(x) for x in model.get("feature_columns", []) or []],
            "structural_branches": structural,
            "active_structural_branches": active_structural,
            "active_activity_branches": active_activity,
            "unresolved_structural_branches": [str(x) for x in model.get("unresolved_structural_branches", []) or []],
            "unresolved_active_structural_branches": unresolved_active,
            "transition_to_activity_branch": transition_to_activity,
            "predicted_activity_probs": {str(k): float(v) for k, v in active_probs.items()},
            "runtime_policy": runtime_policy,
            "is_free_choice": bool(model.get("is_free_choice", False)),
            "has_invisible_branch": bool(model.get("has_invisible_branch", False)),
        })

    return rows

def discover_petri_structural_branch_models(
        net,
        df_features: pd.DataFrame,
        net_transition_labels: List[str],
        max_depths_cv: List[int] = None,
        label_data_attributes: List[str] = None,
        label_data_attributes_categorical: List[str] = None,
        values_categorical: Dict[str, list] = None,
        min_samples_per_decision_point: int = 30,
        dominance_threshold: float = 0.95,
        random_state: int = 72,
        structure_source: str = "petri_net",
        max_tau_depth: int = 8,
) -> Dict[str, Any]:
    """Preferred control-flow discovery for both uploaded PNML and IM nets.

    The Petri net is the single source of truth for decision-point structure.
    Event-log replay/features are used only to learn conditional branch
    probabilities.  This keeps uploaded PNML decisions intact and also makes
    the no-PNML path consistent: IM -> Petri net -> the same extractor.
    """
    if max_depths_cv is None:
        max_depths_cv = [1, 2, 3]
    max_depths_cv = list(max_depths_cv)
    net_transition_labels = net_transition_labels or []
    label_data_attributes = label_data_attributes or []
    label_data_attributes_categorical = label_data_attributes_categorical or []
    values_categorical = values_categorical or {}

    required_cols = ["transition", "prev_enabled_transitions"]
    missing_cols = [c for c in required_cols if c not in df_features.columns]
    if missing_cols:
        result = _disabled_params(
            reason=f"df_features misses required columns: {missing_cols}"
        )
        result["model_type"] = "petri_structural_conflict_decision_point_branch_tree"
        result["metadata"] = {"structure_source": str(structure_source)}
        return result

    structural_points = extract_petri_structural_decision_points(
        net=net,
        max_tau_depth=max_tau_depth,
    )

    if not structural_points:
        result = _disabled_params(
            reason="No Petri-net place with multiple outgoing transitions was found."
        )
        result["model_type"] = "petri_structural_conflict_decision_point_branch_tree"
        result["metadata"] = {
            "structure_source": str(structure_source),
            "candidate_structural_decision_points": 0,
        }
        return result

    replay_audit = build_petri_structural_replay_audit(
        df_features=df_features,
        structural_decision_points=structural_points,
    )

    training_df, metadata, diagnostics = build_petri_structural_training_df(
        df_features=df_features,
        structural_decision_points=structural_points,
        net_transition_labels=net_transition_labels,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
    )

    if not training_df.empty:
        params = _fit_decision_point_models_from_training_df(
            training_df=training_df,
            metadata=metadata,
            max_depths_cv=max_depths_cv,
            min_samples_per_decision_point=min_samples_per_decision_point,
            dominance_threshold=dominance_threshold,
            random_state=random_state,
            model_type="petri_structural_conflict_decision_point_branch_tree",
            description=(
                "Petri-net XOR/conflict decision-point model with visible activity "
                "branches. Decision points come from shared input places; replay "
                "maps the fired Petri transition to its first-visible activity "
                "branch before fitting one tree per choice point."
            ),
            extra_metadata={
                "discovery_source": "petri_net_structural_conflict",
                "structure_source": str(structure_source),
                "decision_point_definition": "shared_input_place_with_multiple_outgoing_transitions",
                "runtime_policy": "visible_activity_probability_mapped_to_enabled_petri_transition",
                "max_tau_depth_for_diagnostics": int(max_tau_depth),
                "diagnostics": diagnostics,
            },
        )
    else:
        params = {
            "enabled": True,
            "model_type": "petri_structural_conflict_decision_point_branch_tree",
            "description": (
                "Petri-net structural conflict decision points with uniform "
                "branch probabilities because replay produced no direct samples."
            ),
            "num_decision_points": 0,
            "decision_points": {},
            "metadata": {
                "max_depths_cv": max_depths_cv,
                "min_samples_per_decision_point": int(min_samples_per_decision_point),
                "dominance_threshold": float(dominance_threshold),
                "discovery_source": "petri_net_structural_conflict",
                "structure_source": str(structure_source),
                "decision_point_definition": "shared_input_place_with_multiple_outgoing_transitions",
                "runtime_policy": "visible_activity_probability_mapped_to_enabled_petri_transition",
                "max_tau_depth_for_diagnostics": int(max_tau_depth),
                "diagnostics": diagnostics,
            },
            "fallback": {
                "single_enabled_transition": "fire_directly",
                "missing_decision_point": "uniform_over_enabled_transitions",
                "missing_feature": 0.0,
            },
        }

    # Keep *all* PNML/IM structural decision points in the model, even when a
    # particular branch group was not directly observed in replay.  Silently
    # dropping such points would make the learned control-flow structure differ
    # from the supplied PNML.
    models = params.setdefault("decision_points", {})
    observed_count = 0
    for signature, node in metadata.items():
        if signature not in models:
            models[signature] = _uniform_structural_decision_model(signature, node)
        else:
            observed_count += 1
            model = models[signature]
            model["source_place"] = node.get("source_place")
            model["structural_branches"] = copy.deepcopy(node.get("structural_branches", []))
            model["structural_branch_display_names"] = copy.deepcopy(node.get("structural_branch_display_names", {}))
            model["structural_branch_label_sets"] = copy.deepcopy(node.get("structural_branch_label_sets", {}))
            model["transition_to_activity_branch"] = copy.deepcopy(node.get("transition_to_activity_branch", {}))
            model["activity_branch_to_transitions"] = copy.deepcopy(node.get("activity_branch_to_transitions", {}))
            model["unresolved_structural_branches"] = copy.deepcopy(node.get("unresolved_structural_branches", []))
            model["is_observable_activity_decision"] = bool(node.get("is_observable_activity_decision", False))
            model["branch_is_invisible"] = copy.deepcopy(node.get("branch_is_invisible", {}))
            model["branch_presets"] = copy.deepcopy(node.get("branch_presets", {}))
            model["is_free_choice"] = bool(node.get("is_free_choice", False))
            model["is_simple_free_choice"] = bool(node.get("is_simple_free_choice", False))
            model["has_invisible_branch"] = bool(node.get("has_invisible_branch", False))
            model["unobserved_in_replay"] = False

    params["num_decision_points"] = int(len(models))
    params.setdefault("metadata", {})["candidate_structural_decision_points"] = int(len(metadata))
    params["metadata"]["observed_structural_decision_points"] = int(observed_count)
    params["metadata"]["unobserved_structural_decision_points"] = int(len(metadata) - observed_count)
    params["metadata"]["free_choice_decision_points"] = int(
        sum(1 for node in metadata.values() if node.get("is_free_choice", False))
    )
    params["metadata"]["non_free_choice_decision_points"] = int(
        sum(1 for node in metadata.values() if not node.get("is_free_choice", False))
    )
    params["metadata"]["invisible_branch_decision_points"] = int(
        sum(1 for node in metadata.values() if node.get("has_invisible_branch", False))
    )
    params["metadata"]["observable_activity_decision_points"] = int(
        sum(1 for node in metadata.values() if node.get("is_observable_activity_decision", False))
    )
    params["metadata"]["unresolved_activity_decision_points"] = int(
        sum(1 for node in metadata.values() if not node.get("is_observable_activity_decision", False))
    )
    params.setdefault("metadata", {})["control_flow_replay_audit"] = replay_audit
    params.setdefault("fallback", {}).update({
        "single_enabled_transition": "fire_directly",
        "no_active_structural_conflict": "neutral_uniform_concurrency",
        "missing_feature": 0.0,
    })
    return params


def compute_petri_structural_transition_weights(
        decision_point_params: Dict[str, Any],
        enabled_transitions,
        features: Dict[str, Any],
) -> Dict[str, float]:
    """Map visible-activity branch probabilities back to enabled Petri transitions.

    The classifier predicts observable activity branches.  Petri execution still
    fires an actually enabled transition (often tau).  For each active XOR point
    the predicted probability of an activity branch is distributed over the
    enabled structural transitions that implement that branch.  Unrelated
    concurrent transitions keep neutral weight 1.0.
    """
    enabled_list = list(enabled_transitions or [])
    enabled_by_key = {transition_key(t): t for t in enabled_list if transition_key(t)}
    weights = {key: 1.0 for key in enabled_by_key}

    if not isinstance(decision_point_params, dict) or not decision_point_params.get("enabled", False):
        return weights

    decision_points = decision_point_params.get("decision_points", {}) or {}
    for _, model in decision_points.items():
        if not isinstance(model, dict) or not model.get("enabled", True):
            continue
        if str(model.get("operator", "")).upper() != "PETRI_STRUCTURAL_CONFLICT":
            continue

        # New activity-level representation.
        structural_branches = [
            str(b) for b in (
                model.get("structural_branches")
                or []
            )
        ]
        transition_to_activity = {
            str(k): str(v)
            for k, v in (model.get("transition_to_activity_branch", {}) or {}).items()
            if v is not None
        }

        if structural_branches and transition_to_activity:
            active_structural = [b for b in structural_branches if b in enabled_by_key]
            if len(active_structural) < 2:
                continue

            activity_to_active_transitions: Dict[str, List[str]] = {}
            for t_key in active_structural:
                activity_branch = transition_to_activity.get(t_key)
                if activity_branch is None:
                    continue
                activity_to_active_transitions.setdefault(activity_branch, []).append(t_key)

            active_activity = sorted(activity_to_active_transitions.keys())
            if len(active_activity) < 2:
                # Structurally several transitions may be enabled but they are
                # observationally the same branch; no activity-level decision.
                continue

            local_features = dict(features or {})
            local_features["cf_active_branch_count"] = float(len(active_activity))
            all_activity_branches = [str(b) for b in model.get("branches", []) or []]
            for branch in all_activity_branches:
                local_features[f"cf_enabled::{branch}"] = 1.0 if branch in active_activity else 0.0

            probs = _model_branch_probs(model=model, features=local_features)
            if not probs:
                probs = model.get("empirical_probs", {}) or {}
            active_probs = _normalize_probs({
                branch: max(0.0, float(probs.get(branch, 0.0)))
                for branch in active_activity
            })
            if not active_probs:
                active_probs = _uniform_probs(active_activity)

            # Convert P(activity branch) to transition weights.  If several
            # internal tau transitions implement the same visible branch, split
            # that activity mass uniformly among those currently enabled paths.
            n_active_transitions = float(len(active_structural))
            for activity_branch, t_keys in activity_to_active_transitions.items():
                p_activity = float(active_probs.get(activity_branch, 0.0))
                per_transition = p_activity / max(len(t_keys), 1)
                factor = max(1e-12, per_transition * n_active_transitions)
                for t_key in t_keys:
                    weights[t_key] = max(1e-12, float(weights.get(t_key, 1.0)) * factor)
            continue

        # Backward compatibility for older saved transition-level parameters.
        branches = [str(b) for b in model.get("branches", []) or []]
        active = [b for b in branches if b in enabled_by_key]
        if len(active) < 2:
            continue

        local_features = dict(features or {})
        local_features["cf_active_branch_count"] = float(len(active))
        for branch in branches:
            local_features[f"cf_enabled::{branch}"] = 1.0 if branch in active else 0.0

        probs = _model_branch_probs(model=model, features=local_features)
        if not probs:
            probs = model.get("empirical_probs", {}) or {}
        active_probs = _normalize_probs({
            branch: max(0.0, float(probs.get(branch, 0.0)))
            for branch in active
        })
        if not active_probs:
            active_probs = _uniform_probs(active)

        scale = float(len(active))
        for branch in active:
            factor = max(1e-12, float(active_probs.get(branch, 0.0)) * scale)
            weights[branch] = max(1e-12, float(weights.get(branch, 1.0)) * factor)

    return weights


def sample_petri_structural_branch(
        decision_point_params: Dict[str, Any],
        enabled_transitions,
        features: Dict[str, Any],
        rng: Optional[random.Random] = None,
):
    """Sample visible branch first, then map back to concrete Petri transition.

    The classifier semantics are observable activities. Execution semantics remain
    exact Petri transitions, including tau transitions.
    """
    enabled_list = list(enabled_transitions or [])
    rng = rng or random
    if not enabled_list:
        return None
    if len(enabled_list) == 1:
        return enabled_list[0]

    weights_by_key = compute_petri_structural_transition_weights(
        decision_point_params=decision_point_params,
        enabled_transitions=enabled_list,
        features=features,
    )
    # Observable branch aggregation: transitions belonging to the same visible
    # activity branch share one probability mass.
    activity_mapping = {}
    decision_points = (decision_point_params or {}).get("decision_points", {}) if isinstance(decision_point_params, dict) else {}
    for _, model in (decision_points or {}).items():
        mapping = model.get("transition_to_activity_branch", {}) if isinstance(model, dict) else {}
        for t_key, branch in mapping.items():
            activity_mapping[str(t_key)] = str(branch)

    weights = [max(0.0, float(weights_by_key.get(transition_key(t), 1.0))) for t in enabled_list]
    total = float(sum(weights))
    if total <= 0.0:
        return rng.choice(enabled_list)

    threshold = rng.random() * total
    cumulative = 0.0
    for transition, weight in zip(enabled_list, weights):
        cumulative += weight
        if threshold <= cumulative:
            return transition
    return enabled_list[-1]


# ---------------------------------------------------------------------------
# Legacy Petri-net enabled-set decision-point discovery
# ---------------------------------------------------------------------------

def discover_decision_point_branch_models(
        df_features: pd.DataFrame,
        net_transition_labels: List[str],
        max_depths_cv: List[int] = None,
        label_data_attributes: List[str] = None,
        label_data_attributes_categorical: List[str] = None,
        values_categorical: Dict[str, list] = None,
        min_samples_per_decision_point: int = 30,
        dominance_threshold: float = 0.95,
        random_state: int = 72
) -> Dict[str, Any]:
    """
    发现 Petri 网 enabled-set 层面的决策点分支模型。

    说明：该函数保留为兼容 fallback。新版推荐使用
    discover_process_tree_guided_branch_models，即结构识别基于流程树，
    运行执行仍基于 Petri 网。
    """
    if max_depths_cv is None:
        max_depths_cv = [1, 2, 3]

    max_depths_cv = list(max_depths_cv)
    label_data_attributes = label_data_attributes or []
    label_data_attributes_categorical = label_data_attributes_categorical or []
    values_categorical = values_categorical or {}

    required_cols = ["transition", "prev_enabled_transitions"]
    missing_cols = [c for c in required_cols if c not in df_features.columns]
    if missing_cols:
        return _disabled_params(
            reason=f"df_features misses required columns: {missing_cols}"
        )

    training_df, metadata = build_decision_point_training_df(
        df_features=df_features,
        net_transition_labels=net_transition_labels,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical
    )

    if training_df.empty:
        return _disabled_params(
            reason="No decision-point samples are available."
        )

    return _fit_decision_point_models_from_training_df(
        training_df=training_df,
        metadata=metadata,
        max_depths_cv=max_depths_cv,
        min_samples_per_decision_point=min_samples_per_decision_point,
        dominance_threshold=dominance_threshold,
        random_state=random_state,
        model_type="decision_point_branch_tree",
        description=(
            "Petri-net enabled-set decision-point branch model. "
            "This model is kept as a backward-compatible fallback."
        ),
        extra_metadata={
            "discovery_source": "petri_enabled_set",
            "label_data_attributes": label_data_attributes,
            "label_data_attributes_categorical": label_data_attributes_categorical,
            "values_categorical": values_categorical,
        },
    )


def build_decision_point_training_df(
        df_features: pd.DataFrame,
        net_transition_labels: List[str],
        label_data_attributes: List[str],
        label_data_attributes_categorical: List[str],
        values_categorical: Dict[str, list]
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    构造 Petri-net enabled-set 版本的决策点训练数据。

    每一行表示某个可使能变迁集合，即一个 Petri 网层面的选择位置。
    """
    feature_cols = []

    for col in label_data_attributes:
        if col in df_features.columns:
            feature_cols.append(col)

    for col in net_transition_labels:
        if col in df_features.columns:
            feature_cols.append(col)

    # Optional enhanced context features. 这些列并不强制要求存在；
    # 如果 build_df_features 或后续预处理已经提供，则 Petri enabled-set 决策树直接使用。
    optional_context_cols = [
        "prefix_length",
        "case_elapsed_time",
        "hour",
        "weekday",
        "loop_count_at_current_node",
        "gateway_visit_count",
        "loop_state",
        "previous_activity",
        "is_revisit",
    ]
    for col in optional_context_cols:
        if col in df_features.columns and col not in feature_cols:
            feature_cols.append(col)

    for col in df_features.columns:
        col_str = str(col)
        if (
            col_str.startswith("last_activity = ")
            or col_str.startswith("last_2_activities = ")
        ) and col_str not in feature_cols:
            feature_cols.append(col_str)

    rows = []
    metadata = {}

    for _, row in df_features.iterrows():
        fired_transition = row.get("transition")
        enabled_transitions = row.get("prev_enabled_transitions")

        if fired_transition is None or enabled_transitions is None:
            continue

        try:
            enabled_list = list(enabled_transitions)
        except Exception:
            continue

        if len(enabled_list) < 2:
            continue

        fired_key = transition_key(fired_transition)
        enabled_keys = [transition_key(t) for t in enabled_list]

        if fired_key not in enabled_keys:
            continue

        signature = decision_point_signature(enabled_list)

        if signature not in metadata:
            branches = sorted(set(enabled_keys))
            branch_display_names = {
                transition_key(t): transition_display_name(t)
                for t in enabled_list
            }

            metadata[signature] = {
                "signature": signature,
                "operator": "PETRI_ENABLED_SET",
                "branches": branches,
                "branch_display_names": branch_display_names,
                "branch_label_sets": {},
            }

        record = {
            "decision_point_signature": signature,
            "branch": fired_key
        }

        for col in feature_cols:
            record[col] = row.get(col, 0.0)

        rows.append(record)

    df = pd.DataFrame(rows)

    if df.empty:
        return df, metadata

    for attr in label_data_attributes_categorical:
        if attr not in df.columns:
            continue

        for value in values_categorical.get(attr, []):
            new_col = f"{attr} = {value}"
            df[new_col] = (df[attr] == value).astype(int)

        df.drop(columns=[attr], inplace=True)

    for col in df.columns:
        if col in ["decision_point_signature", "branch"]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df, metadata


# ---------------------------------------------------------------------------
# New process-tree-guided decision-point discovery
# ---------------------------------------------------------------------------

def discover_process_tree_guided_branch_models(
        log,
        process_tree,
        df_features: Optional[pd.DataFrame] = None,
        net_transition_labels: List[str] = None,
        max_depths_cv: List[int] = None,
        label_data_attributes: List[str] = None,
        label_data_attributes_categorical: List[str] = None,
        values_categorical: Dict[str, list] = None,
        min_samples_per_decision_point: int = 30,
        dominance_threshold: float = 0.95,
        random_state: int = 72,
        include_petri_fallback: bool = True,
) -> Dict[str, Any]:
    """
    流程树筛选的 Petri enabled-set 控制流分支模型。

    目标结构是：Petri 网负责执行语义，流程树负责发现/筛选真正的 XOR 决策点。
    因此，本函数不再把所有 Petri enabled-set 都作为决策点，也不再训练
    流程树 branch -> Petri transition 的二次映射模型，而是：

    1. 从 IM 流程树中提取安全的 XOR 节点；
    2. 基于 Petri replay 得到的 df_features 构造候选
       prev_enabled_transitions -> fired transition 样本；
    3. 只保留那些当前 enabled-set 覆盖某个流程树 XOR 节点至少两个分支的样本；
    4. 决策树的预测目标仍然是 Petri transition key；
    5. 运行时仍然按当前 Petri enabled-set 签名调用 sample_decision_point_branch，
       不进行流程树 branch 到 Petri transition 的运行时映射。

    include_petri_fallback=True 时，会额外训练一个完整 Petri enabled-set fallback，
    仅用于当前 enabled-set 没有被流程树筛选为主决策点时的技术性选择兜底。
    该 fallback 不会出现在主 decision_points 列表中，前端可编辑的决策点仍来自
    流程树 XOR 筛选结果。
    """
    if max_depths_cv is None:
        max_depths_cv = [1, 2, 3]

    max_depths_cv = list(max_depths_cv)
    net_transition_labels = net_transition_labels or []
    label_data_attributes = label_data_attributes or []
    label_data_attributes_categorical = label_data_attributes_categorical or []
    values_categorical = values_categorical or {}

    if df_features is None:
        return _disabled_params(
            reason=(
                "Process-tree-filtered Petri decision-point discovery requires "
                "df_features with transition and prev_enabled_transitions columns."
            )
        )

    process_tree_available = process_tree is not None and process_tree is not False
    decision_nodes: Dict[str, Dict[str, Any]] = {}
    if process_tree_available:
        try:
            decision_nodes = extract_process_tree_decision_nodes(process_tree)
        except Exception:
            decision_nodes = {}

    # 先准备完整 Petri fallback。主模型仍由流程树筛选，fallback 只处理技术性
    # 多 enabled-set 或流程树没有覆盖到的 Petri 选择。
    petri_fallback = None
    if include_petri_fallback:
        try:
            petri_fallback = discover_decision_point_branch_models(
                df_features=df_features,
                net_transition_labels=net_transition_labels,
                max_depths_cv=max_depths_cv,
                label_data_attributes=label_data_attributes,
                label_data_attributes_categorical=label_data_attributes_categorical,
                values_categorical=values_categorical,
                min_samples_per_decision_point=min_samples_per_decision_point,
                dominance_threshold=dominance_threshold,
                random_state=random_state,
            )
            if isinstance(petri_fallback, dict):
                petri_fallback.setdefault("metadata", {})
                petri_fallback["metadata"].update({
                    "fallback_role": "technical_petri_enabled_set_fallback_for_non_xor_states",
                    "hidden_from_primary_decision_point_editor": True,
                })
        except Exception:
            petri_fallback = None

    if not decision_nodes:
        if isinstance(petri_fallback, dict) and petri_fallback.get("enabled", False):
            petri_fallback.setdefault("metadata", {})
            petri_fallback["metadata"].update({
                "discovery_source": "petri_enabled_set_fallback_no_process_tree_xor_nodes",
                "process_tree_role": "unavailable_or_no_safe_xor_nodes",
                "primary_runtime_model": "petri_enabled_set_decision_tree",
                "fallback_reason": "no_process_tree_xor_nodes_found",
            })
            return petri_fallback
        return _disabled_params(reason="No safe XOR decision nodes were found in the process tree.")

    training_df, metadata = build_process_tree_filtered_petri_training_df(
        df_features=df_features,
        decision_nodes=decision_nodes,
        net_transition_labels=net_transition_labels,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
    )

    if training_df.empty:
        if isinstance(petri_fallback, dict) and petri_fallback.get("enabled", False):
            petri_fallback.setdefault("metadata", {})
            petri_fallback["metadata"].update({
                "discovery_source": "petri_enabled_set_fallback_empty_process_tree_filtered_samples",
                "process_tree_role": "xor_extracted_but_no_matching_petri_enabled_sets",
                "primary_runtime_model": "petri_enabled_set_decision_tree",
                "num_process_tree_xor_nodes": int(len(decision_nodes)),
                "process_tree_xor_nodes": _serialize_process_tree_decision_nodes(decision_nodes),
                "fallback_reason": "process_tree_filtered_training_df_empty",
            })
            return petri_fallback
        return _disabled_params(
            reason="No Petri enabled-set samples matched process-tree XOR decision nodes."
        )

    params = _fit_decision_point_models_from_training_df(
        training_df=training_df,
        metadata=metadata,
        max_depths_cv=max_depths_cv,
        min_samples_per_decision_point=min_samples_per_decision_point,
        dominance_threshold=dominance_threshold,
        random_state=random_state,
        model_type="process_tree_filtered_petri_enabled_set_decision_point_branch_tree",
        description=(
            "Petri enabled-set decision-point branch model filtered by process-tree XOR nodes. "
            "The decision tree predicts Petri-net transition keys directly; the process tree "
            "is used only to decide which Petri enabled-sets are true control-flow decision points."
        ),
        extra_metadata={
            "discovery_source": "process_tree_filtered_petri_enabled_set",
            "primary_runtime_model": "petri_enabled_set_decision_tree",
            "process_tree_role": "xor_filter_for_petri_enabled_sets",
            "process_tree_available": bool(process_tree_available),
            "num_process_tree_xor_nodes": int(len(decision_nodes)),
            "num_filtered_petri_decision_points": int(len(metadata)),
            "process_tree_xor_nodes": _serialize_process_tree_decision_nodes(decision_nodes),
            "runtime_mapping": "petri_enabled_set_signature_only",
            "branch_prediction_target": "petri_transition_key",
            "no_process_tree_branch_to_petri_transition_mapping": True,
            "label_data_attributes": label_data_attributes,
            "label_data_attributes_categorical": label_data_attributes_categorical,
            "values_categorical": values_categorical,
            "filter_rule": (
                "Keep a Petri enabled-set only when its visible enabled transitions cover "
                "at least two branches of a safe XOR node extracted from the process tree."
            ),
            "global_label_probs": _global_label_probs_from_log(log) if log is not None else {},
        },
    )

    params.setdefault("fallback", {})
    params["fallback"].update({
        "single_enabled_transition": "fire_directly",
        "missing_decision_point": (
            "petri_enabled_set_fallback_then_uniform"
            if isinstance(petri_fallback, dict) and petri_fallback.get("enabled", False)
            else "uniform_over_enabled_transitions"
        ),
        "missing_feature": 0.0,
    })

    if isinstance(petri_fallback, dict) and petri_fallback.get("enabled", False):
        params["petri_fallback_decision_points"] = petri_fallback
        params.setdefault("metadata", {})["petri_fallback_enabled"] = True
        params["metadata"]["petri_fallback_num_decision_points"] = int(
            petri_fallback.get("num_decision_points", 0)
        )
    else:
        params.setdefault("metadata", {})["petri_fallback_enabled"] = False

    # 标注每一个主决策点，方便前端识别：这些才是由流程树筛选出来的 Petri 决策点。
    decision_points = params.get("decision_points", {})
    if isinstance(decision_points, dict):
        for signature, model in decision_points.items():
            if not isinstance(model, dict):
                continue
            model.setdefault("operator", "PETRI_ENABLED_SET_XOR_FILTERED")
            model["process_tree_filter_role"] = "primary_decision_point"
            model["prediction_target"] = "petri_transition_key"

    return params

def _serialize_process_tree_decision_nodes(decision_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    将流程树 XOR 节点信息转换为 JSON 友好的结构。

    注意：这里不保存流程树对象本身，只保存可显示、可诊断的轻量元数据。
    """
    serializable: Dict[str, Dict[str, Any]] = {}
    if not isinstance(decision_nodes, dict):
        return serializable

    for signature, node in decision_nodes.items():
        if not isinstance(node, dict):
            continue
        branch_label_sets = node.get("branch_label_sets", {}) or {}
        serializable[str(signature)] = {
            "signature": str(signature),
            "operator": str(node.get("operator", "XOR")),
            "path": list(node.get("path", []) or []),
            "branches": [str(b) for b in node.get("branches", [])],
            "branch_display_names": {
                str(k): str(v)
                for k, v in (node.get("branch_display_names", {}) or {}).items()
            },
            "branch_label_sets": {
                str(k): [str(x) for x in v]
                for k, v in branch_label_sets.items()
            },
            "union_labels": [str(x) for x in node.get("union_labels", [])],
            "mapping_safety": dict(node.get("mapping_safety", {}) or {}),
        }
    return serializable


def _fallback_to_petri_model(
        df_features: Optional[pd.DataFrame],
        net_transition_labels: List[str],
        max_depths_cv: List[int],
        label_data_attributes: List[str],
        label_data_attributes_categorical: List[str],
        values_categorical: Dict[str, list],
        min_samples_per_decision_point: int,
        dominance_threshold: float,
        random_state: int,
        reason: str,
) -> Dict[str, Any]:
    if df_features is None:
        return _disabled_params(reason=f"process-tree fallback failed: {reason}; df_features is None")

    params = discover_decision_point_branch_models(
        df_features=df_features,
        net_transition_labels=net_transition_labels,
        max_depths_cv=max_depths_cv,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
        min_samples_per_decision_point=min_samples_per_decision_point,
        dominance_threshold=dominance_threshold,
        random_state=random_state,
    )
    params.setdefault("metadata", {})
    params["metadata"]["process_tree_fallback_reason"] = reason
    return params


def extract_process_tree_decision_nodes(process_tree) -> Dict[str, Dict[str, Any]]:
    """
    从流程树中提取可安全建模的 XOR 决策节点。

    第一版仅处理 XOR，不处理 LOOP。原因是 IM 流程树中的 LOOP 往往包含
    silent exit 或 redo 分支，如果仅依赖可见活动标签集合进行映射，容易导致
    循环退出失败和仿真 trace 膨胀。

    为了保证“流程树分支 -> Petri 网可使能变迁”的运行时映射更可靠，
    本函数只保留满足以下条件的 XOR 节点：
    1. 至少包含两个具有可见活动的子分支；
    2. 不同子分支的可见活动标签集合两两不重叠。

    返回结构：
    signature -> {
        operator, path, branches, branch_display_names, branch_label_sets, union_labels
    }
    """
    decision_nodes: Dict[str, Dict[str, Any]] = {}

    def visit(node, path: Tuple[int, ...]):
        operator = _operator_name(getattr(node, "operator", None))
        children = list(getattr(node, "children", []) or [])

        # 方案 A：只处理 XOR。LOOP 暂时交给 Petri enabled-set fallback，
        # 避免 silent exit / redo 分支被可见标签集合错误映射。
        if operator == "XOR" and len(children) >= 2:
            signature = f"pt::{'.'.join(map(str, path)) if path else 'root'}::{operator}"
            branches = []
            branch_display_names = {}
            branch_label_sets = {}

            for idx, child in enumerate(children):
                labels = sorted(_visible_labels(child))
                # 没有可见活动的 tau 分支无法通过标签集合稳定建模，跳过。
                if not labels:
                    continue

                branch_key = f"{signature}::b{idx}"
                branches.append(branch_key)
                branch_label_sets[branch_key] = labels
                preview = ", ".join(labels[:4])
                if len(labels) > 4:
                    preview += ", ..."
                branch_display_names[branch_key] = f"{operator}[{idx}]({preview})"

            # 至少两个可见分支才构成可学习的 XOR 决策点。
            if len(branches) >= 2:
                overlap_pairs = _find_branch_label_overlaps(branch_label_sets)
                # 如果不同分支含有重复活动标签，当前 label-set 映射不唯一。
                # 该节点暂不启用流程树决策树，交给 Petri fallback。
                if not overlap_pairs:
                    union_labels = sorted({label for labels in branch_label_sets.values() for label in labels})
                    decision_nodes[signature] = {
                        "signature": signature,
                        "operator": operator,
                        "path": list(path),
                        "branches": branches,
                        "branch_display_names": branch_display_names,
                        "branch_label_sets": branch_label_sets,
                        "union_labels": union_labels,
                        "mapping_safety": {
                            "xor_only": True,
                            "branch_label_sets_disjoint": True,
                            "loop_nodes_excluded": True,
                        },
                    }

        for idx, child in enumerate(children):
            visit(child, path + (idx,))

    visit(process_tree, tuple())
    return decision_nodes



def _transition_visible_label(transition) -> str:
    """Return the visible activity label of a Petri transition; tau transitions return ''."""
    if transition is None:
        return ""
    label = getattr(transition, "label", None)
    if label is None:
        return ""
    label = str(label)
    return label if label else ""


def _best_process_tree_xor_match_for_enabled_set(
        enabled_list,
        decision_nodes: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    判断一个 Petri enabled-set 是否对应某个流程树 XOR 节点。

    只有当当前 enabled-set 中的可见变迁覆盖某个 XOR 节点至少两个 child branch 时，
    才认为该 Petri enabled-set 是流程树筛选出的真实决策点。
    """
    if not enabled_list or not isinstance(decision_nodes, dict):
        return None

    enabled_key_label_pairs = []
    for transition in enabled_list:
        key = transition_key(transition)
        if not key:
            continue
        enabled_key_label_pairs.append((key, _transition_visible_label(transition)))

    if not enabled_key_label_pairs:
        return None

    candidates: List[Dict[str, Any]] = []

    for node_signature, node in decision_nodes.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("operator", "")).upper() != "XOR":
            continue

        branch_label_sets = node.get("branch_label_sets", {}) or {}
        if not isinstance(branch_label_sets, dict) or len(branch_label_sets) < 2:
            continue

        branch_to_transition_keys: Dict[str, List[str]] = {}
        mapped_transition_keys = set()

        for branch, labels in branch_label_sets.items():
            label_set = {str(x) for x in (labels or []) if str(x)}
            if not label_set:
                continue
            for transition_key_value, visible_label in enabled_key_label_pairs:
                if visible_label and visible_label in label_set:
                    branch_to_transition_keys.setdefault(str(branch), []).append(str(transition_key_value))
                    mapped_transition_keys.add(str(transition_key_value))

        active_branch_count = len(branch_to_transition_keys)
        if active_branch_count < 2:
            continue

        union_labels = set()
        for labels in branch_label_sets.values():
            union_labels.update(str(x) for x in (labels or []) if str(x))

        candidates.append({
            "process_tree_node_signature": str(node_signature),
            "process_tree_node": node,
            "active_process_tree_branches": sorted(branch_to_transition_keys.keys()),
            "process_tree_branch_to_transition_keys": {
                str(k): sorted(set(str(x) for x in v))
                for k, v in branch_to_transition_keys.items()
            },
            "mapped_transition_keys": sorted(mapped_transition_keys),
            "unmapped_transition_keys": sorted(
                set(key for key, _ in enabled_key_label_pairs).difference(mapped_transition_keys)
            ),
            "active_branch_count": int(active_branch_count),
            "mapped_transition_count": int(len(mapped_transition_keys)),
            "union_label_count": int(len(union_labels)),
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda row: (
            -int(row.get("active_branch_count", 0)),
            int(row.get("union_label_count", 10 ** 9)),
            -int(row.get("mapped_transition_count", 0)),
            str(row.get("process_tree_node_signature", "")),
        )
    )
    return candidates[0]


def build_process_tree_filtered_petri_training_df(
        df_features: pd.DataFrame,
        decision_nodes: Dict[str, Dict[str, Any]],
        net_transition_labels: List[str],
        label_data_attributes: List[str],
        label_data_attributes_categorical: List[str],
        values_categorical: Dict[str, list],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    构造“流程树筛选 + Petri enabled-set 训练”的控制流训练数据。

    与 build_process_tree_guided_training_df 不同，本函数不预测流程树 branch，
    也不在运行时做 branch -> transition 映射。它只用流程树 XOR 判断哪些
    Petri enabled-set 是真实决策点；被保留下来的样本仍以 fired Petri transition
    key 作为 branch 标签。
    """
    if df_features is None or df_features.empty:
        return pd.DataFrame(), {}

    feature_cols: List[str] = []

    for col in label_data_attributes or []:
        if col in df_features.columns:
            feature_cols.append(col)

    for col in net_transition_labels or []:
        if col in df_features.columns:
            feature_cols.append(col)

    # 保守使用已有上下文列，不在这里额外生成 last_activity/last_2_activities，
    # 避免训练阶段与仿真运行阶段的特征不一致。
    optional_context_cols = [
        "prefix_length",
        "case_elapsed_time",
        "hour",
        "weekday",
        "loop_count_at_current_node",
        "gateway_visit_count",
        "loop_state",
        "previous_activity",
        "is_revisit",
    ]
    for col in optional_context_cols:
        if col in df_features.columns and col not in feature_cols:
            feature_cols.append(col)

    rows: List[Dict[str, Any]] = []
    metadata: Dict[str, Dict[str, Any]] = {}
    signature_match_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    for _, row in df_features.iterrows():
        fired_transition = row.get("transition")
        enabled_transitions = row.get("prev_enabled_transitions")

        if fired_transition is None or enabled_transitions is None:
            continue

        try:
            enabled_list = list(enabled_transitions)
        except Exception:
            continue

        if len(enabled_list) < 2:
            continue

        fired_key = transition_key(fired_transition)
        enabled_keys = [transition_key(t) for t in enabled_list if transition_key(t) != ""]
        if not fired_key or fired_key not in enabled_keys:
            continue

        signature = decision_point_signature(enabled_list)

        if signature not in signature_match_cache:
            signature_match_cache[signature] = _best_process_tree_xor_match_for_enabled_set(
                enabled_list=enabled_list,
                decision_nodes=decision_nodes,
            )

        match = signature_match_cache.get(signature)
        if not match:
            continue

        if signature not in metadata:
            branches = sorted(set(enabled_keys))
            branch_display_names = {
                transition_key(t): transition_display_name(t)
                for t in enabled_list
                if transition_key(t) != ""
            }
            process_tree_signature = str(match.get("process_tree_node_signature", ""))
            process_tree_node = match.get("process_tree_node", {}) or {}

            metadata[signature] = {
                "signature": signature,
                "operator": "PETRI_ENABLED_SET_XOR_FILTERED",
                "branches": branches,
                "branch_display_names": branch_display_names,
                "branch_label_sets": {},
                "process_tree_node_signature": process_tree_signature,
                "process_tree_operator": str(process_tree_node.get("operator", "XOR")),
                "process_tree_path": list(process_tree_node.get("path", []) or []),
                "process_tree_active_branches": list(match.get("active_process_tree_branches", []) or []),
                "process_tree_branch_to_transition_keys": dict(
                    match.get("process_tree_branch_to_transition_keys", {}) or {}
                ),
                "process_tree_branch_label_sets": dict(process_tree_node.get("branch_label_sets", {}) or {}),
                "process_tree_union_labels": list(process_tree_node.get("union_labels", []) or []),
                "process_tree_filter": {
                    "mapped_transition_keys": list(match.get("mapped_transition_keys", []) or []),
                    "unmapped_transition_keys": list(match.get("unmapped_transition_keys", []) or []),
                    "active_branch_count": int(match.get("active_branch_count", 0)),
                    "mapped_transition_count": int(match.get("mapped_transition_count", 0)),
                    "rule": "enabled-set covers at least two process-tree XOR branches",
                },
            }

        record = {
            "decision_point_signature": signature,
            "branch": fired_key,
        }
        for col in feature_cols:
            record[col] = row.get(col, 0.0)
        rows.append(record)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, metadata

    for attr in label_data_attributes_categorical or []:
        if attr not in df.columns:
            continue
        for value in values_categorical.get(attr, []):
            new_col = f"{attr} = {value}"
            df[new_col] = (df[attr] == value).astype(int)
        df.drop(columns=[attr], inplace=True)

    df = df.fillna(0.0)

    for col in df.columns:
        if col in ["decision_point_signature", "branch"]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df, metadata

def build_process_tree_guided_training_df(
        log,
        decision_nodes: Dict[str, Dict[str, Any]],
        net_transition_labels: List[str],
        label_data_attributes: List[str],
        label_data_attributes_categorical: List[str],
        values_categorical: Dict[str, list]
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    基于事件日志构造流程树引导的分支训练样本。

    样本触发原则：当当前事件标签第一次进入某个 XOR 节点的可见活动集合，
    或从该节点的一个 child branch 切换到另一个 child branch 时，认为当前前缀
    触发了一次该流程树节点的分支选择。

    注意：decision_nodes 已经在 extract_process_tree_decision_nodes 中完成
    XOR-only 与分支标签集合两两不重叠过滤，因此这里不再处理 LOOP。
    """
    rows: List[Dict[str, Any]] = []
    metadata = copy.deepcopy(decision_nodes)

    activity_values = sorted(set(str(x) for x in net_transition_labels if str(x)))
    start_token = "__START__"

    for trace in log:
        events = list(trace)
        if not events:
            continue

        labels = [str(event.get("concept:name")) for event in events]
        times = [_event_timestamp(event) for event in events]
        first_time = next((t for t in times if t is not None), None)

        case_attrs = {}
        if events:
            first_event = events[0]
            for attr in label_data_attributes:
                case_attrs[attr] = first_event.get(attr, 0.0)

        history = {activity: 0 for activity in activity_values}
        last_activity = start_token
        last_2_activities = f"{start_token}||{start_token}"

        # 每个流程树节点上一次位于哪个 branch，用于减少同一 branch 内部重复采样。
        previous_branch_by_node: Dict[str, Optional[str]] = {sig: None for sig in decision_nodes}
        loop_seen_count_by_node: Dict[str, int] = {sig: 0 for sig in decision_nodes}

        for idx, (event, label) in enumerate(zip(events, labels)):
            current_time = times[idx]
            prefix_length = idx
            case_elapsed_time = _elapsed_minutes(first_time, current_time)
            hour = _safe_hour(current_time)
            weekday = _safe_weekday(current_time)

            for signature, node_info in decision_nodes.items():
                branch = _branch_for_label(label, node_info.get("branch_label_sets", {}))
                if branch is None:
                    continue

                prev_branch = previous_branch_by_node.get(signature)
                # 第一次进入该结构，或从一个分支切换到另一个分支时，记录样本。
                if prev_branch == branch:
                    loop_seen_count_by_node[signature] += 1
                    continue

                record = {
                    "decision_point_signature": signature,
                    "branch": branch,
                    "prefix_length": float(prefix_length),
                    "case_elapsed_time": float(case_elapsed_time),
                    "hour": float(hour),
                    "weekday": float(weekday),
                    "loop_count_at_current_node": float(loop_seen_count_by_node.get(signature, 0)),
                }

                # 案例级属性
                for attr in label_data_attributes:
                    record[attr] = case_attrs.get(attr, 0.0)

                # 活动历史计数
                for activity in activity_values:
                    record[activity] = history.get(activity, 0)

                # 前缀上下文 one-hot，训练和运行时使用相同列名。
                record[f"last_activity = {last_activity}"] = 1
                record[f"last_2_activities = {last_2_activities}"] = 1

                rows.append(record)
                previous_branch_by_node[signature] = branch
                loop_seen_count_by_node[signature] += 1

            # 更新历史。注意：当前事件之后才进入下一步前缀。
            history[label] = history.get(label, 0) + 1
            prev_last = last_activity
            last_activity = label
            last_2_activities = f"{prev_last}||{label}"

    df = pd.DataFrame(rows)
    if df.empty:
        return df, metadata

    # 类别型案例属性 one-hot。
    for attr in label_data_attributes_categorical:
        if attr not in df.columns:
            continue
        for value in values_categorical.get(attr, []):
            new_col = f"{attr} = {value}"
            df[new_col] = (df[attr] == value).astype(int)
        df.drop(columns=[attr], inplace=True)

    # one-hot 稀疏列补 0。
    df = df.fillna(0.0)

    for col in df.columns:
        if col in ["decision_point_signature", "branch"]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df, metadata


# ---------------------------------------------------------------------------
# Runtime sampling: process-tree branch -> Petri-net enabled transition
# ---------------------------------------------------------------------------

def sample_process_tree_guided_branch(
        decision_point_params: Dict[str, Any],
        enabled_transitions,
        features: Dict[str, Any],
        net=None,
        marking=None,
        rng: Optional[random.Random] = None,
        max_tau_depth: int = 8,
):
    """
    运行时从流程树 XOR 分支模型中采样 Petri 网变迁。

    方案 A 的安全策略：
    1. 只允许 XOR 决策点参与流程树引导采样；
    2. 对当前 enabled transitions 进行 visible-label / tau-closure lookahead；
    3. 只有当“enabled transition -> XOR branch”的映射唯一，且至少覆盖两个分支时，
       才使用流程树分支模型；
    4. 一旦出现 transition 同时命中多个 branch、分支标签重叠、候选节点不唯一等歧义，
       立即回退到 Petri enabled-set 决策模型，再回退到全局活动频率。
    """
    enabled_list = list(enabled_transitions)
    rng = rng or random

    if not enabled_list:
        return None
    if len(enabled_list) == 1:
        return enabled_list[0]

    if not isinstance(decision_point_params, dict) or not decision_point_params.get("enabled", False):
        return _sample_by_global_label_frequency(decision_point_params, enabled_list, net, marking, rng, max_tau_depth)

    model_type = str(decision_point_params.get("model_type", ""))
    if model_type != "process_tree_guided_decision_point_branch_tree":
        return sample_decision_point_branch(
            decision_point_params=decision_point_params,
            enabled_transitions=enabled_list,
            features=features,
            rng=rng,
        )

    transition_label_map = {
        transition_key(t): _reachable_visible_labels_for_transition(
            net=net,
            marking=marking,
            transition=t,
            max_tau_depth=max_tau_depth,
        )
        for t in enabled_list
    }

    decision_points = decision_point_params.get("decision_points", {})
    valid_candidates = []

    for signature, model in decision_points.items():
        if not isinstance(model, dict) or not model.get("enabled", True):
            continue

        # 方案 A：运行时再次强制 XOR-only，防止旧参数或缓存中混入 LOOP。
        if str(model.get("operator", "")).upper() != "XOR":
            continue

        branch_label_sets = model.get("branch_label_sets", {}) or {}
        branches = [str(b) for b in model.get("branches", [])]
        if len(branches) < 2:
            continue

        # 分支标签集合必须两两不重叠，否则 label-set 映射本身不唯一。
        if _find_branch_label_overlaps({b: branch_label_sets.get(b, []) for b in branches}):
            continue

        mapping = _build_unique_enabled_transition_branch_mapping(
            enabled_list=enabled_list,
            transition_label_map=transition_label_map,
            branches=branches,
            branch_label_sets=branch_label_sets,
        )

        if not mapping.get("valid", False):
            continue

        branch_to_transitions = mapping.get("branch_to_transitions", {}) or {}
        if len(branch_to_transitions) < 2:
            continue

        union_labels = set()
        for b in branches:
            union_labels.update(str(x) for x in branch_label_sets.get(b, []))

        # 候选排序：优先更局部的 XOR 节点，即 union label 更小；其次样本更多。
        valid_candidates.append({
            "signature": signature,
            "model": model,
            "branch_to_transitions": branch_to_transitions,
            "union_label_count": int(len(union_labels)),
            "active_branch_count": int(len(branch_to_transitions)),
            "n_samples": int(model.get("n_samples", 0)),
        })

    if not valid_candidates:
        return _sample_with_petri_fallback_then_global(
            decision_point_params=decision_point_params,
            enabled_list=enabled_list,
            features=features,
            net=net,
            marking=marking,
            rng=rng,
            max_tau_depth=max_tau_depth,
        )

    valid_candidates.sort(
        key=lambda row: (
            row.get("union_label_count", 10 ** 9),
            -row.get("active_branch_count", 0),
            -row.get("n_samples", 0),
            str(row.get("signature", "")),
        )
    )
    candidate = valid_candidates[0]
    model = candidate["model"]
    branch_to_transitions = candidate["branch_to_transitions"]

    probs = _model_branch_probs(model=model, features=features)
    if not probs:
        probs = model.get("empirical_probs", {}) or {}
    probs = _normalize_probs({str(k): float(v) for k, v in probs.items()})

    transition_scores = []
    for t in enabled_list:
        t_key = transition_key(t)
        score = 0.0
        for branch, transitions in branch_to_transitions.items():
            if t_key in transitions:
                # 一个分支可能对应多个 enabled transition，此时平均分配该分支概率，
                # 避免同一分支因 transition 数量多而被重复加权。
                score = float(probs.get(str(branch), 0.0)) / max(1, len(transitions))
                break
        transition_scores.append(max(0.0, float(score)))

    total = sum(transition_scores)
    if total <= 0:
        return _sample_with_petri_fallback_then_global(
            decision_point_params=decision_point_params,
            enabled_list=enabled_list,
            features=features,
            net=net,
            marking=marking,
            rng=rng,
            max_tau_depth=max_tau_depth,
        )

    threshold = rng.random() * total
    cumulative = 0.0
    for transition, weight in zip(enabled_list, transition_scores):
        cumulative += weight
        if threshold <= cumulative:
            return transition

    return enabled_list[-1]


def _build_unique_enabled_transition_branch_mapping(
        enabled_list,
        transition_label_map: Dict[str, set[str]],
        branches: List[str],
        branch_label_sets: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    构造当前 enabled transition 到流程树 XOR branch 的唯一映射。

    只有满足以下条件时才返回 valid=True：
    1. 每个 transition 至多命中一个 branch；
    2. 至少两个 branch 被当前 enabled transitions 覆盖；
    3. 没有 branch 标签集合重叠导致的潜在歧义。
    """
    branch_label_sets_clean = {
        str(branch): set(str(x) for x in branch_label_sets.get(branch, []))
        for branch in branches
    }

    if _find_branch_label_overlaps(branch_label_sets_clean):
        return {"valid": False, "reason": "overlapping_branch_label_sets"}

    branch_to_transitions: Dict[str, List[str]] = {}
    transition_to_branch: Dict[str, str] = {}
    unmapped_transitions: List[str] = []

    for transition in enabled_list:
        t_key = transition_key(transition)
        reachable_labels = set(str(x) for x in transition_label_map.get(t_key, set()))
        if not reachable_labels:
            unmapped_transitions.append(t_key)
            continue

        matched_branches = []
        for branch in branches:
            label_set = branch_label_sets_clean.get(str(branch), set())
            if reachable_labels & label_set:
                matched_branches.append(str(branch))

        if len(matched_branches) > 1:
            return {
                "valid": False,
                "reason": "transition_matches_multiple_branches",
                "transition": t_key,
                "matched_branches": matched_branches,
                "reachable_labels": sorted(reachable_labels),
            }

        if len(matched_branches) == 1:
            branch = matched_branches[0]
            transition_to_branch[t_key] = branch
            branch_to_transitions.setdefault(branch, []).append(t_key)
        else:
            unmapped_transitions.append(t_key)

    if len(branch_to_transitions) < 2:
        return {
            "valid": False,
            "reason": "less_than_two_active_branches",
            "branch_to_transitions": branch_to_transitions,
            "unmapped_transitions": unmapped_transitions,
        }

    return {
        "valid": True,
        "branch_to_transitions": branch_to_transitions,
        "transition_to_branch": transition_to_branch,
        "unmapped_transitions": unmapped_transitions,
    }


def _sample_with_petri_fallback_then_global(
        decision_point_params: Dict[str, Any],
        enabled_list,
        features: Dict[str, Any],
        net=None,
        marking=None,
        rng: Optional[random.Random] = None,
        max_tau_depth: int = 8,
):
    """先使用 Petri enabled-set fallback，再使用全局活动频率。"""
    rng = rng or random
    petri_fallback = None
    if isinstance(decision_point_params, dict):
        petri_fallback = decision_point_params.get("petri_fallback_decision_points")

    if isinstance(petri_fallback, dict) and petri_fallback.get("enabled", False):
        chosen = sample_decision_point_branch(
            decision_point_params=petri_fallback,
            enabled_transitions=enabled_list,
            features=features,
            rng=rng,
        )
        if chosen is not None:
            return chosen

    return _sample_by_global_label_frequency(
        decision_point_params,
        enabled_list,
        net,
        marking,
        rng,
        max_tau_depth,
    )


def compute_decision_point_branch_probs(
        decision_point_params: Dict[str, Any],
        enabled_transitions,
        features: Dict[str, Any]
) -> Dict[str, float]:
    """
    兼容原 Petri enabled-set 模型的分支概率计算。
    """
    enabled_list = list(enabled_transitions)

    if len(enabled_list) == 0:
        return {}

    if len(enabled_list) == 1:
        return {transition_key(enabled_list[0]): 1.0}

    enabled_keys = [transition_key(t) for t in enabled_list]
    signature = decision_point_signature(enabled_list)

    if not decision_point_params or not decision_point_params.get("enabled", False):
        return _uniform_probs(enabled_keys)

    model = (
        decision_point_params
        .get("decision_points", {})
        .get(signature)
    )

    if model is None:
        petri_fallback = decision_point_params.get("petri_fallback_decision_points")
        if isinstance(petri_fallback, dict) and petri_fallback.get("enabled", False):
            fallback_probs = compute_decision_point_branch_probs(
                decision_point_params=petri_fallback,
                enabled_transitions=enabled_list,
                features=features,
            )
            fallback_probs = {
                key: float(fallback_probs.get(key, 0.0))
                for key in enabled_keys
            }
            if sum(fallback_probs.values()) > 0:
                return _normalize_probs(fallback_probs)
        return _uniform_probs(enabled_keys)

    probs = _model_branch_probs(model=model, features=features)

    probs = {
        key: float(probs.get(key, 0.0))
        for key in enabled_keys
    }

    if sum(probs.values()) <= 0:
        probs = _uniform_probs(enabled_keys)

    return _normalize_probs(probs)


def sample_decision_point_branch(
        decision_point_params: Dict[str, Any],
        enabled_transitions,
        features: Dict[str, Any],
        rng: Optional[random.Random] = None
):
    """
    从 Petri enabled-set 决策点分支概率中采样一个 Petri 网变迁对象。
    """
    enabled_list = list(enabled_transitions)

    if not enabled_list:
        return None

    if len(enabled_list) == 1:
        return enabled_list[0]

    probs = compute_decision_point_branch_probs(
        decision_point_params=decision_point_params,
        enabled_transitions=enabled_list,
        features=features
    )

    rng = rng or random
    keys = [transition_key(t) for t in enabled_list]
    weights = [float(probs.get(k, 0.0)) for k in keys]

    total = sum(weights)
    if total <= 0:
        return rng.choice(enabled_list)

    threshold = rng.random() * total
    cumulative = 0.0

    for transition, weight in zip(enabled_list, weights):
        cumulative += weight
        if threshold <= cumulative:
            return transition

    return enabled_list[-1]


# ---------------------------------------------------------------------------
# Shared model fitting utilities
# ---------------------------------------------------------------------------

def _fit_decision_point_models_from_training_df(
        training_df: pd.DataFrame,
        metadata: Dict[str, Any],
        max_depths_cv: List[int],
        min_samples_per_decision_point: int,
        dominance_threshold: float,
        random_state: int,
        model_type: str,
        description: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    decision_point_models = {}
    grouped = training_df.groupby("decision_point_signature", sort=False)

    for signature, group in grouped:
        group = group.copy()

        branches = metadata.get(signature, {}).get("branches", [])
        branch_display_names = metadata.get(signature, {}).get("branch_display_names", {})
        branch_label_sets = metadata.get(signature, {}).get("branch_label_sets", {})
        operator = metadata.get(signature, {}).get("operator", None)
        path = metadata.get(signature, {}).get("path", None)

        empirical_probs = _empirical_branch_probs(group["branch"], branches=branches)
        n_samples = int(len(group))
        dominant_prob = max(empirical_probs.values()) if empirical_probs else 1.0

        base_model = {
            "enabled": True,
            "signature": signature,
            "operator": operator,
            "path": path,
            "branches": branches,
            "branch_display_names": branch_display_names,
            "branch_label_sets": branch_label_sets,
            "n_samples": n_samples,
            "empirical_probs": empirical_probs,
            "dominant_prob": float(dominant_prob),
        }

        if (
            not max_depths_cv
            or n_samples < min_samples_per_decision_point
            or dominant_prob >= dominance_threshold
            or group["branch"].nunique() < 2
        ):
            model = dict(base_model)
            model.update({
                "model_kind": "empirical",
                "rules": None,
                "feature_columns": [],
                "max_depth": None,
            })
            decision_point_models[signature] = model
            continue

        feature_columns = [
            c for c in group.columns
            if c not in ["decision_point_signature", "branch"]
        ]

        X = _coerce_features_to_numeric(group[feature_columns])
        y = group["branch"].astype(str)

        clf = _fit_branch_tree(
            X=X,
            y=y,
            max_depths_cv=max_depths_cv,
            random_state=random_state
        )

        rules = _decision_tree_to_branch_rules(
            clf=clf,
            feature_columns=feature_columns
        )

        model = dict(base_model)
        model.update({
            "model_kind": "decision_tree",
            "rules": rules,
            "feature_columns": feature_columns,
            "max_depth": int(clf.get_depth()),
        })
        decision_point_models[signature] = model

    metadata_block = {
        "max_depths_cv": max_depths_cv,
        "min_samples_per_decision_point": int(min_samples_per_decision_point),
        "dominance_threshold": float(dominance_threshold),
    }
    if extra_metadata:
        metadata_block.update(extra_metadata)

    return {
        "enabled": True,
        "model_type": model_type,
        "description": description,
        "num_decision_points": int(len(decision_point_models)),
        "decision_points": decision_point_models,
        "metadata": metadata_block,
        "fallback": {
            "single_enabled_transition": "fire_directly",
            "missing_decision_point": "label_frequency_then_uniform",
            "missing_feature": 0.0,
        }
    }


def _fit_branch_tree(
        X: pd.DataFrame,
        y: pd.Series,
        max_depths_cv: List[int],
        random_state: int
) -> DecisionTreeClassifier:
    if not max_depths_cv:
        max_depths_cv = [2]

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 1

    if min_class_count >= 3 and len(y) >= 10:
        cv = min(3, min_class_count)
        clf = DecisionTreeClassifier(random_state=random_state)

        try:
            grid = GridSearchCV(
                estimator=clf,
                param_grid={"max_depth": list(max_depths_cv)},
                cv=cv
            )
            grid.fit(X, y)
            return grid.best_estimator_
        except Exception:
            pass

    fallback_depth = min(max(max_depths_cv), 3)
    clf = DecisionTreeClassifier(
        max_depth=fallback_depth,
        random_state=random_state
    )
    clf.fit(X, y)
    return clf


def _decision_tree_to_branch_rules(
        clf: DecisionTreeClassifier,
        feature_columns: List[str]
) -> Dict[str, Any]:
    tree = clf.tree_
    classes = [str(c) for c in clf.classes_]

    def build_node(node_id: int) -> Dict[str, Any]:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left == right:
            counts = tree.value[node_id][0].astype(float)
            total = float(counts.sum())

            if total <= 0:
                probs = {cls: 1.0 / max(len(classes), 1) for cls in classes}
            else:
                probs = {cls: float(count / total) for cls, count in zip(classes, counts)}

            dominant_branch = max(probs, key=probs.get) if probs else None

            return {
                "type": "leaf",
                "value": dominant_branch,
                "branch_probs": probs
            }

        feature_index = tree.feature[node_id]
        feature_name = feature_columns[feature_index]
        threshold = float(tree.threshold[node_id])

        return {
            "type": "node",
            "feature": feature_name,
            "threshold": threshold,
            "children": {
                "true": build_node(left),
                "false": build_node(right)
            }
        }

    return build_node(0)


def _apply_branch_rules(
        rules: Dict[str, Any],
        features: Dict[str, Any]
) -> Dict[str, float]:
    if not isinstance(rules, dict):
        return {}

    node = rules

    while isinstance(node, dict) and node.get("type") != "leaf":
        feature = node.get("feature")
        threshold = float(node.get("threshold", 0.0))

        value = features.get(feature, 0.0)
        try:
            value = float(value)
        except Exception:
            value = 0.0

        if value <= threshold:
            node = node.get("children", {}).get("true")
        else:
            node = node.get("children", {}).get("false")

        if node is None:
            return {}

    if not isinstance(node, dict):
        return {}

    probs = node.get("branch_probs", {})

    return {str(k): float(v) for k, v in probs.items()}


def _model_branch_probs(model: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(model, dict):
        return {}
    if model.get("model_kind") == "empirical":
        return {str(k): float(v) for k, v in (model.get("empirical_probs", {}) or {}).items()}

    rules = model.get("rules")
    feature_columns = model.get("feature_columns", [])
    prepared_features = _prepare_feature_dict(features=features, feature_columns=feature_columns)
    probs = _apply_branch_rules(rules=rules, features=prepared_features)
    if not probs:
        probs = model.get("empirical_probs", {}) or {}
    return _normalize_probs({str(k): float(v) for k, v in probs.items()})


def _prepare_feature_dict(
        features: Dict[str, Any],
        feature_columns: List[str]
) -> Dict[str, float]:
    prepared = {}
    for col in feature_columns:
        value = features.get(col, 0.0)
        try:
            prepared[col] = float(value)
        except Exception:
            prepared[col] = 0.0
    return prepared


def _coerce_features_to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X


def _empirical_branch_probs(y: pd.Series, branches: List[str]) -> Dict[str, float]:
    counts = y.astype(str).value_counts(normalize=True).to_dict()
    probs = {str(branch): float(counts.get(str(branch), 0.0)) for branch in branches}
    if sum(probs.values()) <= 0:
        return _uniform_probs(branches)
    return _normalize_probs(probs)


def _uniform_probs(branches: List[str]) -> Dict[str, float]:
    branches = [str(b) for b in branches]
    if not branches:
        return {}
    p = 1.0 / len(branches)
    return {b: p for b in branches}


def _normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(float(v) for v in probs.values())
    if total <= 0:
        return _uniform_probs(list(probs.keys()))
    return {str(k): float(v) / total for k, v in probs.items()}


# ---------------------------------------------------------------------------
# Process-tree helper functions
# ---------------------------------------------------------------------------

def _find_branch_label_overlaps(branch_label_sets: Dict[str, List[str]]) -> List[Tuple[str, str, List[str]]]:
    """返回流程树分支可见标签集合之间的重叠，用于过滤不安全 XOR 节点。"""
    overlaps: List[Tuple[str, str, List[str]]] = []
    items = [
        (str(branch), set(str(x) for x in labels))
        for branch, labels in (branch_label_sets or {}).items()
    ]
    for i in range(len(items)):
        b1, s1 = items[i]
        for j in range(i + 1, len(items)):
            b2, s2 = items[j]
            common = sorted(s1 & s2)
            if common:
                overlaps.append((b1, b2, common))
    return overlaps


def _operator_name(operator) -> str:
    if operator is None:
        return "LEAF"
    raw = getattr(operator, "name", None)
    if raw is None:
        raw = str(operator)
    raw = str(raw).upper()
    if "XOR" in raw:
        return "XOR"
    if "LOOP" in raw:
        return "LOOP"
    if "PARALLEL" in raw or "AND" in raw:
        return "PARALLEL"
    if "SEQUENCE" in raw or "SEQ" in raw:
        return "SEQUENCE"
    return raw


def _visible_labels(node) -> set[str]:
    label = getattr(node, "label", None)
    if label is not None:
        return {str(label)}
    labels = set()
    for child in list(getattr(node, "children", []) or []):
        labels.update(_visible_labels(child))
    return labels


def _branch_for_label(label: str, branch_label_sets: Dict[str, List[str]]) -> Optional[str]:
    label = str(label)
    matches = []
    for branch, labels in branch_label_sets.items():
        if label in set(str(x) for x in labels):
            matches.append(str(branch))
    if not matches:
        return None
    # 如果重复活动导致多个 child 都包含该 label，选择标签集合更小的分支，降低误配风险。
    matches.sort(key=lambda b: len(branch_label_sets.get(b, [])))
    return matches[0]


def _event_timestamp(event):
    for key in ["start:timestamp", "time:timestamp"]:
        value = event.get(key)
        if value is not None:
            try:
                return pd.Timestamp(value)
            except Exception:
                continue
    return None


def _elapsed_minutes(start, current) -> float:
    if start is None or current is None:
        return 0.0
    try:
        delta = pd.Timestamp(current) - pd.Timestamp(start)
        value = float(delta.total_seconds() / 60.0)
        if math.isfinite(value) and value >= 0.0:
            return value
    except Exception:
        pass
    return 0.0


def _safe_hour(ts) -> int:
    try:
        return int(pd.Timestamp(ts).hour)
    except Exception:
        return 0


def _safe_weekday(ts) -> int:
    try:
        return int(pd.Timestamp(ts).weekday())
    except Exception:
        return 0


def _global_label_probs_from_log(log) -> Dict[str, float]:
    counter = Counter()
    for trace in log:
        for event in trace:
            label = event.get("concept:name")
            if label is not None:
                counter[str(label)] += 1
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {str(k): float(v) / float(total) for k, v in counter.items()}


# ---------------------------------------------------------------------------
# Petri-net lookahead helper functions
# ---------------------------------------------------------------------------

def _sample_by_global_label_frequency(decision_point_params, enabled_list, net, marking, rng, max_tau_depth: int = 8):
    metadata = {}
    if isinstance(decision_point_params, dict):
        metadata = decision_point_params.get("metadata", {}) or {}
    label_probs = metadata.get("global_label_probs", {}) or {}

    weights = []
    for t in enabled_list:
        labels = _reachable_visible_labels_for_transition(
            net=net,
            marking=marking,
            transition=t,
            max_tau_depth=max_tau_depth,
        )
        score = 0.0
        for label in labels:
            score += float(label_probs.get(str(label), 0.0))
        # 如果当前变迁没有可见 lookahead，则给一个极小正权重，避免 tau 完全不能走。
        weights.append(score if score > 0.0 else 1e-9)

    total = sum(weights)
    if total <= 0:
        return rng.choice(enabled_list)

    threshold = rng.random() * total
    cumulative = 0.0
    for transition, weight in zip(enabled_list, weights):
        cumulative += weight
        if threshold <= cumulative:
            return transition
    return enabled_list[-1]


def _reachable_visible_labels_for_transition(net, marking, transition, max_tau_depth: int = 8) -> set[str]:
    if transition is None:
        return set()
    label = getattr(transition, "label", None)
    if label is not None:
        return {str(label)}
    if net is None or marking is None:
        return set()

    try:
        start_marking = _fire_transition_local(marking, transition)
    except Exception:
        return set()

    labels = set()
    queue = [(start_marking, 0)]
    visited = set()

    while queue:
        current_marking, depth = queue.pop(0)
        sig = _marking_signature_local(current_marking)
        if sig in visited:
            continue
        visited.add(sig)

        enabled = _enabled_transitions_local(net, current_marking)
        for t in enabled:
            t_label = getattr(t, "label", None)
            if t_label is not None:
                labels.add(str(t_label))
            elif depth < max_tau_depth:
                try:
                    queue.append((_fire_transition_local(current_marking, t), depth + 1))
                except Exception:
                    continue

    return labels


def _enabled_transitions_local(net, marking) -> list:
    enabled = []
    for transition in list(getattr(net, "transitions", []) or []):
        ok = True
        for arc in list(getattr(transition, "in_arcs", []) or []):
            place = arc.source
            weight = int(getattr(arc, "weight", 1) or 1)
            if int(marking.get(place, 0)) < weight:
                ok = False
                break
        if ok:
            enabled.append(transition)
    return enabled


def _fire_transition_local(marking, transition):
    new_marking = copy.copy(marking)
    for arc in list(getattr(transition, "in_arcs", []) or []):
        place = arc.source
        weight = int(getattr(arc, "weight", 1) or 1)
        new_value = int(new_marking.get(place, 0)) - weight
        if new_value <= 0:
            if place in new_marking:
                del new_marking[place]
        else:
            new_marking[place] = new_value

    for arc in list(getattr(transition, "out_arcs", []) or []):
        place = arc.target
        weight = int(getattr(arc, "weight", 1) or 1)
        new_marking[place] = int(new_marking.get(place, 0)) + weight
    return new_marking


def _marking_signature_local(marking) -> tuple:
    try:
        return tuple(sorted((getattr(p, "name", str(p)), int(v)) for p, v in marking.items() if int(v) != 0))
    except Exception:
        return tuple(sorted(str(x) for x in marking.items()))


def _disabled_params(reason: str) -> Dict[str, Any]:
    return {
        "enabled": False,
        "model_type": "decision_point_branch_tree",
        "reason": reason,
        "num_decision_points": 0,
        "decision_points": {},
        "metadata": {},
        "fallback": {
            "single_enabled_transition": "fire_directly",
            "missing_decision_point": "uniform_over_enabled_transitions",
            "missing_feature": 0.0
        }
    }
