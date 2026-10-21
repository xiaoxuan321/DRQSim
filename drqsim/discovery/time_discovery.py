# -*- coding: utf-8 -*-
"""
DBPS-based case arrival discovery for the modular drq simulator.

The model is a white-box probabilistic decision tree for case inter-arrival
minutes:

    temporal context  ->  decision-tree leaf  ->  empirical/fitted distribution

Design choices in this version:
1. The target is raw wall-clock inter-arrival time between consecutive case
   arrivals. It is NOT compressed by the arrival calendar.
2. The tree keeps the DBPS global context features weekday and hour, and adds
   cyclical seasonal features (month, week_of_year) so the model can represent
   season-dependent arrival intensity changes. These features are cyclical and
   always stay within their training range, eliminating the extrapolation
   problem that affected the previous elapsed_ratio/elapsed_week_index design.
3. max_depth can be selected by GridSearchCV over 1..5, or fixed by passing a
   single-depth list from the external stagewise GridSearchCV.
4. Arrival inter-arrival values are not aggressively outlier-filtered by default,
   because long gaps are often real arrival behavior rather than noise.
5. No validation-driven arrival_time_scale or global scaling is applied inside
   this module. Demand changes should be handled as explicit what-if parameters
   outside the as-is discovery model.

Legacy histogram/tuple/DecisionRules-like models are still sampled for backward
compatibility with old saved JSON files.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog

from drqsim.utils.distribution_utils import (
    return_best_distribution,
    sampling_from_dist,
    remove_outliers,
)

# Keep the existing distribution set to avoid breaking return_best_distribution
# implementations that do not support additional distribution names.
DIST_SEARCH = ["fixed", "norm", "expon", "uniform"]

DECISION_TREE_MODEL_TYPE = "case_arrival_decision_tree_interarrival"
CONTEXT_MODEL_TYPE = "case_arrival_calendar_context_histogram"
LEGACY_GLOBAL_MODEL_TYPE = "case_arrival_calendar_observed_histogram"

# DBPS-based feature set:
# - weekday/hour are the original DBPS-style global temporal context.
# - month and week_of_year are cyclical seasonal features that allow the tree
#   to separate seasonal arrival regimes (e.g. summer vs winter). Unlike the
#   previous elapsed_ratio/elapsed_week_index design, these features are
#   cyclical and always stay within their training range, so there is no
#   extrapolation problem when simulating beyond the training period.
ARRIVAL_TREE_FEATURE_NAMES = [
    "weekday",
    "hour",
    "month",
    "week_of_year",
]


def _to_dataframe(log: EventLog | pd.DataFrame) -> pd.DataFrame:
    if isinstance(log, pd.DataFrame):
        return log.copy()
    return pm4py.convert_to_dataframe(log).copy()


def _extract_case_arrival_times(
        log: EventLog | pd.DataFrame,
        case_col: str = "case:concept:name",
        start_col: str = "start:timestamp",
) -> list[pd.Timestamp]:
    """Extract the first event start timestamp of each case, sorted by time."""
    df = _to_dataframe(log)

    if case_col not in df.columns or start_col not in df.columns:
        return []

    df = df[[case_col, start_col]].copy()
    df[case_col] = df[case_col].astype(str)
    df[start_col] = pd.to_datetime(df[start_col], utc=True, errors="coerce")
    df.dropna(subset=[case_col, start_col], inplace=True)

    if df.empty:
        return []

    return (
        df.groupby(case_col)[start_col]
        .min()
        .sort_values()
        .tolist()
    )


def _raw_inter_arrival_minutes(
        previous_arrival: pd.Timestamp,
        current_arrival: pd.Timestamp,
) -> float:
    """Raw wall-clock inter-arrival minutes without calendar compression."""
    try:
        return max(float((current_arrival - previous_arrival).total_seconds() / 60.0), 0.0)
    except Exception:
        return 0.0


def _normalize_utc_timestamp(timestamp) -> pd.Timestamp:
    """Convert a timestamp-like value to a UTC pandas Timestamp."""
    ts = pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp.utcnow()
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _build_temporal_reference(arrivals: list[pd.Timestamp]) -> dict:
    """Build a discovery-log temporal reference for position features."""
    clean = []
    for value in arrivals or []:
        try:
            clean.append(_normalize_utc_timestamp(value))
        except Exception:
            continue

    if not clean:
        now = pd.Timestamp.utcnow()
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        return {
            "start": now.isoformat(),
            "end": now.isoformat(),
            "span_minutes": 1.0,
        }

    start_ts = min(clean)
    end_ts = max(clean)
    span_minutes = max(
        float((end_ts - start_ts).total_seconds() / 60.0),
        1.0,
    )
    return {
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "span_minutes": float(span_minutes),
    }


def _arrival_context_features(
        timestamp,
        temporal_reference: dict | None = None,
) -> dict[str, float]:
    """Global arrival context for the previous case arrival.

    Active features (cyclical, no extrapolation):
        weekday (0-6):
            Day of the week (Monday=0, Sunday=6). Same as DBPS.
        hour (0-23):
            Hour of the day. Same as DBPS.
        month (1-12):
            Calendar month. Cyclical — after December comes January,
            so it never exceeds the training range.
        week_of_year (1-53):
            ISO week number. Cyclical — after week 52/53 comes week 1,
            so it never exceeds the training range.

    Legacy features (computed for backward compatibility with old saved
    models that still use elapsed_week_index / elapsed_ratio in their tree
    rules, but no longer included in ARRIVAL_TREE_FEATURE_NAMES):
        elapsed_week_index, elapsed_ratio
    """
    ts = _normalize_utc_timestamp(timestamp)

    # --- Cyclical seasonal features (active) ---
    month = float(int(ts.month))
    try:
        iso_cal = ts.isocalendar()
        week_of_year = float(int(iso_cal[1]))
    except Exception:
        week_of_year = 1.0

    # --- Legacy temporal-position features (backward compat only) ---
    ref_start = None
    ref_span_minutes = 1.0

    if isinstance(temporal_reference, dict):
        try:
            ref_start = _normalize_utc_timestamp(temporal_reference.get("start"))
        except Exception:
            ref_start = None
        try:
            ref_span_minutes = float(temporal_reference.get("span_minutes", 1.0))
            if not np.isfinite(ref_span_minutes) or ref_span_minutes <= 0:
                ref_span_minutes = 1.0
        except Exception:
            ref_span_minutes = 1.0

    if ref_start is None:
        ref_start = ts

    elapsed_minutes = max(
        float((ts - ref_start).total_seconds() / 60.0),
        0.0,
    )
    elapsed_week_index = float(np.floor(elapsed_minutes / (7.0 * 24.0 * 60.0)))
    elapsed_ratio = float(elapsed_minutes / max(ref_span_minutes, 1.0))

    if not np.isfinite(elapsed_week_index):
        elapsed_week_index = 0.0
    if not np.isfinite(elapsed_ratio):
        elapsed_ratio = 0.0

    return {
        "weekday": float(int(ts.weekday())),
        "hour": float(int(ts.hour)),
        "month": month,
        "week_of_year": week_of_year,
        # Legacy — kept so old saved models can still traverse their trees.
        "elapsed_week_index": float(elapsed_week_index),
        "elapsed_ratio": float(elapsed_ratio),
    }

def _features_to_vector(features: dict[str, float], feature_names: list[str]) -> list[float]:
    vector: list[float] = []
    for name in feature_names:
        try:
            value = float(features.get(name, 0.0))
            if not np.isfinite(value):
                value = 0.0
        except Exception:
            value = 0.0
        vector.append(float(value))
    return vector


def _safe_numeric_values(values: list[float]) -> list[float]:
    clean: list[float] = []
    for value in values or []:
        try:
            value = float(value)
            if np.isfinite(value) and value >= 0.0:
                clean.append(value)
        except Exception:
            continue
    return clean


def _filter_arrival_values(
        values: list[float],
        filter_outliers: bool = False,
        outlier_threshold: float = 20.0,
) -> list[float]:
    """Filter values only when explicitly requested.

    Long inter-arrival gaps are often true process behavior, so outlier removal is
    disabled by default for the DBPS-faithful arrival model.
    """
    clean = _safe_numeric_values(values)
    if not clean:
        return []
    if not bool(filter_outliers):
        return clean
    try:
        filtered = remove_outliers(clean, m=float(outlier_threshold))
        filtered = _safe_numeric_values(list(filtered))
        return filtered if filtered else clean
    except Exception:
        return clean


def _summarize_interarrival_values(values: list[float]) -> dict:
    clean = _safe_numeric_values(values)
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(clean, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def get_observed_inter_arrival_histogram(
        inter_arrival_times: list[float],
        num_bins: int = 20,
        filter_outliers: bool = False,
        outlier_threshold: float = 20.0,
) -> dict:
    """Build a JSON-compatible empirical distribution for inter-arrival times.

    The raw observations are retained and used by the sampler by default. The
    histogram is kept for diagnostics / visualization and as a compact fallback.
    """
    values = _filter_arrival_values(
        inter_arrival_times,
        filter_outliers=filter_outliers,
        outlier_threshold=outlier_threshold,
    )

    if not values:
        return {
            "distribution_name": "histogram_sampling",
            "sampling_strategy": "empirical_observation",
            "histogram_data": {"cdf": [1.0], "bin_midpoints": [0.0]},
            "observations": [0.0],
            "num_observations": 0,
            "min_value": 0.0,
            "max_value": 0.0,
            "mean_value": 0.0,
            "median_value": 0.0,
            "fallback": {
                "dist_name": "fixed",
                "params": [0.0],
                "min_value": 0.0,
                "max_value": 0.0,
                "mean_value": 0.0,
            },
        }

    arr = np.asarray(values, dtype=float)

    if arr.size == 1 or float(np.min(arr)) == float(np.max(arr)):
        cdf = [1.0]
        bin_midpoints = [float(arr[0])]
    else:
        num_bins = int(max(2, min(int(num_bins), arr.size)))
        counts, edges = np.histogram(arr, bins=num_bins)
        probabilities = counts.astype(float) / max(float(counts.sum()), 1.0)
        non_empty = probabilities > 0.0
        probabilities = probabilities[non_empty]
        bin_midpoints_arr = ((edges[:-1] + edges[1:]) / 2.0).astype(float)[non_empty]
        cdf_arr = np.cumsum(probabilities)
        cdf_arr[-1] = 1.0
        cdf = [float(v) for v in cdf_arr.tolist()]
        bin_midpoints = [float(v) for v in bin_midpoints_arr.tolist()]

    try:
        fallback_dist, fallback_params = return_best_distribution(arr.tolist(), dist_search=DIST_SEARCH)
        fallback_dist_name = "fixed" if fallback_dist == "fixed" else fallback_dist.name
        fallback = {
            "dist_name": fallback_dist_name,
            "params": [float(v) for v in tuple(fallback_params)],
            "min_value": float(np.min(arr)),
            "max_value": float(np.max(arr)),
            "mean_value": float(np.mean(arr)),
        }
    except Exception:
        fallback = {
            "dist_name": "fixed",
            "params": [float(np.mean(arr))],
            "min_value": float(np.min(arr)),
            "max_value": float(np.max(arr)),
            "mean_value": float(np.mean(arr)),
        }

    return {
        "distribution_name": "histogram_sampling",
        "sampling_strategy": "empirical_observation",
        "histogram_data": {"cdf": cdf, "bin_midpoints": bin_midpoints},
        "observations": [float(v) for v in arr.tolist()],
        "num_observations": int(arr.size),
        "min_value": float(np.min(arr)),
        "max_value": float(np.max(arr)),
        "mean_value": float(np.mean(arr)),
        "median_value": float(np.median(arr)),
        "fallback": fallback,
    }


def _tree_to_rules(tree, feature_names: list[str], node_id: int = 0,
                   leaf_values: dict[str, dict] | None = None):
    """Serialize sklearn's CART tree into a small JSON-editable rule tree."""
    try:
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
    except Exception:
        return {"type": "leaf", "leaf_id": str(node_id), "dist": (leaf_values or {}).get(str(node_id), {})}

    if left == right or left < 0 or right < 0:
        leaf_id = str(node_id)
        leaf = {"type": "leaf", "leaf_id": leaf_id}
        if leaf_values is not None and leaf_id in leaf_values:
            leaf.update(leaf_values[leaf_id])
        return leaf

    feature_idx = int(tree.feature[node_id])
    feature_name = feature_names[feature_idx] if 0 <= feature_idx < len(feature_names) else f"feature_{feature_idx}"
    threshold = float(tree.threshold[node_id])

    return {
        "type": "node",
        "feature": feature_name,
        "threshold": threshold,
        "operator": "<=",
        "children": {
            "true": _tree_to_rules(tree, feature_names, left, leaf_values),
            "false": _tree_to_rules(tree, feature_names, right, leaf_values),
        },
    }


def _traverse_arrival_tree(rules: dict, features: dict[str, float]) -> tuple[dict | None, str]:
    node = rules
    visited = 0
    while isinstance(node, dict) and node.get("type") != "leaf" and visited < 128:
        visited += 1
        feature = node.get("feature")
        threshold = float(node.get("threshold", 0.0) or 0.0)
        try:
            value = float(features.get(feature, 0.0))
        except Exception:
            value = 0.0
        if value <= threshold:
            node = (node.get("children", {}) or {}).get("true")
        else:
            node = (node.get("children", {}) or {}).get("false")

    if isinstance(node, dict) and node.get("type") == "leaf":
        return node, str(node.get("leaf_id", "unknown"))
    return None, "missing_leaf"


def _sample_from_distribution_dict(dist_dict: dict, deterministic_time: bool = False) -> float:
    if not isinstance(dist_dict, dict):
        return 0.0

    dist_name = dist_dict.get("dist_name", "fixed")
    params = tuple(dist_dict.get("params", [dist_dict.get("mean_value", 0.0)]))
    min_value = float(dist_dict.get("min_value", 0.0))
    max_value = float(dist_dict.get("max_value", min_value))
    mean_value = float(dist_dict.get("mean_value", params[0] if params else 0.0))

    if deterministic_time or dist_name == "fixed":
        return max(0.0, mean_value)

    try:
        import scipy.stats as stats
        dist = getattr(stats, dist_name)
        value = sampling_from_dist(dist, params, min_value, max_value, mean_value, n_sample=1)[0]
        return max(0.0, float(value))
    except Exception:
        return max(0.0, mean_value)


def _sample_from_histogram(histogram: dict, deterministic_time: bool = False) -> float:
    if not isinstance(histogram, dict):
        return 0.0

    if deterministic_time:
        try:
            return max(0.0, float(histogram.get("mean_value", 0.0)))
        except Exception:
            return 0.0

    observations = histogram.get("observations", [])
    if histogram.get("sampling_strategy") == "empirical_observation" and observations:
        try:
            return max(0.0, float(random.choice(observations)))
        except Exception:
            pass

    hist_data = histogram.get("histogram_data", {})
    cdf = hist_data.get("cdf", [])
    bin_midpoints = hist_data.get("bin_midpoints", [])
    if cdf and bin_midpoints:
        u = random.random()
        for idx, boundary in enumerate(cdf):
            try:
                if u <= float(boundary):
                    return max(0.0, float(bin_midpoints[min(idx, len(bin_midpoints) - 1)]))
            except Exception:
                continue
        try:
            return max(0.0, float(bin_midpoints[-1]))
        except Exception:
            return 0.0

    if observations:
        try:
            return max(0.0, float(random.choice(observations)))
        except Exception:
            pass

    return _sample_from_distribution_dict(histogram.get("fallback", {}), deterministic_time)


def _resolve_depth_candidates(max_depths=None, max_depth: int | None = None) -> list[int]:
    if max_depths is None:
        if max_depth is None:
            return [1, 2, 3, 4, 5]
        max_depths = [max_depth]

    candidates: list[int] = []
    try:
        for value in list(max_depths):
            value = int(value)
            if 1 <= value <= 5:
                candidates.append(value)
    except Exception:
        candidates = []

    if not candidates:
        if max_depth is None:
            candidates = [1, 2, 3, 4, 5]
        else:
            candidates = [max(1, min(5, int(max_depth)))]

    return sorted(set(candidates))


def _fit_arrival_tree(
        X: np.ndarray,
        y: np.ndarray,
        depth_candidates: list[int],
        cv_folds: int = 3,
):
    from sklearn.tree import DecisionTreeRegressor

    depth_candidates = _resolve_depth_candidates(depth_candidates)
    n_samples = int(len(y))

    if n_samples <= 1:
        return None, {
            "enabled": False,
            "method": "single_leaf_insufficient_samples",
            "depth_candidates": depth_candidates,
            "selected_max_depth": None,
        }

    if len(depth_candidates) == 1 or n_samples < int(cv_folds):
        selected_depth = int(depth_candidates[0])
        tree = DecisionTreeRegressor(max_depth=selected_depth, random_state=72)
        tree.fit(X, y)
        return tree, {
            "enabled": False,
            "method": "fixed_depth",
            "depth_candidates": depth_candidates,
            "selected_max_depth": selected_depth,
        }

    try:
        from sklearn.model_selection import GridSearchCV
        estimator = DecisionTreeRegressor(random_state=72)
        grid = GridSearchCV(
            estimator=estimator,
            param_grid={"max_depth": depth_candidates},
            cv=min(int(cv_folds), n_samples),
            scoring="neg_mean_absolute_error",
        )
        grid.fit(X, y)
        return grid.best_estimator_, {
            "enabled": True,
            "method": "GridSearchCV",
            "scoring": "neg_mean_absolute_error",
            "cv": int(min(int(cv_folds), n_samples)),
            "depth_candidates": depth_candidates,
            "selected_max_depth": int(grid.best_params_.get("max_depth")),
            "best_score": float(grid.best_score_),
        }
    except Exception as exc:
        selected_depth = int(depth_candidates[0])
        tree = DecisionTreeRegressor(max_depth=selected_depth, random_state=72)
        tree.fit(X, y)
        return tree, {
            "enabled": False,
            "method": "fixed_depth_after_cv_failure",
            "depth_candidates": depth_candidates,
            "selected_max_depth": selected_depth,
            "error": str(exc),
        }


def discover_case_arrival_model_decision_tree_interarrival(
        log: EventLog | pd.DataFrame,
        calendar_arrival: dict | None = None,
        num_bins: int = 20,
        filter_outliers: bool = False,
        outlier_threshold: float = 20.0,
        max_depth: int | None = None,
        max_depths=None,
        min_inter_arrivals_for_tree: int = 4,
        cv_folds: int = 3,
        **_unused_kwargs,
) -> dict:
    """Discover a DBPS-faithful probabilistic decision tree for arrivals.

    Training samples:
        X = [
            weekday(previous_case_arrival),
            hour(previous_case_arrival),
            month(previous_case_arrival),
            week_of_year(previous_case_arrival),
        ]
        y = raw wall-clock minutes until the next case arrival

    Each leaf stores the raw inter-arrival observations and a fitted fallback
    distribution. No arrival calendar compression or validation-driven global
    arrival scale is used.
    """
    arrivals = _extract_case_arrival_times(log)
    temporal_reference = _build_temporal_reference(arrivals)
    samples: list[dict] = []
    global_values: list[float] = []

    for i in range(1, len(arrivals)):
        previous_arrival = pd.Timestamp(arrivals[i - 1])
        current_arrival = pd.Timestamp(arrivals[i])
        value = _raw_inter_arrival_minutes(previous_arrival, current_arrival)
        if not np.isfinite(value) or value < 0.0:
            continue
        features = _arrival_context_features(previous_arrival, temporal_reference=temporal_reference)
        samples.append({"timestamp": previous_arrival, "features": features, "delta_min": float(value)})
        global_values.append(float(value))

    depth_candidates = _resolve_depth_candidates(max_depths=max_depths, max_depth=max_depth)
    feature_names = ARRIVAL_TREE_FEATURE_NAMES[:]

    global_histogram = get_observed_inter_arrival_histogram(
        global_values,
        num_bins=num_bins,
        filter_outliers=filter_outliers,
        outlier_threshold=outlier_threshold,
    )

    def _single_leaf_model(reason: str) -> dict:
        root_leaf = {
            "type": "leaf",
            "leaf_id": "root",
            "n": int(len(global_values)),
            "mean_delta_min": float(global_histogram.get("mean_value", 0.0)),
            "median_delta_min": float(global_histogram.get("median_value", 0.0)),
            "dist": global_histogram,
        }
        return {
            "enabled": True,
            "model_type": DECISION_TREE_MODEL_TYPE,
            "arrival_calendar": calendar_arrival or {},
            "temporal_reference": temporal_reference,
            "feature_names": feature_names,
            "tree_model": root_leaf,
            "leaf_distributions": {"root": root_leaf},
            "global_inter_arrival": global_histogram,
            "fallback": global_histogram.get("fallback", {}),
            "sampling": {
                "inter_arrival_time_mode": "wall_clock_minutes",
                "advance_with_calendar": False,
                "use_intensity_trend": False,
                "use_arrival_time_scale": False,
            },
            "metadata": {
                "dbps_base_features": ["weekday", "hour"],
                "seasonal_enhanced": True,
                "seasonal_features": ["month", "week_of_year"],
                "tree_enabled": False,
                "fallback_reason": reason,
                "num_arrivals": int(len(arrivals)),
                "num_inter_arrivals": int(len(global_values)),
                "feature_names": feature_names,
                "depth_candidates": depth_candidates,
                "global_interarrival": _summarize_interarrival_values(global_values),
                "target": "raw_wall_clock_interarrival_minutes",
                "temporal_reference": temporal_reference,
                "outlier_filtering_enabled": bool(filter_outliers),
            },
            "intensity_trend": {"enabled": False, "method": "disabled_dbps_faithful", "forecast_multiplier": 1.0},
            "runtime_guard": {"enabled": False, "method": "disabled_dbps_faithful"},
        }

    if len(samples) < int(min_inter_arrivals_for_tree):
        return _single_leaf_model("insufficient_inter_arrivals")

    try:
        X = np.asarray([_features_to_vector(row["features"], feature_names) for row in samples], dtype=float)
        y = np.asarray([float(row["delta_min"]) for row in samples], dtype=float)

        tree, cv_info = _fit_arrival_tree(
            X=X,
            y=y,
            depth_candidates=depth_candidates,
            cv_folds=cv_folds,
        )
        if tree is None:
            return _single_leaf_model("tree_training_skipped")

        leaf_ids = tree.apply(X)
        values_by_leaf: dict[str, list[float]] = {}
        for leaf_id, row in zip(leaf_ids, samples):
            values_by_leaf.setdefault(str(int(leaf_id)), []).append(float(row["delta_min"]))

        leaf_distributions: dict[str, dict] = {}
        for leaf_id, values in values_by_leaf.items():
            hist = get_observed_inter_arrival_histogram(
                values,
                num_bins=num_bins,
                filter_outliers=filter_outliers,
                outlier_threshold=outlier_threshold,
            )
            leaf_distributions[str(leaf_id)] = {
                "type": "leaf",
                "leaf_id": str(leaf_id),
                "n": int(len(values)),
                "mean_delta_min": float(hist.get("mean_value", 0.0)),
                "median_delta_min": float(hist.get("median_value", 0.0)),
                "dist": hist,
            }

        tree_rules = _tree_to_rules(tree.tree_, feature_names, node_id=0, leaf_values=leaf_distributions)
        return {
            "enabled": True,
            "model_type": DECISION_TREE_MODEL_TYPE,
            "arrival_calendar": calendar_arrival or {},
            "temporal_reference": temporal_reference,
            "feature_names": feature_names,
            "tree_model": tree_rules,
            "leaf_distributions": leaf_distributions,
            "global_inter_arrival": global_histogram,
            "fallback": global_histogram.get("fallback", {}),
            "sampling": {
                "inter_arrival_time_mode": "wall_clock_minutes",
                "advance_with_calendar": False,
                "use_intensity_trend": False,
                "use_arrival_time_scale": False,
            },
            "metadata": {
                "dbps_base_features": ["weekday", "hour"],
                "seasonal_enhanced": True,
                "seasonal_features": ["month", "week_of_year"],
                "tree_enabled": True,
                "num_arrivals": int(len(arrivals)),
                "num_inter_arrivals": int(len(global_values)),
                "feature_names": feature_names,
                "depth_candidates": depth_candidates,
                "cv": cv_info,
                "selected_max_depth": cv_info.get("selected_max_depth"),
                "n_leaves": int(len(leaf_distributions)),
                "global_interarrival": _summarize_interarrival_values(global_values),
                "target": "raw_wall_clock_interarrival_minutes",
                "temporal_reference": temporal_reference,
                "outlier_filtering_enabled": bool(filter_outliers),
            },
            "intensity_trend": {"enabled": False, "method": "disabled_dbps_faithful", "forecast_multiplier": 1.0},
            "runtime_guard": {"enabled": False, "method": "disabled_dbps_faithful"},
        }
    except Exception as exc:
        model = _single_leaf_model("tree_training_failed")
        model.setdefault("metadata", {})["tree_training_error"] = str(exc)
        return model


def discover_case_arrival_model_context_histogram(
        log: EventLog | pd.DataFrame,
        calendar_arrival: dict | None = None,
        num_bins: int = 20,
        filter_outliers: bool = False,
        outlier_threshold: float = 20.0,
        **_unused_kwargs,
) -> dict:
    """Backward-compatible simple global histogram model.

    This is no longer used by default. It keeps old call sites working if they
    explicitly request CONTEXT_MODEL_TYPE.
    """
    arrivals = _extract_case_arrival_times(log)
    values: list[float] = []
    for i in range(1, len(arrivals)):
        values.append(_raw_inter_arrival_minutes(pd.Timestamp(arrivals[i - 1]), pd.Timestamp(arrivals[i])))
    global_histogram = get_observed_inter_arrival_histogram(values, num_bins, filter_outliers, outlier_threshold)
    return {
        "enabled": True,
        "model_type": CONTEXT_MODEL_TYPE,
        "arrival_calendar": calendar_arrival or {},
        "global_inter_arrival": global_histogram,
        "fallback_order": ["global"],
        "sampling": {
            "inter_arrival_time_mode": "wall_clock_minutes",
            "advance_with_calendar": False,
            "use_intensity_trend": False,
            "use_arrival_time_scale": False,
        },
        "statistics": {
            "num_arrivals": int(len(arrivals)),
            "num_inter_arrivals": int(len(values)),
            "global_interarrival": _summarize_interarrival_values(values),
        },
        "fallback": global_histogram.get("fallback", {}),
        "intensity_trend": {"enabled": False, "method": "disabled_dbps_faithful", "forecast_multiplier": 1.0},
        "runtime_guard": {"enabled": False, "method": "disabled_dbps_faithful"},
    }


def discover_arrival_time(
        log: EventLog | pd.DataFrame,
        calendar_arrival: dict | None = None,
        max_depths=None,
        model_type: str = DECISION_TREE_MODEL_TYPE,
        **kwargs,
):
    """Public entry point used by SimulatorParameters.

    Default behavior is the DBPS-based temporal-position-enhanced probabilistic
    decision tree over raw wall-clock inter-arrival minutes. If max_depths contains a single value, that
    depth is fixed; if it contains multiple values, internal GridSearchCV is used.
    """
    if model_type == CONTEXT_MODEL_TYPE:
        return discover_case_arrival_model_context_histogram(
            log=log,
            calendar_arrival=calendar_arrival or {},
            **kwargs,
        )

    explicit_max_depth = kwargs.pop("max_depth", None)
    return discover_case_arrival_model_decision_tree_interarrival(
        log=log,
        calendar_arrival=calendar_arrival or {},
        max_depths=max_depths,
        max_depth=explicit_max_depth,
        **kwargs,
    )


def _context_keys(current_time) -> tuple[str, str]:
    features = _arrival_context_features(current_time)
    return str(int(features["weekday"])), str(int(features["hour"]))


def _select_context_histogram(arrival_model: dict, current_time) -> tuple[dict | None, str]:
    # Legacy support only. New DBPS-faithful models use tree traversal.
    weekday, hour = _context_keys(current_time)

    weekday_hour = arrival_model.get("weekday_hour_inter_arrivals", {})
    if isinstance(weekday_hour, dict):
        hist = (weekday_hour.get(weekday) or {}).get(hour)
        if isinstance(hist, dict):
            return hist, "weekday_hour"

    weekday_map = arrival_model.get("weekday_inter_arrivals", {})
    hist = weekday_map.get(weekday) if isinstance(weekday_map, dict) else None
    if isinstance(hist, dict):
        return hist, "weekday"

    hour_map = arrival_model.get("hour_inter_arrivals", {})
    hist = hour_map.get(hour) if isinstance(hour_map, dict) else None
    if isinstance(hist, dict):
        return hist, "hour"

    hist = arrival_model.get("global_inter_arrival") or arrival_model.get("inter_arrival_times")
    if isinstance(hist, dict):
        return hist, "global"

    return None, "none"


def sample_case_arrival_delta(
        arrival_model: Any,
        current_time=None,
        deterministic_time: bool = False,
        diagnostics: dict | None = None,
) -> float:
    """Runtime sampler for case-arrival models."""

    def _record_source(source: str, sampled_value: float | None = None):
        if diagnostics is None:
            return
        try:
            counts = diagnostics.setdefault("source_counts", {})
            counts[source] = int(counts.get(source, 0)) + 1
            if sampled_value is not None:
                by_source = diagnostics.setdefault("sampled_delta_min_by_source", {})
                values = by_source.setdefault(source, [])
                if len(values) < int(diagnostics.get("max_values_per_source", 5000)):
                    values.append(float(sampled_value))
        except Exception:
            pass

    def _record_tree_leaf(leaf_id: str, sampled_value: float | None = None):
        if diagnostics is None:
            return
        try:
            counts = diagnostics.setdefault("arrival_tree_leaf_counts", {})
            counts[str(leaf_id)] = int(counts.get(str(leaf_id), 0)) + 1
            if sampled_value is not None:
                by_leaf = diagnostics.setdefault("sampled_delta_min_by_arrival_tree_leaf", {})
                values = by_leaf.setdefault(str(leaf_id), [])
                if len(values) < int(diagnostics.get("max_values_per_source", 5000)):
                    values.append(float(sampled_value))
        except Exception:
            pass

    def _record_no_trend():
        if diagnostics is None:
            return
        try:
            diagnostics["intensity_trend_enabled"] = False
            diagnostics["intensity_trend_forecast_multiplier"] = 1.0
            values = diagnostics.setdefault("trend_multipliers", [])
            if len(values) < int(diagnostics.get("max_values_per_source", 5000)):
                values.append(1.0)
        except Exception:
            pass

    if arrival_model is None:
        _record_source("none", 0.0)
        _record_no_trend()
        return 0.0

    if isinstance(arrival_model, dict):
        model_type = arrival_model.get("model_type")

        if model_type == DECISION_TREE_MODEL_TYPE:
            features = _arrival_context_features(
                current_time,
                temporal_reference=arrival_model.get("temporal_reference", {}),
            )
            leaf, leaf_id = _traverse_arrival_tree(arrival_model.get("tree_model", {}), features)
            histogram = None
            if isinstance(leaf, dict):
                histogram = leaf.get("dist")
            if not isinstance(histogram, dict):
                leaf_model = (arrival_model.get("leaf_distributions", {}) or {}).get(str(leaf_id), {})
                histogram = leaf_model.get("dist") if isinstance(leaf_model, dict) else None
            if not isinstance(histogram, dict):
                histogram = arrival_model.get("global_inter_arrival")
                leaf_id = "global_fallback"

            if isinstance(histogram, dict):
                value = max(0.0, _sample_from_histogram(histogram, deterministic_time))
            else:
                value = max(0.0, _sample_from_distribution_dict(arrival_model.get("fallback", {}), deterministic_time))
                leaf_id = "fallback_distribution"

            _record_source("decision_tree_interarrival", value)
            _record_tree_leaf(leaf_id, value)
            _record_no_trend()
            return float(value)

        if model_type == CONTEXT_MODEL_TYPE:
            histogram, source = _select_context_histogram(arrival_model, current_time)
            if isinstance(histogram, dict):
                value = max(0.0, _sample_from_histogram(histogram, deterministic_time))
                _record_source(source, value)
                _record_no_trend()
                return float(value)
            value = max(0.0, _sample_from_distribution_dict(arrival_model.get("fallback", {}), deterministic_time))
            _record_source("fallback_distribution", value)
            _record_no_trend()
            return float(value)

        if model_type == LEGACY_GLOBAL_MODEL_TYPE:
            histogram = arrival_model.get("inter_arrival_times", {})
            value = max(0.0, _sample_from_histogram(histogram, deterministic_time))
            _record_source("legacy_global", value)
            _record_no_trend()
            return float(value)

        if arrival_model.get("distribution_name") == "histogram_sampling":
            value = max(0.0, _sample_from_histogram(arrival_model, deterministic_time))
            _record_source("histogram_sampling", value)
            _record_no_trend()
            return float(value)

        if "dist_name" in arrival_model:
            value = max(0.0, _sample_from_distribution_dict(arrival_model, deterministic_time))
            _record_source("distribution", value)
            _record_no_trend()
            return float(value)

        if "fallback" in arrival_model:
            value = max(0.0, _sample_from_distribution_dict(arrival_model["fallback"], deterministic_time))
            _record_source("fallback_distribution", value)
            _record_no_trend()
            return float(value)

        _record_source("unsupported_dict", 0.0)
        _record_no_trend()
        return 0.0

    # Legacy tuple: (dist, params, min, max, mean).
    if isinstance(arrival_model, tuple) and len(arrival_model) >= 5:
        if deterministic_time:
            value = max(0.0, float(arrival_model[4]))
            _record_source("tuple_mean", value)
            _record_no_trend()
            return float(value)
        value = sampling_from_dist(
            arrival_model[0],
            arrival_model[1],
            arrival_model[2],
            arrival_model[3],
            arrival_model[4],
            n_sample=1,
        )[0]
        value = max(0.0, float(value))
        _record_source("tuple_distribution", value)
        _record_no_trend()
        return float(value)

    # Legacy DecisionRules-like objects.
    try:
        sampled = arrival_model.get_value({})
        value = max(0.0, float(sampled))
        _record_source("decision_rules_legacy", value)
        _record_no_trend()
        return float(value)
    except Exception:
        _record_source("unsupported", 0.0)
        _record_no_trend()
        return 0.0


# ---------------------------------------------------------------------------
# Backward-compatible imports for execution-time and waiting-time discovery.
# The DBPS-faithful arrival model above is the active arrival implementation.
# The following imports keep older simulator code paths working when they still
# call discover_execution_time_distributions() or discover_waiting_time().
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from drqsim.utils.common_utils import count_false_hours
from drqsim.utils.rule_utils import DecisionRules


# ===========================================================================
# BACKWARD-COMPATIBLE EXECUTION-TIME AND WAITING-TIME DISCOVERY
# ===========================================================================

def discover_execution_time_distributions(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        resources: list,
        calendars: dict,
        max_depths: list = range(1, 6),
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
    ) -> dict:
    """
    Discover execution-time distributions / probabilistic decision rules.

    This function is kept for compatibility with older drq/DBPS code paths.
    It does not affect the DBPS-faithful case-arrival model above.
    """
    if not max_depths:
        return find_best_distribution_ex(
            df_features=df_features,
            activity_labels=net_transition_labels,
            calendars=calendars,
        )

    return build_models_ex(
        df_features=df_features,
        activity_labels=net_transition_labels,
        resources=resources,
        calendars=calendars,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
        max_depths=max_depths,
    )


def discover_waiting_time(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        resources: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths: list = range(1, 6),
    ) -> dict:
    """
    Discover resource-level waiting-time distributions / probabilistic rules.

    This is activity-instance waiting time, not case arrival time.
    """
    if not max_depths:
        return find_best_distribution_wt(
            df_features=df_features,
            resources=resources,
            calendars=calendars,
        )

    return build_models_wt(
        df_features=df_features,
        activity_labels=net_transition_labels,
        resources=resources,
        calendars=calendars,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
        max_depths=max_depths,
    )


def _empty_decision_rules(default_value: float = 0.0) -> DecisionRules:
    clf = DecisionRules()
    clf.rules = {
        0: {
            "value": float(default_value),
            "sampled": [float(default_value)],
            "dist": ("fixed", (float(default_value),), float(default_value), float(default_value), float(default_value)),
        }
    }
    return clf


def _fit_regression_tree_with_optional_cv(
        X: pd.DataFrame,
        y: pd.Series,
        max_depths: list,
        random_state: int = 72,
):
    """
    Fit a shallow decision tree. If multiple depths and enough samples exist,
    use GridSearchCV. Otherwise fit the first feasible depth.
    """
    candidates = []
    try:
        for depth in list(max_depths or []):
            depth = int(depth)
            if depth >= 1:
                candidates.append(depth)
    except Exception:
        candidates = []

    if not candidates:
        candidates = [5]

    candidates = sorted(set(candidates))

    if len(X) == 0:
        return None

    if len(candidates) > 1 and len(X) >= 4:
        cv = min(3, len(X))
        try:
            grid_search = GridSearchCV(
                estimator=DecisionTreeRegressor(random_state=random_state),
                param_grid={"max_depth": candidates},
                cv=cv,
                scoring="neg_mean_absolute_error",
            )
            grid_search.fit(X, y)
            return grid_search.best_estimator_
        except Exception:
            pass

    clf = DecisionTreeRegressor(
        max_depth=int(candidates[0]),
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def _fit_leaf_distribution_for_rules(
        clf: DecisionRules,
        clf_mean,
        X: pd.DataFrame,
        y: pd.Series,
        remove_leaf_outliers: bool = True,
):
    """Attach fitted probabilistic distributions and sampled values to leaves."""
    leaf_indices = clf_mean.apply(X)

    y_leaf = pd.DataFrame({
        "Leaf": leaf_indices,
        "Y": y,
    })

    for leaf_id in list(y_leaf["Leaf"].unique()):
        values = y_leaf[y_leaf["Leaf"] == leaf_id]["Y"].dropna().tolist()
        if remove_leaf_outliers:
            try:
                filtered = remove_outliers(values)
                if len(filtered) > 0:
                    values = list(filtered)
            except Exception:
                pass

        values = [
            float(v)
            for v in values
            if np.isfinite(float(v)) and float(v) >= 0.0
        ]

        if not values:
            dist = "fixed"
            params = (0.0,)
            min_value = max_value = mean_value = 0.0
            sampled = [0.0]
        else:
            min_value = float(np.min(values))
            max_value = float(np.max(values))
            mean_value = float(np.mean(values))
            try:
                dist, params = return_best_distribution(
                    values,
                    dist_search=DIST_SEARCH,
                )
            except Exception:
                dist = "fixed"
                params = (mean_value,)

            try:
                sampled = sampling_from_dist(
                    dist,
                    params,
                    min_value,
                    max_value,
                    mean_value,
                    n_sample=max(len(values), 1000),
                )
                sampled = [float(v) for v in sampled]
            except Exception:
                sampled = values or [mean_value]

        if leaf_id in clf.rules:
            clf.rules[leaf_id]["dist"] = dist, params, min_value, max_value
            clf.rules[leaf_id]["sampled"] = sampled
        else:
            clf.rules[leaf_id] = {
                "value": mean_value,
                "dist": (dist, params, min_value, max_value),
                "sampled": sampled,
            }


def build_models_ex(
        df_features: pd.DataFrame,
        activity_labels: list,
        resources: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths: list = range(1, 6),
    ) -> dict:
    """
    Build one probabilistic execution-time decision tree per activity.
    Execution time = end_t - start_t, corrected by the executing resource calendar.
    """
    df = build_training_df_ex(
        df_features=df_features,
        resources=resources,
        activity_labels=activity_labels,
        calendars=calendars,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
    )

    models_act = {}

    for act in tqdm(activity_labels):
        df_act = df[df["activity_executed"] == act].iloc[:, 1:]

        if df_act.empty or "execution_time" not in df_act.columns:
            models_act[act] = _empty_decision_rules(0.0)
            continue

        X = df_act.drop(columns=["execution_time"])
        y = df_act["execution_time"]

        if len(X) == 0:
            models_act[act] = _empty_decision_rules(0.0)
            continue

        clf_mean = _fit_regression_tree_with_optional_cv(
            X=X,
            y=y,
            max_depths=max_depths,
            random_state=72,
        )
        if clf_mean is None:
            models_act[act] = _empty_decision_rules(0.0)
            continue

        clf = DecisionRules()
        clf.from_decision_tree(clf_mean)
        _fit_leaf_distribution_for_rules(
            clf=clf,
            clf_mean=clf_mean,
            X=X,
            y=y,
            remove_leaf_outliers=True,
        )

        models_act[act] = clf

    return models_act


def build_models_wt(
        df_features: pd.DataFrame,
        activity_labels: list,
        resources: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths: list = range(1, 6),
    ) -> dict:
    """
    Build one probabilistic waiting-time decision tree per resource.
    Waiting time = start_t - enabled_t, corrected by the selected resource calendar.
    """
    df = build_training_df_wt(
        df_features=df_features,
        net_transition_labels=activity_labels,
        calendars=calendars,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
    )

    models_res = {}

    for res in tqdm(resources):
        df_res = df[df["resource"] == res].iloc[:, 1:]

        if df_res.empty or "waiting_time" not in df_res.columns:
            models_res[res] = _empty_decision_rules(0.0)
            continue

        X = df_res.drop(columns=["waiting_time"])
        y = df_res["waiting_time"]

        if len(X) == 0:
            models_res[res] = _empty_decision_rules(0.0)
            continue

        clf_mean = _fit_regression_tree_with_optional_cv(
            X=X,
            y=y,
            max_depths=max_depths,
            random_state=72,
        )
        if clf_mean is None:
            models_res[res] = _empty_decision_rules(0.0)
            continue

        clf = DecisionRules()
        clf.from_decision_tree(clf_mean)
        _fit_leaf_distribution_for_rules(
            clf=clf,
            clf_mean=clf_mean,
            X=X,
            y=y,
            remove_leaf_outliers=True,
        )

        models_res[res] = clf

    return models_res


def build_training_df_ex(
        df_features: pd.DataFrame,
        resources: list,
        activity_labels: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
    ) -> pd.DataFrame:
    """
    Build training data for execution-time discovery.
    """
    required_columns = ["transition_label", "resource", "start_t", "end_t"]
    available_columns = [
        col for col in required_columns + label_data_attributes + activity_labels
        if col in df_features.columns
    ]

    df_et = df_features[available_columns].copy()

    if "start_t" not in df_et.columns or "end_t" not in df_et.columns:
        return pd.DataFrame(columns=["activity_executed", "execution_time"])

    df_et = df_et[~df_et["start_t"].isna()].copy()

    def _compute_execution_time(row):
        try:
            resource = row["resource"]
            calendar = calendars.get(resource, {}) if isinstance(calendars, dict) else {}
            return max(
                (row["end_t"] - row["start_t"]).total_seconds() / 60.0
                - count_false_hours(calendar, row["start_t"], row["end_t"]) * 60.0,
                0.0,
            )
        except Exception:
            return 0.0

    df_et["execution_time"] = df_et.apply(_compute_execution_time, axis=1)
    df_et.drop(columns=[c for c in ["start_t", "end_t"] if c in df_et.columns], inplace=True)

    if "transition_label" in df_et.columns:
        df_et.rename(columns={"transition_label": "activity_executed"}, inplace=True)
    else:
        df_et["activity_executed"] = None

    if "resource" in df_et.columns:
        for r in resources:
            df_et["resource = " + str(r)] = (df_et["resource"] == r).astype(int)
        del df_et["resource"]

    for a in label_data_attributes_categorical:
        if a not in df_et.columns:
            continue
        for v in values_categorical.get(a, []):
            df_et[str(a) + " = " + str(v)] = (df_et[a] == v).astype(int)
        del df_et[a]

    df_et.reset_index(drop=True, inplace=True)
    return df_et


def build_training_df_wt(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
    ) -> pd.DataFrame:
    """
    Build training data for resource waiting-time discovery.
    """
    required_columns = ["resource", "start_t", "enabled_t", "res_workload"]
    available_columns = [
        col for col in required_columns + label_data_attributes + net_transition_labels
        if col in df_features.columns
    ]

    df_wt = df_features[available_columns].copy()

    if "start_t" not in df_wt.columns or "enabled_t" not in df_wt.columns:
        return pd.DataFrame(columns=["resource", "waiting_time"])

    df_wt = df_wt[~df_wt["start_t"].isna()].copy()

    def _compute_waiting_time(row):
        try:
            resource = row["resource"]
            calendar = calendars.get(resource, {}) if isinstance(calendars, dict) else {}
            return max(
                (row["start_t"] - row["enabled_t"]).total_seconds() / 60.0
                - count_false_hours(calendar, row["enabled_t"], row["start_t"]) * 60.0,
                0.0,
            )
        except Exception:
            return 0.0

    df_wt["waiting_time"] = df_wt.apply(_compute_waiting_time, axis=1)
    df_wt.drop(columns=[c for c in ["start_t", "enabled_t"] if c in df_wt.columns], inplace=True)

    if "res_workload" in df_wt.columns:
        df_wt.rename(columns={"res_workload": "workload"}, inplace=True)
    else:
        df_wt["workload"] = 0

    for a in label_data_attributes_categorical:
        if a not in df_wt.columns:
            continue
        for v in values_categorical.get(a, []):
            df_wt[str(a) + " = " + str(v)] = (df_wt[a] == v).astype(int)
        del df_wt[a]

    df_wt.reset_index(drop=True, inplace=True)
    return df_wt


def find_best_distribution_ex(
        df_features: pd.DataFrame,
        activity_labels: list,
        calendars: dict,
    ) -> dict:
    """
    Fit one global execution-time distribution per activity.
    """
    if not {"transition_label", "resource", "start_t", "end_t"}.issubset(df_features.columns):
        return {
            act: ("fixed", (0.0,), 0.0, 0.0, 0.0)
            for act in activity_labels
        }

    df_et = df_features[["transition_label", "resource", "start_t", "end_t"]].copy()
    df_et = df_et[~df_et["start_t"].isna()].copy()

    def _compute_execution_time(row):
        try:
            calendar = calendars.get(row["resource"], {}) if isinstance(calendars, dict) else {}
            return max(
                (row["end_t"] - row["start_t"]).total_seconds() / 60.0
                - count_false_hours(calendar, row["start_t"], row["end_t"]) * 60.0,
                0.0,
            )
        except Exception:
            return 0.0

    df_et["execution_time"] = df_et.apply(_compute_execution_time, axis=1)

    activity_exec_time_distributions = {}
    for act in activity_labels:
        values = df_et[df_et["transition_label"] == act]["execution_time"].dropna().tolist()
        try:
            filtered = remove_outliers(values)
            if len(filtered) > 0:
                values = list(filtered)
        except Exception:
            pass

        values = [
            float(v)
            for v in values
            if np.isfinite(float(v)) and float(v) >= 0.0
        ]

        if not values:
            activity_exec_time_distributions[act] = ("fixed", (0.0,), 0.0, 0.0, 0.0)
            continue

        dist, params = return_best_distribution(values, dist_search=DIST_SEARCH)
        activity_exec_time_distributions[act] = (
            dist,
            params,
            float(np.min(values)),
            float(np.max(values)),
            float(np.mean(values)),
        )

    return activity_exec_time_distributions


def find_best_distribution_wt(
        df_features: pd.DataFrame,
        resources: list,
        calendars: dict,
    ) -> dict:
    """
    Fit one global waiting-time distribution per resource.
    """
    if not {"resource", "start_t", "enabled_t"}.issubset(df_features.columns):
        return {
            res: ("fixed", (0.0,), 0.0, 0.0, 0.0)
            for res in resources
        }

    df_wt = df_features[["resource", "start_t", "enabled_t"]].copy()
    df_wt = df_wt[~df_wt["start_t"].isna()].copy()

    def _compute_waiting_time(row):
        try:
            calendar = calendars.get(row["resource"], {}) if isinstance(calendars, dict) else {}
            return max(
                (row["start_t"] - row["enabled_t"]).total_seconds() / 60.0
                - count_false_hours(calendar, row["enabled_t"], row["start_t"]) * 60.0,
                0.0,
            )
        except Exception:
            return 0.0

    df_wt["waiting_time"] = df_wt.apply(_compute_waiting_time, axis=1)
    df_wt.reset_index(drop=True, inplace=True)

    res_waiting_time_distributions = {}
    for res in resources:
        values = df_wt[df_wt["resource"] == res]["waiting_time"].dropna().tolist()
        try:
            filtered = remove_outliers(values)
            if len(filtered) > 0:
                values = list(filtered)
        except Exception:
            pass

        values = [
            float(v)
            for v in values
            if np.isfinite(float(v)) and float(v) >= 0.0
        ]

        if not values:
            res_waiting_time_distributions[res] = ("fixed", (0.0,), 0.0, 0.0, 0.0)
            continue

        dist, params = return_best_distribution(values, dist_search=DIST_SEARCH)
        res_waiting_time_distributions[res] = (
            dist,
            params,
            float(np.min(values)),
            float(np.max(values)),
            float(np.mean(values)),
        )

    return res_waiting_time_distributions

