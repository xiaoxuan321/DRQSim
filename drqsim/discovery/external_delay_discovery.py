# drq/discovery/external_delay_discovery.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from bisect import bisect_right

import math
import warnings

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from drqsim.discovery.queue_scheduling_discovery import QUEUE_FEATURE_NAMES
from drqsim.utils.distribution_utils import (
    remove_outliers,
    return_best_distribution,
)


DIST_SEARCH = ["fixed", "norm", "expon", "uniform"]
ACTIVITY_PREFIX = "activity = "
RESOURCE_PREFIX = "resource = "


# ---------------------------------------------------------------------
# Waiting-time decomposition shared by Queue / Calendar / External Delay
# ---------------------------------------------------------------------
_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_WEEKDAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _calendar_is_active_hour(calendar: dict, weekday: int, hour: int, default: bool = False) -> bool:
    if not isinstance(calendar, dict) or not calendar:
        return bool(default)
    try:
        weekday = max(0, min(6, int(weekday)))
        hour = max(0, min(23, int(hour)))
    except Exception:
        return bool(default)

    wd_keys = [
        weekday, str(weekday), _WEEKDAY_NAMES[weekday], _WEEKDAY_SHORT[weekday],
        _WEEKDAY_NAMES[weekday].lower(), _WEEKDAY_SHORT[weekday].lower(),
    ]
    hour_keys = [hour, str(hour)]
    for wd_key in wd_keys:
        day = calendar.get(wd_key)
        if not isinstance(day, dict):
            continue
        for hour_key in hour_keys:
            if hour_key in day:
                return bool(day.get(hour_key))
    return bool(default)


def _active_calendar_hour_count(calendar: dict) -> int:
    if not isinstance(calendar, dict) or not calendar:
        return 0
    return int(sum(
        1
        for wd in range(7)
        for hour in range(24)
        if _calendar_is_active_hour(calendar, wd, hour, default=False)
    ))


def _subtract_intervals(base_start, base_end, blocked: list[tuple]) -> list[tuple]:
    """Return portions of [base_start, base_end) not covered by blocked intervals."""
    if base_end <= base_start:
        return []
    clean = []
    for start, end in blocked or []:
        try:
            start = max(base_start, start)
            end = min(base_end, end)
            if end > start:
                clean.append((start, end))
        except Exception:
            continue
    if not clean:
        return [(base_start, base_end)]
    clean.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for start, end in clean:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    result = []
    cursor = base_start
    for start, end in merged:
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < base_end:
        result.append((cursor, base_end))
    return result


def _capacity_saturated_intervals(
        window_start,
        window_end,
        resource_intervals: list,
        current_index: int | None,
        capacity: int,
) -> list[tuple]:
    """Intervals where the assigned resource has no free capacity."""
    capacity = max(1, int(capacity or 1))
    points = []
    for item in resource_intervals or []:
        try:
            idx, _other_enabled, other_start, other_end = item
        except Exception:
            continue
        if current_index is not None and idx == current_index:
            continue
        if pd.isna(other_start) or pd.isna(other_end):
            continue
        start = max(window_start, other_start)
        end = min(window_end, other_end)
        if end <= start:
            continue
        # End before start at identical timestamp (half-open interval semantics).
        points.append((start, +1))
        points.append((end, -1))

    if not points:
        return []

    by_time: dict[Any, int] = {}
    for when, delta in points:
        by_time[when] = by_time.get(when, 0) + int(delta)
    times = sorted(by_time)

    active = 0
    saturated = []
    previous = window_start
    for when in times:
        when = max(window_start, min(window_end, when))
        if when > previous and active >= capacity:
            saturated.append((previous, when))
        active += by_time.get(when, 0)
        previous = when
    if previous < window_end and active >= capacity:
        saturated.append((previous, window_end))
    return saturated


def _resource_capacity_for_decomposition(
        resource: str,
        resource_capacity_profile: dict,
        multitasking_set: set[str],
) -> int:
    """Return the exact per-resource capacity used by waiting decomposition.

    This is factored out so the value can be computed once per resource rather
    than once per event.  It preserves the previous fallback rules exactly.
    """
    profile = resource_capacity_profile.get(str(resource), {})
    try:
        capacity = int(
            profile.get("capacity", 2 if str(resource) in multitasking_set else 1)
        ) if isinstance(profile, dict) else (2 if str(resource) in multitasking_set else 1)
    except Exception:
        capacity = 2 if str(resource) in multitasking_set else 1
    return max(1, int(capacity))


def _build_saturated_interval_index(
        intervals_by_resource: Dict[str, list],
        resource_capacity_profile: dict,
        multitasking_set: set[str],
) -> Dict[str, dict]:
    """Precompute capacity-saturated execution intervals per resource.

    The previous implementation rebuilt the same overlap sweep for every event,
    which is O(sum_r n_r^2) on logs with few resources (notably CVS).  Capacity
    is resource-level and an event's own execution starts exactly at the end of
    its waiting window, so excluding the current event has no effect on the
    half-open [enabled_t, start_t) waiting interval.  A single global sweep is
    therefore behavior-equivalent and can be queried with binary search.
    """
    result: Dict[str, dict] = {}
    for resource, items in (intervals_by_resource or {}).items():
        capacity = _resource_capacity_for_decomposition(
            resource=resource,
            resource_capacity_profile=resource_capacity_profile,
            multitasking_set=multitasking_set,
        )
        by_time: dict[Any, int] = {}
        for item in items or []:
            try:
                _idx, _enabled, start, end = item
            except Exception:
                continue
            if pd.isna(start) or pd.isna(end) or end <= start:
                continue
            by_time[start] = by_time.get(start, 0) + 1
            by_time[end] = by_time.get(end, 0) - 1

        segments: list[tuple] = []
        if by_time:
            times = sorted(by_time)
            active = 0
            for pos, when in enumerate(times[:-1]):
                active += int(by_time.get(when, 0))
                next_when = times[pos + 1]
                if active >= capacity and next_when > when:
                    if segments and segments[-1][1] == when:
                        segments[-1] = (segments[-1][0], next_when)
                    else:
                        segments.append((when, next_when))

        result[str(resource)] = {
            "capacity": int(capacity),
            "segments": segments,
            "starts": [seg[0] for seg in segments],
            "ends": [seg[1] for seg in segments],
        }
    return result


def _query_saturated_intervals(
        window_start,
        window_end,
        index_entry: dict | None,
) -> list[tuple]:
    """Return precomputed saturated portions intersecting one waiting window."""
    if window_end <= window_start or not isinstance(index_entry, dict):
        return []
    segments = index_entry.get("segments") or []
    if not segments:
        return []
    starts = index_entry.get("starts") or []
    ends = index_entry.get("ends") or []

    # Half-open semantics: intervals ending exactly at window_start do not overlap.
    pos = bisect_right(ends, window_start)
    overlaps: list[tuple] = []
    while pos < len(segments):
        seg_start = starts[pos]
        if seg_start >= window_end:
            break
        seg_end = ends[pos]
        start = max(window_start, seg_start)
        end = min(window_end, seg_end)
        if end > start:
            overlaps.append((start, end))
        pos += 1
    return overlaps


def _calendar_off_duty_minutes(
        intervals: list[tuple],
        calendar: dict,
        min_calendar_active_hours: int = 12,
        active_hours: int | None = None,
) -> float:
    """Off-duty elapsed minutes inside non-queue intervals."""
    if not isinstance(calendar, dict) or not calendar:
        return 0.0
    if active_hours is None:
        active_hours = _active_calendar_hour_count(calendar)
    if active_hours <= 0 or active_hours < max(1, int(min_calendar_active_hours)):
        # Sparse calendars are treated as unreliable, matching runtime behavior.
        return 0.0

    minutes = 0.0
    for start, end in intervals or []:
        probe = start
        while probe < end:
            try:
                next_hour = probe.floor("h") + pd.Timedelta(hours=1)
            except Exception:
                next_hour = probe + pd.Timedelta(hours=1)
            segment_end = min(end, next_hour)
            if not _calendar_is_active_hour(
                calendar,
                probe.weekday(),
                probe.hour,
                default=True,
            ):
                minutes += max(0.0, float((segment_end - probe).total_seconds() / 60.0))
            probe = segment_end
    return float(minutes)


def decompose_waiting_times(
        df_features: pd.DataFrame,
        calendars: dict | None = None,
        resource_capacity_profile: dict | None = None,
        multitasking_resources: list | None = None,
        min_calendar_active_hours: int = 12,
) -> pd.DataFrame:
    """Split observed [enabled_t, start_t) into Queue, Calendar and External.

    Precedence is Queue > Calendar > External, so every elapsed minute belongs
    to at most one component.  Queue means the *actual assigned resource* had
    no free capacity.  Calendar is evaluated only on the remaining intervals.
    External is the non-negative residual.
    """
    if df_features is None:
        return pd.DataFrame()
    df = df_features.copy()
    required = {"resource", "enabled_t", "start_t", "end_t"}
    if not required.issubset(df.columns):
        return df

    for col in ("enabled_t", "start_t", "end_t"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    intervals_by_resource = _build_resource_intervals(df)
    calendars = calendars or {}
    resource_capacity_profile = resource_capacity_profile or {}
    multitasking_set = {str(r) for r in (multitasking_resources or [])}

    # Behavior-preserving performance index.  For a resource with n events the
    # old code rescanned all n execution intervals for every waiting event.
    # Building saturated intervals once removes that quadratic hot path.
    saturated_index_by_resource = _build_saturated_interval_index(
        intervals_by_resource=intervals_by_resource,
        resource_capacity_profile=resource_capacity_profile,
        multitasking_set=multitasking_set,
    )
    calendar_active_hours = {
        str(resource): _active_calendar_hour_count(calendar)
        for resource, calendar in calendars.items()
    } if isinstance(calendars, dict) else {}

    totals = []
    queues = []
    calendar_waits = []
    externals = []
    residuals = []

    for idx, row in df.iterrows():
        enabled_t = row.get("enabled_t")
        start_t = row.get("start_t")
        resource = str(row.get("resource"))
        if pd.isna(enabled_t) or pd.isna(start_t) or start_t <= enabled_t:
            total_min = 0.0
            queue_min = 0.0
            calendar_min = 0.0
            external_min = 0.0
        else:
            total_min = max(0.0, float((start_t - enabled_t).total_seconds() / 60.0))
            saturated = _query_saturated_intervals(
                window_start=enabled_t,
                window_end=start_t,
                index_entry=saturated_index_by_resource.get(resource),
            )
            queue_min = float(sum(
                max(0.0, (end - start).total_seconds() / 60.0)
                for start, end in saturated
            ))

            non_queue = _subtract_intervals(enabled_t, start_t, saturated)
            calendar_min = _calendar_off_duty_minutes(
                non_queue,
                calendars.get(resource, {}) if isinstance(calendars, dict) else {},
                min_calendar_active_hours=min_calendar_active_hours,
                active_hours=calendar_active_hours.get(resource),
            )
            external_min = max(0.0, total_min - queue_min - calendar_min)

        residual = float(total_min - queue_min - calendar_min - external_min)
        totals.append(float(total_min))
        queues.append(float(queue_min))
        calendar_waits.append(float(calendar_min))
        externals.append(float(external_min))
        residuals.append(float(residual))

    df["total_wait_min"] = totals
    df["queue_wait_observed_min"] = queues
    df["calendar_wait_observed_min"] = calendar_waits
    df["external_delay_min"] = externals
    df["wait_decomposition_residual_min"] = residuals
    return df


def _compute_delay_quantiles(values, levels=(0.50, 0.75, 0.90, 0.95, 0.98, 1.00)) -> dict:
    """Compute non-negative finite delay quantiles used by runtime capping."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0.0]

    if arr.size == 0:
        return {f"{q:.2f}": 0.0 for q in levels}

    result = {}
    for q in levels:
        q = max(0.0, min(1.0, float(q)))
        result[f"{q:.2f}"] = float(np.quantile(arr, q))
    return result


def _clean_nonnegative_delay_values(values) -> list[float]:
    """Return finite non-negative delay values as plain floats."""
    clean: list[float] = []
    for value in values or []:
        try:
            value = float(value)
            if np.isfinite(value) and value >= 0.0:
                clean.append(float(value))
        except Exception:
            continue
    return clean


def _build_empirical_delay_samples(values, max_samples: int = 5000) -> list[float]:
    """Build compact empirical samples while preserving the long tail.

    Low-frequency activities keep all samples.  For very frequent activities,
    use deterministic quantile-spaced samples plus the largest tail samples so
    serialization stays bounded without deleting long waits.
    """
    clean = _clean_nonnegative_delay_values(values)
    if not clean:
        return []

    try:
        max_samples = max(1, int(max_samples))
    except Exception:
        max_samples = 5000

    arr = np.asarray(clean, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0.0]
    if arr.size == 0:
        return []

    if arr.size <= max_samples:
        return [float(v) for v in arr.tolist()]

    arr = np.sort(arr)
    tail_count = max(1, int(round(max_samples * 0.20)))
    body_count = max(1, max_samples - tail_count)

    body_idx = np.linspace(0, arr.size - 1, num=body_count, dtype=int)
    tail_idx = np.arange(max(0, arr.size - tail_count), arr.size, dtype=int)
    idx = np.unique(np.concatenate([body_idx, tail_idx]))
    samples = arr[idx]
    return [float(v) for v in samples.tolist()]


def _summarize_activity_residual(values: list[float], total_wait_values: list[float] | None = None) -> dict:
    clean = []
    for value in values or []:
        try:
            value = float(value)
            if np.isfinite(value) and value >= 0.0:
                clean.append(float(value))
        except Exception:
            continue

    if not clean:
        return {
            "n": 0,
            "positive_n": 0,
            "mean": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p98": 0.0,
            "max": 0.0,
            "positive_ratio": 0.0,
            "residual_wait_share": 0.0,
            "total_residual_min": 0.0,
            "total_wait_min": 0.0,
        }

    arr = np.asarray(clean, dtype=float)
    positive = arr[arr > 1e-9]

    total_wait_clean = []
    for value in total_wait_values or []:
        try:
            value = float(value)
            if np.isfinite(value) and value >= 0.0:
                total_wait_clean.append(float(value))
        except Exception:
            continue

    total_residual = float(np.sum(arr))
    total_wait = float(np.sum(total_wait_clean)) if total_wait_clean else 0.0

    return {
        "n": int(arr.size),
        "positive_n": int(positive.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p98": float(np.quantile(arr, 0.98)),
        "max": float(np.max(arr)),
        "positive_ratio": float(positive.size / max(arr.size, 1)),
        "residual_wait_share": float(total_residual / max(total_wait, 1e-9)) if total_wait > 0.0 else 0.0,
        "total_residual_min": total_residual,
        "total_wait_min": total_wait,
    }


def _activity_significance_score(stats: dict) -> float:
    """Score used to keep only the most meaningful activity timers."""
    n = max(1.0, float(stats.get("n", 0) or 0))
    mean_v = max(0.0, float(stats.get("mean", 0.0) or 0.0))
    p90_v = max(0.0, float(stats.get("p90", 0.0) or 0.0))
    p95_v = max(0.0, float(stats.get("p95", 0.0) or 0.0))
    positive_ratio = max(0.0, min(1.0, float(stats.get("positive_ratio", 0.0) or 0.0)))
    share = max(0.0, float(stats.get("residual_wait_share", 0.0) or 0.0))

    # Long-tail residuals should be kept even when the mean is moderate.
    robust_delay = 0.45 * mean_v + 0.35 * p90_v + 0.20 * p95_v
    reliability = math.log1p(n)
    return float(robust_delay * (0.35 + positive_ratio) * reliability * (1.0 + min(share, 2.0)))


def _is_significant_external_delay_activity(
        stats: dict,
        min_samples: int,
        min_mean_delay_min: float,
        min_p90_delay_min: float,
        min_positive_ratio: float,
        min_residual_wait_share: float,
        long_delay_keep_min: float,
) -> tuple[bool, str]:
    n = int(stats.get("n", 0) or 0)
    mean_v = float(stats.get("mean", 0.0) or 0.0)
    p90_v = float(stats.get("p90", 0.0) or 0.0)
    p95_v = float(stats.get("p95", 0.0) or 0.0)
    positive_ratio = float(stats.get("positive_ratio", 0.0) or 0.0)
    wait_share = float(stats.get("residual_wait_share", 0.0) or 0.0)

    if n <= 0:
        return False, "no_samples"

    # Simod-lite principle: preserve long-tail business waiting even when
    # the activity is infrequent.  These rare timers are often exactly what
    # stretches case duration and reduces CTD.
    if p95_v >= float(long_delay_keep_min):
        return True, "long_tail_business_delay"

    if n < int(min_samples):
        return False, "insufficient_samples"

    if positive_ratio < float(min_positive_ratio):
        return False, "positive_ratio_below_threshold"

    mean_ok = mean_v >= float(min_mean_delay_min)
    p90_ok = p90_v >= float(min_p90_delay_min)
    share_ok = wait_share >= float(min_residual_wait_share)

    if mean_ok and (p90_ok or share_ok):
        return True, "significant_residual_wait"

    return False, "residual_not_significant"


def default_external_delay_params() -> Dict[str, Any]:
    return {
        "model_type": "activity_specific_external_delay_timer",
        "description": (
            "Activity-specific external-delay module with zero-delay fallback when "
            "the log provides no significant external timer evidence."
        ),
        "activities": {},
        "enabled_activities": [],
        "disabled_activities": [],
        # Global capped empirical distribution used as a shrinkage fallback for
        # activities with too few observations to support their own timer.
        "global_empirical_delay_values": [],
        "global_empirical_delay_sample_count": 0,
        "metadata": {},

        # compatibility shortcuts; global scale is only a weak runtime knob.
        "global_scale": 1.0,
        "max_delay_quantile": 1.0,

        "runtime": {
            "max_delay_quantile": 1.0,
            "global_scale": 1.0,
            "activity_scale_mode": "global_weak_scale",
            "add_delay_to_first_activity": False,
            "residual_after_queue_only": False,
            "queue_credit_ratio": 0.0,
            "queue_residual_decay": 1.0,
            "max_external_delay_ratio": None,
            "residual_guard_in_runtime": False,
            "activity_specific_enabled": True,
            "empirical_global_scale_mode": "none",
            "empirical_global_scale_cap": 1.5,
            "empirical_body_tail_mixture": True,
            "empirical_tail_quantile": 0.95,
            "empirical_tail_probability": 0.05,
            # Do not trust an activity-specific empirical timer below this
            # sample count; runtime falls back to the pooled global distribution.
            "min_activity_empirical_samples": 4,
        },
    }


def discover_external_delay_models(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        activities: list,
        max_depths_cv: Optional[List[int]] = None,
        label_data_attributes: Optional[List[str]] = None,
        label_data_attributes_categorical: Optional[List[str]] = None,
        values_categorical: Optional[Dict[str, list]] = None,
        min_samples_per_activity: int = 80,
        min_samples_leaf: Optional[int] = None,
        use_outlier_removal: bool = False,
        max_delay_quantile: float = 1.0,
        random_state: int = 72,
        max_training_events: int | None = 10000,
        max_leaf_nodes: int | None = 64,
        # New activity-specific timer selection controls.
        min_significant_samples: int | None = None,
        min_significant_mean_delay_min: float = 5.0,
        min_significant_p90_delay_min: float = 10.0,
        min_significant_positive_ratio: float = 0.10,
        min_significant_residual_wait_share: float = 0.05,
        long_delay_keep_min: float = 8.0 * 60.0,
        max_enabled_activity_fraction: float | None = None,
        max_enabled_activities: int | None = None,
        simod_lite_activity_timer: bool = True,
        disable_delay_trees: bool = True,
        calendars: dict | None = None,
        resource_capacity_profile: dict | None = None,
        multitasking_resources: list | None = None,
        min_calendar_active_hours: int = 12,
        min_activity_empirical_samples: int = 4,
) -> Dict[str, Any]:
    """
    Discover activity-specific external timer delays.

    Simod-lite version: discover activity-level timer distributions for all
    activities with positive residual waiting.  It intentionally does not keep
    only the Top-K / Top-fraction timers and does not train residual-delay trees
    by default.  The goal is to preserve cumulative long-tail inter-event
    waiting so that simulated case duration is not systematically compressed.

    Activities with no observed residual waiting are still stored explicitly as
    disabled zero timers.
    """

    if max_depths_cv is None:
        max_depths_cv = [1, 2, 3]

    label_data_attributes = label_data_attributes or []
    label_data_attributes_categorical = label_data_attributes_categorical or []
    values_categorical = values_categorical or {}
    resources = [str(r) for r in (resources or [])]
    activities = [str(a) for a in (activities or [])]
    try:
        min_activity_empirical_samples = max(1, int(min_activity_empirical_samples))
    except Exception:
        min_activity_empirical_samples = 4

    required = ["transition_label", "resource", "enabled_t", "start_t", "end_t"]
    missing = [c for c in required if c not in df_features.columns]
    if missing:
        params = default_external_delay_params()
        params["reason"] = f"df_features misses columns: {missing}"
        return params

    df = df_features.copy()
    df = df[(~df["transition_label"].isna()) & (~df["resource"].isna())].copy()
    df["transition_label"] = df["transition_label"].astype(str)
    df["resource"] = df["resource"].astype(str)
    df["enabled_t"] = pd.to_datetime(df["enabled_t"], errors="coerce")
    df["start_t"] = pd.to_datetime(df["start_t"], errors="coerce")
    df["end_t"] = pd.to_datetime(df["end_t"], errors="coerce")
    df.dropna(subset=["enabled_t", "start_t", "end_t"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        params = default_external_delay_params()
        params["reason"] = "No valid synchronized events for external delay discovery."
        return params

    n_events_original = int(len(df))

    # IMPORTANT: decompose on the complete synchronized event set.  Sampling
    # before this step would remove resource-occupancy intervals and inflate the
    # external residual.
    decomposition_cols = {
        "total_wait_min",
        "queue_wait_observed_min",
        "calendar_wait_observed_min",
        "external_delay_min",
    }
    if not decomposition_cols.issubset(df.columns):
        df = decompose_waiting_times(
            df_features=df,
            calendars=calendars,
            resource_capacity_profile=resource_capacity_profile,
            multitasking_resources=multitasking_resources,
            min_calendar_active_hours=min_calendar_active_hours,
        )

    if max_training_events is not None:
        try:
            max_training_events = int(max_training_events)
        except Exception:
            max_training_events = None

        if max_training_events is not None and max_training_events > 0 and len(df) > max_training_events:
            df = (
                df.sample(n=max_training_events, random_state=random_state)
                .sort_values(["start_t", "end_t"])
                .reset_index(drop=True)
            )

    df["external_delay_min"] = pd.to_numeric(df["external_delay_min"], errors="coerce")
    df["total_wait_min"] = pd.to_numeric(df["total_wait_min"], errors="coerce").fillna(0.0)
    df["queue_wait_observed_min"] = pd.to_numeric(
        df.get("queue_wait_observed_min", 0.0), errors="coerce"
    ).fillna(0.0)
    df["calendar_wait_observed_min"] = pd.to_numeric(
        df.get("calendar_wait_observed_min", 0.0), errors="coerce"
    ).fillna(0.0)
    df = df[np.isfinite(df["external_delay_min"].astype(float))].copy()

    if df.empty:
        params = default_external_delay_params()
        params["reason"] = "All external-delay values are invalid."
        return params

    df["external_delay_min"] = df["external_delay_min"].astype(float).clip(lower=0.0)
    df["external_delay_min_raw"] = df["external_delay_min"].copy()
    df["total_wait_min"] = df["total_wait_min"].astype(float).clip(lower=0.0)

    global_delay_quantiles = _compute_delay_quantiles(
        df["external_delay_min_raw"].dropna().astype(float).tolist()
    )

    if max_delay_quantile is None:
        discovery_max_delay_quantile = 1.0
    else:
        discovery_max_delay_quantile = float(np.clip(max_delay_quantile, 0.50, 1.0))

    if discovery_max_delay_quantile < 1.0:
        cap = float(df["external_delay_min_raw"].quantile(discovery_max_delay_quantile))
        if np.isfinite(cap) and cap > 0:
            df["external_delay_min"] = df["external_delay_min_raw"].clip(upper=cap)

    # Pooled fallback distribution for sparse activities.  IMPORTANT: build it
    # from the capped values, not the raw long-tail residuals.  This prevents a
    # singleton/very-low-frequency outlier from becoming a repeatedly replayed
    # activity timer while still preserving the global empirical shape.
    global_empirical_delay_values = _build_empirical_delay_samples(
        df["external_delay_min"].dropna().astype(float).tolist()
    )

    if min_significant_samples is None:
        min_significant_samples = max(10, min(30, int(min_samples_per_activity)))
    min_significant_samples = max(1, int(min_significant_samples))

    # First pass: compute activity-level residual statistics and candidate flags.
    activity_stats: Dict[str, dict] = {}
    candidate_flags: Dict[str, tuple[bool, str]] = {}

    for activity in activities:
        df_act = df[df["transition_label"] == str(activity)].copy()
        raw_values = df_act["external_delay_min_raw"].dropna().astype(float).tolist()
        total_wait_values = df_act["total_wait_min"].dropna().astype(float).tolist()
        stats = _summarize_activity_residual(raw_values, total_wait_values)
        stats["significance_score"] = _activity_significance_score(stats)
        activity_stats[str(activity)] = stats

        if bool(simod_lite_activity_timer):
            # Simod-lite keeps positive-residual activities enabled, but sparse
            # activities are NOT allowed to learn an independent empirical timer.
            # They are marked for pooled/global fallback at runtime.
            n_samples_activity = int(stats.get("n", 0) or 0)
            positive_samples_activity = int(stats.get("positive_n", 0) or 0)
            has_positive_residual = (
                float(stats.get("positive_ratio", 0.0) or 0.0) > 0.0
                or float(stats.get("max", 0.0) or 0.0) > 0.0
            )
            if n_samples_activity <= 0:
                candidate_flags[str(activity)] = (False, "no_samples")
            elif not has_positive_residual:
                candidate_flags[str(activity)] = (False, "zero_residual_timer")
            elif (
                n_samples_activity < int(min_activity_empirical_samples)
                or positive_samples_activity < int(min_activity_empirical_samples)
            ):
                candidate_flags[str(activity)] = (True, "simod_lite_low_sample_global_fallback")
            else:
                candidate_flags[str(activity)] = (True, "simod_lite_positive_residual_timer")
        else:
            candidate_flags[str(activity)] = _is_significant_external_delay_activity(
                stats=stats,
                min_samples=min_significant_samples,
                min_mean_delay_min=float(min_significant_mean_delay_min),
                min_p90_delay_min=float(min_significant_p90_delay_min),
                min_positive_ratio=float(min_significant_positive_ratio),
                min_residual_wait_share=float(min_significant_residual_wait_share),
                long_delay_keep_min=float(long_delay_keep_min),
            )

    enabled_candidates = [
        activity for activity, (enabled, _) in candidate_flags.items() if enabled
    ]

    if bool(simod_lite_activity_timer):
        # Do not cap the number of enabled activity timers.  The earlier
        # max_enabled_activity_fraction filter made simulation faster only by
        # deleting timers, which systematically shortened case duration.
        enabled_set = set(enabled_candidates)
        max_enabled_activities_effective = None
    else:
        if max_enabled_activities is None:
            if max_enabled_activity_fraction is None:
                max_enabled_activities_effective = len(enabled_candidates)
            else:
                max_enabled_activities_effective = int(math.ceil(max(1, len(activities)) * float(max_enabled_activity_fraction)))
                max_enabled_activities_effective = max(1, max_enabled_activities_effective)
        else:
            max_enabled_activities_effective = max(1, int(max_enabled_activities))

        # Keep the most significant activities, but never remove extremely long-tail
        # business delays because they are essential for P2P-like cases.
        if len(enabled_candidates) > max_enabled_activities_effective:
            must_keep = [
                activity for activity in enabled_candidates
                if float(activity_stats.get(activity, {}).get("p95", 0.0) or 0.0) >= float(long_delay_keep_min)
            ]
            ranked = sorted(
                enabled_candidates,
                key=lambda activity: (
                    -float(activity_stats.get(activity, {}).get("significance_score", 0.0) or 0.0),
                    str(activity),
                ),
            )
            selected = []
            for activity in must_keep + ranked:
                if activity not in selected:
                    selected.append(activity)
                if len(selected) >= max(max_enabled_activities_effective, len(must_keep)):
                    break
            enabled_set = set(selected)
        else:
            enabled_set = set(enabled_candidates)

    activity_models: Dict[str, Any] = {}

    for activity in activities:
        df_act = df[df["transition_label"] == str(activity)].copy()

        values = df_act["external_delay_min"].dropna().astype(float).tolist()
        raw_values = df_act["external_delay_min_raw"].dropna().astype(float).tolist()
        # Use capped discovery values for empirical sampling.  Sparse activities
        # deliberately store no private empirical samples and use the pooled
        # global distribution at runtime instead.
        stats = activity_stats.get(str(activity), _summarize_activity_residual([], []))
        positive_sample_count = int(stats.get("positive_n", 0) or 0)
        use_activity_empirical = (
            len(raw_values) >= int(min_activity_empirical_samples)
            and positive_sample_count >= int(min_activity_empirical_samples)
        )
        empirical_values = (
            _build_empirical_delay_samples(values)
            if use_activity_empirical
            else []
        )

        candidate_enabled, candidate_reason = candidate_flags.get(str(activity), (False, "not_evaluated"))
        selected_enabled = str(activity) in enabled_set
        disabled_reason = None
        if not selected_enabled:
            if candidate_enabled:
                disabled_reason = "not_selected_by_activity_timer_policy"
            else:
                disabled_reason = candidate_reason

        activity_delay_quantiles = _compute_delay_quantiles(raw_values)
        activity_default = _fit_distribution_tuple(
            values=values if selected_enabled else [0.0],
            use_outlier_removal=use_outlier_removal,
            default_value=0.0,
        )

        model_entry = {
            "activity": str(activity),
            "enabled": bool(selected_enabled),
            "external_delay_enabled": bool(selected_enabled),
            "timer_delay_enabled": bool(selected_enabled),
            "disabled_reason": disabled_reason,
            "selection_reason": candidate_reason if selected_enabled else disabled_reason,
            "model_type": "activity_specific_timer_delay",
            "n_samples": int(len(values)),
            "n_samples_raw": int(len(raw_values)),
            "positive_sample_count": int(positive_sample_count),
            "activity_default": activity_default,
            "timer_delay_distribution": activity_default,
            "activity_delay_quantiles": activity_delay_quantiles,
            "empirical_delay_values": empirical_values if selected_enabled else [],
            "empirical_delay_sample_count": int(len(empirical_values)) if selected_enabled else 0,
            "empirical_delay_total_count": int(len(raw_values)),
            "empirical_delay_preserves_long_tail": bool(selected_enabled and use_activity_empirical),
            "empirical_delay_source": (
                "external_delay_min_capped"
                if use_activity_empirical
                else "global_external_delay_min_capped_fallback"
            ),
            "activity_default_p90": float(activity_delay_quantiles.get("0.90", 0.0)),
            "activity_default_p95": float(activity_delay_quantiles.get("0.95", 0.0)),
            "activity_default_p98": float(activity_delay_quantiles.get("0.98", 0.0)),
            "activity_default_max": float(activity_delay_quantiles.get("1.00", 0.0)),
            "residual_statistics": stats,
            "significance": {
                "candidate_enabled": bool(candidate_enabled),
                "selected_enabled": bool(selected_enabled),
                "reason": candidate_reason if selected_enabled else disabled_reason,
                "score": float(stats.get("significance_score", 0.0) or 0.0),
                "thresholds": {
                    "min_samples": int(min_significant_samples),
                    "min_mean_delay_min": float(min_significant_mean_delay_min),
                    "min_p90_delay_min": float(min_significant_p90_delay_min),
                    "min_positive_ratio": float(min_significant_positive_ratio),
                    "min_residual_wait_share": float(min_significant_residual_wait_share),
                    "long_delay_keep_min": float(long_delay_keep_min),
                },
            },
            "model_kind": (
                "disabled_zero_timer"
                if not selected_enabled
                else (
                    "empirical_activity_distribution"
                    if use_activity_empirical
                    else "global_empirical_fallback"
                )
            ),
            "feature_columns": [],
            "rules": None,
            "max_depth": 0,
            "min_samples_leaf": 0,
            "fallback": (
                "zero_delay"
                if not selected_enabled
                else (
                    "activity_empirical_capped"
                    if use_activity_empirical
                    else "global_empirical_capped"
                )
            ),
            "runtime": {
                "enabled": bool(selected_enabled),
                "activity_scale": 1.0,
                "max_delay_quantile": None,
                "min_activity_empirical_samples": int(min_activity_empirical_samples),
            },
            "what_if": {
                "editable": True,
                "analyst_action": (
                    "Reduce, remove, or scale this activity timer"
                    if selected_enabled else
                    "This activity has no significant external timer by default"
                ),
            },
        }

        if (not bool(simod_lite_activity_timer)) and (not bool(disable_delay_trees)) and selected_enabled and len(values) >= int(min_samples_per_activity) and len(set(np.round(values, 6))) > 1:
            train_df = _build_activity_training_df(
                df=df_act,
                df_all=df,
                activity=str(activity),
                resources=resources,
                activities=activities,
                act_to_resources=act_to_resources,
                label_data_attributes=label_data_attributes,
                label_data_attributes_categorical=label_data_attributes_categorical,
                values_categorical=values_categorical,
            )

            if not train_df.empty and train_df["target"].nunique() > 1:
                feature_columns = [c for c in train_df.columns if c != "target"]
                X = _coerce_numeric(train_df[feature_columns])
                y = pd.to_numeric(train_df["target"], errors="coerce").fillna(0.0)

                if min_samples_leaf is None:
                    leaf_size = max(10, int(len(train_df) * 0.05))
                else:
                    leaf_size = max(1, int(min_samples_leaf))

                reg = _fit_delay_tree(
                    X=X,
                    y=y,
                    max_depths_cv=max_depths_cv,
                    min_samples_leaf=leaf_size,
                    random_state=random_state,
                    max_leaf_nodes=max_leaf_nodes,
                )

                if not isinstance(reg, DummyRegressor):
                    rules = _decision_tree_to_delay_rules(
                        reg=reg,
                        X=X,
                        y=y,
                        feature_columns=feature_columns,
                        use_outlier_removal=use_outlier_removal,
                    )
                    model_entry.update({
                        "model_kind": "decision_tree_distribution",
                        "feature_columns": feature_columns,
                        "rules": rules,
                        "max_depth": int(reg.get_depth()),
                        "min_samples_leaf": int(leaf_size),
                    })

        activity_models[str(activity)] = model_entry

    runtime_default_max_delay_quantile = (
        1.0 if bool(simod_lite_activity_timer)
        else (0.95 if discovery_max_delay_quantile >= 0.95 else discovery_max_delay_quantile)
    )

    enabled_activities = sorted([str(a) for a in enabled_set])
    disabled_activities = sorted([str(a) for a in activities if str(a) not in enabled_set])

    return {
        "model_type": "activity_specific_external_delay_timer",
        "description": (
            "Simod-lite activity-specific timer model for non-resource residual waiting. "
            "Activities with sufficient positive residual support receive an editable "
            "activity-level timer distribution; sparse activities shrink to a pooled "
            "capped global empirical distribution."
        ),
        "activities": activity_models,
        "enabled_activities": enabled_activities,
        "disabled_activities": disabled_activities,
        "global_empirical_delay_values": global_empirical_delay_values,
        "global_empirical_delay_sample_count": int(len(global_empirical_delay_values)),
        "global_empirical_delay_source": "external_delay_min_capped",
        "global_scale": 1.0,
        "max_delay_quantile": float(runtime_default_max_delay_quantile),
        "metadata": {
            "n_events_original": n_events_original,
            "n_events_used": int(len(df)),
            "n_events": int(len(df)),
            "max_training_events": None if max_training_events is None else int(max_training_events),
            "activities": activities,
            "resources": resources,
            "enabled_activity_count": int(len(enabled_activities)),
            "disabled_activity_count": int(len(disabled_activities)),
            "enabled_activities": enabled_activities,
            "disabled_activities": disabled_activities,
            "activity_selection": {
                "min_significant_samples": int(min_significant_samples),
                "min_significant_mean_delay_min": float(min_significant_mean_delay_min),
                "min_significant_p90_delay_min": float(min_significant_p90_delay_min),
                "min_significant_positive_ratio": float(min_significant_positive_ratio),
                "min_significant_residual_wait_share": float(min_significant_residual_wait_share),
                "long_delay_keep_min": float(long_delay_keep_min),
                "max_enabled_activity_fraction": (None if max_enabled_activity_fraction is None else float(max_enabled_activity_fraction)),
                "max_enabled_activities": (None if max_enabled_activities_effective is None else int(max_enabled_activities_effective)),
                "selection_policy": "simod_lite_all_positive_residual_activity_timers" if bool(simod_lite_activity_timer) else "significance_top_activity_timers",
                "simod_lite_activity_timer": bool(simod_lite_activity_timer),
                "delay_tree_training_enabled": bool((not bool(simod_lite_activity_timer)) and (not bool(disable_delay_trees))),
            },
            "activity_residual_statistics": activity_stats,
            "label_data_attributes": label_data_attributes,
            "label_data_attributes_categorical": label_data_attributes_categorical,
            "values_categorical": values_categorical,
            "target": (
                "external_delay_min = total_wait_min - queue_wait_observed_min "
                "- calendar_wait_observed_min"
            ),
            "waiting_decomposition_policy": "queue_capacity_first_then_calendar_then_external",
            "queue_residual_policy": (
                "Resource-capacity waiting is credited to Queue; off-duty waiting on the "
                "remaining intervals is credited to Calendar; External is the residual."
            ),
            "decomposition_summary": {
                "mean_total_wait_min": float(df["total_wait_min"].mean()) if len(df) else 0.0,
                "mean_queue_wait_observed_min": float(df["queue_wait_observed_min"].mean()) if len(df) else 0.0,
                "mean_calendar_wait_observed_min": float(df["calendar_wait_observed_min"].mean()) if len(df) else 0.0,
                "mean_external_delay_min": float(df["external_delay_min_raw"].mean()) if len(df) else 0.0,
            },
            "discovery_max_delay_quantile": float(discovery_max_delay_quantile),
            "global_delay_quantiles": global_delay_quantiles,
            "use_outlier_removal": bool(use_outlier_removal),
            "min_samples_per_activity_for_tree": int(min_samples_per_activity),
            "min_activity_empirical_samples": int(min_activity_empirical_samples),
            "global_empirical_delay_sample_count": int(len(global_empirical_delay_values)),
            "max_leaf_nodes": None if max_leaf_nodes is None else int(max_leaf_nodes),
            "modeling_note": (
                "Simod-lite mode: disabled activities sample zero external delay; "
                "activities with enough positive residual observations use capped "
                "activity-level empirical timers; sparse activities use the pooled "
                "capped global distribution instead of replaying rare outliers."
            ),
        },
        "runtime": {
            "max_delay_quantile": float(runtime_default_max_delay_quantile),
            "global_scale": 1.0,
            "activity_scale_mode": "global_weak_scale",
            "add_delay_to_first_activity": False,
            "residual_after_queue_only": False,
            "queue_credit_ratio": 0.0,
            "queue_residual_decay": 1.0,
            "max_external_delay_ratio": None,
            "residual_guard_in_runtime": False,
            "activity_specific_enabled": True,
            "empirical_global_scale_mode": "none",
            "empirical_global_scale_cap": 1.5,
            "empirical_body_tail_mixture": True,
            "empirical_tail_quantile": 0.95,
            "empirical_tail_probability": 0.05,
            "min_activity_empirical_samples": int(min_activity_empirical_samples),
        },
    }

def _build_activity_training_df(
        df: pd.DataFrame,
        df_all: pd.DataFrame,
        activity: str,
        resources: list,
        activities: list,
        act_to_resources: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
) -> pd.DataFrame:
    rows = []

    for idx, row in df.iterrows():
        enabled_time = row["enabled_t"]
        resource = str(row.get("resource"))

        record = {
            "target": float(row.get("external_delay_min", 0.0)),
            "res_workload": _safe_float(row.get("res_workload", 0.0)),
            "enabled_hour": float(enabled_time.hour),
            "enabled_weekday": float(enabled_time.weekday()),
        }

        for resource_name in resources:
            record[f"{RESOURCE_PREFIX}{resource_name}"] = int(resource == str(resource_name))
            if resource_name in df.columns:
                record[str(resource_name)] = _safe_float(row.get(resource_name, 0.0))

        for activity_name in activities:
            record[f"{ACTIVITY_PREFIX}{activity_name}"] = int(str(activity_name) == str(activity))
            if activity_name in df.columns:
                record[str(activity_name)] = _safe_float(row.get(activity_name, 0.0))

        queue_features = _compute_queue_features_for_event(
            df=df_all,
            current_index=idx,
            current_time=enabled_time,
            activity=activity,
            resource=resource,
            act_to_resources=act_to_resources,
        )
        for name in QUEUE_FEATURE_NAMES:
            record[name] = float(queue_features.get(name, 0.0))

        for attr in label_data_attributes:
            if attr in label_data_attributes_categorical:
                raw_value = row.get(attr, None)
                for value in values_categorical.get(attr, []):
                    record[f"{attr} = {value}"] = int(raw_value == value)
            else:
                if attr in df.columns:
                    record[attr] = _safe_float(row.get(attr, 0.0))

        rows.append(record)

    train_df = pd.DataFrame(rows)
    for col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)
    return train_df


def _fit_delay_tree(X, y, max_depths_cv, min_samples_leaf, random_state, max_leaf_nodes=64):
    if len(y) < 10 or y.nunique() < 2:
        reg = DummyRegressor(strategy="mean")
        reg.fit(X, y)
        return reg

    depths = sorted({max(1, int(d)) for d in (max_depths_cv or [2])})
    if not depths:
        depths = [2]

    # Important speed-up: when the outer tuner has already fixed the depth,
    # fit one tree directly instead of running CV on the same depth.
    if len(depths) == 1:
        reg = DecisionTreeRegressor(
            max_depth=depths[0],
            random_state=random_state,
            min_samples_leaf=int(min_samples_leaf),
            max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes else None,
        )
        reg.fit(X, y)
        return reg

    base = DecisionTreeRegressor(
        random_state=random_state,
        min_samples_leaf=int(min_samples_leaf),
        max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes else None,
    )

    if len(y) >= 30:
        try:
            cv = min(3, max(2, int(len(y) // max(1, min_samples_leaf))))
            grid = GridSearchCV(
                estimator=base,
                param_grid={"max_depth": depths},
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            grid.fit(X, y)
            return grid.best_estimator_
        except Exception:
            pass

    reg = DecisionTreeRegressor(
        max_depth=min(max(depths), 3),
        random_state=random_state,
        min_samples_leaf=int(min_samples_leaf),
        max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes else None,
    )
    reg.fit(X, y)
    return reg


def _decision_tree_to_delay_rules(
        reg: DecisionTreeRegressor,
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: list[str],
        use_outlier_removal: bool,
) -> Dict[str, Any]:
    tree = reg.tree_
    leaf_ids = reg.apply(X)
    y_values = np.asarray(y, dtype=float)

    values_by_leaf: Dict[int, list[float]] = {}
    for leaf_id, value in zip(leaf_ids, y_values):
        values_by_leaf.setdefault(int(leaf_id), []).append(float(max(0.0, value)))

    def build_node(node_id: int):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left == right:
            values = values_by_leaf.get(int(node_id), [])
            dist_tuple = _fit_distribution_tuple(
                values=values,
                use_outlier_removal=use_outlier_removal,
                default_value=0.0,
            )
            leaf_quantiles = _compute_delay_quantiles(values)
            return {
                "type": "leaf",
                "value": float(dist_tuple[4]),
                "dist": dist_tuple,
                "n_node_samples": int(tree.n_node_samples[node_id]),
                "delay_quantiles": leaf_quantiles,
                "p90": float(leaf_quantiles.get("0.90", 0.0)),
                "p95": float(leaf_quantiles.get("0.95", 0.0)),
                "p98": float(leaf_quantiles.get("0.98", 0.0)),
                "max": float(leaf_quantiles.get("1.00", 0.0)),
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


def _fit_distribution_tuple(
        values: list[float],
        use_outlier_removal: bool,
        default_value: float = 0.0,
) -> Tuple[Any, tuple, float, float, float]:
    cleaned = []
    for value in values or []:
        try:
            value = float(value)
            if np.isfinite(value):
                cleaned.append(max(0.0, value))
        except Exception:
            continue

    if use_outlier_removal and cleaned:
        cleaned = remove_outliers(cleaned)

    if not cleaned:
        default_value = max(0.0, float(default_value))
        return "fixed", (default_value,), default_value, default_value, default_value

    cleaned = [max(0.0, float(v)) for v in cleaned]
    if max(cleaned) <= 0.0:
        return "fixed", (0.0,), 0.0, 0.0, 0.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            dist, params = return_best_distribution(cleaned, dist_search=DIST_SEARCH)
    except Exception:
        mean_value = float(np.mean(cleaned))
        return "fixed", (mean_value,), float(np.min(cleaned)), float(np.max(cleaned)), mean_value

    return (
        dist,
        params,
        float(np.min(cleaned)),
        float(np.max(cleaned)),
        float(np.mean(cleaned)),
    )


def _build_resource_intervals(df: pd.DataFrame) -> Dict[str, list]:
    intervals: Dict[str, list] = {}
    for idx, row in df.iterrows():
        resource = str(row.get("resource"))
        intervals.setdefault(resource, []).append((
            idx,
            row.get("enabled_t"),
            row.get("start_t"),
            row.get("end_t"),
        ))
    for resource in intervals:
        intervals[resource].sort(key=lambda x: (x[1], x[2], x[3]))
    return intervals


def _estimate_external_delay_min(row: pd.Series, intervals_by_resource: Dict[str, list]) -> float:
    """
    Estimate extraneous waiting as the interval between the optimal start
    and the recorded start.

    Simod-style definition:
        optimal_start = max(enabled_t, resource_available_t)
        external_delay_min = max(0, recorded_start - optimal_start)

    Here resource_available_t is estimated only from activities that were
    already occupying the same resource at enabled_t. We do not credit
    later-starting queue-ahead jobs, because they may themselves have been
    scheduled during the current activity's extraneous waiting period.
    """
    try:
        enabled_t = row["enabled_t"]
        start_t = row["start_t"]
        resource = str(row.get("resource"))
        current_idx = row.name
    except Exception:
        return 0.0

    if pd.isna(enabled_t) or pd.isna(start_t):
        return 0.0

    blocking_ends = []

    for item in intervals_by_resource.get(resource, []):
        try:
            idx, other_enabled, other_start, other_end = item
        except Exception:
            continue

        if idx == current_idx:
            continue

        if pd.isna(other_start) or pd.isna(other_end):
            continue

        # Only activities already occupying the same resource at enabled_t
        # make the resource unavailable at the optimal start.
        if other_start <= enabled_t < other_end:
            blocking_ends.append(other_end)

    resource_available_t = max(blocking_ends) if blocking_ends else enabled_t
    optimal_start_t = max(enabled_t, resource_available_t)

    try:
        return max(0.0, float((start_t - optimal_start_t).total_seconds() / 60.0))
    except Exception:
        return 0.0


def _compute_queue_features_for_event(
        df: pd.DataFrame,
        current_index: int,
        current_time,
        activity: str,
        resource: str,
        act_to_resources: dict,
) -> Dict[str, float]:
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
        if str(resource) in possible_resources:
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
        disable_global_queue_features: bool = True,
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


def _coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X


def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return float(value)
    except Exception:
        pass
    return float(default)
