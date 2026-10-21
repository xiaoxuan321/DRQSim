# -*- coding: utf-8 -*-
"""
Time-decomposition diagnostics for modular drq simulations.

This module is intentionally independent from the evaluator.  It does not
produce quality scores; instead, it decomposes the simulated time structure into
queue waiting, external residual delay, and service time so that model errors in
AED/RED/CTD can be traced back to the responsible time component.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pm4py


CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
RESOURCE_COL = "org:resource"
START_COL = "start:timestamp"
END_COL = "time:timestamp"
ENABLED_COL = "enabled:timestamp"
QUEUE_START_COL = "queue_start:timestamp"


def build_time_decomposition_diagnostics(
        real_log,
        simulated_log,
        top_k: int = 10,
) -> Dict[str, Any]:
    """
    Build a diagnostic report comparing real and simulated temporal structure.

    The simulated log may contain extra columns produced by SimulatorEngine:
        - enabled:timestamp
        - queue_start:timestamp
        - queue_wait_min
        - resource_contention_wait_min
        - priority_wait_min
        - calendar_wait_min
        - external_delay_min
        - service_time_min
        - total_wait_min
        - base_service_time_min
        - adjusted_service_time_min

    If some columns are missing, the function reconstructs what it can from
    standard event-log timestamps.
    """

    real_df = _prepare_log_dataframe(real_log)
    sim_df = _prepare_log_dataframe(simulated_log)

    real_event = _add_basic_time_columns(real_df, is_simulated=False)
    sim_event = _add_basic_time_columns(sim_df, is_simulated=True)

    real_case = _build_case_level(real_event)
    sim_case = _build_case_level(sim_event)

    real_activity = _build_activity_level(real_event)
    sim_activity = _build_activity_level(sim_event)

    diagnostics = {
        "summary": {
            "real": _build_summary(real_event, real_case, is_simulated=False),
            "simulated": _build_summary(sim_event, sim_case, is_simulated=True),
            "delta": _build_delta_summary(real_event, sim_event, real_case, sim_case),
        },
        "simulated_decomposition": _build_sim_decomposition(sim_event, sim_case),
        "top_activities": {
            "external_delay": _top_activity_records(sim_activity, "external_delay_min", top_k),
            "queue_wait": _top_activity_records(sim_activity, "queue_wait_min", top_k),
            "resource_contention_wait": _top_activity_records(sim_activity, "resource_contention_wait_min", top_k),
            "priority_wait": _top_activity_records(sim_activity, "priority_wait_min", top_k),
            "calendar_wait": _top_activity_records(sim_activity, "calendar_wait_min", top_k),
            "service_time": _top_activity_records(sim_activity, "service_time_min", top_k),
            "total_wait": _top_activity_records(sim_activity, "total_wait_min", top_k),
        },
        "activity_comparison": _build_activity_comparison(real_activity, sim_activity, top_k),
        "case_duration_comparison": {
            "real_case_duration_min": _series_stats(real_case.get("case_duration_min")),
            "sim_case_duration_min": _series_stats(sim_case.get("case_duration_min")),
            "sim_component_sum_min": _series_stats(sim_case.get("component_total_min")),
        },
        "notes": [
            "queue_wait_min = resource_contention_wait_min + priority_wait_min in the CPT path.",
            "priority_wait_min is the extra queue residence after baseline resource/calendar availability caused by dispatch ordering.",
            "external_delay_min is the pre-resource non-resource timer: queue_entry - enabled_time.",
            "service_time_min is event completion time minus actual start time.",
            "Real logs usually do not contain queue/external split, so real-side comparison focuses on service time and case duration unless enabled timestamps exist.",
        ],
    }

    return _to_jsonable(diagnostics)


def aggregate_time_decomposition_diagnostics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate diagnostics produced across repeated simulation runs.

    The aggregation focuses on the most important scalar values.  Per-activity
    rankings are kept from the run whose simulated CTD proxy, i.e. mean case
    duration, is closest to the repeated-run average.
    """

    valid = [r for r in reports if isinstance(r, dict) and r]
    if not valid:
        return {}

    scalar_paths = [
        ("summary", "simulated", "event_count"),
        ("summary", "simulated", "case_count"),
        ("summary", "simulated", "case_duration_min", "mean"),
        ("summary", "simulated", "service_time_min", "mean"),
        ("summary", "simulated", "total_wait_min", "mean"),
        ("summary", "simulated", "queue_wait_min", "mean"),
        ("summary", "simulated", "resource_contention_wait_min", "mean"),
        ("summary", "simulated", "priority_wait_min", "mean"),
        ("summary", "simulated", "calendar_wait_min", "mean"),
        ("summary", "simulated", "external_delay_min", "mean"),
        ("simulated_decomposition", "component_totals_min", "queue_wait_min"),
        ("simulated_decomposition", "component_totals_min", "resource_contention_wait_min"),
        ("simulated_decomposition", "component_totals_min", "priority_wait_min"),
        ("simulated_decomposition", "component_totals_min", "calendar_wait_min"),
        ("simulated_decomposition", "component_totals_min", "external_delay_min"),
        ("simulated_decomposition", "component_totals_min", "service_time_min"),
        ("simulated_decomposition", "component_ratios", "queue_wait_ratio"),
        ("simulated_decomposition", "component_ratios", "resource_contention_wait_ratio"),
        ("simulated_decomposition", "component_ratios", "priority_wait_ratio"),
        ("simulated_decomposition", "component_ratios", "calendar_wait_ratio"),
        ("simulated_decomposition", "component_ratios", "external_delay_ratio"),
        ("simulated_decomposition", "component_ratios", "service_time_ratio"),
    ]

    aggregated = {
        "n_runs": len(valid),
        "mean_std": {},
        "representative_report": None,
    }

    for path in scalar_paths:
        values = []
        for report in valid:
            value = _nested_get(report, path)
            if isinstance(value, (int, float)) and np.isfinite(value):
                values.append(float(value))
        if values:
            aggregated["mean_std"][".".join(path)] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }

    # Pick a representative report whose simulated mean case duration is close
    # to the average, so top-activity tables remain readable.
    key_path = ("summary", "simulated", "case_duration_min", "mean")
    case_means = [
        _nested_get(report, key_path)
        for report in valid
    ]
    numeric_case_means = [float(v) for v in case_means if isinstance(v, (int, float)) and np.isfinite(v)]
    if numeric_case_means:
        target = float(np.mean(numeric_case_means))
        best_idx = min(
            range(len(valid)),
            key=lambda i: abs(float(case_means[i]) - target)
            if isinstance(case_means[i], (int, float)) and np.isfinite(case_means[i])
            else float("inf"),
        )
        aggregated["representative_report"] = valid[best_idx]
    else:
        aggregated["representative_report"] = valid[-1]

    # Keep real summary from the representative report; it is constant across runs.
    representative = aggregated.get("representative_report") or {}
    if isinstance(representative, dict):
        aggregated["real_summary"] = representative.get("summary", {}).get("real", {})
        aggregated["top_activities"] = representative.get("top_activities", {})
        aggregated["activity_comparison"] = representative.get("activity_comparison", {})
        aggregated["case_duration_comparison"] = representative.get("case_duration_comparison", {})

    return _to_jsonable(aggregated)


def format_time_diagnostics_for_log(report: Dict[str, Any], prefix: str = "[Time Decomposition]") -> List[str]:
    """Return compact log lines for a diagnostic report."""

    if not isinstance(report, dict) or not report:
        return [f"{prefix} no diagnostics available"]

    # If this is an aggregate report, log its mean/std values first.
    if "mean_std" in report:
        mean_std = report.get("mean_std", {})
        lines = [f"{prefix} repeated diagnostics over {report.get('n_runs', 0)} runs"]
        for key in [
            "summary.simulated.case_duration_min.mean",
            "summary.simulated.service_time_min.mean",
            "summary.simulated.total_wait_min.mean",
            "summary.simulated.queue_wait_min.mean",
            "summary.simulated.resource_contention_wait_min.mean",
            "summary.simulated.priority_wait_min.mean",
            "summary.simulated.calendar_wait_min.mean",
            "summary.simulated.external_delay_min.mean",
            "simulated_decomposition.component_ratios.queue_wait_ratio",
            "simulated_decomposition.component_ratios.external_delay_ratio",
            "simulated_decomposition.component_ratios.service_time_ratio",
        ]:
            value = mean_std.get(key)
            if isinstance(value, dict):
                lines.append(
                    f"{prefix} {key}: mean={value.get('mean'):.3f}, std={value.get('std'):.3f}"
                )
        representative = report.get("representative_report") or {}
        lines.extend(format_time_diagnostics_for_log(representative, prefix=f"{prefix} representative"))
        return lines

    summary = report.get("summary", {})
    sim = summary.get("simulated", {})
    real = summary.get("real", {})
    decomp = report.get("simulated_decomposition", {})
    ratios = decomp.get("component_ratios", {})

    lines = [
        (
            f"{prefix} real_case_duration_mean="
            f"{_fmt(_nested_get(real, ('case_duration_min', 'mean')))}, "
            f"sim_case_duration_mean={_fmt(_nested_get(sim, ('case_duration_min', 'mean')))}"
        ),
        (
            f"{prefix} sim_event_mean: queue_wait="
            f"{_fmt(_nested_get(sim, ('queue_wait_min', 'mean')))}, "
            f"priority_wait={_fmt(_nested_get(sim, ('priority_wait_min', 'mean')))}, "
            f"contention_wait={_fmt(_nested_get(sim, ('resource_contention_wait_min', 'mean')))}, "
            f"calendar_wait={_fmt(_nested_get(sim, ('calendar_wait_min', 'mean')))}, "
            f"external_delay={_fmt(_nested_get(sim, ('external_delay_min', 'mean')))}, "
            f"service={_fmt(_nested_get(sim, ('service_time_min', 'mean')))}, "
            f"total_wait={_fmt(_nested_get(sim, ('total_wait_min', 'mean')))}"
        ),
        (
            f"{prefix} sim_component_ratios: queue="
            f"{_fmt(ratios.get('queue_wait_ratio'))}, "
            f"external={_fmt(ratios.get('external_delay_ratio'))}, "
            f"service={_fmt(ratios.get('service_time_ratio'))}"
        ),
    ]

    for group_name, label in [
        ("external_delay", "Top external-delay activities"),
        ("queue_wait", "Top queue-wait activities"),
        ("priority_wait", "Top priority-wait activities"),
        ("resource_contention_wait", "Top contention-wait activities"),
        ("service_time", "Top service-time activities"),
    ]:
        records = report.get("top_activities", {}).get(group_name, [])
        if records:
            head = records[:3]
            text = "; ".join(
                f"{r.get('activity')} mean={_fmt(r.get('mean'))} n={r.get('count')}"
                for r in head
            )
            lines.append(f"{prefix} {label}: {text}")

    return lines


def _prepare_log_dataframe(log) -> pd.DataFrame:
    if isinstance(log, pd.DataFrame):
        df = log.copy()
    else:
        df = pm4py.convert_to_dataframe(log).copy()

    if CASE_COL in df.columns:
        df[CASE_COL] = df[CASE_COL].astype(str)

    for col in [START_COL, END_COL, ENABLED_COL, QUEUE_START_COL]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if START_COL in df.columns and END_COL in df.columns:
        df.dropna(subset=[CASE_COL, START_COL, END_COL], inplace=True)
        df.sort_values(by=[CASE_COL, START_COL, END_COL], inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def _add_basic_time_columns(df: pd.DataFrame, is_simulated: bool) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    df["service_time_min"] = _minutes_between(df.get(START_COL), df.get(END_COL))

    if ENABLED_COL in df.columns:
        df["total_wait_min"] = _minutes_between(df.get(ENABLED_COL), df.get(START_COL))
    else:
        df["total_wait_min"] = np.nan

    if is_simulated:
        # Preserve explicit simulator decomposition whenever it is available.
        # With CPT, queue_start:timestamp means *entry into the ready/resource
        # queue*. Therefore queue wait is start - queue_start, while the
        # pre-queue external timer is queue_start - enabled.
        if "queue_wait_min" not in df.columns:
            if QUEUE_START_COL in df.columns and START_COL in df.columns:
                df["queue_wait_min"] = _minutes_between(df.get(QUEUE_START_COL), df.get(START_COL))
            else:
                df["queue_wait_min"] = np.nan

        if "external_delay_min" not in df.columns:
            if QUEUE_START_COL in df.columns and ENABLED_COL in df.columns:
                df["external_delay_min"] = _minutes_between(df.get(ENABLED_COL), df.get(QUEUE_START_COL))
            else:
                df["external_delay_min"] = 0.0

        if "total_wait_min" not in df.columns or df["total_wait_min"].isna().all():
            calendar = pd.to_numeric(df.get("calendar_wait_min", 0.0), errors="coerce").fillna(0.0) \
                if isinstance(df.get("calendar_wait_min", 0.0), pd.Series) else 0.0
            df["total_wait_min"] = (
                df["queue_wait_min"].fillna(0.0)
                + df["external_delay_min"].fillna(0.0)
                + calendar
            )

        for col in [
            "queue_wait_min",
            "resource_contention_wait_min",
            "priority_wait_min",
            "calendar_wait_min",
            "external_delay_min",
            "service_time_min",
            "total_wait_min",
            "base_service_time_min",
            "adjusted_service_time_min",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0.0)
            else:
                df[col] = np.nan
    else:
        for col in ["queue_wait_min", "external_delay_min", "base_service_time_min", "adjusted_service_time_min"]:
            if col not in df.columns:
                df[col] = np.nan

    return df


def _build_case_level(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty or CASE_COL not in event_df.columns:
        return pd.DataFrame()

    agg_dict = {
        START_COL: "min",
        END_COL: "max",
        "service_time_min": "sum",
        "total_wait_min": "sum",
        "queue_wait_min": "sum",
        "resource_contention_wait_min": "sum",
        "priority_wait_min": "sum",
        "calendar_wait_min": "sum",
        "external_delay_min": "sum",
    }
    existing = {k: v for k, v in agg_dict.items() if k in event_df.columns}
    case_df = event_df.groupby(CASE_COL).agg(existing)

    if START_COL in case_df.columns and END_COL in case_df.columns:
        case_df["case_duration_min"] = _minutes_between(case_df[START_COL], case_df[END_COL])

    component_cols = ["queue_wait_min", "calendar_wait_min", "external_delay_min", "service_time_min"]
    existing_components = [c for c in component_cols if c in case_df.columns]
    if existing_components:
        case_df["component_total_min"] = case_df[existing_components].sum(axis=1)

    case_df.reset_index(inplace=True)
    return case_df


def _build_activity_level(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty or ACTIVITY_COL not in event_df.columns:
        return pd.DataFrame()

    numeric_cols = [
        "service_time_min",
        "total_wait_min",
        "queue_wait_min",
        "resource_contention_wait_min",
        "priority_wait_min",
        "calendar_wait_min",
        "external_delay_min",
        "base_service_time_min",
        "adjusted_service_time_min",
    ]
    numeric_cols = [c for c in numeric_cols if c in event_df.columns]

    rows = []
    for activity, group in event_df.groupby(ACTIVITY_COL):
        row = {"activity": str(activity), "count": int(len(group))}
        for col in numeric_cols:
            stats = _series_stats(group[col])
            for key in ["mean", "median", "p90", "p95", "sum"]:
                row[f"{col}.{key}"] = stats.get(key)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_summary(event_df: pd.DataFrame, case_df: pd.DataFrame, is_simulated: bool) -> Dict[str, Any]:
    return {
        "event_count": int(len(event_df)),
        "case_count": int(event_df[CASE_COL].nunique()) if CASE_COL in event_df.columns else 0,
        "service_time_min": _series_stats(event_df.get("service_time_min")),
        "total_wait_min": _series_stats(event_df.get("total_wait_min")),
        "queue_wait_min": _series_stats(event_df.get("queue_wait_min")),
        "resource_contention_wait_min": _series_stats(event_df.get("resource_contention_wait_min")),
        "priority_wait_min": _series_stats(event_df.get("priority_wait_min")),
        "calendar_wait_min": _series_stats(event_df.get("calendar_wait_min")),
        "external_delay_min": _series_stats(event_df.get("external_delay_min")),
        "base_service_time_min": _series_stats(event_df.get("base_service_time_min")),
        "adjusted_service_time_min": _series_stats(event_df.get("adjusted_service_time_min")),
        "case_duration_min": _series_stats(case_df.get("case_duration_min")),
        "is_simulated": bool(is_simulated),
    }


def _build_delta_summary(real_event, sim_event, real_case, sim_case) -> Dict[str, Any]:
    def mean_delta(real_series, sim_series):
        r = _series_stats(real_series).get("mean")
        s = _series_stats(sim_series).get("mean")
        if r is None or s is None:
            return None
        return float(s - r)

    return {
        "service_time_mean_delta_min": mean_delta(real_event.get("service_time_min"), sim_event.get("service_time_min")),
        "total_wait_mean_delta_min": mean_delta(real_event.get("total_wait_min"), sim_event.get("total_wait_min")),
        "case_duration_mean_delta_min": mean_delta(real_case.get("case_duration_min"), sim_case.get("case_duration_min")),
        "event_count_delta": int(len(sim_event) - len(real_event)),
        "case_count_delta": int(
            (sim_event[CASE_COL].nunique() if CASE_COL in sim_event.columns else 0)
            - (real_event[CASE_COL].nunique() if CASE_COL in real_event.columns else 0)
        ),
    }


def _build_sim_decomposition(sim_event: pd.DataFrame, sim_case: pd.DataFrame) -> Dict[str, Any]:
    totals = {}
    for col in [
        "queue_wait_min", "resource_contention_wait_min", "priority_wait_min",
        "calendar_wait_min", "external_delay_min", "service_time_min", "total_wait_min"
    ]:
        if col in sim_event.columns:
            totals[col] = float(pd.to_numeric(sim_event[col], errors="coerce").fillna(0.0).sum())
        else:
            totals[col] = 0.0

    denominator = (
        totals.get("queue_wait_min", 0.0)
        + totals.get("calendar_wait_min", 0.0)
        + totals.get("external_delay_min", 0.0)
        + totals.get("service_time_min", 0.0)
    )
    ratios = {
        "queue_wait_ratio": _safe_div(totals.get("queue_wait_min", 0.0), denominator),
        "resource_contention_wait_ratio": _safe_div(totals.get("resource_contention_wait_min", 0.0), denominator),
        "priority_wait_ratio": _safe_div(totals.get("priority_wait_min", 0.0), denominator),
        "calendar_wait_ratio": _safe_div(totals.get("calendar_wait_min", 0.0), denominator),
        "external_delay_ratio": _safe_div(totals.get("external_delay_min", 0.0), denominator),
        "service_time_ratio": _safe_div(totals.get("service_time_min", 0.0), denominator),
    }

    return {
        "component_totals_min": totals,
        "component_ratios": ratios,
        "case_level": {
            "total_queue_wait_min": _series_stats(sim_case.get("queue_wait_min")),
            "total_resource_contention_wait_min": _series_stats(sim_case.get("resource_contention_wait_min")),
            "total_priority_wait_min": _series_stats(sim_case.get("priority_wait_min")),
            "total_calendar_wait_min": _series_stats(sim_case.get("calendar_wait_min")),
            "total_external_delay_min": _series_stats(sim_case.get("external_delay_min")),
            "total_service_time_min": _series_stats(sim_case.get("service_time_min")),
            "component_total_min": _series_stats(sim_case.get("component_total_min")),
        },
    }


def _build_activity_comparison(real_activity: pd.DataFrame, sim_activity: pd.DataFrame, top_k: int) -> List[Dict[str, Any]]:
    if real_activity.empty or sim_activity.empty:
        return []

    real = real_activity.set_index("activity")
    sim = sim_activity.set_index("activity")
    activities = sorted(set(real.index) | set(sim.index))
    rows = []
    for activity in activities:
        r_service = _safe_get(real, activity, "service_time_min.mean")
        s_service = _safe_get(sim, activity, "service_time_min.mean")
        r_count = _safe_get(real, activity, "count")
        s_count = _safe_get(sim, activity, "count")
        row = {
            "activity": activity,
            "real_count": r_count,
            "sim_count": s_count,
            "count_delta": _sub_or_none(s_count, r_count),
            "real_service_mean": r_service,
            "sim_service_mean": s_service,
            "service_mean_delta": _sub_or_none(s_service, r_service),
            "sim_queue_wait_mean": _safe_get(sim, activity, "queue_wait_min.mean"),
            "sim_external_delay_mean": _safe_get(sim, activity, "external_delay_min.mean"),
            "sim_total_wait_mean": _safe_get(sim, activity, "total_wait_min.mean"),
        }
        rows.append(row)

    rows.sort(key=lambda r: abs(r.get("service_mean_delta") or 0.0), reverse=True)
    return rows[:max(1, int(top_k))]


def _top_activity_records(activity_df: pd.DataFrame, metric_prefix: str, top_k: int) -> List[Dict[str, Any]]:
    if activity_df.empty:
        return []
    mean_col = f"{metric_prefix}.mean"
    sum_col = f"{metric_prefix}.sum"
    if mean_col not in activity_df.columns:
        return []
    df = activity_df.copy()
    df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")
    df = df.dropna(subset=[mean_col])
    df = df.sort_values(by=mean_col, ascending=False).head(max(1, int(top_k)))
    records = []
    for _, row in df.iterrows():
        records.append({
            "activity": str(row.get("activity")),
            "count": int(row.get("count", 0)) if pd.notna(row.get("count", np.nan)) else 0,
            "mean": _none_if_nan(row.get(mean_col)),
            "median": _none_if_nan(row.get(f"{metric_prefix}.median")),
            "p90": _none_if_nan(row.get(f"{metric_prefix}.p90")),
            "p95": _none_if_nan(row.get(f"{metric_prefix}.p95")),
            "sum": _none_if_nan(row.get(sum_col)),
        })
    return records


def _series_stats(series) -> Dict[str, Any]:
    if series is None:
        return _empty_stats()
    try:
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    except Exception:
        return _empty_stats()
    if s.empty:
        return _empty_stats()
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "min": float(s.min()),
        "median": float(s.median()),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
        "sum": float(s.sum()),
    }


def _empty_stats() -> Dict[str, Any]:
    return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "p90": None, "p95": None, "max": None, "sum": 0.0}


def _minutes_between(start, end):
    if start is None or end is None:
        return np.nan
    try:
        return (end - start).dt.total_seconds().astype(float).clip(lower=0.0) / 60.0
    except Exception:
        try:
            return max(0.0, float((end - start).total_seconds()) / 60.0)
        except Exception:
            return np.nan


def _safe_div(num: float, den: float):
    try:
        num = float(num)
        den = float(den)
        if den > 0 and np.isfinite(num) and np.isfinite(den):
            return float(num / den)
    except Exception:
        pass
    return None


def _safe_get(df: pd.DataFrame, idx, col):
    try:
        value = df.loc[idx, col]
        return _none_if_nan(value)
    except Exception:
        return None


def _sub_or_none(a, b):
    if a is None or b is None:
        return None
    try:
        return float(a) - float(b)
    except Exception:
        return None


def _none_if_nan(value):
    try:
        if value is None:
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(value):
                if float(value).is_integer():
                    return int(value) if abs(float(value)) < 10**12 else float(value)
                return float(value)
            return None
        return value
    except Exception:
        return None


def _nested_get(data: Dict[str, Any], path: Iterable[str]):
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _fmt(value) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
        if np.isfinite(value):
            return f"{value:.3f}"
    except Exception:
        pass
    return "NA"


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if pd.isna(obj) if not isinstance(obj, (dict, list, tuple, str)) else False:
        return None
    return obj
