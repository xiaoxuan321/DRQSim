# -*- coding: utf-8 -*-
"""
Resource discovery utilities for the modular drq simulator.

This resource-calibrated version keeps the original transparent resource
mapping functions and adds lightweight diagnostics / recommendations used by
CWD-oriented resource calibration. Resource assignment is still handled by the
queue scheduling runtime; this file only discovers capability and empirical
workload information.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List

import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog


def _to_dataframe(log: EventLog | pd.DataFrame) -> pd.DataFrame:
    """Convert a pm4py EventLog or DataFrame to a defensive DataFrame copy."""
    if isinstance(log, pd.DataFrame):
        return log.copy()
    return pm4py.convert_to_dataframe(log).copy()


def _safe_str_list(values: Iterable) -> List[str]:
    """Return unique non-empty string values while preserving order."""
    result = []
    seen = set()

    for value in values:
        if pd.isna(value):
            continue

        value = str(value)
        if value == "":
            continue

        if value not in seen:
            result.append(value)
            seen.add(value)

    return result


def discover_resources_list(
        log: EventLog | pd.DataFrame,
        thr: float = 1.0,
        resource_col: str = "org:resource"
) -> list[str]:
    """Discover the ordered resource set from the event log."""
    df = _to_dataframe(log)

    if resource_col not in df.columns:
        return ["auto"]

    values = df[resource_col].dropna().astype(str)
    values = values[values != ""]

    if values.empty:
        return ["auto"]

    counts = values.value_counts(normalize=True).sort_values(ascending=False)
    thr = max(0.0, min(float(thr), 1.0))

    if thr >= 1.0:
        resources = counts.index.tolist()
    else:
        resources = []
        cumulative = 0.0
        for resource, probability in counts.items():
            resources.append(str(resource))
            cumulative += float(probability)
            if cumulative >= thr:
                break

    return resources or ["auto"]


def discover_resources_per_act(
        log: EventLog | pd.DataFrame,
        activities: list[str],
        resources: list[str],
        thr: float = 1.0,
        activity_col: str = "concept:name",
        resource_col: str = "org:resource"
) -> dict[str, list[str]]:
    """Discover the candidate resource set for each activity."""
    df = _to_dataframe(log)

    resources = _safe_str_list(resources)
    if not resources:
        resources = ["auto"]

    activities = _safe_str_list(activities)

    if activity_col not in df.columns or resource_col not in df.columns:
        return {activity: resources[:] for activity in activities}

    df = df[[activity_col, resource_col]].dropna().copy()
    df[activity_col] = df[activity_col].astype(str)
    df[resource_col] = df[resource_col].astype(str)

    resource_set = set(resources)
    df = df[df[resource_col].isin(resource_set)]

    thr = max(0.0, min(float(thr), 1.0))
    act_to_resources: dict[str, list[str]] = {}

    for activity in activities:
        df_act = df[df[activity_col] == activity]

        if df_act.empty:
            act_to_resources[activity] = resources[:]
            continue

        counts = df_act[resource_col].value_counts(normalize=True).sort_values(
            ascending=False
        )

        if thr >= 1.0:
            selected = counts.index.astype(str).tolist()
        else:
            selected = []
            cumulative = 0.0
            for resource, probability in counts.items():
                selected.append(str(resource))
                cumulative += float(probability)
                if cumulative >= thr:
                    break

        act_to_resources[activity] = selected or resources[:]

    return act_to_resources


def return_multitasking_resources(
        df_features: pd.DataFrame,
        thr: float = 0.1,
        resource_col: str = "resource",
        workload_col: str = "res_workload"
) -> list[str]:
    """Identify resources with historical multitasking behavior."""
    if df_features is None or df_features.empty:
        return []

    if resource_col not in df_features.columns or workload_col not in df_features.columns:
        return []

    df = df_features[[resource_col, workload_col]].dropna().copy()

    if df.empty:
        return []

    df[resource_col] = df[resource_col].astype(str)
    df[workload_col] = pd.to_numeric(df[workload_col], errors="coerce").fillna(0.0)

    multitasking = []
    for resource, group in df.groupby(resource_col):
        if len(group) == 0:
            continue
        ratio = float((group[workload_col] > 0).mean())
        if ratio >= float(thr):
            multitasking.append(str(resource))

    return multitasking


def discover_activity_resource_frequency(
        log_or_df: EventLog | pd.DataFrame,
        activities: list[str] | None = None,
        resources: list[str] | None = None,
        activity_col: str = "concept:name",
        resource_col: str = "org:resource"
) -> dict[str, dict[str, float]]:
    """Compute empirical activity-resource frequencies."""
    df = _to_dataframe(log_or_df)

    if activity_col not in df.columns or resource_col not in df.columns:
        return {}

    df = df[[activity_col, resource_col]].dropna().copy()
    df[activity_col] = df[activity_col].astype(str)
    df[resource_col] = df[resource_col].astype(str)

    if activities is not None:
        activities = _safe_str_list(activities)
        df = df[df[activity_col].isin(activities)]
    else:
        activities = sorted(df[activity_col].unique().tolist())

    if resources is not None:
        resources = _safe_str_list(resources)
        resource_set = set(resources)
        df = df[df[resource_col].isin(resource_set)]
    else:
        resources = sorted(df[resource_col].unique().tolist())

    result: dict[str, dict[str, float]] = {}

    for activity in activities:
        df_act = df[df[activity_col] == activity]

        if df_act.empty:
            result[activity] = {resource: 1.0 / len(resources) for resource in resources} if resources else {}
            continue

        counts = df_act[resource_col].value_counts()
        total = float(counts.sum())

        result[activity] = {
            resource: float(counts.get(resource, 0.0) / total)
            for resource in resources
        }

    return result


def discover_resource_workload_profile(
        log_or_df: EventLog | pd.DataFrame,
        activity_col: str = "concept:name",
        resource_col: str = "org:resource",
) -> dict:
    """Return workload-profile statistics for analyst-facing diagnostics."""
    df = _to_dataframe(log_or_df)
    if activity_col not in df.columns or resource_col not in df.columns:
        return {"enabled": False, "reason": "missing activity/resource columns"}

    df = df[[activity_col, resource_col]].dropna().copy()
    df[activity_col] = df[activity_col].astype(str)
    df[resource_col] = df[resource_col].astype(str)
    if df.empty:
        return {"enabled": False, "reason": "empty activity-resource table"}

    resource_counts = df[resource_col].value_counts()
    activity_resource_counts = (
        df.groupby([activity_col, resource_col]).size().rename("count").reset_index()
    )

    top_resource_share = float(resource_counts.iloc[0] / max(resource_counts.sum(), 1)) if len(resource_counts) else 0.0
    activity_concentration = {}
    for activity, sub in activity_resource_counts.groupby(activity_col):
        total = float(sub["count"].sum())
        if total <= 0:
            continue
        top_row = sub.sort_values("count", ascending=False).iloc[0]
        activity_concentration[str(activity)] = {
            "top_resource": str(top_row[resource_col]),
            "top_share": float(top_row["count"] / total),
            "n_resources": int(len(sub)),
        }

    return {
        "enabled": True,
        "n_events": int(len(df)),
        "n_resources": int(df[resource_col].nunique()),
        "n_activities": int(df[activity_col].nunique()),
        "top_resource_share": float(top_resource_share),
        "activity_concentration": activity_concentration,
    }


def recommend_candidate_resource_filter(
        log_or_df: EventLog | pd.DataFrame,
        n_trace_threshold_large: int = 5000,
        n_resource_threshold_small: int = 100,
        case_col: str = "case:concept:name",
        resource_col: str = "org:resource",
) -> dict:
    """Recommend bounded candidate-resource filtering settings.

    The recommendation is deterministic and log-size aware. It is designed as
    an internal helper for CWD-oriented resource calibration; it is not a
    user-facing tuning space.
    """
    df = _to_dataframe(log_or_df)
    n_traces = int(df[case_col].nunique()) if case_col in df.columns else 0
    n_resources = int(df[resource_col].dropna().astype(str).nunique()) if resource_col in df.columns else 0

    if n_resources <= n_resource_threshold_small:
        coverage, max_k, min_k = 0.95, 20, 3
        profile = "small_or_medium_resource_pool"
    elif n_traces and n_traces <= n_trace_threshold_large:
        coverage, max_k, min_k = 0.95, 20, 3
        profile = "long_tail_small_log"
    else:
        coverage, max_k, min_k = 0.90, 15, 3
        profile = "large_log_speed_control"

    return {
        "enabled": True,
        "profile": profile,
        "n_traces": n_traces,
        "n_resources": n_resources,
        "coverage": float(coverage),
        "max_k": int(max_k),
        "min_k": int(min_k),
    }
