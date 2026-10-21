# -*- coding: utf-8 -*-
"""
Modular drq simulator.

This refactored version removes the old DBPS five-tree discovery pipeline
from the simulator entry point. The active simulation model is organized into
five modules:

1. preprocessing and state reconstruction;
2. control-flow modeling with decision-point branch trees;
3. case-arrival modeling with arrival calendar + weekday-hour context histogram + intensity trend;
4. dynamic resource modeling with base service-time distributions + DBN state;
5. queue scheduling with a resource-aware scheduling tree.

Legacy JSON fields are still saved in a minimal form to keep the existing web
application and saved-parameter workflow compatible.
"""

from __future__ import annotations

import copy
import heapq
import json
import math
import os
import random
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import Marking, PetriNet
from tqdm import tqdm

from drqsim.config.modular_schema import (
    attach_module_params,
    build_empty_modular_params,
)
from drqsim.discovery.base_service_time_discovery import (
    deserialize_distribution_container,
    discover_base_service_time_distributions,
    sample_base_service_time,
    serialize_distribution_container,
)
from drqsim.discovery.calendar_discovery import (
    discover_arrival_calendar,
    discover_res_calendars,
)
from drqsim.discovery.data_discovery import (
    discover_attributes_distribution,
    return_label_data_attributes,
)
from drqsim.discovery.decision_point_discovery import (
    discover_decision_point_branch_models,
    discover_process_tree_guided_branch_models,
    discover_petri_structural_branch_models,
    sample_decision_point_branch,
    sample_petri_structural_branch,
    transition_key,
)
from drqsim.discovery.queue_scheduling_discovery import (
    discover_queue_scheduling_model,
)
from drqsim.discovery.conditional_prioritization_discovery import (
    discover_conditional_prioritization_model,
)
from drqsim.discovery.external_delay_discovery import (
    default_external_delay_params,
    discover_external_delay_models,
    decompose_waiting_times,
)
from drqsim.discovery.resource_discovery import (
    discover_resources_list,
    discover_resources_per_act,
    return_multitasking_resources,
)
from drqsim.discovery.resource_state_discovery import (
    discover_resource_state_params,
)
from drqsim.discovery.time_discovery import (
    discover_arrival_time,
    sample_case_arrival_delta,
)
from drqsim.runtime.dynamic_resource_state import DynamicResourceStateRuntime
from drqsim.runtime.external_delay_runtime import ExternalDelayRuntime
from drqsim.runtime.queue_scheduling_runtime import QueueSchedulingRuntime
from drqsim.runtime.conditional_prioritization_runtime import ConditionalPrioritizationRuntime
from drqsim.runtime.queue_state_runtime import QueueStateRuntime
from drqsim.utils.common_utils import (
    add_minutes_with_calendar,
    build_df_features,
    count_concurrent_events,
    return_enabled_transitions,
    update_current_marking,
)
from drqsim.utils.distribution_utils import sampling_from_dist
from drqsim.utils.save_and_load_utils import (
    convert_calendar_names,
    decision_rules_to_dict,
    fromstr_to_scipy,
    name_to_transition,
    transition_to_name,
)


QUEUE_STATE_FEATURE_NAMES = [
    "global_queue_length",
    "activity_queue_length",
    "resource_queue_length",
    "resource_activity_queue_length",
    "queue_pressure",
    "system_wip",
]


def _fixed_distribution(value: float = 0.0) -> tuple:
    value = float(value)
    return "fixed", (value,), value, value, value


def _default_decision_point_params() -> dict:
    return {
        "model_type": "decision_point_branch_tree",
        "num_decision_points": 0,
        "decision_points": {},
        "metadata": {},
        "fallback": {
            "single_enabled_transition": "fire_directly",
            "missing_decision_point": "uniform_over_enabled_transitions",
            "missing_feature": 0.0,
        },
    }


def _default_resource_state_params() -> dict:
    return {
        "model_type": "discrete_two_slice_dbn",
        "state_variables": [
            "fatigue",
            "workload",
            "skill",
            "efficiency",
            "availability",
        ],
        "context_variables": [
            "activity",
            "idle_gap",
            "work_duration",
            "hour",
            "weekday",
        ],
        "variable_states": {},
        "structure": {},
        "cpds": {},
        "initial_state": {},
        "discretization": {},
        "effect_params": {},
    }


def _default_queue_params() -> dict:
    return {
        "model_type": "runtime_queue_state_for_scheduling",
        "feature_names": QUEUE_STATE_FEATURE_NAMES[:],
        "use_log_scaled_features": True,
        "max_queue_count": 8,
        "max_wip_count": 30,
    }


def _default_queue_scheduling_params() -> dict:
    return {
        "model_type": "global_resource_aware_queue_scheduling_tree",
        "model_kind": "empirical",
        "feature_columns": [],
        "rules": None,
        "positive_rate": 0.0,
        "metadata": {},
        "fallback": {
            "missing_model": "activity_resource_frequency",
            "empty_candidate_resources": "first_candidate",
            "all_zero_scores": "uniform_random",
        },
        "runtime_wait_penalty": 0.02,
        "conditional_prioritization": {
            "model_type": "conditional_prioritization_tree",
            "model_kind": "fcfs_fallback",
            "feature_columns": [],
            "rules": None,
            "fallback_policy": "fcfs",
            "tie_breaker": "enabled_time_then_case_id",
        },
    }


def _default_base_service_params() -> dict:
    return {
        "model_type": "activity_resource_base_service_time_distribution",
        "activities": {},
    }


def _is_distribution_dict(value: Any) -> bool:
    return isinstance(value, dict) and "dist_name" in value


def _deserialize_distribution_dict(value: dict) -> tuple:
    return (
        fromstr_to_scipy(value["dist_name"]),
        tuple(value.get("params", [])),
        value.get("min_value", 0.0),
        value.get("max_value", 0.0),
        value.get("mean_value", 0.0),
    )


class SimulatorParameters:
    """
    Container of the modular simulation model.
    """

    def __init__(
            self,
            net: PetriNet,
            initial_marking: Marking,
            final_marking: Marking,
            process_tree: Any | None = None,
            control_flow_structure_source: str = "unknown",
    ):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        # Process tree is retained only as optional provenance/visualization metadata.
        # Control-flow decision points are always extracted from ``self.net`` so that
        # uploaded PNML and IM-discovered models follow exactly the same route:
        # Petri net -> structural conflict points -> replay -> branch model.
        self.process_tree = process_tree
        self.control_flow_structure_source = str(control_flow_structure_source or "unknown")

        self.net_transition_labels = list({
            t.label for t in self.net.transitions if t.label
        })

        self.label_data_attributes: list[str] = []
        self.label_data_attributes_categorical: list[str] = []
        self.attribute_values_label_categorical: dict = {}
        self.distribution_data_attributes: dict = {}

        self.rules_mode: bool = True

        # Kept only as a lightweight compatibility fallback.
        self.transition_weights = {t: 1.0 for t in self.net.transitions}

        self.decision_point_params = _default_decision_point_params()
        self.modular_params = build_empty_modular_params()

        self.resources: list[str] = ["auto"]
        self.resource_weights: dict[str, float] = {"auto": 1.0}
        self.act_to_resources: dict[str, list[str]] = {
            activity: ["auto"] for activity in self.net_transition_labels
        }
        self.multitasking_resources: list[str] = []

        self.calendars = {
            "auto": {wd: {h: True for h in range(24)} for wd in range(7)}
        }

        self.arrival_calendar = {
            wd: {h: True for h in range(24)} for wd in range(7)
        }

        self.arrival_time_distribution: Any = _fixed_distribution(1.0)
        self.arrival_time_scale: float = 1.0
        self.arrival_demand_multiplier: float = 1.0
        self.arrival_time_shift_min: float = 0.0
        self.arrival_decision_tree_max_depth: int = 3

        # Legacy-compatible shallow fallback only. The active model is
        # base_service_time_distributions.
        self.execution_time_distributions = {
            activity: _fixed_distribution(1.0)
            for activity in self.net_transition_labels
        }
        self.base_service_time_distributions = _default_base_service_params()

        # Waiting time is generated by scheduling. These fields are kept only
        # for JSON/UI compatibility.
        self.waiting_time_distributions = {"auto": _fixed_distribution(0.0)}
        self.waiting_time_distributions_base = {"auto": _fixed_distribution(0.0)}

        self.resource_state_params = _default_resource_state_params()
        self.queue_params = _default_queue_params()
        self.queue_scheduling_params = _default_queue_scheduling_params()
        self.external_delay_params = default_external_delay_params()
    @staticmethod
    def _build_activity_resource_counts(log) -> tuple[dict[str, Counter], Counter]:
        """
        Count activity-resource frequencies from the original event log.

        返回：
        1. activity_resource_counts: activity -> Counter(resource)
        2. global_resource_counts: Counter(resource)
        """
        activity_resource_counts: dict[str, Counter] = {}
        global_resource_counts: Counter = Counter()

        for trace in log:
            for event in trace:
                activity = event.get("concept:name")
                resource = event.get("org:resource")

                if activity is None or resource is None:
                    continue

                activity = str(activity)
                resource = str(resource)

                if not activity or not resource or resource.lower() == "nan":
                    continue

                if activity not in activity_resource_counts:
                    activity_resource_counts[activity] = Counter()

                activity_resource_counts[activity][resource] += 1
                global_resource_counts[resource] += 1

        return activity_resource_counts, global_resource_counts

    @staticmethod
    def _build_resource_capacity_profile(
            df_features: pd.DataFrame,
            resource_col: str = "resource",
            workload_col: str = "res_workload",
            min_samples: int = 5,
            max_capacity: int = 8,
    ) -> dict:
        """Estimate a conservative resource capacity profile from training features.

        ``res_workload`` in the feature table describes how many jobs were already
        active for a resource when an event started.  Therefore ``workload + 1``
        is an observed concurrency sample.  The runtime scheduler uses the
        resulting p90-based capacity as a *soft upper bound* for queue simulation;
        it does not disable queue scheduling and it does not make every resource
        infinitely available.
        """
        profile: dict[str, dict] = {}

        if df_features is None or getattr(df_features, "empty", True):
            return profile

        if resource_col not in df_features.columns or workload_col not in df_features.columns:
            return profile

        df = df_features[[resource_col, workload_col]].dropna().copy()
        if df.empty:
            return profile

        df[resource_col] = df[resource_col].astype(str)
        df[workload_col] = pd.to_numeric(df[workload_col], errors="coerce").fillna(0.0)

        try:
            max_capacity = max(1, int(max_capacity))
        except Exception:
            max_capacity = 8

        for resource, group in df.groupby(resource_col):
            if len(group) < int(min_samples):
                continue

            observed_concurrency = (group[workload_col].clip(lower=0.0) + 1.0)
            try:
                p50 = float(observed_concurrency.quantile(0.50))
                p90 = float(observed_concurrency.quantile(0.90))
                p95 = float(observed_concurrency.quantile(0.95))
                max_seen = float(observed_concurrency.max())
            except Exception:
                continue

            if not math.isfinite(p90):
                continue

            capacity = int(math.ceil(max(1.0, p90)))
            capacity = max(1, min(capacity, max_capacity))

            profile[str(resource)] = {
                "n_samples": int(len(group)),
                "p50_observed_concurrency": p50,
                "p90_observed_concurrency": p90,
                "p95_observed_concurrency": p95,
                "max_observed_concurrency": max_seen,
                "capacity": int(capacity),
                "estimation": "ceil_p90_observed_concurrency",
            }

        return profile

    @staticmethod
    def _build_adaptive_wait_cap_profile(
            df_features: pd.DataFrame,
            activity_col: str = "transition_label",
            enabled_col: str = "enabled_t",
            start_col: str = "start_t",
            wait_col: str = "queue_wait_observed_min",
            quantile: float = 0.99,
            min_cap_min: float = 480.0,
            max_cap_min: float = 7 * 24 * 60.0,
            min_samples_per_activity: int = 5,
    ) -> dict:
        """Estimate queue-wait caps from the training log.

        A fixed 480-minute cap can cut off real cross-day/cross-week waits.
        This profile uses p99 with a 7-day upper bound by default: generic
        enough for five-log experiments, but less aggressive than p99.5 + 14 days.
        """
        result = {
            "enabled": False,
            "method": "training_log_wait_quantile",
            "quantile": float(quantile),
            "min_cap_min": float(min_cap_min),
            "max_cap_min": float(max_cap_min),
            "global_cap_min": float(min_cap_min),
            "activity_cap_min": {},
            "diagnostics": {},
        }

        if df_features is None or getattr(df_features, "empty", True):
            result["reason"] = "empty_df_features"
            return result

        if enabled_col not in df_features.columns or start_col not in df_features.columns:
            result["reason"] = "missing_enabled_or_start_time"
            return result

        df = df_features.copy()
        df[enabled_col] = pd.to_datetime(df[enabled_col], errors="coerce")
        df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
        df = df.dropna(subset=[enabled_col, start_col]).copy()

        if df.empty:
            result["reason"] = "no_valid_time_rows"
            return result

        if wait_col in df.columns:
            wait_min = pd.to_numeric(df[wait_col], errors="coerce").dropna()
        else:
            # Backward-compatible fallback for old feature tables.
            wait_min = (df[start_col] - df[enabled_col]).dt.total_seconds() / 60.0
            wait_min = pd.to_numeric(wait_min, errors="coerce").dropna()
        wait_min = wait_min[(wait_min >= 0.0) & (wait_min.map(math.isfinite))]

        if wait_min.empty:
            result["reason"] = "no_nonnegative_wait_samples"
            return result

        try:
            quantile = float(quantile)
        except Exception:
            quantile = 0.99
        quantile = max(0.50, min(1.0, quantile))

        try:
            min_cap_min = max(0.0, float(min_cap_min))
        except Exception:
            min_cap_min = 480.0
        try:
            max_cap_min = max(min_cap_min, float(max_cap_min))
        except Exception:
            max_cap_min = 7 * 24 * 60.0

        global_q = float(wait_min.quantile(quantile))
        global_cap = min(max_cap_min, max(min_cap_min, global_q))

        result["enabled"] = True
        result["global_cap_min"] = float(global_cap)
        result["diagnostics"]["global"] = {
            "n": int(len(wait_min)),
            "mean": float(wait_min.mean()),
            "p90": float(wait_min.quantile(0.90)),
            "p95": float(wait_min.quantile(0.95)),
            "p99": float(wait_min.quantile(0.99)),
            "p995": float(wait_min.quantile(0.995)),
            "max": float(wait_min.max()),
        }

        if activity_col in df.columns:
            activity_caps: dict[str, float] = {}
            activity_diag: dict[str, dict] = {}
            tmp_cols = [activity_col, enabled_col, start_col]
            if wait_col in df.columns:
                tmp_cols.append(wait_col)
            tmp = df[tmp_cols].copy()
            tmp[activity_col] = tmp[activity_col].astype(str)
            if wait_col in tmp.columns:
                tmp["wait_min"] = pd.to_numeric(tmp[wait_col], errors="coerce")
            else:
                tmp["wait_min"] = (tmp[start_col] - tmp[enabled_col]).dt.total_seconds() / 60.0
            tmp["wait_min"] = pd.to_numeric(tmp["wait_min"], errors="coerce")
            tmp = tmp.dropna(subset=["wait_min"])
            tmp = tmp[(tmp["wait_min"] >= 0.0) & (tmp["wait_min"].map(math.isfinite))]

            for activity, group in tmp.groupby(activity_col):
                values = group["wait_min"]
                if len(values) < int(min_samples_per_activity):
                    continue

                qv = float(values.quantile(quantile))
                cap = min(max_cap_min, max(min_cap_min, qv))
                activity_caps[str(activity)] = float(cap)
                activity_diag[str(activity)] = {
                    "n": int(len(values)),
                    "p90": float(values.quantile(0.90)),
                    "p95": float(values.quantile(0.95)),
                    "p99": float(values.quantile(0.99)),
                    "p995": float(values.quantile(0.995)),
                    "max": float(values.max()),
                    "cap_min": float(cap),
                }

            result["activity_cap_min"] = activity_caps
            result["diagnostics"]["activity"] = activity_diag

        return result

    def _apply_candidate_resource_filter(
            self,
            log,
            enable_candidate_resource_filter: bool = False,
            candidate_resource_coverage: float = 0.90,
            candidate_resource_max_k: int | None = 20,
            candidate_resource_min_k: int = 3,
            verbose: bool = True,
    ):
        """
        Limit candidate resources per activity.

        这个函数的作用不是简单截断全局 resources，而是：
        1. 对每个 activity，根据该活动下资源出现频率排序；
        2. 保留最多 candidate_resource_max_k 个资源；
        3. 同时尽量覆盖 candidate_resource_coverage 的活动-资源频率；
        4. 最后将全局 self.resources 缩小为所有活动候选资源的并集。

        注意：
        - candidate_resource_max_k=20 表示每个活动最多 20 个候选资源；
        - 全局 resources 数量可能大于 20，因为不同活动的 top20 资源可能不同。
        """
        if not enable_candidate_resource_filter:
            return

        try:
            candidate_resource_coverage = float(candidate_resource_coverage)
            candidate_resource_coverage = max(0.0, min(candidate_resource_coverage, 1.0))
        except Exception:
            candidate_resource_coverage = 0.90

        try:
            candidate_resource_min_k = max(1, int(candidate_resource_min_k))
        except Exception:
            candidate_resource_min_k = 1

        if candidate_resource_max_k is not None:
            try:
                candidate_resource_max_k = max(1, int(candidate_resource_max_k))
            except Exception:
                candidate_resource_max_k = 20

            candidate_resource_max_k = max(
                candidate_resource_min_k,
                candidate_resource_max_k,
            )

        original_resources = [str(r) for r in self.resources]
        original_resource_set = set(original_resources)

        activity_counts, global_counts = self._build_activity_resource_counts(log)

        filtered_act_to_resources: dict[str, list[str]] = {}
        selected_resources: set[str] = set()

        for activity in self.net_transition_labels:
            current_candidates = [
                str(r)
                for r in self.act_to_resources.get(activity, original_resources)
                if str(r) in original_resource_set
            ]

            if not current_candidates:
                current_candidates = original_resources[:]

            current_candidate_set = set(current_candidates)

            # 优先使用该活动自己的 activity-resource 频率
            counter = activity_counts.get(activity, Counter())

            ranked = [
                (str(resource), int(count))
                for resource, count in counter.most_common()
                if str(resource) in current_candidate_set
            ]

            # 如果该活动没有频率统计，则退回到全局资源频率
            if not ranked:
                ranked = sorted(
                    [
                        (resource, int(global_counts.get(resource, 0)))
                        for resource in current_candidates
                    ],
                    key=lambda x: (-x[1], x[0]),
                )

            # 兜底：仍然没有 ranked 时，直接使用当前候选
            if not ranked:
                ranked = [(resource, 1) for resource in current_candidates]

            total_count = sum(max(0, count) for _, count in ranked)
            if total_count <= 0:
                total_count = len(ranked)

            max_keep = (
                len(ranked)
                if candidate_resource_max_k is None
                else min(candidate_resource_max_k, len(ranked))
            )
            max_keep = max(1, max_keep)

            min_keep = min(candidate_resource_min_k, len(ranked))

            chosen: list[str] = []
            covered = 0

            for resource, count in ranked:
                if resource in chosen:
                    continue

                chosen.append(resource)
                covered += max(0, int(count))

                coverage = covered / max(float(total_count), 1.0)

                if len(chosen) >= min_keep:
                    if len(chosen) >= max_keep or coverage >= candidate_resource_coverage:
                        break

            if not chosen:
                chosen = [ranked[0][0]]

            filtered_act_to_resources[activity] = chosen
            selected_resources.update(chosen)

        if not selected_resources:
            selected_resources = set(original_resources)

        # 保持原资源顺序，方便日志和前端展示稳定
        self.resources = [
            resource
            for resource in original_resources
            if resource in selected_resources
        ]

        # 如果有新增资源不在原顺序中，补到最后
        for resource in sorted(selected_resources):
            if resource not in self.resources:
                self.resources.append(resource)

        self.act_to_resources = filtered_act_to_resources

        if verbose:
            candidate_sizes = [
                len(v)
                for v in self.act_to_resources.values()
                if isinstance(v, list)
            ]

            if candidate_sizes:
                print(
                    "[Candidate Resource Filter] "
                    f"enabled=True, resources {len(original_resources)} -> {len(self.resources)}, "
                    f"activity candidate size min={min(candidate_sizes)}, "
                    f"mean={sum(candidate_sizes) / len(candidate_sizes):.2f}, "
                    f"max={max(candidate_sizes)}, "
                    f"coverage={candidate_resource_coverage}, "
                    f"max_k={candidate_resource_max_k}, "
                    f"min_k={candidate_resource_min_k}"
                )
            else:
                print(
                    "[Candidate Resource Filter] "
                    f"enabled=True, resources {len(original_resources)} -> {len(self.resources)}, "
                    "but no activity candidate sizes found."
                )
    # ------------------------------------------------------------------
    # Adaptive runtime-recency helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _calendar_active_hour_count(calendar: dict) -> int:
        if not isinstance(calendar, dict):
            return 0
        count = 0
        for day in calendar.values():
            if not isinstance(day, dict):
                continue
            count += sum(1 for value in day.values() if bool(value))
        return int(count)

    @classmethod
    def _merge_runtime_calendars(
            cls,
            stable_calendars: dict,
            recent_calendars: dict,
            min_recent_active_hours: int = 12,
    ) -> dict:
        """Overlay recent calendars while preserving stable fallbacks for sparse resources."""
        stable_calendars = stable_calendars or {}
        recent_calendars = recent_calendars or {}
        merged = {}
        for resource in sorted(set(stable_calendars) | set(recent_calendars)):
            recent = recent_calendars.get(resource)
            stable = stable_calendars.get(resource)
            if cls._calendar_active_hour_count(recent) >= int(min_recent_active_hours):
                merged[str(resource)] = copy.deepcopy(recent)
            elif stable is not None:
                merged[str(resource)] = copy.deepcopy(stable)
            elif recent is not None:
                merged[str(resource)] = copy.deepcopy(recent)
        return merged

    @staticmethod
    def _merge_runtime_service_time_models(
            stable_model: dict,
            recent_model: dict,
            min_recent_activity_samples: int = 3,
    ) -> dict:
        """Use recent service-time estimates where supported, otherwise retain full-history estimates."""
        stable_model = stable_model or {}
        recent_model = recent_model or {}
        merged = copy.deepcopy(stable_model)
        stable_acts = merged.setdefault("activities", {}) if isinstance(merged, dict) else {}
        recent_acts = recent_model.get("activities", {}) if isinstance(recent_model, dict) else {}
        overlaid = []
        fallback = []

        for activity, recent_entry in recent_acts.items():
            if not isinstance(recent_entry, dict):
                continue
            try:
                n_recent = int(recent_entry.get("n_samples", 0) or 0)
            except Exception:
                n_recent = 0
            if n_recent < int(min_recent_activity_samples):
                if activity in stable_acts:
                    fallback.append(str(activity))
                continue

            target = copy.deepcopy(stable_acts.get(activity, {}))
            if not isinstance(target, dict):
                target = {}
            for key in ("activity", "n_samples", "activity_default", "fallback"):
                if key in recent_entry:
                    target[key] = copy.deepcopy(recent_entry[key])

            target_resources = target.setdefault("resources", {})
            recent_resources = recent_entry.get("resources", {}) or {}
            if isinstance(target_resources, dict) and isinstance(recent_resources, dict):
                for resource, model in recent_resources.items():
                    target_resources[str(resource)] = copy.deepcopy(model)

            target_counts = target.setdefault("resource_sample_counts", {})
            recent_counts = recent_entry.get("resource_sample_counts", {}) or {}
            if isinstance(target_counts, dict) and isinstance(recent_counts, dict):
                for resource, count in recent_counts.items():
                    target_counts[str(resource)] = int(count or 0)

            stable_acts[str(activity)] = target
            overlaid.append(str(activity))

        if isinstance(merged, dict):
            merged["runtime_recency_overlay"] = {
                "enabled": True,
                "min_recent_activity_samples": int(min_recent_activity_samples),
                "recent_activities_used": sorted(overlaid),
                "stable_fallback_activities": sorted(fallback),
            }
        return merged

    @staticmethod
    def _merge_runtime_external_delay_models(stable_model: dict, recent_model: dict) -> dict:
        """Prefer recent external-delay timers, with full-history fallback for unobserved activities."""
        stable_model = stable_model or {}
        recent_model = recent_model or {}
        merged = copy.deepcopy(recent_model) if isinstance(recent_model, dict) else {}
        if not merged:
            return copy.deepcopy(stable_model)

        stable_acts = stable_model.get("activities", {}) if isinstance(stable_model, dict) else {}
        recent_acts = merged.setdefault("activities", {})
        fallback_activities = []

        for activity, stable_entry in stable_acts.items():
            recent_entry = recent_acts.get(activity)
            try:
                n_recent = int((recent_entry or {}).get("n_samples_raw", (recent_entry or {}).get("n_samples", 0)) or 0)
            except Exception:
                n_recent = 0
            if not isinstance(recent_entry, dict) or n_recent <= 0:
                fallback_entry = copy.deepcopy(stable_entry)
                # Preserve the stable activity's enabled/disabled decision, but do
                # not replay an old activity-specific timer when the activity is
                # absent from the recent window. If recent pooled timing exists,
                # shrink the timer to that recent global distribution instead.
                if merged.get("global_empirical_delay_values") and isinstance(fallback_entry, dict):
                    fallback_entry["empirical_delay_values"] = []
                    fallback_entry["empirical_delay_sample_count"] = 0
                    fallback_entry["empirical_delay_source"] = "recent_global_external_delay_fallback"
                    if bool(fallback_entry.get("enabled", fallback_entry.get("external_delay_enabled", False))):
                        fallback_entry["model_kind"] = "global_empirical_fallback"
                        fallback_entry["fallback"] = "recent_global_empirical_capped"
                        fallback_entry["selection_reason"] = "unobserved_in_recent_window_recent_global_fallback"
                recent_acts[str(activity)] = fallback_entry
                fallback_activities.append(str(activity))

        # If the recent window has no usable pooled distribution, keep the stable one.
        if not merged.get("global_empirical_delay_values"):
            for key in (
                "global_empirical_delay_values",
                "global_empirical_delay_sample_count",
                "global_empirical_delay_source",
            ):
                if key in stable_model:
                    merged[key] = copy.deepcopy(stable_model[key])

        enabled = sorted([
            str(activity)
            for activity, entry in recent_acts.items()
            if isinstance(entry, dict) and bool(entry.get("enabled", entry.get("external_delay_enabled", False)))
        ])
        merged["enabled_activities"] = enabled
        merged["disabled_activities"] = sorted([
            str(activity) for activity in recent_acts if str(activity) not in set(enabled)
        ])
        metadata = merged.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["runtime_recency_overlay"] = {
                "enabled": True,
                "stable_fallback_activities": sorted(fallback_activities),
                "fallback_policy": "full_history_only_when_activity_unobserved_in_recent_window",
            }
        return merged

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_from_eventlog(
            self,
            log: EventLog,
            max_depth_tree: int = 3,

            decision_point_max_depth: int | None = None,
            arrival_decision_tree_max_depth: int | None = None,
            case_arrival_max_depth: int | None = None,
            queue_scheduling_max_depth: int | None = None,
            conditional_prioritization_max_depth: int | None = None,
            external_delay_max_depth: int | None = None,

            verbose: bool = True,
            queue_scheduling_force_empirical: bool = False,
            queue_scheduling_max_training_events: int | None = 15000,
            queue_scheduling_max_negative_per_positive: int = 3,
            conditional_prioritization_max_training_winners: int | None = 12000,
            conditional_prioritization_min_training_pairs: int = 80,
            conditional_prioritization_max_competitors_per_winner: int = 8,
            external_delay_discovery_max_quantile: float = 1.0,
            external_delay_runtime_max_quantile: float = 1.0,
            external_delay_min_samples_per_activity: int = 80,
            external_delay_max_training_events: int | None = None,
            external_delay_max_leaf_nodes: int | None = None,

            # 新增：候选资源过滤
            enable_candidate_resource_filter: bool = False,
            candidate_resource_coverage: float = 1.00,
            candidate_resource_max_k: int | None = None,
            candidate_resource_min_k: int = 1,

            # 新增：对齐超时保护（秒）
            alignment_timeout_seconds: int | None = None,

            # 兼容 StagewiseTuner / adaptive recency
            precomputed_df_features=None,
            runtime_event_log=None,
            runtime_recency_modules=None,
    ):
        """
        Discover the five-module simulation model from an event log.

        The complete five-module model is always discovered. There are no
        module-level discovery switches. Conditional prioritization is always
        analyzed; the data selects a pairwise CPT or the FCFS fallback.
        """

        self.rules_mode = True

        runtime_modules = {str(x).strip().lower() for x in (runtime_recency_modules or ()) if str(x).strip()}
        runtime_log = runtime_event_log if runtime_event_log is not None else log
        runtime_recency_active = bool(runtime_event_log is not None and runtime_log is not log and runtime_modules)
        self.runtime_recency_metadata = {
            "enabled": bool(runtime_recency_active),
            "modules": sorted(runtime_modules),
            "structure_cases": int(len(log)) if hasattr(log, "__len__") else None,
            "runtime_cases": int(len(runtime_log)) if hasattr(runtime_log, "__len__") else None,
        }
        if verbose and runtime_recency_active:
            print(
                "[Adaptive Runtime Recency] "
                f"structure_cases={self.runtime_recency_metadata['structure_cases']}, "
                f"runtime_cases={self.runtime_recency_metadata['runtime_cases']}, "
                f"modules={self.runtime_recency_metadata['modules']}"
            )

        max_depth_tree = max(1, int(max_depth_tree or 1))

        decision_point_max_depth = (
            max_depth_tree
            if decision_point_max_depth is None
            else max(1, int(decision_point_max_depth or 1))
        )

        # Arrival tree depth is selected by StagewiseTuner Stage 2.
        # ``case_arrival_max_depth`` is accepted as a backward-compatible alias.
        if arrival_decision_tree_max_depth is None and case_arrival_max_depth is not None:
            arrival_decision_tree_max_depth = case_arrival_max_depth

        arrival_decision_tree_max_depth = (
            max_depth_tree
            if arrival_decision_tree_max_depth is None
            else max(1, int(arrival_decision_tree_max_depth or 1))
        )
        self.arrival_decision_tree_max_depth = int(arrival_decision_tree_max_depth)

        queue_scheduling_max_depth = (
            max_depth_tree
            if queue_scheduling_max_depth is None
            else max(1, int(queue_scheduling_max_depth or 1))
        )
        conditional_prioritization_max_depth = (
            queue_scheduling_max_depth
            if conditional_prioritization_max_depth is None
            else max(1, int(conditional_prioritization_max_depth or 1))
        )

        external_delay_max_depth = (
            max_depth_tree
            if external_delay_max_depth is None
            else max(1, int(external_delay_max_depth or 1))
        )

        decision_point_depth_cv = list(range(1, decision_point_max_depth + 1))
        # 外层 StagewiseTuner 已经枚举了 arrival_depth 和 queue_depth。
        # 因此这里每次 discovery 只训练当前候选深度，保证外层 GridSearchCV 真正生效。
        arrival_depth_cv = [int(arrival_decision_tree_max_depth)]
        queue_scheduling_depth_cv = [int(queue_scheduling_max_depth)]
        conditional_prioritization_depth_cv = [int(conditional_prioritization_max_depth)]
        external_delay_depth_cv = [int(external_delay_max_depth)]

        # -------------------------------
        # 1. Preprocessing and resources
        # -------------------------------
        self.label_data_attributes, self.label_data_attributes_categorical = (
            return_label_data_attributes(log)
        )

        self.attribute_values_label_categorical = {}
        for attr in self.label_data_attributes_categorical:
            self.attribute_values_label_categorical[attr] = list(
                pm4py.get_event_attribute_values(log, attr).keys()
            )

        if verbose:
            print("Discovering resources...")

        self.resources = [str(r) for r in discover_resources_list(log)]
        if not self.resources:
            self.resources = ["auto"]

        self.act_to_resources = discover_resources_per_act(
            log=log,
            activities=self.net_transition_labels,
            resources=self.resources,
        )

        # Ensure all activities have at least one candidate resource.
        for activity in self.net_transition_labels:
            candidates = self.act_to_resources.get(activity) or self.resources
            self.act_to_resources[activity] = [str(r) for r in candidates]

        # 新增：在构建 df_features、资源日历、DBN、队列调度、external delay 之前过滤候选资源
        self._apply_candidate_resource_filter(
            log=log,
            enable_candidate_resource_filter=bool(enable_candidate_resource_filter),
            candidate_resource_coverage=float(candidate_resource_coverage),
            candidate_resource_max_k=candidate_resource_max_k,
            candidate_resource_min_k=int(candidate_resource_min_k),
            verbose=verbose,
        )

        if self.label_data_attributes:
            if verbose:
                print("Discovering data attributes...")

            self.distribution_data_attributes = discover_attributes_distribution(
                log,
                self.label_data_attributes,
                self.label_data_attributes_categorical,
            )
        else:
            self.distribution_data_attributes = {}

        if verbose:
            print("Building feature dataset...")

        # v3 path verification: confirms which common_utils.py is really loaded.
        try:
            import inspect as _inspect
            print(
                f"[CF-DEBUG-V3] build_df_features module={getattr(build_df_features, '__module__', None)}, "
                f"file={_inspect.getsourcefile(build_df_features)}, "
                f"alignment_timeout_seconds={alignment_timeout_seconds!r}",
                flush=True,
            )
        except Exception as _debug_exc:
            print(f"[CF-DEBUG-V3] unable to resolve build_df_features source: {_debug_exc}", flush=True)

        df_features = build_df_features(
            log,
            self.net,
            self.initial_marking,
            self.final_marking,
            self.act_to_resources,
            self.net_transition_labels,
            self.resources,
            self.label_data_attributes,
            alignment_timeout_seconds=alignment_timeout_seconds,
        )

        if "resource" in df_features.columns:
            df_features = df_features[
                (df_features["resource"].astype(str).isin(self.resources))
                | (df_features["resource"].isna())
            ].copy()

        df_features.reset_index(drop=True, inplace=True)
        df_features_base = df_features.copy()

        self.multitasking_resources = return_multitasking_resources(
            df_features_base
        )

        # Runtime-recency features are built from a recent complete-case window,
        # but always against the full-history Petri net/resource universe.
        runtime_df_features_base = df_features_base
        runtime_multitasking_resources = list(self.multitasking_resources)
        if runtime_recency_active:
            if verbose:
                print("Building recent-window runtime feature dataset...")
            runtime_df_features = build_df_features(
                runtime_log,
                self.net,
                self.initial_marking,
                self.final_marking,
                self.act_to_resources,
                self.net_transition_labels,
                self.resources,
                self.label_data_attributes,
                alignment_timeout_seconds=alignment_timeout_seconds,
            )
            if "resource" in runtime_df_features.columns:
                runtime_df_features = runtime_df_features[
                    (runtime_df_features["resource"].astype(str).isin(self.resources))
                    | (runtime_df_features["resource"].isna())
                ].copy()
            runtime_df_features.reset_index(drop=True, inplace=True)
            runtime_df_features_base = runtime_df_features.copy()
            runtime_multitasking_resources = return_multitasking_resources(runtime_df_features_base)

        self.queue_params = _default_queue_params()

        # -------------------------------
        # 2. Control-flow module
        # -------------------------------
        if verbose:
            print(
                "Discovering Petri-net structural conflict decision points "
                f"(source={self.control_flow_structure_source})..."
            )

        # IMPORTANT: the Petri net is the single source of truth for routing
        # structure.  If a PNML was uploaded, this is that exact net.  If no
        # PNML was provided, the integration layer first mines one with IM and
        # passes the resulting Petri net here.  We never re-mine a process tree
        # inside parameter discovery.
        self.decision_point_params = discover_petri_structural_branch_models(
            net=self.net,
            df_features=df_features_base,
            net_transition_labels=self.net_transition_labels,
            max_depths_cv=decision_point_depth_cv,
            label_data_attributes=self.label_data_attributes,
            label_data_attributes_categorical=self.label_data_attributes_categorical,
            values_categorical=self.attribute_values_label_categorical,
            min_samples_per_decision_point=30,
            dominance_threshold=0.95,
            random_state=72,
            structure_source=self.control_flow_structure_source,
            max_tau_depth=8,
        )

        if verbose and isinstance(self.decision_point_params, dict):
            metadata = self.decision_point_params.get("metadata", {}) or {}
            diagnostics = metadata.get("diagnostics", {}) or {}
            print(
                "[Petri Decision Discovery] "
                f"source={metadata.get('structure_source', self.control_flow_structure_source)}, "
                f"decision_points={self.decision_point_params.get('num_decision_points', 0)}, "
                f"observed={metadata.get('observed_structural_decision_points', 0)}, "
                f"unobserved={metadata.get('unobserved_structural_decision_points', 0)}, "
                f"free_choice={metadata.get('free_choice_decision_points', 0)}, "
                f"non_free_choice={metadata.get('non_free_choice_decision_points', 0)}, "
                f"invisible_branch_points={metadata.get('invisible_branch_decision_points', 0)}, "
                f"visible_activity_points={metadata.get('observable_activity_decision_points', 0)}, "
                f"unresolved_activity_points={metadata.get('unresolved_activity_decision_points', 0)}"
            )
            print(
                "[Petri Decision Samples] "
                f"training_samples={diagnostics.get('training_samples', 0)}, "
                f"multi_enabled_rows={diagnostics.get('rows_with_multiple_enabled_transitions', 0)}, "
                f"structural_conflict_rows={diagnostics.get('rows_with_structural_conflict_active', 0)}, "
                f"visible_activity_conflict_rows={diagnostics.get('rows_with_observable_activity_conflict_active', 0)}, "
                f"concurrency_only_rows={diagnostics.get('rows_with_only_concurrency_or_unrelated_enabled', 0)}, "
                f"activity_proxy_samples={diagnostics.get('activity_proxy_training_samples', 0)}, "
                f"ambiguous_activity_proxy={diagnostics.get('ambiguous_activity_proxy_rows', 0)}, "
                f"unresolved_activity_rows={diagnostics.get('unresolved_activity_branch_rows', 0)}, "
                f"fired_outside_conflict={diagnostics.get('fired_transition_outside_active_conflict', 0)}"
            )
            replay_audit = metadata.get("control_flow_replay_audit", {}) or {}
            replay_dp = replay_audit.get("decision_points", {}) or {}
            if replay_dp:
                print(
                    "[CF-REAL] Per-decision-point replay baseline "
                    f"(cases={replay_audit.get('total_cases', 0)}):"
                )
                for signature, real_dp in sorted(
                        replay_dp.items(),
                        key=lambda kv: float((kv[1] or {}).get("visits_per_case", 0.0)),
                        reverse=True,
                ):
                    model = (self.decision_point_params.get("decision_points", {}) or {}).get(signature, {}) or {}
                    print(
                        "  "
                        f"place={real_dp.get('source_place')} "
                        f"reach={float(real_dp.get('reach_rate', 0.0)):.3f} "
                        f"visits/case={float(real_dp.get('visits_per_case', 0.0)):.3f} "
                        f"mean_when_reached={float(real_dp.get('mean_visits_when_reached', 0.0)):.3f} "
                        f"max_visits={int(real_dp.get('max_visits_in_case', 0))} "
                        f"branches={real_dp.get('activity_branch_freq', {})} "
                        f"unresolved={real_dp.get('unresolved_structural_branches', [])} "
                        f"model_enabled={model.get('enabled')} "
                        f"uses_loop_feature={bool(set(model.get('feature_columns', []) or []) & {'loop_count_at_current_node', 'gateway_visit_count', 'is_revisit', 'loop_state::revisit', 'loop_state::deep_loop'})}"
                    )
                    if real_dp.get("unresolved_structural_branches"):
                        print(
                            "     "
                            f"structural_branches={real_dp.get('structural_branches', [])} "
                            f"label_sets={real_dp.get('structural_branch_label_sets', {})}"
                        )
        self.transition_weights = {t: 1.0 for t in self.net.transitions}

        self.modular_params = build_empty_modular_params()
        self.modular_params = attach_module_params(
            self.modular_params,
            "control_flow",
            {
                "model_type": "petri_structural_conflict_decision_point_branch_trees",
                "structure_source": self.control_flow_structure_source,
                "decision_point_branch_models": self.decision_point_params,
                "fallback": "neutral_uniform_concurrency",
            },
        )

        # -------------------------------
        # 3. Calendars and arrival module
        # -------------------------------
        if verbose:
            print("Discovering resource calendars...")

        # Queue/resource behavior remains full-history. Runtime calendars may be
        # overlaid from the recent window, with sparse-resource fallback.
        stable_calendars = discover_res_calendars(log, self.resources)
        if runtime_recency_active and "calendar" in runtime_modules:
            recent_calendars = discover_res_calendars(runtime_log, self.resources)
            self.calendars = self._merge_runtime_calendars(
                stable_calendars, recent_calendars, min_recent_active_hours=12
            )
        else:
            self.calendars = stable_calendars

        arrival_discovery_log = (
            runtime_log
            if runtime_recency_active and "arrival" in runtime_modules
            else log
        )
        self.arrival_calendar = discover_arrival_calendar(arrival_discovery_log)

        # Full-history waiting decomposition is used for stable Queue/Dispatch discovery.
        resource_capacity_profile = self._build_resource_capacity_profile(df_features_base)
        df_features_wait = decompose_waiting_times(
            df_features=df_features_base,
            calendars=stable_calendars,
            resource_capacity_profile=resource_capacity_profile,
            multitasking_resources=self.multitasking_resources,
        )

        # External Delay is a temporal residual. When recency is active, recompute
        # its decomposition using the recent feature window and recent calendars.
        runtime_resource_capacity_profile = resource_capacity_profile
        df_features_runtime_wait = df_features_wait
        if runtime_recency_active and "external_delay" in runtime_modules:
            runtime_resource_capacity_profile = self._build_resource_capacity_profile(runtime_df_features_base)
            df_features_runtime_wait = decompose_waiting_times(
                df_features=runtime_df_features_base,
                calendars=self.calendars,
                resource_capacity_profile=runtime_resource_capacity_profile,
                multitasking_resources=runtime_multitasking_resources,
            )

        if verbose and not df_features_wait.empty:
            try:
                print(
                    "[Waiting Decomposition] "
                    f"total={df_features_wait['total_wait_min'].mean():.3f}, "
                    f"queue={df_features_wait['queue_wait_observed_min'].mean():.3f}, "
                    f"calendar={df_features_wait['calendar_wait_observed_min'].mean():.3f}, "
                    f"external={df_features_wait['external_delay_min'].mean():.3f}"
                )
                if runtime_recency_active and "external_delay" in runtime_modules and not df_features_runtime_wait.empty:
                    print(
                        "[Runtime Recency Waiting Decomposition] "
                        f"total={df_features_runtime_wait['total_wait_min'].mean():.3f}, "
                        f"queue={df_features_runtime_wait['queue_wait_observed_min'].mean():.3f}, "
                        f"calendar={df_features_runtime_wait['calendar_wait_observed_min'].mean():.3f}, "
                        f"external={df_features_runtime_wait['external_delay_min'].mean():.3f}"
                    )
            except Exception:
                pass

        if verbose:
            print(
                "Discovering weekday-hour context case arrival model "
                f"(arrival_decision_tree_max_depth={int(arrival_decision_tree_max_depth)})..."
            )

        self.arrival_time_distribution = discover_arrival_time(
            log=arrival_discovery_log,
            calendar_arrival=self.arrival_calendar,
            max_depths=arrival_depth_cv,
        )

        self.modular_params = attach_module_params(
            self.modular_params,
            "arrival",
            {
                "model_type": "case_arrival_decision_tree_interarrival",
                "arrival_calendar": self.arrival_calendar,
                "inter_arrival_distribution": self.arrival_time_distribution,
                "arrival_time_scale": float(self.arrival_time_scale),
                "arrival_demand_multiplier": float(self.arrival_demand_multiplier),
                "arrival_time_shift_min": float(self.arrival_time_shift_min),
                "arrival_decision_tree_max_depth": int(self.arrival_decision_tree_max_depth),
                "case_arrival_max_depth": int(self.arrival_decision_tree_max_depth),
                "depth_search_mode": "stagewise_outer_grid_fixed_candidate",
            },
        )

        # -------------------------------
        # 4. Dynamic resource module
        # -------------------------------
        if verbose:
            print("Discovering base service-time distributions...")

        stable_service_time_distributions = discover_base_service_time_distributions(
            df_features=df_features_base,
            activities=self.net_transition_labels,
            resources=self.resources,
            calendars=stable_calendars,
            act_to_resources=self.act_to_resources,
            min_samples_per_activity_resource=5,
            use_outlier_removal=True,
        )
        if runtime_recency_active and "service_time" in runtime_modules:
            recent_service_time_distributions = discover_base_service_time_distributions(
                df_features=runtime_df_features_base,
                activities=self.net_transition_labels,
                resources=self.resources,
                calendars=self.calendars,
                act_to_resources=self.act_to_resources,
                min_samples_per_activity_resource=5,
                use_outlier_removal=True,
            )
            self.base_service_time_distributions = self._merge_runtime_service_time_models(
                stable_service_time_distributions,
                recent_service_time_distributions,
                min_recent_activity_samples=3,
            )
        else:
            self.base_service_time_distributions = stable_service_time_distributions
        # Keep a minimal activity-level fallback for legacy UI fields.
        self.execution_time_distributions = {
            activity: _fixed_distribution(1.0)
            for activity in self.net_transition_labels
        }

        if verbose:
            print("Discovering DBN-based dynamic resource-state model...")

        self.resource_state_params = discover_resource_state_params(
            log=log,
            resources=self.resources,
            activities=self.net_transition_labels,
            df_features=df_features_base,
            calendars=stable_calendars,
            smoothing=1.0,
        )
        self.modular_params = attach_module_params(
            self.modular_params,
            "dynamic_resource",
            {
                "model_type": "base_service_distribution_plus_resource_state_dbn",
                "resources": self.resources,
                "activity_to_resources": self.act_to_resources,
                "resource_calendars": self.calendars,
                "multitasking_resources": self.multitasking_resources,
                "base_service_time_distributions": self.base_service_time_distributions,
                "resource_state_dbn": self.resource_state_params,
            },
        )

        # -------------------------------
        # 5. Queue scheduling module
        # -------------------------------
        self.resource_weights = {str(r): 1.0 for r in self.resources}
        self.waiting_time_distributions_base = {
            str(r): _fixed_distribution(0.0) for r in self.resources
        }
        self.waiting_time_distributions = self.waiting_time_distributions_base

        if verbose:
            print(
                "Discovering global resource-aware queue scheduling tree "
                f"(queue_scheduling_max_depth={int(queue_scheduling_max_depth)})..."
            )

        self.queue_scheduling_params = discover_queue_scheduling_model(
            df_features=df_features_wait,
            act_to_resources=self.act_to_resources,
            resources=self.resources,
            activities=self.net_transition_labels,
            max_depths_cv=queue_scheduling_depth_cv,
            label_data_attributes=self.label_data_attributes,
            label_data_attributes_categorical=self.label_data_attributes_categorical,
            values_categorical=self.attribute_values_label_categorical,
            min_samples_leaf=None,
            max_negative_per_positive=int(queue_scheduling_max_negative_per_positive),
            random_state=72,
            force_empirical=bool(queue_scheduling_force_empirical),
            max_training_events=queue_scheduling_max_training_events,
            use_fast_queue_features=True,
            max_leaf_nodes=64,
            resource_capacity_profile=resource_capacity_profile,
        )

        self.queue_scheduling_params["runtime_wait_penalty"] = 0.02

        if verbose:
            print(
                "Discovering Conditional Prioritization Tree "
                f"(max_depth={int(conditional_prioritization_max_depth)})..."
            )
        self.queue_scheduling_params["conditional_prioritization"] = (
            discover_conditional_prioritization_model(
                df_features=df_features_wait,
                act_to_resources=self.act_to_resources,
                resources=self.resources,
                activities=self.net_transition_labels,
                max_depths_cv=conditional_prioritization_depth_cv,
                label_data_attributes=self.label_data_attributes,
                label_data_attributes_categorical=self.label_data_attributes_categorical,
                values_categorical=self.attribute_values_label_categorical,
                min_samples_leaf=None,
                min_training_pairs=int(conditional_prioritization_min_training_pairs),
                max_competitors_per_winner=int(conditional_prioritization_max_competitors_per_winner),
                max_training_winners=conditional_prioritization_max_training_winners,
                random_state=72,
                max_leaf_nodes=64,
            )
        )
        self.queue_scheduling_params["model_type"] = (
            "resource_allocation_tree_plus_conditional_prioritization_tree"
        )

        # Runtime queue scheduling remains active, but we provide two
        # safeguards learned from the training log: (1) a conservative
        # resource capacity profile; (2) bounded queue-wait candidate
        # filtering.  These are unified mechanisms for long-tail resource
        # logs and are not tied to any single dataset.
        self.queue_scheduling_params.setdefault("metadata", {})
        wait_cap_profile = self._build_adaptive_wait_cap_profile(
            df_features=df_features_wait,
            wait_col="queue_wait_observed_min",
            quantile=0.99,
            min_cap_min=480.0,
            max_cap_min=7 * 24 * 60.0,
            min_samples_per_activity=5,
        )

        if isinstance(self.queue_scheduling_params["metadata"], dict):
            self.queue_scheduling_params["metadata"]["resource_capacity_profile"] = (
                resource_capacity_profile
            )
            self.queue_scheduling_params["metadata"]["adaptive_wait_cap_profile"] = wait_cap_profile

        self.queue_scheduling_params.setdefault("adaptive_resource_capacity", True)
        self.queue_scheduling_params["adaptive_wait_cap"] = bool(wait_cap_profile.get("enabled", False))
        self.queue_scheduling_params["max_queue_wait_min"] = float(
            wait_cap_profile.get("global_cap_min", 480.0)
        )
        self.queue_scheduling_params["activity_max_queue_wait_min"] = dict(
            wait_cap_profile.get("activity_cap_min", {}) or {}
        )
        self.queue_scheduling_params.setdefault("fallback_top_k_when_all_blocked", 3)

        if verbose:
            print(
                "[Adaptive Wait Cap] "
                f"enabled={bool(wait_cap_profile.get('enabled', False))}, "
                f"global_max_queue_wait_min={float(wait_cap_profile.get('global_cap_min', 480.0)):.3f}, "
                f"activity_caps={len(wait_cap_profile.get('activity_cap_min', {}) or {})}, "
                f"quantile={float(wait_cap_profile.get('quantile', 0.995)):.3f}"
            )
        self.modular_params = attach_module_params(
            self.modular_params,
            "queue_scheduling",
            {
                "model_type": "resource_allocation_plus_conditional_prioritization",
                "queue_scheduling_model": self.queue_scheduling_params,
                "conditional_prioritization_model": self.queue_scheduling_params.get("conditional_prioritization", {}),
                "replaces": [
                    "legacy_resource_selection_tree",
                    "legacy_waiting_time_tree",
                ],
            },
        )

        # -------------------------------
        # 5b. External residual-delay module
        # -------------------------------
        if verbose:
            print("Discovering Simod-lite activity external timers...")

        def _discover_external_for(
                feature_df,
                calendars_for_external,
                capacity_profile_for_external,
                multitasking_for_external,
        ):
            return discover_external_delay_models(
                df_features=feature_df,
                act_to_resources=self.act_to_resources,
                resources=self.resources,
                activities=self.net_transition_labels,
                max_depths_cv=external_delay_depth_cv,
                label_data_attributes=self.label_data_attributes,
                label_data_attributes_categorical=self.label_data_attributes_categorical,
                values_categorical=self.attribute_values_label_categorical,
                min_samples_per_activity=int(external_delay_min_samples_per_activity),
                min_samples_leaf=None,
                use_outlier_removal=False,
                max_delay_quantile=1.0,
                random_state=72,
                max_training_events=external_delay_max_training_events,
                max_leaf_nodes=external_delay_max_leaf_nodes,
                max_enabled_activity_fraction=None,
                max_enabled_activities=None,
                simod_lite_activity_timer=True,
                disable_delay_trees=True,
                calendars=calendars_for_external,
                resource_capacity_profile=capacity_profile_for_external,
                multitasking_resources=multitasking_for_external,
            )

        stable_external_delay_params = _discover_external_for(
            df_features_wait,
            stable_calendars,
            resource_capacity_profile,
            self.multitasking_resources,
        )
        if runtime_recency_active and "external_delay" in runtime_modules:
            recent_external_delay_params = _discover_external_for(
                df_features_runtime_wait,
                self.calendars,
                runtime_resource_capacity_profile,
                runtime_multitasking_resources,
            )
            self.external_delay_params = self._merge_runtime_external_delay_models(
                stable_external_delay_params, recent_external_delay_params
            )
        else:
            self.external_delay_params = stable_external_delay_params

        self.external_delay_params.setdefault("runtime", {})
        if isinstance(self.external_delay_params["runtime"], dict):
            self.external_delay_params["runtime"]["global_scale"] = float(
                self.external_delay_params["runtime"].get("global_scale", 1.0)
            )
            self.external_delay_params["runtime"]["max_delay_quantile"] = 1.0
            self.external_delay_params["max_delay_quantile"] = 1.0
            self.external_delay_params["runtime"]["residual_guard_in_runtime"] = False
            self.external_delay_params["runtime"]["residual_after_queue_only"] = False
            self.external_delay_params["runtime"]["queue_credit_ratio"] = 0.0
            self.external_delay_params["runtime"]["queue_residual_decay"] = 1.0
            self.external_delay_params["runtime"]["max_external_delay_ratio"] = None
            self.external_delay_params["runtime"]["calendar_aware_timer"] = False
            self.external_delay_params["runtime"]["calendar_aware_mode"] = "elapsed_time"
        if isinstance(self.modular_params, dict):
            self.modular_params["runtime_recency"] = copy.deepcopy(self.runtime_recency_metadata)
        self.modular_params = attach_module_params(
            self.modular_params,
            "external_delay",
            {
                "model_type": "activity_specific_external_delay_timer",
                "external_delay_model": self.external_delay_params,
                "complements": "queue_scheduling",
                "note": (
                    "Queue scheduling explains resource-competition waiting; "
                    "external-delay trees add non-resource residual start delays."
                ),
            },
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        arrival_distribution = serialize_distribution_container(
            self.arrival_time_distribution
        )

        return {
            "rules_mode": True,
            "runtime_recency": copy.deepcopy(getattr(self, "runtime_recency_metadata", {"enabled": False})),

            # Minimal compatibility field. No DBPS transition trees are saved.
            "transition_params": {
                "transition_weights": {
                    transition_to_name(t): float(w)
                    for t, w in self.transition_weights.items()
                }
            },

            "control_flow_params": {
                "structure_source": self.control_flow_structure_source,
                "decision_point_branch_models": self.decision_point_params,
            },

            "modular_params": serialize_distribution_container(self.modular_params),

            "resource_params": {
                "resources": self.resources,
                "multitasking_resource": self.multitasking_resources,
                "act_to_resources": self.act_to_resources,
                "resource_weights": {
                    str(r): float(w) if isinstance(w, (int, float)) else w
                    for r, w in self.resource_weights.items()
                },
                "calendars": {
                    str(r): convert_calendar_names(cal)
                    for r, cal in self.calendars.items()
                },
            },

            "arrival_params": {
                "arrival_calendar": convert_calendar_names(self.arrival_calendar),
                "arrival_time_distributions": arrival_distribution,
                "arrival_time_scale": float(self.arrival_time_scale),
                "arrival_demand_multiplier": float(self.arrival_demand_multiplier),
                "arrival_time_shift_min": float(self.arrival_time_shift_min),
                "arrival_decision_tree_max_depth": int(getattr(self, "arrival_decision_tree_max_depth", 3)),
                "case_arrival_max_depth": int(getattr(self, "arrival_decision_tree_max_depth", 3)),
            },

            "execution_time_params": {
                "execution_time_distributions": {
                    activity: decision_rules_to_dict(dist)
                    for activity, dist in self.execution_time_distributions.items()
                },
                "base_service_time_distributions": serialize_distribution_container(
                    self.base_service_time_distributions
                ),
            },

            "waiting_time_params": {
                "waiting_time_distributions": {
                    resource: decision_rules_to_dict(dist)
                    for resource, dist in self.waiting_time_distributions.items()
                },
                "waiting_time_distributions_base": {
                    resource: decision_rules_to_dict(dist)
                    for resource, dist in self.waiting_time_distributions_base.items()
                },
            },

            "data_attribute_params": {
                "label_data_attributes": self.label_data_attributes,
                "label_data_attributes_categorical": self.label_data_attributes_categorical,
                "attribute_values_label_categorical": self.attribute_values_label_categorical,
                "distribution_data_attributes": {
                    attr: decision_rules_to_dict(value)
                    for attr, value in self.distribution_data_attributes.items()
                },
            },

            "resource_state_params": self.resource_state_params,
            "queue_params": self.queue_params,
            "queue_scheduling_params": self.queue_scheduling_params,
            "external_delay_params": serialize_distribution_container(self.external_delay_params),
        }

    def to_json(self, path: str = "simulator_params.json"):
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(self.to_dict(), json_file, indent=4)

    def from_dict(self, dict_params: dict):
        self.rules_mode = bool(dict_params.get("rules_mode", True))
        self.runtime_recency_metadata = copy.deepcopy(
            dict_params.get("runtime_recency", {"enabled": False}) or {"enabled": False}
        )

        data_params = dict_params.get("data_attribute_params", {})
        self.label_data_attributes = data_params.get("label_data_attributes", [])
        self.label_data_attributes_categorical = data_params.get(
            "label_data_attributes_categorical",
            [],
        )
        self.attribute_values_label_categorical = data_params.get(
            "attribute_values_label_categorical",
            {},
        )

        raw_attr_dists = data_params.get("distribution_data_attributes", {})
        self.distribution_data_attributes = {}

        for attr, value in raw_attr_dists.items():
            if attr in self.label_data_attributes_categorical:
                self.distribution_data_attributes[attr] = value
            elif _is_distribution_dict(value):
                self.distribution_data_attributes[attr] = _deserialize_distribution_dict(value)
            else:
                self.distribution_data_attributes[attr] = value

        resource_params = dict_params.get("resource_params", {})
        self.resources = [str(r) for r in resource_params.get("resources", self.resources)]

        self.act_to_resources = {
            str(activity): [str(r) for r in resources]
            for activity, resources in resource_params.get(
                "act_to_resources",
                self.act_to_resources
            ).items()
        }

        self.multitasking_resources = [
            str(r) for r in resource_params.get("multitasking_resource", [])
        ]

        self.resource_weights = resource_params.get(
            "resource_weights",
            {r: 1.0 for r in self.resources}
        )

        raw_calendars = resource_params.get("calendars", {})
        self.calendars = {
            str(r): convert_calendar_names(cal, to_number=True)
            for r, cal in raw_calendars.items()
        } or {
            r: {wd: {h: True for h in range(24)} for wd in range(7)}
            for r in self.resources
        }

        arrival_params = dict_params.get("arrival_params", {})
        raw_arrival_calendar = arrival_params.get("arrival_calendar", None)
        if raw_arrival_calendar:
            self.arrival_calendar = convert_calendar_names(
                raw_arrival_calendar,
                to_number=True,
            )

        self.arrival_time_scale = float(
            arrival_params.get("arrival_time_scale", 1.0)
        )
        self.arrival_demand_multiplier = float(
            arrival_params.get("arrival_demand_multiplier", 1.0)
        )
        if self.arrival_demand_multiplier <= 0:
            self.arrival_demand_multiplier = 1.0
        self.arrival_time_shift_min = float(
            arrival_params.get("arrival_time_shift_min", 0.0)
        )
        self.arrival_decision_tree_max_depth = int(
            arrival_params.get(
                "arrival_decision_tree_max_depth",
                arrival_params.get("case_arrival_max_depth", getattr(self, "arrival_decision_tree_max_depth", 3)),
            )
        )

        raw_arrival = arrival_params.get(
            "arrival_time_distributions",
            _fixed_distribution(1.0),
        )
        self.arrival_time_distribution = deserialize_distribution_container(
            raw_arrival
        )

        self.modular_params = deserialize_distribution_container(
            dict_params.get("modular_params", build_empty_modular_params())
        )

        control_flow_params = dict_params.get("control_flow_params", {})
        self.control_flow_structure_source = str(
            control_flow_params.get(
                "structure_source",
                getattr(self, "control_flow_structure_source", "unknown"),
            )
            or "unknown"
        )
        self.decision_point_params = (
            control_flow_params.get("decision_point_branch_models")
            or self.modular_params
            .get("modules", {})
            .get("control_flow", {})
            .get("decision_point_branch_models")
            or _default_decision_point_params()
        )

        transition_params = dict_params.get("transition_params", {})
        raw_transition_weights = transition_params.get("transition_weights", {})

        self.transition_weights = {}
        for t_name, value in raw_transition_weights.items():
            transition = name_to_transition(t_name, self.net)
            if transition is not None:
                self.transition_weights[transition] = value

        for transition in self.net.transitions:
            self.transition_weights.setdefault(transition, 1.0)

        execution_params = dict_params.get("execution_time_params", {})
        raw_exec_dists = execution_params.get("execution_time_distributions", {})
        self.execution_time_distributions = {}

        for activity, value in raw_exec_dists.items():
            if _is_distribution_dict(value):
                self.execution_time_distributions[activity] = _deserialize_distribution_dict(value)
            else:
                self.execution_time_distributions[activity] = deserialize_distribution_container(value)

        for activity in self.net_transition_labels:
            self.execution_time_distributions.setdefault(activity, _fixed_distribution(1.0))

        dynamic_resource_module = (
            self.modular_params
            .get("modules", {})
            .get("dynamic_resource", {})
        )

        self.base_service_time_distributions = deserialize_distribution_container(
            execution_params.get(
                "base_service_time_distributions",
                dynamic_resource_module.get(
                    "base_service_time_distributions",
                    _default_base_service_params(),
                ),
            )
        )

        waiting_params = dict_params.get("waiting_time_params", {})
        raw_waiting = waiting_params.get("waiting_time_distributions", {})
        raw_waiting_base = waiting_params.get(
            "waiting_time_distributions_base",
            raw_waiting,
        )

        self.waiting_time_distributions = {
            str(resource): (
                _deserialize_distribution_dict(value)
                if _is_distribution_dict(value)
                else deserialize_distribution_container(value)
            )
            for resource, value in raw_waiting.items()
        } or {str(r): _fixed_distribution(0.0) for r in self.resources}

        self.waiting_time_distributions_base = {
            str(resource): (
                _deserialize_distribution_dict(value)
                if _is_distribution_dict(value)
                else deserialize_distribution_container(value)
            )
            for resource, value in raw_waiting_base.items()
        } or self.waiting_time_distributions

        self.resource_state_params = dict_params.get(
            "resource_state_params",
            _default_resource_state_params(),
        )

        self.queue_params = dict_params.get(
            "queue_params",
            _default_queue_params(),
        )

        queue_scheduling_from_modular = (
            self.modular_params
            .get("modules", {})
            .get("queue_scheduling", {})
            .get("queue_scheduling_model", {})
        )

        self.queue_scheduling_params = (
            dict_params.get("queue_scheduling_params")
            or queue_scheduling_from_modular
            or _default_queue_scheduling_params()
        )

        external_delay_from_modular = (
            self.modular_params
            .get("modules", {})
            .get("external_delay", {})
            .get("external_delay_model", {})
        )

        self.external_delay_params = deserialize_distribution_container(
            dict_params.get("external_delay_params")
            or external_delay_from_modular
            or default_external_delay_params()
        )

        # Re-attach modules to ensure loaded old files obtain the new schema.
        self.modular_params = attach_module_params(
            self.modular_params,
            "control_flow",
            {
                "model_type": "petri_structural_conflict_decision_point_branch_trees",
                "structure_source": self.control_flow_structure_source,
                "decision_point_branch_models": self.decision_point_params,
                "fallback": "neutral_uniform_concurrency",
            },
        )

        self.modular_params = attach_module_params(
            self.modular_params,
            "arrival",
            {
                "model_type": "case_arrival_decision_tree_interarrival",
                "arrival_calendar": self.arrival_calendar,
                "inter_arrival_distribution": self.arrival_time_distribution,
                "arrival_time_scale": self.arrival_time_scale,
                "arrival_demand_multiplier": self.arrival_demand_multiplier,
                "arrival_time_shift_min": self.arrival_time_shift_min,
            },
        )

        self.modular_params = attach_module_params(
            self.modular_params,
            "dynamic_resource",
            {
                "model_type": "base_service_distribution_plus_resource_state_dbn",
                "resources": self.resources,
                "activity_to_resources": self.act_to_resources,
                "resource_calendars": self.calendars,
                "multitasking_resources": self.multitasking_resources,
                "base_service_time_distributions": self.base_service_time_distributions,
                "resource_state_dbn": self.resource_state_params,
            },
        )

        self.modular_params = attach_module_params(
            self.modular_params,
            "queue_scheduling",
            {
                "model_type": "resource_allocation_plus_conditional_prioritization",
                "queue_scheduling_model": self.queue_scheduling_params,
                "conditional_prioritization_model": self.queue_scheduling_params.get("conditional_prioritization", {}),
            },
        )

        self.modular_params = attach_module_params(
            self.modular_params,
            "external_delay",
            {
                "model_type": "activity_specific_external_delay_timer",
                "external_delay_model": self.external_delay_params,
                "complements": "queue_scheduling",
            },
        )

    def from_json(self, path: str = "simulator_params.json"):
        with open(path, "r", encoding="utf-8") as file:
            self.from_dict(json.load(file))


class SimulatorEngine:
    """Execute simulation from ``SimulatorParameters``."""

    def __init__(self, simulation_parameters: SimulatorParameters):
        self.net = simulation_parameters.net
        self.initial_marking = simulation_parameters.initial_marking
        self.final_marking = simulation_parameters.final_marking
        self.simulation_parameters = simulation_parameters

        # Arrival sampler diagnostics. These counters are filled by
        # sample_case_arrival_delta() during a simulation run and then read by
        # drqIntegration.run_simulation().
        self.arrival_diagnostics = {
            "source_counts": {},
            "sampled_delta_min_by_source": {},
            "max_values_per_source": 5000,
        }

        self.resource_state_runtime = DynamicResourceStateRuntime(
            resource_state_params=getattr(
                simulation_parameters,
                "resource_state_params",
                {},
            ),
            resources=simulation_parameters.resources,
            activities=simulation_parameters.net_transition_labels,
            multitasking_resources=simulation_parameters.multitasking_resources,
        )

        self.queue_state_runtime = QueueStateRuntime(
            queue_params=getattr(
                simulation_parameters,
                "queue_params",
                {},
            ),
            act_to_resources=simulation_parameters.act_to_resources,
            resources=simulation_parameters.resources,
        )

        self.queue_scheduling_runtime = QueueSchedulingRuntime(
            queue_scheduling_params=getattr(
                simulation_parameters,
                "queue_scheduling_params",
                {},
            ),
            act_to_resources=simulation_parameters.act_to_resources,
            resources=simulation_parameters.resources,
            activities=simulation_parameters.net_transition_labels,
        )

        queue_scheduling_params = getattr(
            simulation_parameters, "queue_scheduling_params", {}
        ) or {}
        self.conditional_prioritization_runtime = ConditionalPrioritizationRuntime(
            prioritization_params=(queue_scheduling_params.get("conditional_prioritization", {})
                                   if isinstance(queue_scheduling_params, dict) else {}),
            activities=simulation_parameters.net_transition_labels,
        )

        self.external_delay_runtime = ExternalDelayRuntime(
            external_delay_params=getattr(
                simulation_parameters,
                "external_delay_params",
                {},
            ),
            resources=simulation_parameters.resources,
            activities=simulation_parameters.net_transition_labels,
        )

        # Control-flow debug is observational only. It never changes weights or
        # consumes random numbers. Set drq_CONTROL_FLOW_DEBUG=0 to silence it.
        self.control_flow_debug_enabled = str(
            os.environ.get("drq_CONTROL_FLOW_DEBUG", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.control_flow_debug_rows = []
        self.control_flow_diagnostics = {}

        # Performance cache (behavior-preserving):
        # The Petri-net structure is immutable during simulation, but the old
        # runtime repeatedly scanned self.net.arcs to find input places of a
        # transition. This becomes expensive for long CVS-style simulations.
        # Cache only structural information; no simulation state is changed.
        self._transition_input_places_cache = {
            transition: [arc.source for arc in self.net.arcs if arc.target == transition]
            for transition in self.net.transitions
        }
        try:
            self.control_flow_debug_max_rows = max(1000, int(
                os.environ.get("drq_CONTROL_FLOW_DEBUG_MAX_ROWS", "50000")
            ))
        except Exception:
            self.control_flow_debug_max_rows = 50000
        self.control_flow_debug_dropped_rows = 0

    def _sample_arrival_delta(self, current_time, deterministic_time: bool) -> float:
        return float(
            sample_case_arrival_delta(
                self.simulation_parameters.arrival_time_distribution,
                current_time=current_time,
                deterministic_time=deterministic_time,
                diagnostics=self.arrival_diagnostics,
            )
        )

    def _should_advance_arrival_with_calendar(self) -> bool:
        """
        Decide how to advance case-arrival timestamps.

        The previous implementation always used add_minutes_with_calendar(). That
        is correct only when the sampled inter-arrival is measured in active
        arrival-calendar minutes. The new arrival model stores wall-clock
        inter-arrival minutes by default, so applying the calendar again would
        distort the learned no-arrival gaps. The model therefore carries an
        explicit sampling.advance_with_calendar flag.
        """
        model = getattr(self.simulation_parameters, "arrival_time_distribution", None)
        if isinstance(model, dict):
            sampling = model.get("sampling", {})
            if isinstance(sampling, dict) and "advance_with_calendar" in sampling:
                return bool(sampling.get("advance_with_calendar"))
        return True

    def _arrival_runtime_guard(self) -> dict:
        """Return runtime guard settings saved inside the arrival model."""
        model = getattr(self.simulation_parameters, "arrival_time_distribution", None)
        if isinstance(model, dict):
            guard = model.get("runtime_guard", {})
            if isinstance(guard, dict):
                return guard
        return {}

    def _record_arrival_guard_event(self, name: str, **payload):
        """Record bounded guard diagnostics without interrupting simulation."""
        try:
            key = f"{name}_count"
            self.arrival_diagnostics[key] = int(
                self.arrival_diagnostics.get(key, 0)
            ) + 1

            samples_key = f"{name}_samples"
            samples = self.arrival_diagnostics.setdefault(samples_key, [])
            if len(samples) < int(self.arrival_diagnostics.get("max_values_per_source", 5000)):
                samples.append(payload)
        except Exception:
            pass

    def _runtime_delta_cap_min(self) -> float | None:
        guard = self._arrival_runtime_guard()
        if not bool(guard.get("enabled", False)):
            return None

        try:
            cap = float(guard.get("delta_cap_min", 0.0))
            if math.isfinite(cap) and cap > 0.0:
                return cap
        except Exception:
            pass

        return None

    def _cap_arrival_delta_after_runtime_controls(self, delta_minutes: float) -> float:
        """
        Apply a second safety cap after arrival_time_scale and demand multiplier.

        The discovery sampler already caps raw sampled deltas, but user-facing
        what-if controls may enlarge them. This guard prevents these runtime
        controls from creating impossible timestamps.
        """
        try:
            delta = float(delta_minutes)
        except Exception:
            delta = 0.0

        if not math.isfinite(delta) or delta < 0.0:
            self._record_arrival_guard_event(
                "arrival_delta_invalid",
                raw_value=str(delta_minutes),
            )
            return 0.0

        cap = self._runtime_delta_cap_min()
        if cap is None or delta <= cap:
            return delta

        self.arrival_diagnostics["arrival_delta_runtime_cap_min"] = float(cap)
        self._record_arrival_guard_event(
            "arrival_delta_runtime_cap",
            raw_delta_min=float(delta),
            capped_delta_min=float(cap),
        )
        return float(cap)

    def _compute_arrival_timestamp_upper_bound(self, t_start, n_traces: int):
        """
        Compute a safe simulation horizon for case-arrival timestamps.

        The horizon is a runtime safety bound, not a calibration target. It is
        derived from the discovery-log guard and the requested trace count.
        """
        start = pd.Timestamp(t_start)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")

        guard = self._arrival_runtime_guard()

        try:
            horizon_min = float(guard.get("max_future_horizon_min", 0.0))
        except Exception:
            horizon_min = 0.0

        cap = self._runtime_delta_cap_min()
        if cap is None:
            cap = 24.0 * 60.0

        if not math.isfinite(horizon_min) or horizon_min <= 0.0:
            horizon_min = max(
                float(cap) * max(1.0, float(n_traces + 1)),
                30.0 * 24.0 * 60.0,
            )

        # Keep the safety horizon finite even if an old/bad JSON has no guard.
        horizon_min = min(
            max(horizon_min, 1.0 * 24.0 * 60.0),
            10.0 * 365.0 * 24.0 * 60.0,
        )

        try:
            upper = start + pd.Timedelta(minutes=float(horizon_min))
        except Exception:
            upper = pd.Timestamp.max.tz_localize("UTC") - pd.Timedelta(days=1)

        pandas_upper = pd.Timestamp.max.tz_localize("UTC") - pd.Timedelta(days=1)
        if upper > pandas_upper:
            upper = pandas_upper

        self.arrival_diagnostics["arrival_timestamp_upper_bound"] = str(upper)
        self.arrival_diagnostics["arrival_max_future_horizon_min"] = float(horizon_min)
        return upper

    def _safe_timestamp(self, value, fallback):
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts
        except Exception:
            return pd.Timestamp(fallback)

    def _advance_arrival_timestamp(self, current_timestamp, delta_minutes: float):
        delta_minutes = self._cap_arrival_delta_after_runtime_controls(delta_minutes)

        current_ts = self._safe_timestamp(current_timestamp, fallback=pd.Timestamp.utcnow())
        upper_bound = getattr(self, "_arrival_timestamp_upper_bound", None)
        if upper_bound is None:
            upper_bound = self._compute_arrival_timestamp_upper_bound(
                current_ts,
                n_traces=1,
            )

        try:
            if self._should_advance_arrival_with_calendar():
                candidate = add_minutes_with_calendar(
                    current_ts,
                    int(round(delta_minutes)),
                    self.simulation_parameters.arrival_calendar,
                )
            else:
                # Wall-clock arrival model: preserve long no-arrival gaps directly.
                candidate = current_ts + pd.Timedelta(minutes=float(delta_minutes))

            candidate = self._safe_timestamp(candidate, fallback=current_ts)
        except Exception as exc:
            self._record_arrival_guard_event(
                "arrival_timestamp_overflow",
                current_timestamp=str(current_ts),
                delta_min=float(delta_minutes),
                error=str(exc),
            )
            return upper_bound

        if candidate > upper_bound:
            self._record_arrival_guard_event(
                "arrival_timestamp_horizon_clamp",
                current_timestamp=str(current_ts),
                candidate_timestamp=str(candidate),
                upper_bound=str(upper_bound),
                delta_min=float(delta_minutes),
            )
            return upper_bound

        return candidate

    # ------------------------------------------------------------------
    # Runtime safety guards for service time and case completion
    # ------------------------------------------------------------------
    def _record_simulation_guard_event(self, name: str, **payload):
        """
        Record bounded diagnostics for runtime guards.

        These diagnostics are attached to ``arrival_diagnostics`` for backward
        compatibility because drqIntegration already reads this dictionary
        after each simulation run.
        """
        try:
            key = f"{name}_count"
            self.arrival_diagnostics[key] = int(
                self.arrival_diagnostics.get(key, 0)
            ) + 1

            samples_key = f"{name}_samples"
            samples = self.arrival_diagnostics.setdefault(samples_key, [])
            if len(samples) < int(self.arrival_diagnostics.get("max_values_per_source", 5000)):
                samples.append(payload)
        except Exception:
            pass

    def _scan_numeric_distribution_stats(self, value: Any) -> list[float]:
        """Collect useful numeric statistics from nested distribution dictionaries."""
        stats = []

        def _walk(obj, parent_key: str = ""):
            if isinstance(obj, dict):
                for key, val in obj.items():
                    key_str = str(key).lower()
                    if isinstance(val, (int, float)):
                        if any(token in key_str for token in (
                                "mean", "median", "p90", "p95", "p99",
                                "q90", "q95", "q99", "max_value", "max"
                        )):
                            try:
                                fval = float(val)
                                if math.isfinite(fval) and fval > 0.0:
                                    stats.append(fval)
                            except Exception:
                                pass
                    elif isinstance(val, (list, tuple)):
                        if any(token in key_str for token in ("observations", "samples", "values")):
                            for item in val[:5000]:
                                try:
                                    fval = float(item)
                                    if math.isfinite(fval) and fval > 0.0:
                                        stats.append(fval)
                                except Exception:
                                    pass
                        else:
                            for item in val:
                                _walk(item, key_str)
                    else:
                        _walk(val, key_str)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _walk(item, parent_key)

        _walk(value)
        return stats

    def _compute_service_time_cap_min(self) -> float:
        """
        Derive a service-time safety cap from the discovered service-time model.

        This is not a validation/test calibration. It only prevents extreme
        sampled service times from making add_minutes_with_calendar() run for
        an excessive amount of time or pushing timestamps too far into the future.
        """
        params = getattr(
            self.simulation_parameters,
            "base_service_time_distributions",
            {},
        )

        values = self._scan_numeric_distribution_stats(params)
        clean = [
            float(v) for v in values
            if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0
        ]

        # Conservative generic bounds: keep legitimate long tasks possible, but
        # prevent pathological samples from lasting months/years.
        min_cap = 24.0 * 60.0          # 1 day
        hard_cap = 30.0 * 24.0 * 60.0  # 30 days

        if not clean:
            cap = 7.0 * 24.0 * 60.0
        else:
            arr = sorted(clean)
            n = len(arr)

            def _q(q):
                idx = int(round((n - 1) * q))
                idx = max(0, min(idx, n - 1))
                return float(arr[idx])

            median_v = _q(0.50)
            p90_v = _q(0.90)
            p95_v = _q(0.95)
            p99_v = _q(0.99)
            mean_v = float(sum(arr) / max(1, n))

            candidates = [
                min_cap,
                median_v * 80.0,
                mean_v * 20.0,
                p90_v * 8.0,
                p95_v * 5.0,
                p99_v * 3.0,
            ]
            cap = max(c for c in candidates if math.isfinite(c) and c > 0.0)
            cap = min(cap, hard_cap)

        cap = max(min_cap, min(float(cap), hard_cap))
        self.arrival_diagnostics["service_time_runtime_cap_min"] = float(cap)
        return float(cap)

    def _guard_service_time(
            self,
            value: Any,
            activity: str | None = None,
            resource: str | None = None,
            case_id: int | None = None,
            section: str = "service_time",
    ) -> float:
        """Validate and cap sampled service/execution times."""
        try:
            value = float(value)
        except Exception:
            self._record_simulation_guard_event(
                "service_time_invalid",
                case_id=case_id,
                activity=str(activity),
                resource=str(resource),
                section=str(section),
                raw_value=str(value),
            )
            return 1.0

        if not math.isfinite(value) or value <= 0.0:
            self._record_simulation_guard_event(
                "service_time_invalid",
                case_id=case_id,
                activity=str(activity),
                resource=str(resource),
                section=str(section),
                raw_value=str(value),
            )
            return 1.0

        cap = getattr(self, "_service_time_runtime_cap_min", None)
        if cap is None:
            cap = self._compute_service_time_cap_min()
            self._service_time_runtime_cap_min = cap

        if value <= cap:
            return max(1.0, value)

        self._record_simulation_guard_event(
            "service_time_cap",
            case_id=case_id,
            activity=str(activity),
            resource=str(resource),
            section=str(section),
            raw_value_min=float(value),
            capped_value_min=float(cap),
        )
        return max(1.0, float(cap))

    def _marking_signature(self, marking) -> tuple:
        """Create a hashable signature for detecting repeated Petri-net markings."""
        try:
            return tuple(
                sorted(
                    (
                        getattr(place, "name", str(place)),
                        int(tokens),
                    )
                    for place, tokens in marking.items()
                    if int(tokens) != 0
                )
            )
        except Exception:
            return tuple(sorted(str(item) for item in getattr(marking, "items", lambda: [])()))

    def _compute_case_completion_guard_config(self, n_traces: int) -> dict:
        """Adaptive case-completion guard based on model size."""
        n_activities = max(1, len(getattr(self.simulation_parameters, "net_transition_labels", []) or []))
        n_transitions = max(1, len(getattr(self.net, "transitions", []) or []))

        max_events_per_case = max(
            100,
            25 * n_activities,
            10 * n_transitions,
        )
        max_events_per_case = int(min(max_events_per_case, 2000))

        max_same_marking_visits = max(
            20,
            2 * n_transitions,
        )
        max_same_marking_visits = int(min(max_same_marking_visits, 200))

        max_total_simulation_steps = int(
            max(1, int(n_traces)) * max_events_per_case
        )

        cfg = {
            "max_events_per_case": int(max_events_per_case),
            "max_same_marking_visits": int(max_same_marking_visits),
            "max_total_simulation_steps": int(max_total_simulation_steps),
        }
        self.arrival_diagnostics["case_completion_guard"] = cfg
        return cfg

    def _abort_case(
            self,
            case_id: int,
            case: dict,
            reason: str,
            completed_cases: set,
            pbar,
            current_time=None,
            activity: str | None = None,
    ) -> bool:
        """
        Stop a non-terminating/deadlocked case and count it in the progress bar.

        Returning True means the case was newly aborted/closed.
        """
        if case_id in completed_cases:
            return False

        case["enabled"] = {}
        case["aborted"] = True
        case["abort_reason"] = str(reason)
        completed_cases.add(case_id)

        try:
            pbar.update(1)
        except Exception:
            pass

        self._record_simulation_guard_event(
            "case_completion_abort",
            case_id=int(case_id),
            reason=str(reason),
            activity=str(activity),
            event_count=int(case.get("event_count", 0)),
            current_time=str(current_time) if current_time is not None else None,
            marking=str(self._marking_signature(case.get("marking"))),
        )
        return True

    def _sample_data_attributes(self, n_traces: int) -> list[list[Any]]:
        params = self.simulation_parameters

        if not params.label_data_attributes:
            return [[] for _ in range(n_traces)]

        attr_values_by_attr = []

        for attr in params.label_data_attributes:
            dist = params.distribution_data_attributes.get(attr)

            if attr in params.label_data_attributes_categorical:
                if isinstance(dist, dict) and dist:
                    choices = list(dist.keys())
                    weights = list(dist.values())
                    values = random.choices(choices, weights=weights, k=n_traces)
                else:
                    values = [None] * n_traces
            else:
                try:
                    values = sampling_from_dist(*dist, n_sample=n_traces)
                except Exception:
                    values = [0.0] * n_traces

            attr_values_by_attr.append(values)

        return [list(row) for row in zip(*attr_values_by_attr)]

    def _build_control_flow_runtime_features(self, case: dict, current_time=None) -> dict:
        """Build runtime features for Petri enabled-set decision-point branch trees."""
        features = {}
        features.update(case.get("attributes", {}))
        features.update(case.get("history", {}))

        last_activity = str(case.get("last_activity", "__START__"))
        last_2_activities = str(case.get("last_2_activities", "__START__||__START__"))

        features["prefix_length"] = float(sum(case.get("history", {}).values()))
        features["last_activity"] = last_activity
        features["last_2_activities"] = last_2_activities
        features[f"last_activity = {last_activity}"] = 1.0
        features[f"last_2_activities = {last_2_activities}"] = 1.0

        try:
            now_ts = pd.Timestamp(current_time if current_time is not None else case.get("arrival_time"))
            start_ts = pd.Timestamp(case.get("arrival_time", now_ts))
            elapsed = (now_ts - start_ts).total_seconds() / 60.0
            features["case_elapsed_time"] = max(0.0, float(elapsed))
            features["hour"] = float(now_ts.hour)
            features["weekday"] = float(now_ts.weekday())
        except Exception:
            features["case_elapsed_time"] = 0.0
            features["hour"] = 0.0
            features["weekday"] = 0.0

        # 运行时无法直接知道当前处于哪个流程树节点，先提供全局循环/重复程度特征。
        # 具体节点的 loop_count_at_current_node 在训练阶段按流程树节点计算；运行时
        # 通过活动重复次数近似，缺失列会在规则应用时置 0。
        repeated_count = 0
        for value in case.get("history", {}).values():
            try:
                repeated_count += max(0, int(value) - 1)
            except Exception:
                pass
        features["loop_count_at_current_node"] = float(repeated_count)

        # Third version: gateway-level loop context.
        # The sampler resolves the concrete gateway when it selects a branch.
        visit_total = sum(int(v) for v in case.get("decision_visit_counts", {}).values())
        features["gateway_visit_count"] = float(visit_total)
        features["is_revisit"] = float(1.0 if visit_total > 0 else 0.0)
        features["loop_state"] = (
            "first_visit" if visit_total <= 1
            else ("revisit" if visit_total <= 3 else "deep_loop")
        )

        return features

    @staticmethod
    def _control_flow_visit_bucket(visit_index: int) -> str:
        if int(visit_index) <= 1:
            return "1"
        if int(visit_index) == 2:
            return "2"
        if int(visit_index) == 3:
            return "3"
        return "4+"

    def _record_control_flow_decision_debug(
            self,
            case: dict,
            enabled_transitions: list,
            chosen_transition,
            features: dict,
            current_time=None,
            selection_policy: str = "unknown",
            selection_error: str | None = None,
    ):
        """Record active Petri conflicts without re-running branch prediction.

        Important: this function is deliberately lightweight.  The previous
        debug build called the full decision-explanation path after every
        transition choice, which repeated tree probability evaluation and could
        make tuning much slower.  Here we inspect only already-discovered model
        metadata and the enabled/chosen Petri transitions.  No random number is
        consumed and routing is unchanged.
        """
        if not self.control_flow_debug_enabled:
            return

        if len(self.control_flow_debug_rows) >= int(self.control_flow_debug_max_rows):
            self.control_flow_debug_dropped_rows += 1
            return

        params = getattr(self.simulation_parameters, "decision_point_params", {}) or {}
        if str(params.get("model_type", "")) != "petri_structural_conflict_decision_point_branch_tree":
            return

        models = params.get("decision_points", {}) or {}
        chosen_key = transition_key(chosen_transition)
        enabled_keys = [transition_key(t) for t in enabled_transitions if transition_key(t)]
        enabled_key_set = set(enabled_keys)
        visit_counts = case.setdefault("decision_visit_counts", {})

        for signature, model in models.items():
            if not isinstance(model, dict):
                continue
            if str(model.get("operator", "")).upper() != "PETRI_STRUCTURAL_CONFLICT":
                continue

            structural = [str(x) for x in model.get("structural_branches", []) or []]
            active_structural = [b for b in structural if b in enabled_key_set]
            if len(active_structural) < 2:
                continue

            signature = str(signature)
            chosen_is_branch = chosen_key in set(active_structural)
            previous_visits = int(visit_counts.get(signature, 0) or 0)
            visit_index = previous_visits + 1 if chosen_is_branch else 0
            if chosen_is_branch:
                visit_counts[signature] = int(visit_index)

            mapping = {
                str(k): str(v) for k, v in (model.get("transition_to_activity_branch", {}) or {}).items()
                if v is not None
            }
            chosen_activity = mapping.get(chosen_key) if chosen_is_branch else None
            chosen_resolved = chosen_activity is not None
            if chosen_is_branch and chosen_activity is None:
                chosen_activity = f"__UNRESOLVED__::{chosen_key}"

            active_activity = sorted({
                mapping[b] for b in active_structural if b in mapping
            })
            unresolved_active = [b for b in active_structural if b not in mapping]
            model_enabled = bool(model.get("enabled", True))
            if not model_enabled:
                runtime_policy = "model_disabled_neutral_weights"
            elif len(active_activity) < 2:
                runtime_policy = "no_observable_activity_choice"
            else:
                runtime_policy = "learned_or_empirical"

            feature_columns = [str(x) for x in model.get("feature_columns", []) or []]
            self.control_flow_debug_rows.append({
                "record_type": "decision",
                "case_id": str(case.get("case_id", "")),
                "decision_point_signature": signature,
                "source_place": model.get("source_place"),
                "current_time": str(current_time),
                "selection_policy": str(selection_policy),
                "selection_error": str(selection_error) if selection_error else "",
                "chosen_transition": chosen_key,
                "chosen_is_dp_branch": bool(chosen_is_branch),
                "chosen_activity_branch": str(chosen_activity) if chosen_activity is not None else "",
                "chosen_branch_resolved": bool(chosen_resolved) if chosen_is_branch else True,
                "visit_index": int(visit_index),
                "previous_visit_count": int(previous_visits),
                "visit_bucket": self._control_flow_visit_bucket(visit_index) if chosen_is_branch else "deferred",
                "enabled_transition_count": int(len(enabled_keys)),
                "enabled_transitions": list(enabled_keys),
                "active_structural_branches": active_structural,
                "active_activity_branches": active_activity,
                "unresolved_active_structural_branches": unresolved_active,
                "model_enabled": model_enabled,
                "runtime_policy": runtime_policy,
                "model_kind": str(model.get("model_kind", "")),
                "model_n_samples": int(model.get("n_samples", 0) or 0),
                "feature_columns": feature_columns,
                "uses_loop_count_feature": "loop_count_at_current_node" in set(feature_columns),
                "runtime_global_repeat_count": float((features or {}).get("loop_count_at_current_node", 0.0) or 0.0),
                # Deliberately omitted in lightweight mode: recomputing model
                # probabilities here would duplicate the actual routing work.
                "predicted_activity_probs": {},
                "predicted_chosen_prob": None,
                "prefix_length": float((features or {}).get("prefix_length", 0.0) or 0.0),
            })

    @staticmethod
    def _summarize_sim_control_flow_rows(rows: list, total_cases: int) -> dict:
        decision_rows = [
            row for row in (rows or [])
            if row.get("record_type") == "decision" and row.get("chosen_is_dp_branch", False)
        ]
        per_dp = {}
        for row in decision_rows:
            sig = str(row.get("decision_point_signature", ""))
            dp = per_dp.setdefault(sig, {
                "signature": sig,
                "source_place": row.get("source_place"),
                "cases": set(),
                "case_visits": Counter(),
                "activity_branch_counts": Counter(),
                "branch_by_visit_counts": {},
                "unresolved_fired_rows": 0,
                "model_disabled_rows": 0,
                "loop_feature_used_rows": 0,
                "runtime_policies": Counter(),
                "selection_policies": Counter(),
                "selection_error_rows": 0,
            })
            case_id = str(row.get("case_id", ""))
            dp["cases"].add(case_id)
            dp["case_visits"][case_id] += 1
            branch = str(row.get("chosen_activity_branch", ""))
            dp["activity_branch_counts"][branch] += 1
            bucket = str(row.get("visit_bucket", ""))
            dp["branch_by_visit_counts"].setdefault(bucket, Counter())[branch] += 1
            if not bool(row.get("chosen_branch_resolved", True)):
                dp["unresolved_fired_rows"] += 1
            if not bool(row.get("model_enabled", True)):
                dp["model_disabled_rows"] += 1
            if bool(row.get("uses_loop_count_feature", False)):
                dp["loop_feature_used_rows"] += 1
            dp["runtime_policies"][str(row.get("runtime_policy", ""))] += 1
            dp["selection_policies"][str(row.get("selection_policy", ""))] += 1
            if str(row.get("selection_error", "")):
                dp["selection_error_rows"] += 1

        result = {}
        for sig, dp in per_dp.items():
            total_visits = int(sum(dp["case_visits"].values()))
            cases_reached = int(len(dp["cases"]))
            denom_cases = max(int(total_cases), 1)
            branch_total = float(sum(dp["activity_branch_counts"].values()))
            branch_freq = {
                str(k): float(v / branch_total) if branch_total else 0.0
                for k, v in dp["activity_branch_counts"].items()
            }
            by_visit = {}
            for bucket, counts in dp["branch_by_visit_counts"].items():
                n = float(sum(counts.values()))
                by_visit[str(bucket)] = {
                    "n": int(n),
                    "counts": {str(k): int(v) for k, v in counts.items()},
                    "freq": {str(k): float(v / n) if n else 0.0 for k, v in counts.items()},
                }
            result[sig] = {
                "signature": sig,
                "source_place": dp.get("source_place"),
                "cases_reached": cases_reached,
                "reach_rate": float(cases_reached / denom_cases),
                "total_visits": total_visits,
                "visits_per_case": float(total_visits / denom_cases),
                "mean_visits_when_reached": float(total_visits / cases_reached) if cases_reached else 0.0,
                "max_visits_in_case": int(max(dp["case_visits"].values())) if dp["case_visits"] else 0,
                "activity_branch_counts": {str(k): int(v) for k, v in dp["activity_branch_counts"].items()},
                "activity_branch_freq": branch_freq,
                "branch_by_visit": by_visit,
                "unresolved_fired_rows": int(dp["unresolved_fired_rows"]),
                "model_disabled_rows": int(dp["model_disabled_rows"]),
                "loop_feature_used_rows": int(dp["loop_feature_used_rows"]),
                "runtime_policies": {str(k): int(v) for k, v in dp["runtime_policies"].items()},
                "selection_policies": {str(k): int(v) for k, v in dp["selection_policies"].items()},
                "selection_error_rows": int(dp["selection_error_rows"]),
            }
        return result

    @staticmethod
    def _branch_tv_distance(real_freq: dict, sim_freq: dict) -> float:
        keys = set((real_freq or {}).keys()) | set((sim_freq or {}).keys())
        return 0.5 * sum(
            abs(float((real_freq or {}).get(k, 0.0)) - float((sim_freq or {}).get(k, 0.0)))
            for k in keys
        )

    def _finalize_control_flow_debug(self, total_cases: int):
        if not self.control_flow_debug_enabled:
            self.control_flow_diagnostics = {}
            return

        params = getattr(self.simulation_parameters, "decision_point_params", {}) or {}
        real_audit = (params.get("metadata", {}) or {}).get("control_flow_replay_audit", {}) or {}
        real_dp = real_audit.get("decision_points", {}) or {}
        sim_dp = self._summarize_sim_control_flow_rows(
            self.control_flow_debug_rows,
            total_cases=int(total_cases),
        )

        all_sigs = sorted(set(real_dp.keys()) | set(sim_dp.keys()))
        comparisons = []
        for sig in all_sigs:
            real = real_dp.get(sig, {}) or {}
            sim = sim_dp.get(sig, {}) or {}
            rv = float(real.get("visits_per_case", 0.0) or 0.0)
            sv = float(sim.get("visits_per_case", 0.0) or 0.0)
            visit_ratio = (sv / rv) if rv > 1e-12 else (float("inf") if sv > 0 else 1.0)
            tv = self._branch_tv_distance(
                real.get("activity_branch_freq", {}) or {},
                sim.get("activity_branch_freq", {}) or {},
            )
            max_visit_tv = 0.0
            visit_tv = {}
            for bucket in ["1", "2", "3", "4+"]:
                rf = ((real.get("branch_by_visit", {}) or {}).get(bucket, {}) or {}).get("freq", {}) or {}
                sf = ((sim.get("branch_by_visit", {}) or {}).get(bucket, {}) or {}).get("freq", {}) or {}
                if rf or sf:
                    d = self._branch_tv_distance(rf, sf)
                    visit_tv[bucket] = float(d)
                    max_visit_tv = max(max_visit_tv, float(d))

            unresolved = len(real.get("unresolved_structural_branches", []) or [])
            model_disabled_rows = int(sim.get("model_disabled_rows", 0) or 0)
            ratio_component = abs(math.log(max(visit_ratio, 1e-9))) if math.isfinite(visit_ratio) else 5.0
            suspicion_score = float(
                ratio_component + tv + max_visit_tv
                + (1.0 if unresolved else 0.0)
                + (1.0 if model_disabled_rows else 0.0)
            )
            comparisons.append({
                "decision_point_signature": sig,
                "source_place": real.get("source_place", sim.get("source_place")),
                "real_reach_rate": float(real.get("reach_rate", 0.0) or 0.0),
                "sim_reach_rate": float(sim.get("reach_rate", 0.0) or 0.0),
                "real_visits_per_case": rv,
                "sim_visits_per_case": sv,
                "visit_ratio": float(visit_ratio),
                "real_max_visits": int(real.get("max_visits_in_case", 0) or 0),
                "sim_max_visits": int(sim.get("max_visits_in_case", 0) or 0),
                "branch_tv_distance": float(tv),
                "branch_tv_by_visit": visit_tv,
                "max_visit_branch_tv": float(max_visit_tv),
                "unresolved_structural_branches": list(real.get("unresolved_structural_branches", []) or []),
                "real_unresolved_fired_rows": int(real.get("unresolved_fired_rows", 0) or 0),
                "sim_unresolved_fired_rows": int(sim.get("unresolved_fired_rows", 0) or 0),
                "sim_model_disabled_rows": model_disabled_rows,
                "sim_loop_feature_used_rows": int(sim.get("loop_feature_used_rows", 0) or 0),
                "real_branch_freq": real.get("activity_branch_freq", {}) or {},
                "sim_branch_freq": sim.get("activity_branch_freq", {}) or {},
                "real_branch_by_visit": real.get("branch_by_visit", {}) or {},
                "sim_branch_by_visit": sim.get("branch_by_visit", {}) or {},
                "runtime_policies": sim.get("runtime_policies", {}) or {},
                "selection_policies": sim.get("selection_policies", {}) or {},
                "selection_error_rows": int(sim.get("selection_error_rows", 0) or 0),
                "suspicion_score": suspicion_score,
            })

        comparisons.sort(key=lambda x: float(x.get("suspicion_score", 0.0)), reverse=True)
        deferred_rows = sum(
            1 for row in self.control_flow_debug_rows
            if row.get("record_type") == "decision" and not row.get("chosen_is_dp_branch", False)
        )
        self.control_flow_diagnostics = {
            "real_total_cases": int(real_audit.get("total_cases", 0) or 0),
            "sim_total_cases": int(total_cases),
            "num_decision_points": int(len(all_sigs)),
            "active_conflict_debug_rows": int(sum(
                1 for r in self.control_flow_debug_rows if r.get("record_type") == "decision"
            )),
            "deferred_conflict_rows": int(deferred_rows),
            "dropped_debug_rows": int(getattr(self, "control_flow_debug_dropped_rows", 0) or 0),
            "per_decision_point": comparisons,
        }
        self.arrival_diagnostics["control_flow_audit"] = copy.deepcopy(self.control_flow_diagnostics)

        print(
            "[CF-AUDIT] "
            f"real_cases={self.control_flow_diagnostics['real_total_cases']}, "
            f"sim_cases={self.control_flow_diagnostics['sim_total_cases']}, "
            f"decision_points={len(all_sigs)}, "
            f"active_conflict_rows={self.control_flow_diagnostics['active_conflict_debug_rows']}, "
            f"deferred_conflicts={deferred_rows}, "
            f"dropped_debug_rows={self.control_flow_diagnostics.get('dropped_debug_rows', 0)}"
        )
        print("[CF-AUDIT] Top suspicious decision points (diagnostic ranking only):")
        for idx, row in enumerate(comparisons[:5], start=1):
            ratio_text = "inf" if not math.isfinite(float(row["visit_ratio"])) else f"{row['visit_ratio']:.3f}"
            print(
                f"  #{idx} place={row.get('source_place')} "
                f"real_visits/case={row['real_visits_per_case']:.3f} "
                f"sim_visits/case={row['sim_visits_per_case']:.3f} "
                f"ratio={ratio_text} "
                f"branch_TV={row['branch_tv_distance']:.3f} "
                f"visit_TV={row['max_visit_branch_tv']:.3f} "
                f"real_max={row['real_max_visits']} sim_max={row['sim_max_visits']} "
                f"unresolved={len(row['unresolved_structural_branches'])} "
                f"model_disabled_rows={row['sim_model_disabled_rows']} "
                f"loop_feature_rows={row['sim_loop_feature_used_rows']} "
                f"selection_errors={row['selection_error_rows']}"
            )
            print(f"     real_branch={row['real_branch_freq']}")
            print(f"     sim_branch ={row['sim_branch_freq']}")
            print(
                f"     runtime_policy={row.get('runtime_policies', {})} "
                f"selection_policy={row.get('selection_policies', {})}"
            )
            if row.get("branch_tv_by_visit"):
                print(f"     branch_TV_by_visit={row['branch_tv_by_visit']}")
                for bucket in ["1", "2", "3", "4+"]:
                    real_bucket = (row.get("real_branch_by_visit", {}) or {}).get(bucket, {}) or {}
                    sim_bucket = (row.get("sim_branch_by_visit", {}) or {}).get(bucket, {}) or {}
                    if real_bucket or sim_bucket:
                        print(
                            f"     visit={bucket} "
                            f"real(n={real_bucket.get('n', 0)})={real_bucket.get('freq', {})} "
                            f"sim(n={sim_bucket.get('n', 0)})={sim_bucket.get('freq', {})}"
                        )

        unresolved_rows = [row for row in comparisons if row.get("unresolved_structural_branches")]
        if unresolved_rows:
            print("[CF-AUDIT] Unresolved structural decision points:")
            models = params.get("decision_points", {}) or {}
            for row in unresolved_rows:
                model = models.get(row["decision_point_signature"], {}) or {}
                print(
                    f"  place={row.get('source_place')} enabled={model.get('enabled')} "
                    f"unresolved={row.get('unresolved_structural_branches')} "
                    f"label_sets={model.get('structural_branch_label_sets', {})}"
                )

    def _choose_transition(self, case: dict, enabled_transitions: list, current_time=None):
        decision_point_params = getattr(
            self.simulation_parameters,
            "decision_point_params",
            {},
        )

        chosen_transition = None
        selection_policy = "uniform_fallback"
        selection_error = None
        features = self._build_control_flow_runtime_features(
            case=case,
            current_time=current_time,
        )
        # Loop-aware observable gateway context.
        features.setdefault("gateway_visit_count", float(case.get("control_flow_visit_count", 0)))
        if features.get("gateway_visit_count", 0) <= 1:
            features.setdefault("loop_state", "first_visit")
        elif features.get("gateway_visit_count", 0) <= 3:
            features.setdefault("loop_state", "revisit")
        else:
            features.setdefault("loop_state", "deep_loop")
        features.setdefault("previous_activity", case.get("last_activity", ""))

        # V4.1: runtime must expose the same numeric loop-state features used
        # during decision-tree training. The training dataframe expands
        # loop_state into one-hot columns, but the simulator previously only
        # passed the raw string (which was converted to 0.0).
        loop_state = str(features.get("loop_state", "first_visit"))
        for state in ("first_visit", "revisit", "deep_loop"):
            features.setdefault(f"loop_state::{state}", 1.0 if loop_state == state else 0.0)
        try:
            features["gateway_visit_count"] = float(features.get("gateway_visit_count", 0.0))
            features["is_revisit"] = 1.0 if features["gateway_visit_count"] > 1 else 0.0
        except Exception:
            features["gateway_visit_count"] = 0.0
            features["is_revisit"] = 0.0

        if isinstance(decision_point_params, dict):

            runtime_decision_params = decision_point_params
            model_type = str(decision_point_params.get("model_type", ""))

            try:
                if model_type == "petri_structural_conflict_decision_point_branch_tree":
                    # Preferred runtime: the tree predicts visible activity
                    # branches for one Petri XOR/conflict point. The sampler maps
                    # those activity probabilities back to the currently enabled
                    # Petri transitions (often tau) before firing. Unrelated
                    # concurrency keeps neutral weight.
                    chosen_transition = sample_petri_structural_branch(
                        decision_point_params=runtime_decision_params,
                        enabled_transitions=enabled_transitions,
                        features=features,
                    )
                    selection_policy = "observable_branch_sampler"
                else:
                    # Backward compatibility for parameters saved by older
                    # versions of drq.
                    if (
                            model_type == "process_tree_guided_decision_point_branch_tree"
                            and isinstance(decision_point_params.get("petri_fallback_decision_points"), dict)
                    ):
                        runtime_decision_params = decision_point_params.get("petri_fallback_decision_points")

                    chosen_transition = sample_decision_point_branch(
                        decision_point_params=runtime_decision_params,
                        enabled_transitions=enabled_transitions,
                        features=features,
                    )
                    selection_policy = "legacy_decision_sampler"
            except Exception as exc:
                selection_error = f"{type(exc).__name__}: {exc}"
                chosen_transition = None

        if chosen_transition is None:
            chosen_transition = random.choice(enabled_transitions)
            selection_policy = "uniform_fallback"

        self._record_control_flow_decision_debug(
            case=case,
            enabled_transitions=enabled_transitions,
            chosen_transition=chosen_transition,
            features=features,
            current_time=current_time,
            selection_policy=selection_policy,
            selection_error=selection_error,
        )
        return chosen_transition

    def _earliest_resource_start(
            self,
            activity: str,
            t_enabled,
            candidate_resources: list[str],
            workloads: dict,
            resource_schedule: dict,
            case: dict
    ) -> dict:
        """Fallback scheduler using the same Queue/Calendar decomposition as runtime."""
        candidates = []

        for resource in candidate_resources:
            resource = str(resource)
            try:
                self.resource_state_runtime.recover_by_idle_time(resource, t_enabled)
            except Exception:
                pass

            workload = int(workloads.get(resource, 0))
            is_first_activity = (sum(case.get("history", {}).values()) == 0)
            base_multitasking = resource in self.simulation_parameters.multitasking_resources
            try:
                capacity = int(self.resource_state_runtime.get_capacity(
                    resource,
                    base_multitasking=base_multitasking,
                    current_workload=workload,
                    is_first_activity=is_first_activity,
                ))
            except Exception:
                capacity = 2 if base_multitasking else 1

            try:
                capacity = self.queue_scheduling_runtime._capacity_from_profile(
                    resource=resource,
                    fallback_capacity=capacity,
                )
            except Exception:
                capacity = max(1, int(capacity))

            feasible = self.queue_scheduling_runtime.find_feasible_start_components(
                schedule=resource_schedule.get(resource, []),
                enabled_time=t_enabled,
                capacity=max(1, int(capacity)),
                calendar=self.simulation_parameters.calendars.get(resource, {}),
            )
            candidates.append((
                feasible["start_time"],
                resource,
                workload,
                feasible,
            ))

        if not candidates:
            resource = str(self.simulation_parameters.resources[0])
            return {
                "resource": resource,
                "start_time": t_enabled,
                "resource_ready_time": t_enabled,
                "resource_queue_wait_min": 0.0,
                "calendar_wait_min": 0.0,
                "waiting_time_min": 0.0,
                "workload": int(workloads.get(resource, 0)),
            }

        start_time, resource, workload, feasible = min(candidates, key=lambda x: (x[0], x[1]))
        queue_wait = float(feasible.get("resource_queue_wait_min", 0.0))
        calendar_wait = float(feasible.get("calendar_wait_min", 0.0))
        return {
            "resource": resource,
            "start_time": start_time,
            "resource_ready_time": feasible.get("resource_ready_time", t_enabled),
            "resource_queue_wait_min": queue_wait,
            "calendar_wait_min": calendar_wait,
            "waiting_time_min": float(queue_wait + calendar_wait),
            "workload": int(workload),
        }

    def _apply_with_ready_queue(
            self,
            n_traces: int = 1,
            t_start: datetime = datetime.now(),
            deterministic_time: bool = False,
    ) -> pd.DataFrame:
        """Run the always-on explicit ready-queue DES.

        This path is intentionally separate from the legacy reservation scheduler.
        A visible task first reaches its queue-entry time.  The existing
        resource-allocation model then assigns it to a resource queue using the
        same enabled-time context as discovery.  When that resource has free
        capacity, simultaneously waiting tasks are ranked by the Conditional
        Prioritization Tree (CPT) when learned from the log; otherwise dispatch uses FCFS.
        """
        # Reset per-run control-flow observations.
        self.control_flow_debug_rows = []
        self.control_flow_diagnostics = {}

        event_log = []

        # Queue-state features are maintained incrementally for this run.
        # Reset here because one SimulatorEngine instance may be reused across
        # repeated evaluation runs.
        self.queue_state_runtime.reset(start_time=t_start)

        # CPT DES uses only already-started executions; completed historical
        # intervals never affect future dispatch decisions.  Keep the public
        # ``resource_schedule`` shape for the existing allocation runtime, but
        # compact it to ACTIVE intervals only instead of accumulating the full
        # simulation history.  A per-resource min-heap of active end times makes
        # workload checks O(log c), where c is the current resource concurrency.
        resource_schedule = {
            str(resource): [] for resource in self.simulation_parameters.resources
        }
        resource_active_end_heaps: dict[str, list] = {
            str(resource): [] for resource in self.simulation_parameters.resources
        }

        # ``ready_jobs`` remains the source of truth for compatibility and final
        # diagnostics. ``ready_jobs_by_resource`` is a secondary index only; it
        # does NOT change CPT eligibility/ranking semantics.  Jobs are inserted
        # into this index after the existing resource-allocation step assigns
        # them to exactly one resource queue.
        ready_jobs: dict[Any, dict] = {}
        ready_jobs_by_resource: dict[str, dict[Any, dict]] = {
            str(resource): {} for resource in self.simulation_parameters.resources
        }
        event_heap = []
        event_seq = 0
        job_seq = 0
        # Dispatch is resource-specific: at most one pending dispatch event exists
        # for the same (timestamp, resource) pair.  This removes the previous
        # global dispatch wake-up that rescanned every resource queue whenever any
        # resource changed state.
        scheduled_dispatch_events: set[tuple[int, str]] = set()
        resource_dispatch_stats = {
            "scheduled": 0,
            "deduplicated": 0,
            "processed": 0,
            "future_wakeups": 0,
        }

        def _ts(value, fallback=None):
            if fallback is None:
                fallback = t_start
            return self._safe_timestamp(value, fallback=fallback)

        def _event_time_key(value) -> int:
            try:
                return int(pd.Timestamp(value).value)
            except Exception:
                return 0

        def push_event(when, priority: int, kind: str, payload=None):
            nonlocal event_seq
            when = _ts(when)
            if kind == "dispatch":
                resource = str(payload or "")
                if not resource:
                    # The ready-queue DES no longer uses global dispatch events.
                    return
                key = (_event_time_key(when), resource)
                if key in scheduled_dispatch_events:
                    resource_dispatch_stats["deduplicated"] += 1
                    return
                scheduled_dispatch_events.add(key)
                resource_dispatch_stats["scheduled"] += 1
                payload = resource
            event_seq += 1
            heapq.heappush(event_heap, (when, int(priority), event_seq, str(kind), payload))

        def enqueue_job(job: dict, current_time=None):
            job_id = job["job_id"]
            ready_jobs[job_id] = job
            resource = str(job.get("assigned_resource") or "")
            if resource:
                ready_jobs_by_resource.setdefault(resource, {})[job_id] = job
            self.queue_state_runtime.enqueue_ready_job(
                job,
                current_time=current_time,
            )

        def dequeue_job(job: dict):
            job_id = job.get("job_id")
            ready_jobs.pop(job_id, None)
            resource = str(job.get("assigned_resource") or "")
            if resource:
                queue = ready_jobs_by_resource.get(resource)
                if queue is not None:
                    queue.pop(job_id, None)
            self.queue_state_runtime.remove_ready_job(job_id)

        def _purge_finished_resource_intervals(resource: str, when) -> None:
            """Drop executions completed by ``when`` from CPT runtime state.

            This is semantics-preserving for the CPT DES path: it never creates
            future reservations, so an interval with end <= current simulation
            time can no longer affect workload, capacity, or feasible-start
            calculations.
            """
            resource = str(resource)
            when = _ts(when)
            heap = resource_active_end_heaps.setdefault(resource, [])
            while heap and heap[0] <= when:
                heapq.heappop(heap)

            # Keep the schedule interface expected by QueueSchedulingRuntime,
            # but retain active intervals only.  This list is now bounded by
            # current concurrency instead of total historical executions.
            schedule = resource_schedule.setdefault(resource, [])
            if schedule:
                resource_schedule[resource] = [
                    (start, end) for start, end in schedule if _ts(end, fallback=when) > when
                ]

        def _register_resource_execution(resource: str, start_time, end_time) -> None:
            resource = str(resource)
            start_time = _ts(start_time)
            end_time = _ts(end_time, fallback=start_time)
            _purge_finished_resource_intervals(resource, start_time)
            resource_schedule.setdefault(resource, []).append((start_time, end_time))
            heapq.heappush(resource_active_end_heaps.setdefault(resource, []), end_time)

        def current_workload(resource: str, when) -> int:
            resource = str(resource)
            _purge_finished_resource_intervals(resource, when)
            return int(len(resource_active_end_heaps.setdefault(resource, [])))

        def resource_capacity(resource: str, when, case: dict | None = None) -> int:
            resource = str(resource)
            workload = current_workload(resource, when)
            base_multitasking = resource in self.simulation_parameters.multitasking_resources
            is_first_activity = bool(case is not None and sum(case.get("history", {}).values()) == 0)
            try:
                self.resource_state_runtime.recover_by_idle_time(resource, when)
            except Exception:
                pass
            try:
                cap = int(self.resource_state_runtime.get_capacity(
                    resource,
                    base_multitasking=base_multitasking,
                    current_workload=workload,
                    is_first_activity=is_first_activity,
                ))
            except Exception:
                cap = 2 if base_multitasking else 1
            try:
                cap = self.queue_scheduling_runtime._capacity_from_profile(
                    resource=resource,
                    fallback_capacity=max(1, cap),
                )
            except Exception:
                cap = max(1, cap)
            return max(1, int(cap))

        def _case_first_activity_flag(case: dict | None) -> bool:
            return bool(case is not None and sum(case.get("history", {}).values()) == 0)

        def cached_resource_capacity(
                resource: str,
                when,
                case: dict | None = None,
                capacity_cache: dict | None = None,
        ) -> int:
            """Cache capacity within one immutable dispatch snapshot only.

            Capacity can differ only by resource and first-activity protection at
            a fixed timestamp/workload.  The cache is intentionally created anew
            after every started job, so resource-state and workload semantics are
            unchanged.
            """
            key = (str(resource), _case_first_activity_flag(case))
            if capacity_cache is not None and key in capacity_cache:
                return int(capacity_cache[key])
            cap = resource_capacity(resource, when, case=case)
            if capacity_cache is not None:
                capacity_cache[key] = int(cap)
            return int(cap)

        def feasible_components(
                resource: str,
                when,
                case: dict | None = None,
                capacity_cache: dict | None = None,
                feasible_cache: dict | None = None,
        ) -> dict:
            resource = str(resource)
            _purge_finished_resource_intervals(resource, when)
            capacity = cached_resource_capacity(
                resource, when, case=case, capacity_cache=capacity_cache
            )
            # At a fixed dispatch snapshot, schedule/calendar/time are identical
            # for every waiting job.  Capacity is the only case-dependent input.
            # Reuse the deterministic feasibility result for equal capacities.
            if feasible_cache is not None and capacity in feasible_cache:
                return feasible_cache[capacity]
            result = self.queue_scheduling_runtime.find_feasible_start_components(
                schedule=resource_schedule.setdefault(resource, []),
                enabled_time=when,
                capacity=capacity,
                calendar=self.simulation_parameters.calendars.get(resource, {}),
            )
            if feasible_cache is not None:
                feasible_cache[capacity] = result
            return result

        def is_resource_available_now(
                resource: str,
                when,
                case: dict | None = None,
                capacity_cache: dict | None = None,
                feasible_cache: dict | None = None,
        ) -> bool:
            try:
                feasible = feasible_components(
                    resource, when, case=case,
                    capacity_cache=capacity_cache,
                    feasible_cache=feasible_cache,
                )
                start = _ts(feasible.get("start_time", when), fallback=when)
                return start <= _ts(when)
            except Exception:
                return False

        def next_dispatch_time_for_resource(resource: str, now):
            """Return the next time one specific resource queue may make progress.

            The old implementation searched every resource queue after each
            dispatch.  Jobs are already assigned to exactly one resource, so the
            wake-up calculation can be local to that resource without changing
            queue eligibility or CPT/FCFS ranking semantics.
            """
            resource = str(resource)
            queue = ready_jobs_by_resource.get(resource) or {}
            if not queue:
                return None

            candidates = []
            capacity_cache = {}
            feasible_cache = {}
            for job in list(queue.values()):
                case_id = int(job["case_id"])
                if case_id in completed_cases:
                    continue

                queue_entry_time = _ts(job.get("queue_entry_time"), fallback=now)
                if queue_entry_time > now:
                    candidates.append(queue_entry_time)
                    continue

                case = cases[case_id]
                try:
                    start = _ts(
                        feasible_components(
                            resource, now, case=case,
                            capacity_cache=capacity_cache,
                            feasible_cache=feasible_cache,
                        ).get("start_time", now),
                        fallback=now,
                    )
                    if start > now:
                        candidates.append(start)
                except Exception:
                    continue
            return min(candidates) if candidates else None

        def enable_next_transitions(case_id: int, transition, end_time, activity, pbar, completed_cases,
                                    max_same_marking_visits: int):
            case = cases[case_id]
            for arc in transition.out_arcs:
                case["place_token_time"][arc.target] = end_time

            case["enabled"] = {}
            case["marking"] = update_current_marking(case["marking"], transition)

            marking_sig = self._marking_signature(case["marking"])
            marking_counts = case.setdefault("marking_visit_counts", {})
            marking_counts[marking_sig] = int(marking_counts.get(marking_sig, 0)) + 1
            if marking_counts[marking_sig] > max_same_marking_visits:
                self._abort_case(
                    case_id=case_id,
                    case=case,
                    reason="max_same_marking_visits_exceeded",
                    completed_cases=completed_cases,
                    pbar=pbar,
                    current_time=end_time,
                    activity=str(activity),
                )
                return

            if case["marking"] == self.final_marking:
                if case_id not in completed_cases:
                    completed_cases.add(case_id)
                    pbar.update(1)
                return

            enabled = return_enabled_transitions(self.net, case["marking"])
            for next_transition in enabled:
                input_places = self._transition_input_places_cache.get(next_transition, [])
                if not input_places:
                    enabled_time = end_time
                else:
                    enabled_time = max(case["place_token_time"][place] for place in input_places)
                case["enabled"][next_transition] = enabled_time

            # Incremental Queue State: replace this case's enabled
            # representative once, instead of rediscovering it by scanning all
            # cases on every resource-candidate feature query.
            self.queue_state_runtime.register_enabled_case(
                case_id=case_id,
                case=case,
                current_time=end_time,
            )

            if case["enabled"]:
                push_event(min(case["enabled"].values()), 1, "case_ready", case_id)
            else:
                self._abort_case(
                    case_id=case_id,
                    case=case,
                    reason="deadlock_no_enabled_transition",
                    completed_cases=completed_cases,
                    pbar=pbar,
                    current_time=end_time,
                )

        x_attr_list = self._sample_data_attributes(n_traces)
        current_arr_ts = t_start
        self._arrival_timestamp_upper_bound = self._compute_arrival_timestamp_upper_bound(
            current_arr_ts, n_traces=n_traces
        )
        self._service_time_runtime_cap_min = self._compute_service_time_cap_min()
        completion_guard = self._compute_case_completion_guard_config(n_traces)
        max_events_per_case = int(completion_guard["max_events_per_case"])
        max_same_marking_visits = int(completion_guard["max_same_marking_visits"])
        max_total_simulation_steps = int(completion_guard["max_total_simulation_steps"])
        total_simulation_steps = 0

        arrival_shift = float(getattr(self.simulation_parameters, "arrival_time_shift_min", 0.0))
        if arrival_shift != 0.0:
            current_arr_ts = add_minutes_with_calendar(
                current_arr_ts,
                int(round(arrival_shift)),
                self.simulation_parameters.arrival_calendar,
            )

        cases = []
        for i in range(n_traces):
            trace_attributes = {}
            if x_attr_list[i]:
                for j, attr in enumerate(self.simulation_parameters.label_data_attributes):
                    if attr in self.simulation_parameters.label_data_attributes_categorical:
                        for value in self.simulation_parameters.attribute_values_label_categorical.get(attr, []):
                            trace_attributes[f"{attr} = {value}"] = int(x_attr_list[i][j] == value)
                    else:
                        trace_attributes[attr] = x_attr_list[i][j]

            if i == 0:
                case_arrival_time = current_arr_ts
            else:
                try:
                    arrival_delta = self._sample_arrival_delta(
                        current_time=current_arr_ts,
                        deterministic_time=deterministic_time,
                    )
                except Exception:
                    arrival_delta = 0.0
                arrival_delta = max(0.0, float(arrival_delta))
                arrival_delta *= float(getattr(self.simulation_parameters, "arrival_time_scale", 1.0))
                demand_multiplier = float(getattr(self.simulation_parameters, "arrival_demand_multiplier", 1.0))
                if demand_multiplier > 0:
                    arrival_delta /= demand_multiplier
                arrival_delta = self._cap_arrival_delta_after_runtime_controls(arrival_delta)
                current_arr_ts = self._advance_arrival_timestamp(current_arr_ts, arrival_delta)
                case_arrival_time = current_arr_ts

            case = {
                "case_id": i,
                "marking": self.initial_marking,
                "arrival_time": case_arrival_time,
                "place_token_time": {place: None for place in self.net.places},
                "enabled": {},
                "history": {activity: 0 for activity in self.simulation_parameters.net_transition_labels},
                "res_history": {resource: 0 for resource in self.simulation_parameters.resources},
                "attributes": trace_attributes,
                "event_count": 0,
                "last_activity": "__START__",
                "last_2_activities": "__START__||__START__",
                "marking_visit_counts": {},
                # Diagnostic-only per structural decision-point visit counter.
                "decision_visit_counts": {},
                "aborted": False,
                "pending_job_id": None,
            }
            case["marking_visit_counts"][self._marking_signature(case["marking"])] = 1
            initial_places = list(self.initial_marking.keys())
            if initial_places:
                case["place_token_time"][initial_places[0]] = case_arrival_time

            enabled = return_enabled_transitions(self.net, case["marking"])
            for transition in enabled:
                input_places = self._transition_input_places_cache.get(transition, [])
                enabled_time = max(case["place_token_time"][place] for place in input_places) if input_places else case_arrival_time
                case["enabled"][transition] = enabled_time
            cases.append(case)
            self.queue_state_runtime.register_enabled_case(
                case_id=i,
                case=case,
            )
            if case["enabled"]:
                push_event(min(case["enabled"].values()), 1, "case_ready", i)
            else:
                case["abort_reason"] = "deadlock_no_enabled_transition"

        completed_cases = set()
        pbar = tqdm(total=n_traces, desc="Simulating Cases (Ready Queue)")
        for case in cases:
            if not case.get("enabled"):
                self._abort_case(
                    case_id=int(case["case_id"]),
                    case=case,
                    reason=case.get("abort_reason", "no_initial_enabled_transition"),
                    completed_cases=completed_cases,
                    pbar=pbar,
                    current_time=case.get("arrival_time"),
                )

        while event_heap:
            current_time, _priority, _seq, kind, payload = heapq.heappop(event_heap)
            self.queue_state_runtime.advance_to(current_time)
            if kind == "dispatch":
                dispatch_resource = str(payload or "")
                scheduled_dispatch_events.discard(
                    (_event_time_key(current_time), dispatch_resource)
                )
                resource_dispatch_stats["processed"] += 1

            if total_simulation_steps >= max_total_simulation_steps:
                for case in cases:
                    cid = int(case["case_id"])
                    if cid not in completed_cases:
                        self._abort_case(
                            case_id=cid,
                            case=case,
                            reason="max_total_simulation_steps_exceeded",
                            completed_cases=completed_cases,
                            pbar=pbar,
                            current_time=current_time,
                        )
                break

            if kind == "case_ready":
                case_id = int(payload)
                if case_id in completed_cases:
                    continue
                case = cases[case_id]
                if case.get("pending_job_id") is not None or not case.get("enabled"):
                    continue
                current_sim_time = min(case["enabled"].values())
                if current_sim_time > current_time:
                    push_event(current_sim_time, 1, "case_ready", case_id)
                    continue

                # The case is now leaving the pre-allocation enabled state.
                # Removing it here preserves the previous rule that the current
                # case does not count itself in resource-allocation queue
                # features, while all other due cases remain incrementally
                # represented until their own case_ready event is processed.
                self.queue_state_runtime.remove_enabled_case(case_id)

                if int(case.get("event_count", 0)) >= max_events_per_case:
                    self._abort_case(
                        case_id=case_id,
                        case=case,
                        reason="max_events_per_case_exceeded",
                        completed_cases=completed_cases,
                        pbar=pbar,
                        current_time=current_time,
                    )
                    continue

                enabled_transitions = list(case["enabled"].keys())
                chosen_transition = self._choose_transition(case, enabled_transitions, current_time=current_time)
                activity = chosen_transition.label
                t_enabled = _ts(case["enabled"][chosen_transition], fallback=current_time)
                case["event_count"] = int(case.get("event_count", 0)) + 1
                total_simulation_steps += 1

                if activity is None:
                    enable_next_transitions(
                        case_id, chosen_transition, t_enabled, activity,
                        pbar, completed_cases, max_same_marking_visits,
                    )
                    continue

                enabled_resources_act = [
                    str(resource)
                    for resource in self.simulation_parameters.act_to_resources.get(
                        activity, self.simulation_parameters.resources
                    )
                ] or [str(self.simulation_parameters.resources[0])]

                external_delay_min = 0.0
                delayed_enabled_time = t_enabled
                pre_resource_delay_context = {"external_delay_phase": "pre_resource_timer", "resource": None}
                try:
                    external_delay_min = self.external_delay_runtime.sample_delay(
                        activity=activity,
                        resource=None,
                        enabled_time=t_enabled,
                        queue_start_time=t_enabled,
                        case=case,
                        scheduling_result=pre_resource_delay_context,
                        deterministic_time=deterministic_time,
                    )
                    delayed_enabled_time = self.external_delay_runtime.add_delay_to_start_time(
                        t_enabled, external_delay_min
                    )
                except Exception:
                    external_delay_min = 0.0
                    delayed_enabled_time = t_enabled
                delayed_enabled_time = _ts(delayed_enabled_time, fallback=t_enabled)

                job_seq += 1
                job = {
                    "job_id": int(job_seq),
                    "case_id": case_id,
                    "transition": chosen_transition,
                    "activity": str(activity),
                    "enabled_time": t_enabled,
                    "queue_entry_time": delayed_enabled_time,
                    "candidate_resources": enabled_resources_act,
                    "external_delay_min": float(external_delay_min),
                    "history": dict(case.get("history", {})),
                    "res_history": dict(case.get("res_history", {})),
                    "attributes": dict(case.get("attributes", {})),
                    "entry_resource_components": {},
                    "external_delay_activity_timer": pre_resource_delay_context.get("external_delay_activity_timer"),
                }
                case["enabled"] = {}
                case["pending_job_id"] = job["job_id"]
                push_event(delayed_enabled_time, 1, "queue_entry", job)

            elif kind == "queue_entry":
                job = payload
                case_id = int(job["case_id"])
                if case_id in completed_cases or cases[case_id].get("pending_job_id") != job.get("job_id"):
                    continue
                case = cases[case_id]

                # Resource allocation and queue prioritization are intentionally
                # separated.  The existing white-box allocation model assigns the
                # task to one resource queue at queue-entry time; CPT later decides
                # which task leaves that queue when the resource becomes available.
                original_candidates = [str(r) for r in job.get("candidate_resources", [])]
                workloads = {r: current_workload(r, current_time) for r in original_candidates}
                try:
                    allocation = self.queue_scheduling_runtime.select_resource_for_task(
                        activity=job["activity"],
                        enabled_time=current_time,
                        case=case,
                        cases=cases,
                        completed_cases=completed_cases,
                        current_case_id=case_id,
                        candidate_resources=original_candidates,
                        workloads=workloads,
                        resource_schedule=resource_schedule,
                        calendars=self.simulation_parameters.calendars,
                        multitasking_resources=self.simulation_parameters.multitasking_resources,
                        resource_state_runtime=self.resource_state_runtime,
                        queue_state_runtime=self.queue_state_runtime,
                        deterministic=deterministic_time,
                    )
                except Exception:
                    allocation = {}

                assigned_resource = allocation.get("resource")
                if assigned_resource is None or str(assigned_resource) not in original_candidates:
                    if original_candidates:
                        try:
                            fallback = self._earliest_resource_start(
                                activity=job["activity"],
                                t_enabled=current_time,
                                candidate_resources=original_candidates,
                                workloads=workloads,
                                resource_schedule=resource_schedule,
                                case=case,
                            )
                            assigned_resource = fallback.get("resource")
                            allocation = dict(allocation or {})
                            allocation.update(fallback)
                        except Exception:
                            assigned_resource = sorted(original_candidates)[0]
                    else:
                        assigned_resource = str(self.simulation_parameters.resources[0])

                assigned_resource = str(assigned_resource)
                job["original_candidate_resources"] = original_candidates
                job["assigned_resource"] = assigned_resource
                job["resource_allocation_result"] = dict(allocation or {})
                # A real resource queue contains tasks assigned to that resource.
                job["candidate_resources"] = [assigned_resource]

                try:
                    entry_components = feasible_components(assigned_resource, current_time, case=case)
                except Exception:
                    entry_components = {}
                # Prefer the allocation model's own decomposition when available.
                if allocation:
                    entry_components = dict(entry_components or {})
                    for key in ("start_time", "resource_ready_time", "resource_queue_wait_min", "calendar_wait_min"):
                        if key in allocation:
                            entry_components[key] = allocation.get(key)
                job["entry_resource_components"][assigned_resource] = entry_components

                enqueue_job(job, current_time=current_time)
                push_event(current_time, 2, "dispatch", assigned_resource)

            elif kind == "completion":
                info = payload or {}
                job = info.get("job", {})
                case_id = int(job.get("case_id"))
                if case_id in completed_cases:
                    continue
                case = cases[case_id]
                resource = str(info.get("resource"))
                activity = str(job.get("activity"))
                start_time = info.get("start_time", current_time)
                end_time = info.get("end_time", current_time)
                workload = int(info.get("workload", 0))

                try:
                    self.resource_state_runtime.update_after_execution(
                        resource=resource,
                        activity=activity,
                        start_time=start_time,
                        end_time=end_time,
                        workload=workload,
                    )
                except Exception:
                    pass

                prev_last_activity = str(case.get("last_activity", "__START__"))
                case["history"][activity] = case["history"].get(activity, 0) + 1
                case["res_history"][resource] = case["res_history"].get(resource, 0) + 1
                case["last_activity"] = activity
                case["last_2_activities"] = f"{prev_last_activity}||{activity}"
                case["pending_job_id"] = None

                enable_next_transitions(
                    case_id, job["transition"], end_time, activity,
                    pbar, completed_cases, max_same_marking_visits,
                )
                push_event(current_time, 2, "dispatch", resource)

            elif kind == "dispatch":
                resource = str(payload or "")
                if not resource:
                    continue

                queue = ready_jobs_by_resource.get(resource)
                if not queue:
                    continue

                # Process only the affected resource queue.  Multi-capacity
                # resources may start more than one job at the same timestamp,
                # so keep dispatching this resource until capacity/calendar
                # constraints prevent further progress.
                while queue:
                    # One dispatch snapshot has an immutable resource schedule.
                    # Cache capacity/feasibility only for this iteration; starting
                    # a job mutates the schedule and the next iteration rebuilds
                    # both caches.  This removes O(queue_length) duplicate calendar
                    # and capacity calculations without changing eligibility.
                    capacity_cache = {}
                    feasible_cache = {}
                    eligible_jobs = []
                    for job in list(queue.values()):
                        if _ts(job.get("queue_entry_time"), fallback=current_time) > current_time:
                            continue
                        case_id = int(job["case_id"])
                        if case_id in completed_cases:
                            dequeue_job(job)
                            continue
                        case = cases[case_id]
                        if not is_resource_available_now(
                            resource, current_time, case=case,
                            capacity_cache=capacity_cache,
                            feasible_cache=feasible_cache,
                        ):
                            continue
                        eligible_jobs.append(job)

                    if not eligible_jobs:
                        break

                    workload_before = current_workload(resource, current_time)
                    first_case = cases[int(eligible_jobs[0]["case_id"])]
                    if workload_before >= cached_resource_capacity(
                        resource,
                        current_time,
                        case=first_case,
                        capacity_cache=capacity_cache,
                    ):
                        break

                    selection = self.conditional_prioritization_runtime.select_next_job(
                        resource=resource,
                        ready_jobs=eligible_jobs,
                        current_time=current_time,
                        resource_workload=workload_before,
                        deterministic=deterministic_time,
                    )
                    job = selection.get("job")
                    if not isinstance(job, dict):
                        break

                    case_id = int(job["case_id"])
                    case = cases[case_id]
                    if case_id in completed_cases:
                        dequeue_job(job)
                        continue

                    dequeue_job(job)
                    t_start_exec = _ts(current_time, fallback=job["queue_entry_time"])

                    entry_comp = (job.get("entry_resource_components") or {}).get(resource, {}) or {}
                    elapsed_queue_window = max(
                        0.0,
                        (t_start_exec - _ts(job["queue_entry_time"], fallback=t_start_exec)).total_seconds() / 60.0,
                    )
                    contention_wait_min = max(
                        0.0,
                        float(entry_comp.get("resource_queue_wait_min", 0.0) or 0.0),
                    )
                    calendar_wait_min = max(
                        0.0,
                        float(entry_comp.get("calendar_wait_min", 0.0) or 0.0),
                    )
                    contention_wait_min = min(contention_wait_min, elapsed_queue_window)
                    calendar_wait_min = min(
                        calendar_wait_min,
                        max(0.0, elapsed_queue_window - contention_wait_min),
                    )
                    priority_wait_min = max(
                        0.0,
                        elapsed_queue_window - contention_wait_min - calendar_wait_min,
                    )
                    queue_wait_min = float(contention_wait_min + priority_wait_min)
                    total_wait_min = max(
                        0.0,
                        (t_start_exec - _ts(job["enabled_time"], fallback=t_start_exec)).total_seconds() / 60.0,
                    )
                    external_delay_min = max(
                        0.0,
                        float(job.get("external_delay_min", 0.0) or 0.0),
                    )
                    residual_min = float(
                        total_wait_min - queue_wait_min - calendar_wait_min - external_delay_min
                    )

                    try:
                        base_ex_time = sample_base_service_time(
                            base_service_time_params=self.simulation_parameters.base_service_time_distributions,
                            activity=job["activity"],
                            resource=resource,
                            deterministic_time=deterministic_time,
                        )
                    except Exception:
                        base_ex_time = 1.0
                    base_ex_time = self._guard_service_time(
                        base_ex_time,
                        activity=job["activity"],
                        resource=resource,
                        case_id=case_id,
                        section="base_service_time",
                    )
                    try:
                        adjusted_ex_time = self.resource_state_runtime.adjust_execution_time(
                            resource=resource,
                            activity=job["activity"],
                            base_execution_time=base_ex_time,
                            workload=workload_before,
                        )
                        ex_time = self.resource_state_runtime.guard_temporal_adjustment(
                            base_value=base_ex_time,
                            adjusted_value=adjusted_ex_time,
                            section_name="execution_time",
                            is_first_activity=(sum(case["history"].values()) == 0),
                        )
                    except Exception:
                        adjusted_ex_time = base_ex_time
                        ex_time = base_ex_time
                    adjusted_ex_time = self._guard_service_time(
                        adjusted_ex_time,
                        job["activity"],
                        resource,
                        case_id,
                        "adjusted_service_time",
                    )
                    ex_time = self._guard_service_time(
                        ex_time,
                        job["activity"],
                        resource,
                        case_id,
                        "execution_time",
                    )
                    try:
                        resource_calendar = self.simulation_parameters.calendars.get(
                            resource,
                            {wd: {h: True for h in range(24)} for wd in range(7)},
                        )

                        # Use the exact same sparse-calendar gate as
                        # QueueSchedulingRuntime._respect_calendar_start().
                        # If a calendar is too sparse to be trusted as a hard
                        # start-time constraint, it must not stretch service
                        # completion across days/weeks either.
                        if self.queue_scheduling_runtime.is_sparse_resource_calendar(
                                resource_calendar
                        ):
                            t_end = _ts(
                                t_start_exec,
                                fallback=job["enabled_time"],
                            ) + pd.Timedelta(minutes=max(1.0, float(ex_time)))
                        else:
                            t_end = add_minutes_with_calendar(
                                t_start_exec,
                                int(round(ex_time)),
                                resource_calendar,
                            )
                            t_end = _ts(t_end, fallback=t_start_exec)
                    except Exception:
                        t_end = _ts(
                            t_start_exec,
                            fallback=job["enabled_time"],
                        ) + pd.Timedelta(minutes=max(1.0, float(ex_time)))

                    _register_resource_execution(resource, t_start_exec, t_end)
                    priority_score = selection.get("selected_score")
                    if priority_score is None:
                        priority_score = np.nan
                    event_log.append((
                        case_id,
                        job["activity"],
                        resource,
                        job["enabled_time"],
                        job["queue_entry_time"],
                        t_start_exec,
                        t_end,
                        float(queue_wait_min),
                        float(calendar_wait_min),
                        float(external_delay_min),
                        float(total_wait_min),
                        float(residual_min),
                        float(base_ex_time),
                        float(adjusted_ex_time),
                        float(ex_time),
                        float(contention_wait_min),
                        float(priority_wait_min),
                        str(selection.get("policy", "fcfs")),
                        float(priority_score) if np.isfinite(priority_score) else np.nan,
                    ) + tuple(x_attr_list[case_id]))

                    push_event(t_end, 0, "completion", {
                        "job": job,
                        "resource": resource,
                        "start_time": t_start_exec,
                        "end_time": t_end,
                        "workload": workload_before,
                    })

                    # dequeue_job() mutates the same per-resource queue dict.
                    queue = ready_jobs_by_resource.get(resource) or {}

                # If this resource still has waiting jobs, schedule only its own
                # next wake-up.  This covers both capacity release and calendar
                # reopening without rescanning unrelated resource queues.
                queue = ready_jobs_by_resource.get(resource) or {}
                if queue:
                    next_time = next_dispatch_time_for_resource(resource, current_time)
                    if next_time is None or next_time <= current_time:
                        # Defensive fallback for malformed calendar/capacity
                        # profiles.  Normal capacity releases are also awakened by
                        # the corresponding completion event and deduplicated by
                        # (timestamp, resource).
                        next_time = current_time + pd.Timedelta(minutes=1.0)
                    resource_dispatch_stats["future_wakeups"] += 1
                    push_event(next_time, 2, "dispatch", resource)

        pbar.close()
        self.arrival_diagnostics["completed_or_closed_cases"] = int(len(completed_cases))
        self.arrival_diagnostics["total_simulation_steps"] = int(total_simulation_steps)
        cpt_kind = str(self.conditional_prioritization_runtime.params.get("model_kind", "fcfs_fallback"))
        self.arrival_diagnostics["dispatch_policy"] = (
            "conditional_prioritization_tree" if cpt_kind == "pairwise_decision_tree" else "fcfs"
        )
        self.arrival_diagnostics["remaining_ready_jobs"] = int(len(ready_jobs))
        self.arrival_diagnostics["ready_queue_index_mode"] = "per_resource_secondary_index"
        self.arrival_diagnostics["dispatch_event_mode"] = "resource_specific"
        self.arrival_diagnostics["resource_dispatch_stats"] = dict(resource_dispatch_stats)
        self.arrival_diagnostics["resource_schedule_mode"] = "active_intervals_with_end_heap"
        self.arrival_diagnostics["queue_state_runtime"] = self.queue_state_runtime.diagnostics()
        self._finalize_control_flow_debug(total_cases=int(n_traces))

        df_log = pd.DataFrame(
            event_log,
            columns=[
                "case:concept:name",
                "concept:name",
                "org:resource",
                "enabled:timestamp",
                "queue_start:timestamp",
                "start:timestamp",
                "time:timestamp",
                "queue_wait_min",
                "calendar_wait_min",
                "external_delay_min",
                "total_wait_min",
                "wait_decomposition_residual_min",
                "base_service_time_min",
                "adjusted_service_time_min",
                "service_time_min",
                "resource_contention_wait_min",
                "priority_wait_min",
                "dispatch_policy",
                "conditional_priority_score",
            ] + self.simulation_parameters.label_data_attributes,
        )
        if df_log.empty:
            return df_log
        df_log["case:concept:name"] = df_log["case:concept:name"].apply(lambda x: f"case_{x + 1}")
        df_log.sort_values(by=["start:timestamp", "time:timestamp"], inplace=True)
        df_log.reset_index(drop=True, inplace=True)
        return df_log


    def apply(
            self,
            n_traces: int = 1,
            t_start: datetime = datetime.now(),
            deterministic_time: bool = False
    ) -> pd.DataFrame:
        """Run the complete simulation pipeline.

        The explicit ready-queue engine is always used. Resource allocation,
        dynamic resource state and external delay are always evaluated. The
        dataset-discovered dispatch model decides only the queue policy:
        pairwise CPT when supported by overtaking evidence, otherwise FCFS.
        """
        return self._apply_with_ready_queue(
            n_traces=n_traces,
            t_start=t_start,
            deterministic_time=deterministic_time,
        )
