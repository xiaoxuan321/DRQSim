# drq/runtime/queue_scheduling_runtime.py

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any, Dict, Optional

import numpy as np

from drqsim.discovery.queue_scheduling_discovery import (
    ACTIVITY_PREFIX,
    apply_queue_scheduling_score,
)

try:
    from drqsim.discovery.queue_scheduling_discovery import QUEUE_FEATURE_NAMES
except ImportError:
    QUEUE_FEATURE_NAMES = [
        "global_queue_length",
        "activity_queue_length",
        "resource_queue_length",
        "resource_activity_queue_length",
        "queue_pressure",
        "system_wip",
    ]


class QueueSchedulingRuntime:
    """
    五模块版队列调度运行时。

    当前优化点：
    1. 若混合权重中 tree=0，则不构造完整树特征，也不调用队列状态特征，
       只计算 activity_resource_frequency 和 availability_score。
    2. 避免在每个候选资源内重复构造 multitasking_resources 的字符串列表。
    3. 仅在需要 tree_score 时才构建 activity one-hot、case history、DBN 状态和 queue features。
    """

    DEFAULT_SCORE_WEIGHTS = {
        "tree": 0.30,
        "frequency": 0.35,
        "availability": 0.20,
        "calendar_fit": 0.15,
    }

    RESOURCE_POLICY_MODES = {
        "frequency_oriented": {
            "tree": 0.0,
            "frequency": 0.80,
            "availability": 0.20,
            "calendar_fit": 0.0,
        },
        "calendar_frequency_oriented": {
            "tree": 0.0,
            "frequency": 0.55,
            "availability": 0.15,
            "calendar_fit": 0.30,
        },
        "stable_calendar_frequency": {
            "tree": 0.0,
            "frequency": 0.65,
            "availability": 0.10,
            "calendar_fit": 0.25,
        },
        "balanced": {
            "tree": 0.30,
            "frequency": 0.35,
            "availability": 0.20,
            "calendar_fit": 0.15,
        },
        "queue_oriented": {
            "tree": 0.40,
            "frequency": 0.20,
            "availability": 0.25,
            "calendar_fit": 0.15,
        },
    }

    def __init__(
            self,
            queue_scheduling_params: dict,
            act_to_resources: dict,
            resources: list,
            activities: list,
            random_seed: Optional[int] = None,
    ):
        self.params = queue_scheduling_params or {}
        self.act_to_resources = act_to_resources or {}
        self.resources = [str(r) for r in (resources or [])]
        self.activities = [str(a) for a in (activities or [])]
        self.rng = random.Random(random_seed)
        metadata = self.params.get("metadata", {}) if isinstance(self.params, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        self._freq_table = metadata.get("activity_resource_frequency", {}) or {}
        self._capacity_profile = metadata.get("resource_capacity_profile", {}) or {}
        if not isinstance(self._capacity_profile, dict):
            self._capacity_profile = {}
        self.resource_policy_mode = str(
            self.params.get("resource_policy_mode")
            or self.params.get("selected_strategy")
            or self.params.get("selected_resource_policy")
            or "balanced"
        )

    def select_resource_for_task(
            self,
            activity: str,
            enabled_time,
            case: dict,
            cases: list,
            completed_cases: set,
            current_case_id: int,
            candidate_resources: list,
            workloads: dict,
            resource_schedule: dict,
            calendars: dict,
            multitasking_resources: list,
            resource_state_runtime,
            queue_state_runtime=None,
            deterministic: bool = False,
    ) -> Dict[str, Any]:
        activity = str(activity)
        candidate_resources = [str(r) for r in (candidate_resources or [])]

        if not candidate_resources:
            candidate_resources = [str(r) for r in self.act_to_resources.get(activity, self.resources)]

        if not candidate_resources:
            return {
                "resource": None,
                "start_time": enabled_time,
                "resource_queue_wait_min": 0.0,
                "calendar_wait_min": 0.0,
                "waiting_time_min": 0.0,
                "workload": 0,
                "scores": {},
                "score_components": {},
                "selected_features": {},
                "resource_policy_mode": self._get_policy_mode(),
                "resource_policy_weights": self._get_score_weights(),
            }

        score_weights = self._get_score_weights()
        policy_mode = self._get_policy_mode()
        tree_weight = float(score_weights.get("tree", 0.0))
        need_tree_features = bool(
            tree_weight > 1e-12
            and self.params.get("model_kind") == "decision_tree"
            and self.params.get("rules") is not None
        )

        case_progress = int(sum(case.get("history", {}).values()))
        is_first_activity = case_progress == 0
        first_activity_can_queue = bool(self.params.get("first_activity_can_queue", True))
        multitasking_set = {str(r) for r in (multitasking_resources or [])}

        candidate_records = []

        for resource in candidate_resources:
            resource = str(resource)

            try:
                resource_state_runtime.recover_by_idle_time(resource, enabled_time)
            except Exception:
                pass

            workload = int(workloads.get(resource, 0))
            base_multitasking = resource in multitasking_set

            try:
                capacity = int(
                    resource_state_runtime.get_capacity(
                        resource,
                        base_multitasking=base_multitasking,
                        current_workload=workload,
                        is_first_activity=is_first_activity,
                    )
                )
            except Exception:
                capacity = 2 if base_multitasking else 1

            capacity = self._capacity_from_profile(
                resource=resource,
                fallback_capacity=capacity,
            )

            calendar = calendars.get(resource, {}) if isinstance(calendars, dict) else {}
            if is_first_activity and not first_activity_can_queue:
                feasible = {
                    "start_time": enabled_time,
                    "resource_ready_time": enabled_time,
                    "resource_queue_wait_min": 0.0,
                    "calendar_wait_min": 0.0,
                }
            else:
                feasible = self.find_feasible_start_components(
                    schedule=resource_schedule.get(resource, []),
                    enabled_time=enabled_time,
                    capacity=max(1, capacity),
                    calendar=calendar,
                )

            earliest_start = feasible["start_time"]
            resource_queue_wait_min = float(feasible["resource_queue_wait_min"])
            calendar_wait_min = float(feasible["calendar_wait_min"])
            waiting_time_min = float(resource_queue_wait_min + calendar_wait_min)

            if need_tree_features:
                features = self._build_candidate_features(
                    activity=activity,
                    resource=resource,
                    case=case,
                    cases=cases,
                    completed_cases=completed_cases,
                    current_case_id=current_case_id,
                    current_time=enabled_time,
                    candidate_resources=candidate_resources,
                    workloads=workloads,
                    capacity=capacity,
                    waiting_time_min=resource_queue_wait_min,
                    resource_state_runtime=resource_state_runtime,
                    queue_state_runtime=queue_state_runtime,
                )
                tree_score = apply_queue_scheduling_score(
                    queue_scheduling_params=self.params,
                    features=features,
                )
                frequency_score = self._clip01(features.get("activity_resource_frequency", 0.0))
            else:
                frequency_score = self._activity_resource_frequency(activity, resource)
                tree_score = 0.0
                features = {
                    "resource_policy_mode_encoded": float(self._policy_mode_code(policy_mode)),
                    "activity_resource_frequency": float(frequency_score),
                    "candidate_workload": float(workload),
                    "candidate_capacity": float(max(1, int(capacity))),
                    "candidate_available_now": float(workload < max(1, int(capacity))),
                    "candidate_resource_wait_min": float(resource_queue_wait_min),
                    "candidate_waiting_time_min": float(resource_queue_wait_min),
                    "candidate_waiting_time_log": self._log_scale_minutes(resource_queue_wait_min),
                    "candidate_queue_blocked": float(resource_queue_wait_min > 0.0),
                    "candidate_calendar_wait_min": float(calendar_wait_min),
                }

            calendar_fit_score = self._resource_calendar_fit_score(
                resource=resource,
                current_time=enabled_time,
                calendar=calendars.get(resource, {}) if isinstance(calendars, dict) else {},
            )
            features["candidate_calendar_fit_score"] = float(calendar_fit_score)
            features["candidate_resource_calendar_fit"] = float(calendar_fit_score)

            candidate_records.append(
                {
                    "resource": resource,
                    "start_time": earliest_start,
                    "resource_ready_time": feasible.get("resource_ready_time", enabled_time),
                    "resource_queue_wait_min": float(resource_queue_wait_min),
                    "calendar_wait_min": float(calendar_wait_min),
                    "waiting_time_min": waiting_time_min,
                    "workload": workload,
                    "capacity": capacity,
                    "tree_score": self._clip01(tree_score),
                    "frequency_score": self._clip01(frequency_score),
                    "availability_score": 0.0,
                    "calendar_fit_score": self._clip01(calendar_fit_score),
                    "score": 0.0,
                    "features": features,
                    "calendar": calendars.get(resource, {}) if isinstance(calendars, dict) else {},
                }
            )

        if not candidate_records:
            r = candidate_resources[0]
            return {
                "resource": r,
                "start_time": enabled_time,
                "resource_queue_wait_min": 0.0,
                "calendar_wait_min": 0.0,
                "waiting_time_min": 0.0,
                "workload": int(workloads.get(r, 0)),
                "scores": {},
                "score_components": {},
                "selected_features": {},
                "resource_policy_mode": self._get_policy_mode(),
                "resource_policy_weights": self._get_score_weights(),
            }

        candidate_records = self._apply_queue_wait_cap(candidate_records)

        self._attach_availability_scores(candidate_records)

        for rec in candidate_records:
            final_score = self._hybrid_score(
                tree_score=rec["tree_score"],
                frequency_score=rec["frequency_score"],
                availability_score=rec["availability_score"],
                calendar_fit_score=rec.get("calendar_fit_score", 0.0),
                weights=score_weights,
            )

            wait_penalty = float(self.params.get("runtime_wait_penalty", 0.0))
            if wait_penalty > 0:
                # Use a log-scaled waiting penalty. A linear penalty in raw minutes
                # over-penalizes realistic resource waits and makes queue_wait vanish.
                wait_norm = self._log_scale_minutes(
                    rec.get("resource_queue_wait_min", rec["waiting_time_min"]),
                    cap_min=float(self.params.get("wait_penalty_cap_min", 1440.0)),
                )
                final_score = final_score / (1.0 + wait_penalty * wait_norm)

            rec["score"] = max(0.0, float(final_score))

        if deterministic:
            selected = sorted(
                candidate_records,
                key=lambda rec: (-rec["score"], rec["waiting_time_min"], rec["resource"])
            )[0]
        else:
            total = sum(rec["score"] for rec in candidate_records)
            if total <= 0:
                selected = self.rng.choice(candidate_records)
            else:
                threshold = self.rng.random() * total
                cumulative = 0.0
                selected = candidate_records[-1]
                for rec in candidate_records:
                    cumulative += rec["score"]
                    if threshold <= cumulative:
                        selected = rec
                        break

        return {
            "resource": selected["resource"],
            "start_time": selected["start_time"],
            "resource_ready_time": selected.get("resource_ready_time", enabled_time),
            "resource_queue_wait_min": float(selected.get("resource_queue_wait_min", 0.0)),
            "calendar_wait_min": float(selected.get("calendar_wait_min", 0.0)),
            "waiting_time_min": selected["waiting_time_min"],
            "workload": selected["workload"],
            "capacity": int(selected.get("capacity", 1)),
            "queue_wait_cap_applied": bool(selected.get("queue_wait_cap_applied", False)),
            "queue_wait_cap_min": selected.get("queue_wait_cap_min"),
            "scores": {rec["resource"]: float(rec["score"]) for rec in candidate_records},
            "score_components": {
                rec["resource"]: {
                    "tree_score": float(rec["tree_score"]),
                    "frequency_score": float(rec["frequency_score"]),
                    "availability_score": float(rec["availability_score"]),
                    "calendar_fit_score": float(rec.get("calendar_fit_score", 0.0)),
                    "final_score": float(rec["score"]),
                    "waiting_time_min": float(rec["waiting_time_min"]),
                    "resource_queue_wait_min": float(rec.get("resource_queue_wait_min", 0.0)),
                    "calendar_wait_min": float(rec.get("calendar_wait_min", 0.0)),
                    "weights": score_weights,
                }
                for rec in candidate_records
            },
            "selected_features": selected["features"],
            "selected_resource_calendar": selected.get("calendar", {}),
            "selected_resource_calendar_fit_score": float(selected.get("calendar_fit_score", 0.0)),
            "resource_policy_mode": self._get_policy_mode(),
            "resource_policy_weights": score_weights,
        }

    def _activity_resource_frequency(self, activity: str, resource: str) -> float:
        freq_key = f"{activity}|||{resource}"
        return self._clip01(self._freq_table.get(freq_key, 0.0))

    def _build_candidate_features(
            self,
            activity: str,
            resource: str,
            case: dict,
            cases: list,
            completed_cases: set,
            current_case_id: int,
            current_time,
            candidate_resources: list,
            workloads: dict,
            capacity: int,
            waiting_time_min: float,
            resource_state_runtime,
            queue_state_runtime=None,
    ) -> Dict[str, Any]:
        features = {
            "resource_policy_mode_encoded": float(self._policy_mode_code(self._get_policy_mode())),
        }

        for activity_name in self.activities:
            features[f"{ACTIVITY_PREFIX}{activity_name}"] = int(str(activity_name) == str(activity))

        features.update(case.get("history", {}))
        features.update(case.get("attributes", {}))

        features["candidate_workload"] = float(workloads.get(resource, 0))
        features["candidate_capacity"] = float(max(1, int(capacity)))
        features["candidate_available_now"] = float(workloads.get(resource, 0) < max(1, int(capacity)))
        features["candidate_resource_wait_min"] = float(waiting_time_min)
        features["candidate_waiting_time_min"] = float(waiting_time_min)
        features["candidate_waiting_time_log"] = self._log_scale_minutes(waiting_time_min)
        features["candidate_queue_blocked"] = float(waiting_time_min > 0.0)
        features["candidate_resource_history_count"] = float(
            case.get("res_history", {}).get(resource, 0.0)
        )

        features["activity_resource_frequency"] = self._activity_resource_frequency(activity, resource)
        features.setdefault("candidate_calendar_fit_score", 0.0)
        features.setdefault("candidate_resource_calendar_fit", 0.0)

        queue_features = {}
        if queue_state_runtime is not None:
            try:
                queue_features = queue_state_runtime.compute_features(
                    cases=cases,
                    completed_cases=completed_cases,
                    current_case_id=current_case_id,
                    current_time=current_time,
                    activity=activity,
                    resource=resource,
                    candidate_resources=candidate_resources,
                )
            except Exception:
                queue_features = {}

        for name in QUEUE_FEATURE_NAMES:
            features[name] = float(queue_features.get(name, 0.0))

        state = getattr(resource_state_runtime, "states", {}).get(str(resource), {})
        skill_state = state.get("skill", {}).get(str(activity), "medium")

        features.update({
            "resource_fatigue_low": int(state.get("fatigue", "low") == "low"),
            "resource_fatigue_medium": int(state.get("fatigue", "low") == "medium"),
            "resource_fatigue_high": int(state.get("fatigue", "low") == "high"),
            "resource_skill_low": int(skill_state == "low"),
            "resource_skill_medium": int(skill_state == "medium"),
            "resource_skill_high": int(skill_state == "high"),
            "resource_efficiency_low": int(state.get("efficiency", "medium") == "low"),
            "resource_efficiency_medium": int(state.get("efficiency", "medium") == "medium"),
            "resource_efficiency_high": int(state.get("efficiency", "medium") == "high"),
            "resource_available_state": int(state.get("availability", "available") == "available"),
        })

        return features

    def _get_policy_mode(self) -> str:
        mode = str(
            self.params.get("resource_policy_mode")
            or self.params.get("selected_strategy")
            or self.params.get("selected_resource_policy")
            or self.resource_policy_mode
            or "balanced"
        )
        if mode not in self.RESOURCE_POLICY_MODES:
            # Keep backward compatibility with previous strategy names.
            alias = {
                "freq_only": "frequency_oriented",
                "freq_avail": "frequency_oriented",
                "tree_freq": "balanced",
                "calendar_freq": "calendar_frequency_oriented",
                "calendar_frequency": "calendar_frequency_oriented",
                "stable_calendar": "stable_calendar_frequency",
                "stable_calendar_freq": "stable_calendar_frequency",
                "queue_aware": "queue_oriented",
            }.get(mode)
            if alias is not None:
                mode = alias
        return mode

    @staticmethod
    def _policy_mode_code(mode: str) -> int:
        order = {
            "frequency_oriented": 0,
            "calendar_frequency_oriented": 1,
            "stable_calendar_frequency": 2,
            "balanced": 3,
            "queue_oriented": 4,
        }
        return int(order.get(str(mode), 1))

    def _get_score_weights(self) -> Dict[str, float]:
        policy_mode = self._get_policy_mode()
        policy_default = self.RESOURCE_POLICY_MODES.get(policy_mode, self.DEFAULT_SCORE_WEIGHTS)
        raw = (
            self.params.get("score_mixing_weights")
            or self.params.get("hybrid_score_weights")
            or policy_default
            or self.DEFAULT_SCORE_WEIGHTS
        )

        weights = {
            "tree": self._safe_nonnegative(raw.get("tree", self.DEFAULT_SCORE_WEIGHTS["tree"])),
            "frequency": self._safe_nonnegative(raw.get("frequency", self.DEFAULT_SCORE_WEIGHTS["frequency"])),
            "availability": self._safe_nonnegative(raw.get("availability", self.DEFAULT_SCORE_WEIGHTS["availability"])),
            "calendar_fit": self._safe_nonnegative(raw.get("calendar_fit", self.DEFAULT_SCORE_WEIGHTS["calendar_fit"])),
        }

        total = sum(weights.values())
        if total <= 0:
            weights = dict(self.DEFAULT_SCORE_WEIGHTS)
            total = sum(weights.values())

        return {k: float(v) / total for k, v in weights.items()}

    def _hybrid_score(
            self,
            tree_score: float,
            frequency_score: float,
            availability_score: float,
            calendar_fit_score: float,
            weights: Dict[str, float],
    ) -> float:
        return float(
            weights.get("tree", 0.0) * self._clip01(tree_score)
            + weights.get("frequency", 0.0) * self._clip01(frequency_score)
            + weights.get("availability", 0.0) * self._clip01(availability_score)
            + weights.get("calendar_fit", 0.0) * self._clip01(calendar_fit_score)
        )

    def _resource_calendar_fit_score(self, resource: str, current_time, calendar: dict) -> float:
        """Return how well a candidate resource matches the current weekday-hour.

        This is a soft score only.  Sparse calendars should not make a resource
        impossible, because in long-tail logs a resource may have only a few
        observed active hours in the training split.
        """
        if not isinstance(calendar, dict) or not calendar:
            return 0.5

        if self.is_sparse_resource_calendar(calendar):
            # Sparse resource calendars are unreliable as hard evidence.
            # Keep a neutral soft score instead of strongly penalizing them.
            return 0.5

        try:
            if self._calendar_is_active_hour(calendar, current_time.weekday(), current_time.hour, default=False):
                return 1.0
        except Exception:
            return 0.5

        probe = current_time
        max_lookahead_hours = int(self.params.get("calendar_fit_lookahead_hours", 24))
        max_lookahead_hours = max(1, min(max_lookahead_hours, 24 * 7))
        for offset in range(1, max_lookahead_hours + 1):
            try:
                probe = (probe + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                if self._calendar_is_active_hour(calendar, probe.weekday(), probe.hour, default=False):
                    return float(1.0 / (1.0 + offset / 8.0))
            except Exception:
                break

        return 0.25

    def _attach_availability_scores(self, candidate_records: list) -> None:
        if not candidate_records:
            return

        waits = [float(rec.get("waiting_time_min", 0.0)) for rec in candidate_records]
        min_wait = min(waits)
        max_wait = max(waits)

        if max_wait <= min_wait:
            for rec in candidate_records:
                rec["availability_score"] = 1.0
            return

        for rec in candidate_records:
            wait = float(rec.get("waiting_time_min", 0.0))
            rec["availability_score"] = float(1.0 - (wait - min_wait) / (max_wait - min_wait))
            rec["availability_score"] = self._clip01(rec["availability_score"])

    @staticmethod
    def _log_scale_minutes(value: float, cap_min: float = 1440.0) -> float:
        try:
            value = max(0.0, min(float(value), float(cap_min)))
            cap_min = max(1.0, float(cap_min))
            return float(np.log1p(value) / np.log1p(cap_min))
        except Exception:
            return 0.0

    @staticmethod
    def _clip01(value) -> float:
        try:
            value = float(value)
            if np.isfinite(value):
                return float(np.clip(value, 0.0, 1.0))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _safe_nonnegative(value) -> float:
        try:
            value = float(value)
            if np.isfinite(value):
                return max(0.0, value)
        except Exception:
            pass
        return 0.0

    def _capacity_from_profile(self, resource: str, fallback_capacity: int) -> int:
        """Return capacity from discovered profile when available.

        The profile is estimated from training-log concurrent workload and is
        used as a conservative upper bound. This preserves queue scheduling, but
        prevents single-capacity assumptions from creating artificial multi-day
        queues in logs where resources historically handled concurrent work.
        """
        try:
            fallback_capacity = max(1, int(fallback_capacity))
        except Exception:
            fallback_capacity = 1

        if not bool(self.params.get("adaptive_resource_capacity", True)):
            return fallback_capacity

        profile = self._capacity_profile.get(str(resource), {})
        if not isinstance(profile, dict):
            return fallback_capacity

        try:
            prof_capacity = int(profile.get("capacity", fallback_capacity))
        except Exception:
            prof_capacity = fallback_capacity

        try:
            max_capacity = int(self.params.get("max_adaptive_resource_capacity", 8))
        except Exception:
            max_capacity = 8
        max_capacity = max(1, max_capacity)

        return int(max(1, min(max(fallback_capacity, prof_capacity), max_capacity)))

    def _apply_queue_wait_cap(self, candidate_records: list) -> list:
        """Filter extreme queue-wait candidates when alternatives exist.

        This does not disable queue scheduling. It prevents a selected resource
        from waiting thousands of minutes if another capable resource can start
        within a unified bound. If all candidates are blocked, the shortest-wait
        alternatives are kept so the queue module still makes the decision.
        """
        if not candidate_records:
            return candidate_records

        try:
            cap_min = float(self.params.get("max_queue_wait_min", 480.0))
        except Exception:
            cap_min = 480.0

        if not np.isfinite(cap_min) or cap_min <= 0.0:
            return candidate_records

        within = []
        for rec in candidate_records:
            try:
                wait = float(rec.get("resource_queue_wait_min", rec.get("waiting_time_min", 0.0)))
            except Exception:
                wait = 0.0
            if wait <= cap_min:
                within.append(rec)

        if within:
            for rec in within:
                rec["queue_wait_cap_applied"] = len(within) < len(candidate_records)
                rec["queue_wait_cap_min"] = float(cap_min)
            return within

        try:
            top_k = int(self.params.get("fallback_top_k_when_all_blocked", 3))
        except Exception:
            top_k = 3
        top_k = max(1, min(top_k, len(candidate_records)))

        bounded = sorted(
            candidate_records,
            key=lambda rec: (
                float(rec.get("resource_queue_wait_min", rec.get("waiting_time_min", 0.0)) or 0.0),
                str(rec.get("resource", "")),
            ),
        )[:top_k]
        for rec in bounded:
            rec["queue_wait_cap_applied"] = True
            rec["queue_wait_cap_min"] = float(cap_min)
            rec["queue_wait_cap_fallback"] = "all_candidates_exceed_cap_keep_shortest"
        return bounded

    def _earliest_start_by_capacity(self, schedule: list, enabled_time, capacity: int):
        if not schedule:
            return enabled_time

        capacity = max(1, int(capacity or 1))
        active_ends = sorted(
            end for start, end in schedule
            if start <= enabled_time < end
        )
        if len(active_ends) < capacity:
            return enabled_time

        completions_needed = len(active_ends) - capacity + 1
        return active_ends[completions_needed - 1]

    def find_feasible_start_components(
            self,
            schedule: list,
            enabled_time,
            capacity: int,
            calendar: dict,
            max_iterations: int = 32,
    ) -> Dict[str, Any]:
        """Resolve a feasible start while keeping Queue and Calendar separate."""
        probe = enabled_time
        resource_queue_wait_min = 0.0
        calendar_wait_min = 0.0
        resource_ready_time = enabled_time

        for _ in range(max(1, int(max_iterations))):
            previous = probe

            resource_ready = self._earliest_start_by_capacity(
                schedule=schedule,
                enabled_time=probe,
                capacity=max(1, int(capacity)),
            )
            if resource_ready > probe:
                resource_queue_wait_min += max(
                    0.0, (resource_ready - probe).total_seconds() / 60.0
                )
                probe = resource_ready
            resource_ready_time = probe

            calendar_ready = self._respect_calendar_start(probe, calendar)
            if calendar_ready > probe:
                calendar_wait_min += max(
                    0.0, (calendar_ready - probe).total_seconds() / 60.0
                )
                probe = calendar_ready

            if probe == previous:
                break

        return {
            "start_time": probe,
            "resource_ready_time": resource_ready_time,
            "resource_queue_wait_min": float(resource_queue_wait_min),
            "calendar_wait_min": float(calendar_wait_min),
        }

    def _respect_calendar_start(self, current_time, calendar: dict):
        """Apply resource calendars as a bounded soft constraint.

        The previous implementation searched up to 14 days for the next active
        resource hour.  In logs with many long-tail resources, this could create
        thousands of minutes of artificial queue wait and damage CED/CWD.
        Now calendars still guide resource scoring, but hard start-time shifts
        are bounded and disabled for sparse calendars.
        """
        if not calendar:
            return current_time

        try:
            if self._calendar_is_active_hour(calendar, current_time.weekday(), current_time.hour, default=True):
                return current_time
        except Exception:
            return current_time

        if self.is_sparse_resource_calendar(calendar):
            return current_time

        max_shift_hours = int(self.params.get("max_resource_calendar_shift_hours", 24 * 7))
        max_shift_hours = max(0, min(max_shift_hours, 24 * 7))
        if max_shift_hours <= 0:
            return current_time

        probe = current_time
        for _ in range(max_shift_hours):
            probe = (probe + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            if self._calendar_is_active_hour(calendar, probe.weekday(), probe.hour, default=False):
                return probe

        return current_time

    @staticmethod
    def _weekday_key_candidates(weekday: int):
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        try:
            weekday = int(weekday)
        except Exception:
            weekday = 0
        weekday = max(0, min(6, weekday))
        return [weekday, str(weekday), names[weekday], short[weekday], names[weekday].lower(), short[weekday].lower()]

    @staticmethod
    def _hour_key_candidates(hour: int):
        try:
            hour = int(hour)
        except Exception:
            hour = 0
        hour = max(0, min(23, hour))
        return [hour, str(hour)]

    def _calendar_is_active_hour(self, calendar: dict, weekday: int, hour: int, default: bool = False) -> bool:
        if not isinstance(calendar, dict) or not calendar:
            return bool(default)
        for wd_key in self._weekday_key_candidates(weekday):
            day = calendar.get(wd_key)
            if not isinstance(day, dict):
                continue
            for h_key in self._hour_key_candidates(hour):
                if h_key in day:
                    return bool(day.get(h_key))
        return bool(default)

    def is_sparse_resource_calendar(self, calendar: dict) -> bool:
        """Return the shared sparse-calendar decision used at runtime.

        Keep the existing semantics exactly: only non-empty calendars with
        1..(threshold-1) active weekday-hour cells are treated as sparse.
        The threshold remains configurable through
        ``min_resource_calendar_active_hours`` and defaults to 12.
        """
        active_hours = self._active_calendar_hour_count(calendar)
        min_active_hours = int(
            self.params.get("min_resource_calendar_active_hours", 12)
        )
        return bool(active_hours > 0 and active_hours < min_active_hours)

    def _active_calendar_hour_count(self, calendar: dict) -> int:
        if not isinstance(calendar, dict) or not calendar:
            return 0
        active = 0
        for wd in range(7):
            for hour in range(24):
                if self._calendar_is_active_hour(calendar, wd, hour, default=False):
                    active += 1
        return int(active)
