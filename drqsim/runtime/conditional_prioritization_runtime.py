# drq/runtime/conditional_prioritization_runtime.py

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from drqsim.discovery.conditional_prioritization_discovery import (
    LEFT_ACTIVITY_PREFIX,
    RIGHT_ACTIVITY_PREFIX,
)


class ConditionalPrioritizationRuntime:
    """Runtime dispatcher for the pairwise Conditional Prioritization Tree.

    The runtime never creates waiting time by itself.  It only ranks jobs that
    are simultaneously ready and eligible for the resource at the dispatch time.
    If the model is unavailable, the queue deterministically falls back to FCFS.
    """

    def __init__(self, prioritization_params: dict, activities: list):
        self.params = prioritization_params or {"model_kind": "fcfs_fallback"}
        self.activities = [str(a) for a in (activities or [])]

    def select_next_job(
            self,
            resource: str,
            ready_jobs: list[dict],
            current_time,
            resource_workload: int = 0,
            deterministic: bool = False,
    ) -> Dict[str, Any]:
        jobs = [job for job in (ready_jobs or []) if self._job_eligible_for_resource(job, resource)]
        if not jobs:
            return {"job": None, "policy": "empty_queue", "scores": {}}
        if len(jobs) == 1:
            return {"job": jobs[0], "policy": "single_job", "scores": {self._job_key(jobs[0]): 1.0}}

        if self.params.get("model_kind") != "pairwise_decision_tree" or not self.params.get("rules"):
            selected = self._fcfs(jobs)
            return {"job": selected, "policy": "fcfs", "scores": {}}

        scores: Dict[str, float] = {}
        queue_length = len(jobs)

        # Behavior-preserving vectorized evaluation.  The original runtime built
        # a full Python feature dictionary for every ordered job pair, making one
        # dispatch O(q^2) in expensive Python object work.  We still evaluate the
        # exact same ordered pairs and the exact same decision-tree rules, but the
        # right-hand jobs are evaluated as NumPy arrays.  Score order and np.mean
        # semantics are preserved, so dispatch decisions remain unchanged.
        context = self._prepare_vector_context(
            jobs=jobs,
            current_time=current_time,
            resource=str(resource),
            queue_length=queue_length,
            resource_workload=resource_workload,
        )
        rules = self.params.get("rules")
        for i, left in enumerate(jobs):
            pair_scores = self._apply_rules_vectorized(rules, left_index=i, context=context)
            if len(pair_scores) <= 1:
                score = 0.5
            else:
                # np.delete preserves the same right-job order as the old nested
                # loop (j=0..q-1, skipping i), and np.mean therefore receives the
                # same float sequence as before.
                score = float(np.mean(np.delete(pair_scores, i)))
            scores[self._job_key(left)] = score

        # Stable deterministic tie breaker: score -> older enabled -> case id.
        selected = sorted(
            jobs,
            key=lambda job: (
                -scores.get(self._job_key(job), 0.5),
                self._timestamp_value(job.get("queue_entry_time") or job.get("enabled_time")),
                int(job.get("case_id", 0)),
                str(job.get("activity", "")),
            ),
        )[0]

        return {
            "job": selected,
            "policy": "conditional_prioritization_tree",
            "scores": scores,
            "selected_score": float(scores.get(self._job_key(selected), 0.5)),
        }

    @staticmethod
    def _job_eligible_for_resource(job: dict, resource: str) -> bool:
        return str(resource) in {str(r) for r in (job.get("candidate_resources") or [])}

    @staticmethod
    def _job_key(job: dict) -> str:
        return str(job.get("job_id", f"{job.get('case_id')}::{job.get('activity')}"))

    def _fcfs(self, jobs: list[dict]) -> dict:
        return sorted(
            jobs,
            key=lambda job: (
                self._timestamp_value(job.get("queue_entry_time") or job.get("enabled_time")),
                int(job.get("case_id", 0)),
                str(job.get("activity", "")),
            ),
        )[0]

    def _prepare_vector_context(
            self,
            jobs: list[dict],
            current_time,
            resource: str,
            queue_length: int,
            resource_workload: int,
    ) -> Dict[str, Any]:
        """Prepare one dispatch-time cache for pairwise tree evaluation."""
        now = pd.Timestamp(current_time)
        waits = []
        activities = []
        histories = []
        attributes = []
        res_histories = []
        for job in jobs:
            enabled = pd.Timestamp(job.get("queue_entry_time") or job.get("enabled_time") or now)
            wait = max(0.0, (now - enabled).total_seconds() / 60.0)
            waits.append(float(wait))
            activities.append(str(job.get("activity", "")))
            histories.append(job.get("history") or {})
            attributes.append(job.get("attributes") or {})
            res_histories.append(job.get("res_history") or {})

        waits_arr = np.asarray(waits, dtype=float)
        return {
            "jobs": jobs,
            "n": len(jobs),
            "now": now,
            "resource": str(resource),
            "queue_length": int(queue_length),
            "resource_workload": int(resource_workload),
            "waits": waits_arr,
            "wait_logs": np.asarray([self._log_scale_minutes(v) for v in waits], dtype=float),
            "activities": np.asarray(activities, dtype=object),
            "histories": histories,
            "attributes": attributes,
            "res_histories": res_histories,
            "right_feature_cache": {},
        }

    def _right_numeric_array(self, cache_key: str, context: dict, getter) -> np.ndarray:
        cache = context["right_feature_cache"]
        if cache_key not in cache:
            cache[cache_key] = np.asarray(
                [self._safe_float(getter(i)) for i in range(context["n"])],
                dtype=float,
            )
        return cache[cache_key]

    def _pair_feature_vector(self, feature: str, left_index: int, context: dict):
        """Return a scalar or right-job vector matching _build_pair_features."""
        waits = context["waits"]
        wait_logs = context["wait_logs"]
        activities = context["activities"]
        left_activity = str(activities[left_index])
        now = context["now"]

        if feature == "left_waiting_age_min":
            return float(waits[left_index])
        if feature == "right_waiting_age_min":
            return waits
        if feature == "waiting_age_diff_min":
            return float(waits[left_index]) - waits
        if feature == "left_waiting_age_log":
            return float(wait_logs[left_index])
        if feature == "right_waiting_age_log":
            return wait_logs
        if feature == "waiting_age_log_diff":
            return float(wait_logs[left_index]) - wait_logs
        if feature == "same_activity":
            return (activities == left_activity).astype(float)
        if feature == "resource_queue_length":
            return float(max(1, int(context["queue_length"])))
        if feature == "resource_queue_length_log":
            return self._log_scale_count(context["queue_length"])
        if feature == "resource_workload":
            return float(max(0, int(context["resource_workload"])))
        if feature == "hour":
            return float(now.hour)
        if feature == "weekday":
            return float(now.weekday())

        if isinstance(feature, str) and feature.startswith(LEFT_ACTIVITY_PREFIX):
            activity = feature[len(LEFT_ACTIVITY_PREFIX):]
            return float(left_activity == activity)
        if isinstance(feature, str) and feature.startswith(RIGHT_ACTIVITY_PREFIX):
            activity = feature[len(RIGHT_ACTIVITY_PREFIX):]
            return (activities == activity).astype(float)

        if isinstance(feature, str) and feature.startswith("history_diff::"):
            activity = feature[len("history_diff::"):]
            arr = self._right_numeric_array(
                f"history::{activity}",
                context,
                lambda i: (context["histories"][i] or {}).get(activity, 0.0),
            )
            left_value = self._safe_float((context["histories"][left_index] or {}).get(activity, 0.0))
            return float(left_value) - arr

        if isinstance(feature, str) and feature.startswith("left::"):
            raw = feature[len("left::"):]
            if " = " in raw:
                attr, value = raw.split(" = ", 1)
                attrs = context["attributes"][left_index] or {}
                return float(
                    bool(attrs.get(f"{attr} = {value}", False))
                    or str(attrs.get(attr, "")) == value
                )
            attrs = context["attributes"][left_index] or {}
            return self._safe_float(attrs.get(raw, 0.0))

        if isinstance(feature, str) and feature.startswith("right::"):
            raw = feature[len("right::"):]
            cache = context["right_feature_cache"]
            if " = " in raw:
                attr, value = raw.split(" = ", 1)
                cache_key = f"cat::{attr}::{value}"
                if cache_key not in cache:
                    cache[cache_key] = np.asarray([
                        float(
                            bool((attrs or {}).get(f"{attr} = {value}", False))
                            or str((attrs or {}).get(attr, "")) == value
                        )
                        for attrs in context["attributes"]
                    ], dtype=float)
                return cache[cache_key]
            return self._right_numeric_array(
                f"attr::{raw}",
                context,
                lambda i: (context["attributes"][i] or {}).get(raw, 0.0),
            )

        if isinstance(feature, str) and feature.startswith("diff::"):
            attr = feature[len("diff::"):]
            arr = self._right_numeric_array(
                f"attr::{attr}",
                context,
                lambda i: (context["attributes"][i] or {}).get(attr, 0.0),
            )
            left_value = self._safe_float((context["attributes"][left_index] or {}).get(attr, 0.0))
            return float(left_value) - arr

        if feature == "resource_history_diff":
            resource = str(context["resource"])
            arr = self._right_numeric_array(
                f"res_history::{resource}",
                context,
                lambda i: (context["res_histories"][i] or {}).get(resource, 0.0),
            )
            left_value = self._safe_float(
                (context["res_histories"][left_index] or {}).get(resource, 0.0)
            )
            return float(left_value) - arr

        # _apply_rules previously read features.get(feature, 0.0).
        return 0.0

    def _apply_rules_vectorized(self, rules: Optional[dict], left_index: int, context: dict) -> np.ndarray:
        """Evaluate the serialized tree against every right job at once."""
        n = int(context["n"])
        out = np.empty(n, dtype=float)

        def visit(node, mask):
            if not np.any(mask):
                return
            if not isinstance(node, dict):
                out[mask] = 0.5
                return
            if node.get("type") == "leaf":
                score = float(np.clip(self._safe_float(node.get("score", 0.5), 0.5), 0.0, 1.0))
                out[mask] = score
                return

            feature = node.get("feature")
            threshold = self._safe_float(node.get("threshold", 0.0))
            values = self._pair_feature_vector(feature, left_index=left_index, context=context)
            children = node.get("children") or {}

            if np.isscalar(values):
                branch = "true" if self._safe_float(values) <= threshold else "false"
                child = children.get(branch)
                if child is None:
                    out[mask] = 0.5
                else:
                    visit(child, mask)
                return

            values = np.asarray(values, dtype=float)
            cond = values <= threshold
            true_mask = mask & cond
            false_mask = mask & (~cond)
            child_true = children.get("true")
            child_false = children.get("false")
            if child_true is None:
                out[true_mask] = 0.5
            else:
                visit(child_true, true_mask)
            if child_false is None:
                out[false_mask] = 0.5
            else:
                visit(child_false, false_mask)

        visit(rules, np.ones(n, dtype=bool))
        return out

    def _build_pair_features(
            self,
            left: dict,
            right: dict,
            current_time,
            resource: str,
            queue_length: int,
            resource_workload: int,
    ) -> Dict[str, float]:
        now = pd.Timestamp(current_time)
        left_enabled = pd.Timestamp(left.get("queue_entry_time") or left.get("enabled_time") or now)
        right_enabled = pd.Timestamp(right.get("queue_entry_time") or right.get("enabled_time") or now)
        left_wait = max(0.0, (now - left_enabled).total_seconds() / 60.0)
        right_wait = max(0.0, (now - right_enabled).total_seconds() / 60.0)
        left_activity = str(left.get("activity", ""))
        right_activity = str(right.get("activity", ""))

        features: Dict[str, float] = {
            "left_waiting_age_min": float(left_wait),
            "right_waiting_age_min": float(right_wait),
            "waiting_age_diff_min": float(left_wait - right_wait),
            "left_waiting_age_log": self._log_scale_minutes(left_wait),
            "right_waiting_age_log": self._log_scale_minutes(right_wait),
            "waiting_age_log_diff": self._log_scale_minutes(left_wait) - self._log_scale_minutes(right_wait),
            "same_activity": float(left_activity == right_activity),
            "resource_queue_length": float(max(1, int(queue_length))),
            "resource_queue_length_log": self._log_scale_count(queue_length),
            "resource_workload": float(max(0, int(resource_workload))),
            "hour": float(now.hour),
            "weekday": float(now.weekday()),
        }

        for activity in self.activities:
            features[f"{LEFT_ACTIVITY_PREFIX}{activity}"] = float(left_activity == activity)
            features[f"{RIGHT_ACTIVITY_PREFIX}{activity}"] = float(right_activity == activity)
            lv = self._safe_float((left.get("history") or {}).get(activity, 0.0))
            rv = self._safe_float((right.get("history") or {}).get(activity, 0.0))
            features[f"history_diff::{activity}"] = float(lv - rv)

        left_attrs = left.get("attributes") or {}
        right_attrs = right.get("attributes") or {}
        # Only emit keys known by the discovered feature schema. Numeric and
        # categorical case attributes are both reconstructed from serialized names.
        for col in self.params.get("feature_columns", []) or []:
            if col.startswith("left::") and " = " not in col:
                attr = col[len("left::"):]
                features[col] = self._safe_float(left_attrs.get(attr, 0.0))
            elif col.startswith("right::") and " = " not in col:
                attr = col[len("right::"):]
                features[col] = self._safe_float(right_attrs.get(attr, 0.0))
            elif col.startswith("diff::"):
                attr = col[len("diff::"):]
                features[col] = self._safe_float(left_attrs.get(attr, 0.0)) - self._safe_float(right_attrs.get(attr, 0.0))
            elif col.startswith("left::") and " = " in col:
                raw = col[len("left::"):]
                attr, value = raw.split(" = ", 1)
                features[col] = float(bool(left_attrs.get(f"{attr} = {value}", False)) or str(left_attrs.get(attr, "")) == value)
            elif col.startswith("right::") and " = " in col:
                raw = col[len("right::"):]
                attr, value = raw.split(" = ", 1)
                features[col] = float(bool(right_attrs.get(f"{attr} = {value}", False)) or str(right_attrs.get(attr, "")) == value)

        lv = self._safe_float((left.get("res_history") or {}).get(str(resource), 0.0))
        rv = self._safe_float((right.get("res_history") or {}).get(str(resource), 0.0))
        features["resource_history_diff"] = float(lv - rv)
        return features

    def _apply_rules(self, rules: Optional[dict], features: dict) -> float:
        node = rules
        while isinstance(node, dict) and node.get("type") != "leaf":
            feature = node.get("feature")
            threshold = self._safe_float(node.get("threshold", 0.0))
            value = self._safe_float(features.get(feature, 0.0))
            node = (node.get("children") or {}).get("true" if value <= threshold else "false")
            if node is None:
                return 0.5
        if not isinstance(node, dict):
            return 0.5
        return float(np.clip(self._safe_float(node.get("score", 0.5), 0.5), 0.0, 1.0))

    @staticmethod
    def _timestamp_value(value) -> int:
        try:
            return int(pd.Timestamp(value).value)
        except Exception:
            return 0

    @staticmethod
    def _log_scale_minutes(value: float, cap_min: float = 7 * 24 * 60.0) -> float:
        value = max(0.0, min(ConditionalPrioritizationRuntime._safe_float(value), float(cap_min)))
        return float(np.log1p(value) / np.log1p(max(1.0, float(cap_min))))

    @staticmethod
    def _log_scale_count(value: float, cap: float = 50.0) -> float:
        value = max(0.0, min(ConditionalPrioritizationRuntime._safe_float(value), float(cap)))
        return float(np.log1p(value) / np.log1p(max(1.0, float(cap))))

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            value = float(value)
            if np.isfinite(value):
                return value
        except Exception:
            pass
        return float(default)
