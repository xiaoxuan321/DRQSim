# drq/runtime/queue_state_runtime.py

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


class QueueStateRuntime:
    """Incremental runtime queue-state tracker.

    The previous implementation rebuilt queue features on every resource
    candidate evaluation by scanning all cases, ready jobs and scheduled jobs.
    In large logs that made one feature query O(N_cases + N_jobs).

    This implementation keeps the *same runtime feature semantics* but updates
    counts when state changes:

    - enabled visible case representative enters/leaves the pre-allocation queue;
    - ready job enters/leaves an assigned-resource queue;
    - legacy scheduled job moves pending -> waiting -> running -> completed.

    Therefore ``compute_features()`` is O(1) apart from excluding jobs belonging
    to ``current_case_id`` (normally zero or one job).

    Queue semantics are unchanged:
        queue(t) = {job | enabled_time <= t < start_time}
        wip(t)   = {job | enabled_time <= t < end_time}

    For the explicit Ready Queue engine, running jobs are intentionally not
    added to ``system_wip`` unless they are registered through the legacy
    ``register_scheduled_job`` API.  This preserves the behavior of the latest
    Ready Queue code before this optimization.
    """

    def __init__(
            self,
            queue_params: dict,
            act_to_resources: dict,
            resources: list,
    ):
        self.params = queue_params or {}
        self.act_to_resources = {
            str(activity): [str(r) for r in (resource_list or [])]
            for activity, resource_list in (act_to_resources or {}).items()
        }
        self.resources = [str(r) for r in resources]

        self.max_queue_count = float(self.params.get("max_queue_count", 20.0))
        self.max_wip_count = float(self.params.get("max_wip_count", 20.0))
        self.queue_warmup_events = int(self.params.get("queue_warmup_events", 4))

        self.disable_global_queue_features = bool(
            self.params.get("disable_global_queue_features", False)
        )
        self.disable_queue_when_resource_waiting = bool(
            self.params.get("disable_queue_when_resource_waiting", True)
        )
        self.queue_effect_strength = float(
            self.params.get("queue_effect_strength", 0.05)
        )
        self.queue_max_relative_delta = float(
            self.params.get("queue_max_relative_delta", 0.03)
        )
        self.queue_max_abs_delta_min = float(
            self.params.get("queue_max_abs_delta_min", 3.0)
        )

        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle / clock
    # ------------------------------------------------------------------
    @staticmethod
    def _time_key(value) -> int | None:
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                return None
            return int(ts.value)
        except Exception:
            return None

    def reset(self, start_time=None):
        """Reset all incremental state before a simulation run."""

        # Public compatibility stores.
        self.scheduled_jobs: dict[Any, dict] = {}
        self.ready_jobs: dict[Any, dict] = {}

        # Enabled-but-not-yet-converted-to-job representative per case.
        self._enabled_cases: dict[Any, dict] = {}

        # Fast current counts.
        self._global_queue_count = 0
        self._system_wip_count = 0
        self._activity_queue_counts = defaultdict(int)
        self._resource_queue_counts = defaultdict(int)
        self._resource_activity_queue_counts = defaultdict(int)

        # Ready jobs indexed by case only for O(1) current-case exclusion.
        self._ready_job_ids_by_case = defaultdict(set)

        # Deferred state changes (future enabled times / legacy scheduled jobs).
        self._time_events: list[tuple[int, int, str, Any, int]] = []
        self._state_seq = 0
        self._clock_key = self._time_key(start_time)

        # Lightweight diagnostics; useful for confirming that feature queries no
        # longer perform global scans.
        self._diagnostics = {
            "feature_queries": 0,
            "time_events_processed": 0,
            "enabled_case_registrations": 0,
            "ready_job_enqueues": 0,
            "ready_job_dequeues": 0,
        }

    def diagnostics(self) -> dict:
        return {
            **dict(self._diagnostics),
            "raw_global_queue_length": int(max(0, self._global_queue_count)),
            "raw_system_wip": int(max(0, self._system_wip_count)),
            "tracked_enabled_cases": int(len(self._enabled_cases)),
            "tracked_ready_jobs": int(len(self.ready_jobs)),
            "tracked_scheduled_jobs": int(len(self.scheduled_jobs)),
            "pending_state_events": int(len(self._time_events)),
            "feature_mode": "incremental_o1",
        }

    def _next_version(self) -> int:
        self._state_seq += 1
        return int(self._state_seq)

    def _push_time_event(self, when, kind: str, item_id, version: int):
        key = self._time_key(when)
        if key is None:
            return
        seq = self._next_version()
        heapq.heappush(
            self._time_events,
            (int(key), int(seq), str(kind), item_id, int(version)),
        )

    def advance_to(self, current_time):
        """Advance incremental state to ``current_time``.

        The simulator calls this once per DES event. ``compute_features`` also
        calls it defensively, so existing external callers remain correct.
        """

        key = self._time_key(current_time)
        if key is None:
            return

        # DES time is monotone.  If an external caller asks for an older time,
        # do not rewind destructive incremental state; simply leave the latest
        # valid state untouched.
        if self._clock_key is not None and key < self._clock_key:
            return

        self._clock_key = int(key)

        while self._time_events and self._time_events[0][0] <= key:
            _, _, kind, item_id, version = heapq.heappop(self._time_events)
            self._diagnostics["time_events_processed"] += 1

            if kind == "enabled_activate":
                record = self._enabled_cases.get(item_id)
                if not record or int(record.get("version", -1)) != int(version):
                    continue
                if not record.get("active", False):
                    self._activate_queue_record(record)

            elif kind == "ready_activate":
                record = self.ready_jobs.get(item_id)
                if not record or int(record.get("_qsr_version", -1)) != int(version):
                    continue
                if not record.get("_qsr_active", False):
                    self._activate_ready_record(record)

            elif kind.startswith("scheduled_"):
                record = self.scheduled_jobs.get(item_id)
                if not record or int(record.get("version", -1)) != int(version):
                    continue
                if kind == "scheduled_enabled":
                    self._scheduled_to_waiting(record)
                elif kind == "scheduled_start":
                    self._scheduled_to_running(record)
                elif kind == "scheduled_end":
                    self._scheduled_to_completed(item_id, record)

    # ------------------------------------------------------------------
    # Counter primitives
    # ------------------------------------------------------------------
    @staticmethod
    def _unique_resources(resources) -> tuple[str, ...]:
        if not resources:
            return tuple()
        # dict preserves stable input order while removing duplicates.
        return tuple(dict.fromkeys(str(r) for r in resources))

    def _queue_delta(self, activity: str, resources, delta: int):
        activity = str(activity)
        resources = self._unique_resources(resources)
        delta = int(delta)

        self._global_queue_count += delta
        self._activity_queue_counts[activity] += delta
        for resource in resources:
            self._resource_queue_counts[resource] += delta
            self._resource_activity_queue_counts[(resource, activity)] += delta

    def _wip_delta(self, delta: int):
        self._system_wip_count += int(delta)

    def _activate_queue_record(self, record: dict):
        self._queue_delta(
            record.get("activity", ""),
            record.get("candidate_resources", ()),
            +1,
        )
        self._wip_delta(+1)
        record["active"] = True

    def _deactivate_queue_record(self, record: dict):
        if not record.get("active", False):
            return
        self._queue_delta(
            record.get("activity", ""),
            record.get("candidate_resources", ()),
            -1,
        )
        self._wip_delta(-1)
        record["active"] = False

    # ------------------------------------------------------------------
    # Enabled-case state
    # ------------------------------------------------------------------
    def _representative_enabled_item(self, case: dict, current_time=None):
        """Return one visible representative from ``case['enabled']``.

        A marking may expose multiple transitions (e.g. XOR candidates).  The
        previous runtime counted at most one pending task per case, selecting
        the earliest visible transition and then activity name as a stable tie
        breaker.  The same rule is kept here.

        If ``current_time`` is provided, only already-enabled transitions are
        considered.  Registration omits it so future representatives can be
        scheduled incrementally.
        """

        enabled_items = []
        current_key = self._time_key(current_time) if current_time is not None else None

        for transition, enabled_time in (case.get("enabled", {}) or {}).items():
            enabled_key = self._time_key(enabled_time)
            if enabled_key is None:
                continue
            if current_key is not None and enabled_key > current_key:
                continue

            act = getattr(transition, "label", None)
            if act is None:
                continue

            enabled_items.append((enabled_key, str(act), transition, enabled_time))

        if not enabled_items:
            return None

        enabled_items.sort(key=lambda x: (x[0], x[1]))
        _, _, transition, enabled_time = enabled_items[0]
        return transition, enabled_time

    def register_enabled_case(self, case_id, case: dict, current_time=None):
        """Register/replace the visible enabled representative of one case.

        This method is called when the Petri marking enables new transitions.
        Future enabled times are placed in an internal heap and become active
        when simulation time reaches them.
        """

        if current_time is not None:
            self.advance_to(current_time)

        self.remove_enabled_case(case_id)
        representative = self._representative_enabled_item(case, current_time=None)
        if representative is None:
            return

        transition, enabled_time = representative
        activity = getattr(transition, "label", None)
        if activity is None:
            return
        activity = str(activity)

        version = self._next_version()
        record = {
            "case_id": case_id,
            "activity": activity,
            "enabled_time": enabled_time,
            "candidate_resources": self._unique_resources(
                self.act_to_resources.get(activity, ())
            ),
            "active": False,
            "version": int(version),
        }
        self._enabled_cases[case_id] = record
        self._diagnostics["enabled_case_registrations"] += 1

        enabled_key = self._time_key(enabled_time)
        if enabled_key is None:
            return
        if self._clock_key is not None and enabled_key <= self._clock_key:
            self._activate_queue_record(record)
        else:
            self._push_time_event(enabled_time, "enabled_activate", case_id, version)

    def remove_enabled_case(self, case_id):
        record = self._enabled_cases.pop(case_id, None)
        if record is None:
            return None
        self._deactivate_queue_record(record)
        return record

    # ------------------------------------------------------------------
    # Explicit Ready Queue state
    # ------------------------------------------------------------------
    def enqueue_ready_job(self, job: dict, current_time=None):
        """Register a real waiting job in incremental queue counts."""

        if not isinstance(job, dict):
            return
        if current_time is not None:
            self.advance_to(current_time)

        job_id = job.get("job_id")
        if job_id is None:
            job_id = (job.get("case_id"), job.get("activity"), job.get("queue_entry_time"))
            job["job_id"] = job_id

        # Defensive replacement: never double-count a reused job id.
        self.remove_ready_job(job_id)

        record = dict(job)
        version = self._next_version()
        record["_qsr_version"] = int(version)
        record["_qsr_active"] = False
        record["candidate_resources"] = list(
            self._unique_resources(record.get("candidate_resources") or ())
        )
        self.ready_jobs[job_id] = record
        self._ready_job_ids_by_case[record.get("case_id")].add(job_id)
        self._diagnostics["ready_job_enqueues"] += 1

        entry_time = record.get("queue_entry_time", record.get("enabled_time"))
        entry_key = self._time_key(entry_time)

        # In the simulator this method is invoked by the queue_entry event, so
        # it is normally active immediately.  The deferred branch preserves the
        # old get_ready_jobs(current_time) behavior for generic callers.
        if entry_key is None or self._clock_key is None or entry_key <= self._clock_key:
            self._activate_ready_record(record)
        else:
            self._push_time_event(entry_time, "ready_activate", job_id, version)

    def _activate_ready_record(self, record: dict):
        if record.get("_qsr_active", False):
            return
        self._queue_delta(
            str(record.get("activity", "")),
            record.get("candidate_resources") or (),
            +1,
        )
        self._wip_delta(+1)
        record["_qsr_active"] = True

    def remove_ready_job(self, job_id):
        record = self.ready_jobs.pop(job_id, None)
        if record is None:
            return None

        case_id = record.get("case_id")
        ids = self._ready_job_ids_by_case.get(case_id)
        if ids is not None:
            ids.discard(job_id)
            if not ids:
                self._ready_job_ids_by_case.pop(case_id, None)

        if record.get("_qsr_active", False):
            self._queue_delta(
                str(record.get("activity", "")),
                record.get("candidate_resources") or (),
                -1,
            )
            self._wip_delta(-1)
            record["_qsr_active"] = False

        self._diagnostics["ready_job_dequeues"] += 1
        return record

    def get_ready_jobs(self, current_time=None) -> list[dict]:
        if current_time is not None:
            self.advance_to(current_time)
        return [
            job for job in self.ready_jobs.values()
            if bool(job.get("_qsr_active", False))
        ]

    def get_eligible_jobs(self, resource, current_time=None) -> list[dict]:
        resource = str(resource)
        return [
            job for job in self.get_ready_jobs(current_time=current_time)
            if resource in set(str(r) for r in (job.get("candidate_resources") or ()))
        ]

    def ready_queue_size(self, current_time=None) -> int:
        if current_time is not None:
            self.advance_to(current_time)
        return int(max(0, self._global_queue_count))

    # ------------------------------------------------------------------
    # Legacy scheduled-job compatibility (incremental as well)
    # ------------------------------------------------------------------
    def register_scheduled_job(
            self,
            job_id,
            case_id,
            activity,
            resource,
            enabled_time,
            start_time,
            end_time,
            candidate_resources,
    ):
        """Register a legacy scheduled job without future full scans.

        It contributes queue + WIP during enabled <= t < start, then only WIP
        during start <= t < end.  The explicit Ready Queue simulator currently
        does not call this method, but keeping it incremental preserves the
        public runtime API.
        """

        old = self.scheduled_jobs.pop(job_id, None)
        if old is not None:
            self._remove_scheduled_contribution(old)

        version = self._next_version()
        record = {
            "case_id": case_id,
            "activity": str(activity),
            "resource": str(resource),
            "enabled_time": enabled_time,
            "start_time": start_time,
            "end_time": end_time,
            "candidate_resources": list(self._unique_resources(candidate_resources or ())),
            "phase": "pending",
            "version": int(version),
        }
        self.scheduled_jobs[job_id] = record

        enabled_key = self._time_key(enabled_time)
        start_key = self._time_key(start_time)
        end_key = self._time_key(end_time)
        clock = self._clock_key

        if end_key is not None and clock is not None and end_key <= clock:
            self.scheduled_jobs.pop(job_id, None)
            return
        if start_key is not None and clock is not None and start_key <= clock:
            self._wip_delta(+1)
            record["phase"] = "running"
        elif enabled_key is not None and clock is not None and enabled_key <= clock:
            self._queue_delta(record["activity"], record["candidate_resources"], +1)
            self._wip_delta(+1)
            record["phase"] = "waiting"

        if enabled_key is not None and (clock is None or enabled_key > clock):
            self._push_time_event(enabled_time, "scheduled_enabled", job_id, version)
        if start_key is not None and (clock is None or start_key > clock):
            self._push_time_event(start_time, "scheduled_start", job_id, version)
        if end_key is not None and (clock is None or end_key > clock):
            self._push_time_event(end_time, "scheduled_end", job_id, version)

    def _scheduled_to_waiting(self, record: dict):
        if record.get("phase") != "pending":
            return
        self._queue_delta(record["activity"], record["candidate_resources"], +1)
        self._wip_delta(+1)
        record["phase"] = "waiting"

    def _scheduled_to_running(self, record: dict):
        phase = record.get("phase")
        if phase == "pending":
            # Defensive handling for malformed times where the enabled event was
            # absent or ordered after start.
            self._wip_delta(+1)
            record["phase"] = "running"
        elif phase == "waiting":
            self._queue_delta(record["activity"], record["candidate_resources"], -1)
            record["phase"] = "running"

    def _scheduled_to_completed(self, job_id, record: dict):
        self._remove_scheduled_contribution(record)
        record["phase"] = "completed"
        self.scheduled_jobs.pop(job_id, None)

    def _remove_scheduled_contribution(self, record: dict):
        phase = record.get("phase")
        if phase == "waiting":
            self._queue_delta(record["activity"], record["candidate_resources"], -1)
            self._wip_delta(-1)
        elif phase == "running":
            self._wip_delta(-1)
        record["phase"] = "completed"

    def prune(self, current_time):
        # Kept for compatibility.  Heap events now perform the pruning.
        self.advance_to(current_time)

    # ------------------------------------------------------------------
    # O(1) feature query
    # ------------------------------------------------------------------
    def _subtract_record_from_raw(
            self,
            raw: dict,
            record: dict,
            query_activity: str,
            query_resource: str,
    ):
        activity = str(record.get("activity", ""))
        resources = set(str(r) for r in (record.get("candidate_resources") or ()))

        raw["global_queue_length"] -= 1
        raw["system_wip"] -= 1
        if activity == query_activity:
            raw["activity_queue_length"] -= 1
        if query_resource in resources:
            raw["resource_queue_length"] -= 1
        if activity == query_activity and query_resource in resources:
            raw["resource_activity_queue_length"] -= 1

    def compute_features(
            self,
            cases,
            completed_cases,
            current_case_id,
            current_time,
            activity,
            resource,
            candidate_resources,
    ) -> dict:
        """Return queue features from incremental counters.

        ``cases`` and ``completed_cases`` remain in the signature for backward
        compatibility with QueueSchedulingRuntime, but are deliberately not
        scanned.  ``candidate_resources`` is also retained for API compatibility.
        """

        del cases, completed_cases, candidate_resources
        self.advance_to(current_time)
        self._diagnostics["feature_queries"] += 1

        activity = str(activity)
        resource = str(resource)

        raw = {
            "global_queue_length": int(self._global_queue_count),
            "activity_queue_length": int(self._activity_queue_counts.get(activity, 0)),
            "resource_queue_length": int(self._resource_queue_counts.get(resource, 0)),
            "resource_activity_queue_length": int(
                self._resource_activity_queue_counts.get((resource, activity), 0)
            ),
            "system_wip": int(self._system_wip_count),
        }

        # Preserve the old rule: the case currently making a resource decision
        # must not count its own enabled/ready representation in queue features.
        enabled_record = self._enabled_cases.get(current_case_id)
        if enabled_record is not None and enabled_record.get("active", False):
            self._subtract_record_from_raw(raw, enabled_record, activity, resource)

        for job_id in tuple(self._ready_job_ids_by_case.get(current_case_id, ())):
            record = self.ready_jobs.get(job_id)
            if record is not None and record.get("_qsr_active", False):
                self._subtract_record_from_raw(raw, record, activity, resource)

        for key in (
            "global_queue_length",
            "activity_queue_length",
            "resource_queue_length",
            "resource_activity_queue_length",
            "system_wip",
        ):
            raw[key] = max(0, int(raw.get(key, 0)))

        raw["queue_pressure"] = max(
            raw["activity_queue_length"],
            raw["resource_queue_length"],
            raw["resource_activity_queue_length"],
        )
        return self._post_process_queue_features(raw)

    # ------------------------------------------------------------------
    # Feature post-processing (unchanged semantics)
    # ------------------------------------------------------------------
    def zero_features(self) -> dict:
        return {
            "global_queue_length": 0.0,
            "activity_queue_length": 0.0,
            "resource_queue_length": 0.0,
            "resource_activity_queue_length": 0.0,
            "queue_pressure": 0.0,
            "system_wip": 0.0,
            "raw_global_queue_length": 0.0,
            "raw_activity_queue_length": 0.0,
            "raw_resource_queue_length": 0.0,
            "raw_resource_activity_queue_length": 0.0,
            "raw_system_wip": 0.0,
        }

    @staticmethod
    def _compress_count(value, max_count: float) -> float:
        try:
            value = float(value)
        except Exception:
            value = 0.0
        value = max(0.0, min(value, float(max_count)))
        if max_count <= 0:
            return 0.0
        return float(np.log1p(value) / np.log1p(float(max_count)))

    def _post_process_queue_features(self, raw_features: dict) -> dict:
        if self.disable_global_queue_features:
            global_queue_value = 0.0
            system_wip_value = 0.0
        else:
            global_queue_value = self._compress_count(
                raw_features.get("global_queue_length", 0.0), self.max_queue_count
            )
            system_wip_value = self._compress_count(
                raw_features.get("system_wip", 0.0), self.max_wip_count
            )

        activity_q = self._compress_count(
            raw_features.get("activity_queue_length", 0.0), self.max_queue_count
        )
        resource_q = self._compress_count(
            raw_features.get("resource_queue_length", 0.0), self.max_queue_count
        )
        resource_activity_q = self._compress_count(
            raw_features.get("resource_activity_queue_length", 0.0), self.max_queue_count
        )

        queue_pressure = max(activity_q, resource_q, resource_activity_q)

        return {
            "global_queue_length": global_queue_value,
            "activity_queue_length": activity_q,
            "resource_queue_length": resource_q,
            "resource_activity_queue_length": resource_activity_q,
            "queue_pressure": queue_pressure,
            "system_wip": system_wip_value,
            "raw_global_queue_length": float(raw_features.get("global_queue_length", 0.0)),
            "raw_activity_queue_length": float(raw_features.get("activity_queue_length", 0.0)),
            "raw_resource_queue_length": float(raw_features.get("resource_queue_length", 0.0)),
            "raw_resource_activity_queue_length": float(
                raw_features.get("resource_activity_queue_length", 0.0)
            ),
            "raw_system_wip": float(raw_features.get("system_wip", 0.0)),
        }
