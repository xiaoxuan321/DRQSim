# drq/runtime/external_delay_runtime.py

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

import numpy as np

from drqsim.discovery.external_delay_discovery import RESOURCE_PREFIX
from drqsim.discovery.queue_scheduling_discovery import ACTIVITY_PREFIX, QUEUE_FEATURE_NAMES
from drqsim.discovery.base_service_time_discovery import deserialize_distribution_container
from drqsim.utils.distribution_utils import sampling_from_dist


class ExternalDelayRuntime:
    """
    Runtime sampler for the activity-specific external timer-delay model.

    Queue scheduling is responsible for resource-competition waiting.  This
    runtime module samples only the non-resource residual start delay.  When
    ``runtime.residual_guard_in_runtime`` is enabled, the sampled residual is
    additionally discounted by already generated queue waiting.  In the current
    integrated simulator this guard is normally applied in ``simulator.py`` to
    avoid double application, so the internal guard defaults to False.
    """

    def __init__(
            self,
            external_delay_params: dict,
            resources: list,
            activities: list,
    ):
        self.params = external_delay_params or {"activities": {}}
        self.resources = [str(r) for r in (resources or [])]
        self.activities = [str(a) for a in (activities or [])]
        if isinstance(self.params, dict):
            runtime_cfg = self.params.setdefault("runtime", {})
            if isinstance(runtime_cfg, dict):
                runtime_cfg.setdefault("calendar_aware_timer", False)
                runtime_cfg.setdefault("calendar_aware_mode", "elapsed_time")
                # Calendar-aware alignment is only a small correction.  Long-tail
                # business delays are elapsed external waiting and should not be
                # moved by sparse resource calendars.
                runtime_cfg.setdefault("max_calendar_aware_delay_min", 1440.0)
                runtime_cfg.setdefault("max_calendar_snap_shift_min", 240.0)
                runtime_cfg.setdefault("min_calendar_active_hours", 12)
                # Generic safe empirical-timer policy.  Empirical activity
                # samples already come from the training log, so do not multiply
                # them directly by global_scale.  Long tails are sampled by a
                # body/tail mixture to keep rare waits rare.
                runtime_cfg.setdefault("empirical_global_scale_mode", "none")
                runtime_cfg.setdefault("empirical_global_scale_cap", 1.5)
                runtime_cfg.setdefault("empirical_body_tail_mixture", True)
                runtime_cfg.setdefault("empirical_tail_quantile", 0.95)
                runtime_cfg.setdefault("empirical_tail_probability", 0.05)
                # Activity-specific empirical timers require enough support.
                # Sparse activities use the pooled capped global distribution.
                runtime_cfg.setdefault("min_activity_empirical_samples", 4)

    def sample_delay(
            self,
            activity: str,
            resource: str,
            enabled_time,
            queue_start_time,
            case: dict,
            scheduling_result: dict | None = None,
            deterministic_time: bool = False,
    ) -> float:
        if not isinstance(self.params, dict):
            return 0.0

        runtime_cfg = self.params.get("runtime", {}) or {}
        if not bool(runtime_cfg.get("add_delay_to_first_activity", False)):
            try:
                if int(sum(case.get("history", {}).values())) == 0:
                    return 0.0
            except Exception:
                pass

        act_model = (
            self.params
            .get("activities", {})
            .get(str(activity))
        )
        if not isinstance(act_model, dict):
            self._record_activity_decision(
                scheduling_result,
                activity=activity,
                enabled=False,
                reason="missing_activity_model",
                delay_min=0.0,
            )
            return 0.0

        if not self._is_activity_timer_enabled(act_model):
            self._record_activity_decision(
                scheduling_result,
                activity=activity,
                enabled=False,
                reason=str(act_model.get("disabled_reason") or "activity_timer_disabled"),
                delay_min=0.0,
            )
            return 0.0

        scheduling_result = scheduling_result or {}
        features = self._build_features(
            activity=activity,
            resource=resource,
            enabled_time=enabled_time,
            case=case,
            scheduling_result=scheduling_result,
        )

        # ``global_scale`` is kept only as a weak runtime calibration knob.
        # The main model is activity-specific: each enabled activity has its own
        # empirical timer samples/distribution, and can optionally define an
        # activity-level scale.
        global_scale = self._safe_float(runtime_cfg.get("global_scale", 1.0), 1.0)
        activity_runtime_cfg = act_model.get("runtime", {}) if isinstance(act_model.get("runtime", {}), dict) else {}
        activity_scale = self._safe_float(activity_runtime_cfg.get("activity_scale", act_model.get("activity_scale", 1.0)), 1.0)

        # Prefer activity empirical samples for low-frequency/non-tree timers.
        # This preserves rare long-tail waits instead of replacing them with a
        # fitted/global short distribution.  Decision-tree models still use the
        # tree first and keep empirical samples only as fallback metadata.
        # ``None`` at activity level means "inherit the global runtime value".
        # Do not pass None into _safe_float(..., 1.0): that would silently
        # turn a selected global quantile such as 0.90 back into 1.0 and reopen
        # the full long tail for every activity.  The same inheritance rule is
        # applied when runtime.max_delay_quantile itself is None.
        activity_cap_quantile = activity_runtime_cfg.get("max_delay_quantile")
        if activity_cap_quantile is not None:
            cap_quantile_source = "activity_override"
        else:
            activity_cap_quantile = runtime_cfg.get("max_delay_quantile")
            if activity_cap_quantile is not None:
                cap_quantile_source = "global_runtime"
            else:
                activity_cap_quantile = self.params.get("max_delay_quantile", 1.0)
                cap_quantile_source = "global_model"

        cap_quantile = float(np.clip(
            self._safe_float(activity_cap_quantile, 1.0),
            0.0,
            1.0,
        ))

        empirical_first = str(act_model.get("model_kind", "")) in {
            "empirical_activity_distribution",
            "activity_empirical_distribution",
            "activity_distribution",
            "global_empirical_fallback",
        }
        empirical_delay = None
        if empirical_first:
            empirical_delay = self._sample_empirical_delay_from_model(
                act_model=act_model,
                deterministic_time=deterministic_time,
                max_delay_quantile=cap_quantile,
            )

        empirical_global_scale = 1.0
        if empirical_delay is not None:
            delay = float(empirical_delay)
            delay_source = "activity_empirical_body_tail"
            empirical_global_scale = self._empirical_global_scale(
                global_scale=global_scale,
                runtime_cfg=runtime_cfg,
            )
            # Empirical activity timers are observed elapsed waits.  Do not
            # multiply them directly by global_scale; at most apply the
            # configured conservative empirical scale policy.
            delay_scale = max(0.0, empirical_global_scale) * max(0.0, activity_scale)
        else:
            dist_tuple = self._select_distribution(act_model, features)
            delay = self._sample_distribution(dist_tuple, deterministic_time=deterministic_time)
            delay_source = "tree_or_distribution"
            delay_scale = max(0.0, global_scale) * max(0.0, activity_scale)

        delay = max(0.0, delay * delay_scale)

        # Cap by the activity-level quantile when available. With quantile=1.0
        # this keeps the observed activity max, so the long tail is not truncated.

        cap = self._get_activity_delay_cap(
            act_model=act_model,
            cap_quantile=cap_quantile,
        )

        if cap is not None and np.isfinite(cap) and cap > 0:
            delay = min(delay, cap)

        raw_delay = max(0.0, float(delay))
        try:
            scheduling_result["raw_external_delay_min"] = float(raw_delay)
        except Exception:
            pass

        # Optional internal residual guard.  It defaults to False because the
        # integrated simulator applies the same guard after queue_wait_min is
        # measured.  Keep this switch for standalone usage of ExternalDelayRuntime.
        if bool(runtime_cfg.get("residual_guard_in_runtime", False)):
            queue_wait_min = self._queue_wait_from_context(
                enabled_time=enabled_time,
                queue_start_time=queue_start_time,
                scheduling_result=scheduling_result,
            )
            delay = self._apply_queue_residual_guard(
                delay_min=delay,
                queue_wait_min=queue_wait_min,
                runtime_cfg=runtime_cfg,
            )
            try:
                scheduling_result["external_delay_runtime_residual_guard_applied"] = True
                scheduling_result["adjusted_external_delay_min"] = float(delay)
            except Exception:
                pass

        delay = max(0.0, float(delay))

        # Calendar-aware timer:
        # The sampled timer represents an external business delay, but CED/CWD
        # are sensitive to the weekday-hour where the delayed activity starts.
        # Instead of blindly adding minutes, align the delayed start to the
        # selected resource calendar when it is available.  The method returns
        # an effective elapsed delay in minutes, so existing simulator code that
        # does ``queue_start_time + delay_min`` does not need to change.
        calendar_aware_enabled = bool(runtime_cfg.get("calendar_aware_timer", False))
        calendar_aware_mode = str(runtime_cfg.get("calendar_aware_mode", "elapsed_time"))
        calendar_aware_adjusted = False
        calendar_aware_target_time = None
        calendar_aware_raw_delay = float(delay)

        if calendar_aware_enabled and delay > 0.0:
            resource_calendar = {}
            if isinstance(scheduling_result, dict):
                resource_calendar = (
                    scheduling_result.get("selected_resource_calendar")
                    or scheduling_result.get("resource_calendar")
                    or {}
                )

            if isinstance(resource_calendar, dict) and resource_calendar:
                try:
                    calendar_aware_target_time = self.add_delay_to_start_time(
                        queue_start_time,
                        delay,
                        calendar=resource_calendar,
                        calendar_aware=True,
                        mode=calendar_aware_mode,
                    )
                    adjusted_delay = max(
                        0.0,
                        float((calendar_aware_target_time - queue_start_time).total_seconds() / 60.0),
                    )
                    if np.isfinite(adjusted_delay):
                        delay = adjusted_delay
                        calendar_aware_adjusted = True
                        scheduling_result["external_delay_calendar_aware_enabled"] = True
                        scheduling_result["external_delay_calendar_aware_mode"] = calendar_aware_mode
                        scheduling_result["external_delay_calendar_aware_target_time"] = calendar_aware_target_time
                        scheduling_result["external_delay_calendar_aware_raw_delay_min"] = calendar_aware_raw_delay
                        scheduling_result["external_delay_calendar_aware_effective_delay_min"] = float(delay)
                except Exception:
                    calendar_aware_adjusted = False

        self._record_activity_decision(
            scheduling_result,
            activity=activity,
            enabled=True,
            reason=str(act_model.get("selection_reason") or act_model.get("significance", {}).get("reason") or "activity_timer_enabled"),
            delay_min=delay,
            raw_delay_min=raw_delay,
            model_kind=str(act_model.get("model_kind", "activity_distribution")),
            delay_source=delay_source,
            activity_scale=activity_scale,
            global_scale=global_scale,
            empirical_global_scale=empirical_global_scale,
            cap_quantile=cap_quantile,
            cap_quantile_source=cap_quantile_source,
            cap_min=cap,
            calendar_aware_enabled=calendar_aware_enabled,
            calendar_aware_mode=calendar_aware_mode,
            calendar_aware_adjusted=calendar_aware_adjusted,
            calendar_aware_raw_delay_min=calendar_aware_raw_delay,
        )
        return delay

    def _is_activity_timer_enabled(self, act_model: dict) -> bool:
        if not isinstance(act_model, dict):
            return False
        if "enabled" in act_model:
            return bool(act_model.get("enabled"))
        if "external_delay_enabled" in act_model:
            return bool(act_model.get("external_delay_enabled"))
        if "timer_delay_enabled" in act_model:
            return bool(act_model.get("timer_delay_enabled"))
        # Legacy activity-level models did not carry an explicit enabled flag;
        # keep them enabled for backward compatibility.
        return True

    def _record_activity_decision(
            self,
            scheduling_result: dict | None,
            activity: str,
            enabled: bool,
            reason: str,
            delay_min: float = 0.0,
            **extra,
    ):
        if scheduling_result is None:
            return
        try:
            row = {
                "activity": str(activity),
                "enabled": bool(enabled),
                "reason": str(reason),
                "delay_min": float(delay_min or 0.0),
            }
            for key, value in extra.items():
                try:
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        row[str(key)] = float(value)
                    else:
                        row[str(key)] = value
                except Exception:
                    row[str(key)] = str(value)
            scheduling_result["external_delay_activity_timer"] = row
            scheduling_result["external_delay_activity_enabled"] = bool(enabled)
            scheduling_result["external_delay_activity_reason"] = str(reason)
            scheduling_result["external_delay_activity_delay_min"] = float(delay_min or 0.0)
        except Exception:
            pass

    def _queue_wait_from_context(self, enabled_time, queue_start_time, scheduling_result: dict) -> float:
        for key in ("queue_wait_min", "waiting_time_min", "candidate_waiting_time_min"):
            value = scheduling_result.get(key)
            try:
                value = float(value)
                if np.isfinite(value) and value >= 0.0:
                    return float(value)
            except Exception:
                pass
        try:
            return max(0.0, float((queue_start_time - enabled_time).total_seconds() / 60.0))
        except Exception:
            return 0.0

    def _apply_queue_residual_guard(self, delay_min: float, queue_wait_min: float, runtime_cfg: dict) -> float:
        try:
            delay = max(0.0, float(delay_min))
        except Exception:
            delay = 0.0
        try:
            queue_wait = max(0.0, float(queue_wait_min))
        except Exception:
            queue_wait = 0.0

        if delay <= 0.0 or queue_wait <= 0.0:
            return delay
        if not bool(runtime_cfg.get("residual_after_queue_only", True)):
            return delay

        credit_ratio = self._safe_float(runtime_cfg.get("queue_credit_ratio", 0.70), 0.70)
        credit_ratio = float(np.clip(credit_ratio, 0.0, 1.0))
        residual_decay = self._safe_float(runtime_cfg.get("queue_residual_decay", 0.30), 0.30)
        residual_decay = float(np.clip(residual_decay, 0.0, 1.0))

        delay = max(0.0, delay - credit_ratio * queue_wait)
        delay *= residual_decay

        max_ratio = runtime_cfg.get("max_external_delay_ratio", None)
        if max_ratio is not None:
            try:
                max_ratio = float(max_ratio)
                if np.isfinite(max_ratio) and 0.0 < max_ratio < 1.0:
                    max_allowed = (max_ratio / max(1.0 - max_ratio, 1e-9)) * queue_wait
                    delay = min(delay, max_allowed)
            except Exception:
                pass

        return max(0.0, float(delay))

    def _get_activity_delay_cap(self, act_model: dict, cap_quantile: float):
        """
        Return runtime cap for external delay according to selected quantile.

        Preferred source:
            act_model["activity_delay_quantiles"]["0.95" / "0.98" / "1.00"]

        Fallback:
            activity_default_p95
            distribution max_value
        """

        try:
            cap_quantile = float(cap_quantile)
        except Exception:
            cap_quantile = 0.95

        cap_quantile = max(0.0, min(1.0, cap_quantile))

        q_dict = act_model.get("activity_delay_quantiles", {})
        if isinstance(q_dict, dict) and q_dict:
            candidate_keys = [
                f"{cap_quantile:.2f}",
                str(round(cap_quantile, 2)),
                str(cap_quantile),
            ]

            for key in candidate_keys:
                if key in q_dict:
                    try:
                        value = float(q_dict[key])
                        if np.isfinite(value) and value >= 0:
                            return value
                    except Exception:
                        pass

            parsed_items = []
            for key, value in q_dict.items():
                try:
                    parsed_items.append((float(key), float(value)))
                except Exception:
                    continue

            if parsed_items:
                nearest_q, nearest_v = min(
                    parsed_items,
                    key=lambda item: abs(item[0] - cap_quantile),
                )
                if np.isfinite(nearest_v) and nearest_v >= 0:
                    return nearest_v

        try:
            value = float(act_model.get("activity_default_p95"))
            if np.isfinite(value) and value >= 0:
                return value
        except Exception:
            pass

        try:
            default = self._distribution_from_obj(act_model.get("activity_default"))
            value = float(default[3]) if len(default) >= 4 else None
            if value is not None and np.isfinite(value) and value >= 0:
                return value
        except Exception:
            pass

        return None

    def add_delay_to_start_time(
            self,
            start_time,
            delay_min: float,
            calendar: dict | None = None,
            calendar_aware: bool | None = None,
            mode: str | None = None,
    ):
        """Add external delay to a timestamp with bounded calendar alignment.

        Long-tail external delays are treated as elapsed business waiting and
        are not snapped by a sparse resource calendar.  Calendar-aware snapping
        is only used for short delays and only when the snap shift is small.
        """
        delay_min = max(0.0, float(delay_min or 0.0))
        if delay_min <= 0.0:
            return start_time

        raw_target = start_time + timedelta(minutes=int(round(delay_min)))

        if calendar_aware is None:
            try:
                runtime_cfg = self.params.get("runtime", {}) or {}
                calendar_aware = bool(runtime_cfg.get("calendar_aware_timer", False))
            except Exception:
                calendar_aware = False

        if not calendar_aware or not isinstance(calendar, dict) or not calendar:
            return raw_target

        runtime_cfg = self.params.get("runtime", {}) or {}
        max_calendar_aware_delay_min = self._safe_float(
            runtime_cfg.get("max_calendar_aware_delay_min", 1440.0),
            1440.0,
        )
        max_calendar_snap_shift_min = self._safe_float(
            runtime_cfg.get("max_calendar_snap_shift_min", 240.0),
            240.0,
        )
        min_calendar_active_hours = int(self._safe_float(
            runtime_cfg.get("min_calendar_active_hours", 12),
            12.0,
        ))

        # Do not snap long-tail business delays.  They often represent elapsed
        # waiting for documents, approvals, or external parties, not a resource
        # working-time delay.
        if delay_min > max_calendar_aware_delay_min:
            return raw_target

        active_hours = self._active_calendar_hour_count(calendar)
        if active_hours > 0 and active_hours < min_calendar_active_hours:
            return raw_target

        mode = str(mode or runtime_cfg.get("calendar_aware_mode", "elapsed_time"))

        if mode == "business_time":
            candidate = self._add_business_calendar_minutes(
                start_time=start_time,
                delay_min=delay_min,
                calendar=calendar,
                max_elapsed_shift_min=max_calendar_snap_shift_min,
            )
        else:
            candidate = self._snap_to_next_calendar_time(
                raw_target,
                calendar,
                max_shift_min=max_calendar_snap_shift_min,
            )

        try:
            shift_min = abs((candidate - raw_target).total_seconds()) / 60.0
            if shift_min <= max_calendar_snap_shift_min:
                return candidate
        except Exception:
            pass

        return raw_target

    def _is_calendar_active(self, ts, calendar: dict) -> bool:
        try:
            return self._calendar_is_active_hour(calendar, ts.weekday(), ts.hour, default=False)
        except Exception:
            return True

    def _snap_to_next_calendar_time(self, target_time, calendar: dict, max_shift_min: float | None = None):
        if not isinstance(calendar, dict) or not calendar:
            return target_time

        if self._is_calendar_active(target_time, calendar):
            return target_time

        if max_shift_min is None:
            max_shift_min = self._safe_float(
                (self.params.get("runtime", {}) or {}).get("max_calendar_snap_shift_min", 240.0),
                240.0,
            )
        max_shift_hours = max(0, int(np.ceil(max(0.0, float(max_shift_min)) / 60.0)))
        if max_shift_hours <= 0:
            return target_time

        probe = target_time
        for _ in range(max_shift_hours):
            probe = (probe + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            if self._is_calendar_active(probe, calendar):
                return probe

        return target_time

    def _add_business_calendar_minutes(
            self,
            start_time,
            delay_min: float,
            calendar: dict,
            max_elapsed_shift_min: float | None = None,
    ):
        if not isinstance(calendar, dict) or not calendar:
            return start_time + timedelta(minutes=int(round(delay_min)))

        raw_target = start_time + timedelta(minutes=int(round(delay_min)))
        remaining = max(0.0, float(delay_min))
        probe = self._snap_to_next_calendar_time(start_time, calendar, max_shift_min=max_elapsed_shift_min)

        # Step at hour granularity for speed and determinism.  The last partial
        # hour is added exactly.  If the elapsed shift becomes too large, fall
        # back to raw elapsed time.
        max_steps = 24 * 14
        steps = 0
        while remaining > 0 and steps < max_steps:
            steps += 1
            if not self._is_calendar_active(probe, calendar):
                probe = self._snap_to_next_calendar_time(probe, calendar, max_shift_min=max_elapsed_shift_min)
                if max_elapsed_shift_min is not None:
                    try:
                        if abs((probe - raw_target).total_seconds()) / 60.0 > float(max_elapsed_shift_min):
                            return raw_target
                    except Exception:
                        return raw_target
                continue

            minutes_to_next_hour = max(1.0, 60.0 - float(probe.minute))
            consume = min(remaining, minutes_to_next_hour)
            probe = probe + timedelta(minutes=consume)
            remaining -= consume

            if remaining <= 1e-9:
                break

            if probe.minute == 0 and not self._is_calendar_active(probe, calendar):
                probe = self._snap_to_next_calendar_time(probe, calendar, max_shift_min=max_elapsed_shift_min)

            if max_elapsed_shift_min is not None:
                try:
                    if abs((probe - raw_target).total_seconds()) / 60.0 > float(max_elapsed_shift_min):
                        return raw_target
                except Exception:
                    return raw_target

        return probe

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

    def _active_calendar_hour_count(self, calendar: dict) -> int:
        if not isinstance(calendar, dict) or not calendar:
            return 0
        active = 0
        for wd in range(7):
            for hour in range(24):
                if self._calendar_is_active_hour(calendar, wd, hour, default=False):
                    active += 1
        return int(active)

    def _sample_global_empirical_delay(
            self,
            deterministic_time: bool = False,
            max_delay_quantile: float = 1.0,
    ) -> float | None:
        """Sample from the pooled capped external-delay distribution.

        This is the shrinkage fallback for activities with too few observations
        to support their own empirical timer.  New discovery output stores the
        pooled samples at the top level.  For older parameter files, fall back
        conservatively to the global median quantile when available.
        """
        values = []
        if isinstance(self.params, dict):
            values = self.params.get("global_empirical_delay_values") or []

        arr = []
        if isinstance(values, (list, tuple)):
            for value in values:
                try:
                    value = float(value)
                    if np.isfinite(value) and value >= 0.0:
                        arr.append(float(value))
                except Exception:
                    continue

        if arr:
            arr_np = np.asarray(arr, dtype=float)
            try:
                q = float(np.clip(float(max_delay_quantile), 0.0, 1.0))
            except Exception:
                q = 1.0
            if q < 1.0 and arr_np.size > 0:
                try:
                    cap = float(np.quantile(arr_np, q))
                    if np.isfinite(cap):
                        arr_np = np.minimum(arr_np, cap)
                except Exception:
                    pass

            if arr_np.size > 0:
                if deterministic_time:
                    return float(np.mean(arr_np))
                try:
                    idx = int(np.random.randint(0, arr_np.size))
                    return float(arr_np[idx])
                except Exception:
                    return float(np.mean(arr_np))

        # Backward-compatible fallback for older discovered parameter files that
        # do not contain pooled empirical samples.  Use the global median rather
        # than replaying an unsupported singleton activity outlier.
        metadata = self.params.get("metadata", {}) if isinstance(self.params, dict) else {}
        quantiles = metadata.get("global_delay_quantiles", {}) if isinstance(metadata, dict) else {}
        if isinstance(quantiles, dict):
            for key in ("0.50", "0.5", "0.75", "0.90"):
                if key in quantiles:
                    try:
                        value = float(quantiles.get(key))
                        if np.isfinite(value) and value >= 0.0:
                            return float(value)
                    except Exception:
                        continue
        return None

    def _sample_empirical_delay_from_model(
            self,
            act_model: dict,
            deterministic_time: bool = False,
            max_delay_quantile: float = 1.0,
    ) -> float | None:
        """Sample from activity empirical delay values when available.

        Low-frequency activities use this path instead of a decision tree.  The
        sampler uses a body/tail mixture: most draws come from the body of the
        empirical distribution, while rare draws come from its tail.  This keeps
        long waits possible without making rare long-tail samples too frequent.
        """
        if not isinstance(act_model, dict):
            return None

        runtime_cfg = self.params.get("runtime", {}) if isinstance(self.params, dict) else {}
        activity_runtime_cfg = act_model.get("runtime", {}) if isinstance(act_model.get("runtime", {}), dict) else {}
        try:
            min_activity_samples = int(activity_runtime_cfg.get(
                "min_activity_empirical_samples",
                runtime_cfg.get("min_activity_empirical_samples", 4),
            ))
        except Exception:
            min_activity_samples = 4
        min_activity_samples = max(1, min_activity_samples)

        try:
            observed_sample_count = int(act_model.get(
                "positive_sample_count",
                act_model.get(
                    "n_samples_raw",
                    act_model.get("empirical_delay_total_count", 0),
                ),
            ) or 0)
        except Exception:
            observed_sample_count = 0

        if 0 < observed_sample_count < min_activity_samples:
            return self._sample_global_empirical_delay(
                deterministic_time=deterministic_time,
                max_delay_quantile=max_delay_quantile,
            )

        values = (
            act_model.get("empirical_delay_values")
            or act_model.get("empirical_values")
            or act_model.get("activity_empirical_values")
            or []
        )
        if not isinstance(values, (list, tuple)) or not values:
            if str(act_model.get("model_kind", "")) == "global_empirical_fallback":
                return self._sample_global_empirical_delay(
                    deterministic_time=deterministic_time,
                    max_delay_quantile=max_delay_quantile,
                )
            return None

        arr = []
        for value in values:
            try:
                value = float(value)
                if np.isfinite(value) and value >= 0.0:
                    arr.append(float(value))
            except Exception:
                continue

        if not arr:
            return None

        arr_np = np.asarray(arr, dtype=float)
        arr_np = arr_np[np.isfinite(arr_np)]
        arr_np = arr_np[arr_np >= 0.0]
        if arr_np.size == 0:
            return None

        try:
            q = float(max_delay_quantile)
        except Exception:
            q = 1.0
        q = float(np.clip(q, 0.0, 1.0))

        if q < 1.0:
            try:
                cap = float(np.quantile(arr_np, q))
                if np.isfinite(cap):
                    clipped = arr_np[arr_np <= cap]
                    if clipped.size > 0:
                        arr_np = clipped
            except Exception:
                pass

        if arr_np.size == 0:
            return None

        mixture_enabled = bool(activity_runtime_cfg.get(
            "empirical_body_tail_mixture",
            runtime_cfg.get("empirical_body_tail_mixture", True),
        ))
        try:
            tail_quantile = float(activity_runtime_cfg.get(
                "empirical_tail_quantile",
                runtime_cfg.get("empirical_tail_quantile", 0.95),
            ))
        except Exception:
            tail_quantile = 0.95
        tail_quantile = float(np.clip(tail_quantile, 0.50, 0.999))

        try:
            tail_probability = float(activity_runtime_cfg.get(
                "empirical_tail_probability",
                runtime_cfg.get("empirical_tail_probability", 0.05),
            ))
        except Exception:
            tail_probability = 0.05
        tail_probability = float(np.clip(tail_probability, 0.0, 0.50))

        if arr_np.size < min_activity_samples:
            global_delay = self._sample_global_empirical_delay(
                deterministic_time=deterministic_time,
                max_delay_quantile=max_delay_quantile,
            )
            if global_delay is not None:
                return float(global_delay)
            # Last-resort compatibility fallback only when no pooled/global
            # information exists at all.  Use the local mean, never randomly
            # replay a singleton outlier.
            return float(np.mean(arr_np))

        if not mixture_enabled:
            if deterministic_time:
                return float(np.mean(arr_np))
            try:
                idx = int(np.random.randint(0, arr_np.size))
                return float(arr_np[idx])
            except Exception:
                return float(np.mean(arr_np))

        try:
            tail_threshold = float(np.quantile(arr_np, tail_quantile))
        except Exception:
            tail_threshold = float(np.max(arr_np))

        body = arr_np[arr_np <= tail_threshold]
        tail = arr_np[arr_np > tail_threshold]

        if body.size == 0:
            body = arr_np
        if tail.size == 0:
            tail = arr_np[arr_np >= tail_threshold]
        if tail.size == 0:
            tail = body

        if deterministic_time:
            body_mean = float(np.mean(body)) if body.size else float(np.mean(arr_np))
            tail_mean = float(np.mean(tail)) if tail.size else body_mean
            return float((1.0 - tail_probability) * body_mean + tail_probability * tail_mean)

        try:
            use_tail = bool(np.random.random() < tail_probability)
            source = tail if use_tail and tail.size > 0 else body
            idx = int(np.random.randint(0, source.size))
            return float(source[idx])
        except Exception:
            return float(np.mean(arr_np))

    def _empirical_global_scale(self, global_scale: float, runtime_cfg: dict) -> float:
        """Return conservative scale for empirical activity timers.

        Empirical samples are observed delays.  The default policy is "none", so
        runtime global_scale does not amplify long-tail samples.  Other modes are
        kept only for controlled what-if experiments.
        """
        try:
            gs = max(0.0, float(global_scale))
        except Exception:
            gs = 1.0

        mode = str((runtime_cfg or {}).get("empirical_global_scale_mode", "none")).lower()
        try:
            cap = max(0.0, float((runtime_cfg or {}).get("empirical_global_scale_cap", 1.5)))
        except Exception:
            cap = 1.5

        if mode in {"none", "off", "disabled", "no_scale"}:
            return 1.0
        if mode in {"sqrt", "sqrt_capped", "weak"}:
            return float(min(cap, np.sqrt(gs))) if cap > 0 else float(np.sqrt(gs))
        if mode in {"linear_capped", "capped"}:
            return float(min(cap, gs)) if cap > 0 else gs
        return 1.0

    def _select_distribution(self, act_model: dict, features: Dict[str, Any]):
        if not self._is_activity_timer_enabled(act_model):
            return "fixed", (0.0,), 0.0, 0.0, 0.0
        if act_model.get("model_kind") == "decision_tree_distribution" and act_model.get("rules"):
            leaf = self._traverse_rules(act_model.get("rules"), features)
            if isinstance(leaf, dict) and "dist" in leaf:
                return self._distribution_from_obj(leaf.get("dist"))
        return self._distribution_from_obj(
            act_model.get("timer_delay_distribution", act_model.get("activity_default"))
        )

    def _traverse_rules(self, rules: dict, features: Dict[str, Any]):
        node = rules
        while isinstance(node, dict) and node.get("type") != "leaf":
            feature = node.get("feature")
            threshold = self._safe_float(node.get("threshold", 0.0), 0.0)
            value = self._safe_float(features.get(feature, 0.0), 0.0)
            if value <= threshold:
                node = node.get("children", {}).get("true")
            else:
                node = node.get("children", {}).get("false")
            if node is None:
                break
        return node

    def _distribution_from_obj(self, obj):
        obj = deserialize_distribution_container(obj)
        if isinstance(obj, tuple) and len(obj) >= 5:
            return obj
        if isinstance(obj, tuple) and len(obj) >= 4:
            return obj[0], obj[1], obj[2], obj[3], obj[2]
        return "fixed", (0.0,), 0.0, 0.0, 0.0

    def _sample_distribution(self, dist_tuple, deterministic_time: bool) -> float:
        try:
            if deterministic_time:
                return max(0.0, float(dist_tuple[4]))
            return max(0.0, float(sampling_from_dist(
                dist_tuple[0],
                dist_tuple[1],
                dist_tuple[2],
                dist_tuple[3],
                dist_tuple[4],
                n_sample=1,
            )[0]))
        except Exception:
            try:
                return max(0.0, float(dist_tuple[4]))
            except Exception:
                return 0.0

    def _build_features(
            self,
            activity: str,
            resource: str,
            enabled_time,
            case: dict,
            scheduling_result: dict,
    ) -> Dict[str, Any]:
        features = {}
        selected_features = scheduling_result.get("selected_features", {}) or {}
        features.update(selected_features)
        features.update(case.get("history", {}))
        features.update(case.get("attributes", {}))

        for activity_name in self.activities:
            features[f"{ACTIVITY_PREFIX}{activity_name}"] = int(str(activity_name) == str(activity))

        for resource_name in self.resources:
            features[f"{RESOURCE_PREFIX}{resource_name}"] = int(str(resource_name) == str(resource))
            features.setdefault(str(resource_name), float(case.get("res_history", {}).get(str(resource_name), 0.0)))

        try:
            features["enabled_hour"] = float(enabled_time.hour)
            features["enabled_weekday"] = float(enabled_time.weekday())
        except Exception:
            features.setdefault("enabled_hour", 0.0)
            features.setdefault("enabled_weekday", 0.0)

        features["res_workload"] = self._safe_float(
            scheduling_result.get("workload", features.get("candidate_workload", 0.0)),
            0.0,
        )
        features["queue_wait_min"] = self._safe_float(
            scheduling_result.get("waiting_time_min", features.get("candidate_waiting_time_min", 0.0)),
            0.0,
        )

        for name in QUEUE_FEATURE_NAMES:
            features.setdefault(name, 0.0)

        return features

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            value = float(value)
            if np.isfinite(value):
                return float(value)
        except Exception:
            pass
        return float(default)
