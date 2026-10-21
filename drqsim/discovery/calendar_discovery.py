# -*- coding: utf-8 -*-
"""
Calendar discovery for the modular drq simulator.

DBPS-faithful change:
- Case-arrival calendar is discovered only from the first event start timestamp
  of each case.
- Resource calendars keep the original resource-event logic and still use both
  event start and completion timestamps.
"""

from __future__ import annotations

import pm4py
from pm4py.objects.log.obj import EventLog


def _repair_shift_continuity(
        calendar: dict,
        max_shift_span_hours: int = 12,
) -> dict:
    """Fill small gaps between observed active hours on the same weekday.

    Resource-calendar discovery is based on event start/completion endpoints.
    Those endpoints can leave artificial holes inside an otherwise continuous
    work shift.  For a weekday with at least two active hours, fill the hours
    between the earliest and latest active endpoint only when their span is
    bounded by ``max_shift_span_hours``.  Large spans are left untouched so
    split shifts, overnight patterns, and irregular calendars are not turned
    into an all-day shift.
    """
    if not isinstance(calendar, dict):
        return calendar

    try:
        max_span = int(max_shift_span_hours)
    except Exception:
        max_span = 12
    max_span = max(0, min(max_span, 23))
    if max_span <= 0:
        return calendar

    for wd in range(7):
        day = calendar.get(wd)
        if not isinstance(day, dict):
            continue

        active_hours = [
            h for h in range(24)
            if bool(day.get(h, False))
        ]
        if len(active_hours) < 2:
            continue

        h_min = min(active_hours)
        h_max = max(active_hours)
        if (h_max - h_min) > max_span:
            continue

        for h in range(h_min, h_max + 1):
            day[h] = True

    return calendar


def discover_arrival_calendar(
        log: EventLog,
        thr_h: float = 0.95,
        thr_wd: float = 0.95,
) -> dict:
    """
    Discover a case-arrival calendar.

    DBPS-faithful rule:
    A case arrival is the first event start timestamp of a trace. Do not count
    the first event completion timestamp as an additional arrival observation.

    Returns:
        {weekday: {hour: True/False}}
    """
    N_events_per_hour = {
        wd: {h: 0 for h in range(24)}
        for wd in range(7)
    }

    for trace in log:
        if not trace:
            continue
        first_event = trace[0]
        if "start:timestamp" not in first_event:
            continue
        ts = first_event.get("start:timestamp")
        if ts is None:
            continue
        try:
            N_events_per_hour[ts.weekday()][ts.hour] += 1
        except Exception:
            continue

    N_events_per_hour_perc = {
        wd: {h: 0.0 for h in range(24)}
        for wd in range(7)
    }

    for weekday in range(7):
        total_events_weekday = sum(N_events_per_hour[weekday].values())
        for h in range(24):
            if total_events_weekday:
                N_events_per_hour_perc[weekday][h] = (
                    N_events_per_hour[weekday][h] / total_events_weekday
                )
            else:
                N_events_per_hour_perc[weekday][h] = 0.0

    N_events_per_wd = {
        wd: sum(N_events_per_hour[wd].values())
        for wd in range(7)
    }
    total_events = sum(N_events_per_wd.values())

    N_events_per_wd_perc = {
        wd: (N_events_per_wd[wd] / total_events if total_events else 0.0)
        for wd in range(7)
    }

    calendar = {
        wd: {h: False for h in range(24)}
        for wd in range(7)
    }

    sorted_wd = sorted(
        N_events_per_wd_perc.items(),
        key=lambda x: -x[1],
    )

    wd_cumulative = 0.0
    top_wds = []

    for wd, perc in sorted_wd:
        if wd_cumulative >= float(thr_wd):
            break
        if perc <= 0:
            continue
        top_wds.append(wd)
        wd_cumulative += float(perc)

    for wd in top_wds:
        sorted_hours = sorted(
            N_events_per_hour_perc[wd].items(),
            key=lambda x: -x[1],
        )

        hour_cumulative = 0.0
        for h, perc in sorted_hours:
            if hour_cumulative >= float(thr_h):
                break
            if perc <= 0:
                continue
            calendar[wd][h] = True
            hour_cumulative += float(perc)

    return calendar


def discover_res_calendars(
        log: EventLog,
        resources: list = [],
        thr_h: float = 0.95,
        thr_wd: float = 0.95,
        max_shift_span_hours: int = 12,
) -> dict:
    """
    Discover resource working calendars.

    Resource activity is still estimated from event start/completion endpoints.
    After the original thresholding step, a conservative continuity repair fills
    gaps between the earliest and latest active hour of a weekday only when that
    span is at most ``max_shift_span_hours`` (default: 12 hours).

    Returns:
        {resource: {weekday: {hour: True/False}}}
    """
    if not resources:
        resources = list(
            pm4py.get_event_attribute_values(log, "org:resource").keys()
        )

    N_events_per_hour_res = {
        res: {
            wd: {h: 0 for h in range(24)}
            for wd in range(7)
        }
        for res in resources
    }

    for trace in log:
        for event in trace:
            res = event.get("org:resource")
            if res not in resources:
                continue

            ts = event.get("start:timestamp")
            if ts is not None:
                try:
                    N_events_per_hour_res[res][ts.weekday()][ts.hour] += 1
                except Exception:
                    pass

            ts = event.get("time:timestamp")
            if ts is not None:
                try:
                    N_events_per_hour_res[res][ts.weekday()][ts.hour] += 1
                except Exception:
                    pass

    N_events_per_hour_res_perc = {
        res: {
            wd: {h: 0.0 for h in range(24)}
            for wd in range(7)
        }
        for res in resources
    }

    for res in resources:
        for weekday in range(7):
            total_events_res_weekday = sum(
                N_events_per_hour_res[res][weekday].values()
            )
            for h in range(24):
                if total_events_res_weekday:
                    N_events_per_hour_res_perc[res][weekday][h] = (
                        N_events_per_hour_res[res][weekday][h]
                        / total_events_res_weekday
                    )
                else:
                    N_events_per_hour_res_perc[res][weekday][h] = 0.0

    N_events_per_wd_res = {
        res: {wd: 0 for wd in range(7)}
        for res in resources
    }

    for res in resources:
        for weekday in range(7):
            N_events_per_wd_res[res][weekday] += sum(
                N_events_per_hour_res[res][weekday].values()
            )

    N_events_per_wd_res_perc = {
        res: {
            wd: (
                N_events_per_wd_res[res][wd]
                / sum(N_events_per_wd_res[res].values())
                if sum(N_events_per_wd_res[res].values())
                else 0.0
            )
            for wd in range(7)
        }
        for res in resources
    }

    calendar_wd_hour_res = {}

    for res, wd_perc in N_events_per_wd_res_perc.items():
        calendar = {
            wd: {h: False for h in range(24)}
            for wd in range(7)
        }

        sorted_wd = sorted(
            wd_perc.items(),
            key=lambda x: -x[1],
        )

        wd_cumulative = 0.0
        top_wds = []

        for wd, perc in sorted_wd:
            if wd_cumulative >= float(thr_wd):
                break
            if perc <= 0:
                continue
            top_wds.append(wd)
            wd_cumulative += float(perc)

        for wd in top_wds:
            sorted_hours = sorted(
                N_events_per_hour_res_perc[res][wd].items(),
                key=lambda x: -x[1],
            )

            hour_cumulative = 0.0
            for h, perc in sorted_hours:
                if hour_cumulative >= float(thr_h):
                    break
                if perc <= 0:
                    continue
                calendar[wd][h] = True
                hour_cumulative += float(perc)

        calendar = _repair_shift_continuity(
            calendar,
            max_shift_span_hours=max_shift_span_hours,
        )
        calendar_wd_hour_res[res] = calendar

    return calendar_wd_hour_res
