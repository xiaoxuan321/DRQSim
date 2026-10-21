"""

该文件用于构建 DBN 动态资源状态建模所需的资源级二时间片特征。

build_resource_state_dbn_features() 面向动态贝叶斯网络 DBN；
该函数输出的是 S_r(t), C_r(t) -> S_r(t+1) 形式的资源状态转移样本；
每一行表示某个资源在相邻两次任务之间的一次状态转移。

输出结果可以用于学习 DBN 条件概率表，例如：
    P(Fatigue_{t+1} | Fatigue_t, Workload_t, IdleGap_t)
    P(Skill_{t+1} | Skill_t, Activity_t)
    P(Efficiency_{t+1} | Fatigue_{t+1}, Skill_{t+1}, Workload_{t+1})
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog


def build_resource_state_dbn_features(
        log: Union[EventLog, pd.DataFrame],
        df_features: Optional[pd.DataFrame] = None,
        resources: Optional[list[str]] = None,
        activities: Optional[list[str]] = None,
        calendars: Optional[dict] = None,
        case_id_col: str = "case:concept:name",
        activity_col: str = "concept:name",
        resource_col: str = "org:resource",
        start_time_col: str = "start:timestamp",
        end_time_col: str = "time:timestamp",
        state_bins: int = 3,
        rolling_window: int = 5
) -> pd.DataFrame:
    """
    构建 DBN 动态资源状态建模所需的二时间片特征表。

    参数
    ----
    log:
        输入事件日志，可以是 pm4py EventLog，也可以是 pandas.DataFrame。

    df_features:
        原 drq 的事件级特征表。
        当前函数不强依赖它，主要保留该参数是为了后续扩展时复用
        build_df_features() 中的 res_workload、case history 等字段。

    resources:
        资源列表。如果为 None，则从日志中自动提取。

    activities:
        活动列表。如果为 None，则从日志中自动提取。

    calendars:
        资源工作日历。
        用于构造 availability_t 和 availability_t1。
        如果为 None，则默认所有资源在所有时间均可用。

    case_id_col:
        case ID 字段名。

    activity_col:
        活动字段名。

    resource_col:
        资源字段名。

    start_time_col:
        活动开始时间字段名。

    end_time_col:
        活动完成时间字段名。

    state_bins:
        状态离散化数量。
        当前建议使用 3，对应 low / medium / high。

    rolling_window:
        用于计算近期效率的历史窗口大小。

    返回
    ----
    dbn_features:
        DBN 训练特征表。
        每一行表示一个资源状态转移样本：
            S_r(t), Context_r(t) -> S_r(t+1)
    """

    if state_bins != 3:
        raise ValueError(
            "当前版本建议 state_bins=3，对应 low / medium / high。"
        )

    # =========================
    # 一、读取并标准化事件日志
    # =========================

    df_log = _to_dataframe(log)

    _check_required_columns(
        df_log,
        required_cols=[
            case_id_col,
            activity_col,
            resource_col,
            start_time_col,
            end_time_col
        ]
    )

    df_log = df_log[
        [
            case_id_col,
            activity_col,
            resource_col,
            start_time_col,
            end_time_col
        ]
    ].copy()

    df_log.rename(
        columns={
            case_id_col: "case_id",
            activity_col: "activity",
            resource_col: "resource",
            start_time_col: "start_time",
            end_time_col: "end_time"
        },
        inplace=True
    )

    df_log["case_id"] = df_log["case_id"].astype(str)
    df_log["activity"] = df_log["activity"].astype(str)
    df_log["resource"] = df_log["resource"].astype(str)

    df_log["start_time"] = pd.to_datetime(
        df_log["start_time"],
        utc=True,
        errors="coerce"
    )
    df_log["end_time"] = pd.to_datetime(
        df_log["end_time"],
        utc=True,
        errors="coerce"
    )

    df_log.dropna(
        subset=[
            "case_id",
            "activity",
            "resource",
            "start_time",
            "end_time"
        ],
        inplace=True
    )

    df_log["duration_min"] = (
        df_log["end_time"] - df_log["start_time"]
    ).dt.total_seconds() / 60.0

    # 去掉执行时间非正的异常事件
    df_log = df_log[df_log["duration_min"] > 0].copy()

    if df_log.empty:
        return _empty_dbn_feature_table()

    # =====================================================
    # 新增：从 drq 事件级特征表中提取真实等待时间
    # =====================================================
    # waiting_time_min = start_t - enabled_t
    # 该值比 idle_gap_min_t 更适合作为 DBN 等待时间影响的学习目标。
    waiting_time_lookup = _build_waiting_time_lookup_from_df_features(
        df_features=df_features
    )

    if resources is None:
        resources = sorted(df_log["resource"].dropna().unique().tolist())

    if activities is None:
        activities = sorted(df_log["activity"].dropna().unique().tolist())

    # =========================
    # 二、计算全局统计量
    # =========================

    activity_median_duration = (
        df_log.groupby("activity")["duration_min"]
        .median()
        .to_dict()
    )

    # 统计每个资源执行每个活动的总次数，用于后续技能归一化
    resource_activity_total_counts = (
        df_log.groupby(["resource", "activity"])
        .size()
        .to_dict()
    )

    max_count_per_activity = {}
    for activity in activities:
        counts = [
            resource_activity_total_counts.get((resource, activity), 0)
            for resource in resources
        ]
        max_count_per_activity[activity] = max(max(counts), 1)

    # 估计疲劳增长和恢复的基础尺度
    avg_duration = float(df_log["duration_min"].mean())
    avg_duration = max(avg_duration, 1.0)

    # 工作导致疲劳增长的经验系数
    fatigue_increase_per_work_min = 1.0 / max(avg_duration * 6.0, 1.0)

    # 初始设置，后面会根据资源空闲间隔重新估计
    positive_idle_gaps = []

    # =========================
    # 三、先计算每个资源的原始连续状态序列
    # =========================

    state_rows = []

    for resource in resources:
        df_r = df_log[df_log["resource"] == resource].copy()

        if df_r.empty:
            continue

        df_r.sort_values(
            by=["start_time", "end_time", "case_id"],
            inplace=True
        )
        df_r.reset_index(drop=True, inplace=True)

        # 当前资源历史任务列表，用于计算并发负载和空闲时间
        previous_intervals = []

        # 当前资源在各活动上的历史执行次数
        activity_history_counts = {
            activity: 0
            for activity in activities
        }

        # 当前资源的近期效率观测
        recent_efficiency_values = []

        # 当前资源的疲劳状态，使用连续值 [0, 1]
        fatigue_value = 0.0

        for event_index, row in df_r.iterrows():
            activity = row["activity"]
            start_time = row["start_time"]
            end_time = row["end_time"]
            duration_min = float(row["duration_min"])
            waiting_time_min = _lookup_waiting_time_min(
                waiting_time_lookup=waiting_time_lookup,
                row=row
            )

            # 当前时刻资源已有并发任务数量
            workload_value = _count_active_intervals(
                previous_intervals,
                current_time=start_time
            )

            # 当前任务开始前的空闲间隔
            idle_gap_min = _compute_idle_gap(
                previous_intervals,
                current_start=start_time
            )

            if idle_gap_min > 0:
                positive_idle_gaps.append(idle_gap_min)

            # 根据空闲时间恢复疲劳
            # 恢复系数后续会根据全局平均空闲间隔估计，这里先使用保守默认值
            fatigue_recovery_per_idle_min = 1.0 / max(avg_duration * 8.0, 1.0)
            fatigue_value = max(
                0.0,
                fatigue_value - idle_gap_min * fatigue_recovery_per_idle_min
            )

            # 当前资源对当前活动的技能水平
            # 使用“当前历史执行次数 / 该活动在所有资源中的最大执行次数”进行归一化
            current_activity_count = activity_history_counts.get(activity, 0)
            max_count = max_count_per_activity.get(activity, 1)
            skill_value = current_activity_count / max_count

            # 当前资源近期效率
            # 如果近期没有历史执行，则效率设为 1.0
            if recent_efficiency_values:
                efficiency_value = float(
                    np.mean(recent_efficiency_values[-rolling_window:])
                )
            else:
                efficiency_value = 1.0

            efficiency_value = float(
                np.clip(efficiency_value, 0.5, 1.5)
            )

            # 当前时间是否处于资源工作日历可用区间
            availability_value = _is_resource_available(
                calendars=calendars,
                resource=resource,
                timestamp=start_time
            )

            state_rows.append(
                {
                    "resource": resource,
                    "resource_event_index": event_index,
                    "case_id": row["case_id"],
                    "activity": activity,
                    "start_time": start_time,
                    "end_time": end_time,
                    "weekday": start_time.weekday(),
                    "hour": start_time.hour,

                    # 连续状态变量
                    "fatigue_value": fatigue_value,
                    "workload_value": float(workload_value),
                    "skill_value": float(skill_value),
                    "efficiency_value": efficiency_value,
                    "availability_value": int(availability_value),

                    # 上下文变量
                    "idle_gap_min": float(idle_gap_min),
                    "work_duration_min": duration_min,
                    "waiting_time_min": float(waiting_time_min),
                }
            )

            # 当前事件完成后更新疲劳
            fatigue_value = min(
                1.0,
                fatigue_value + duration_min * fatigue_increase_per_work_min
            )

            # 当前事件完成后更新技能计数
            activity_history_counts[activity] = (
                activity_history_counts.get(activity, 0) + 1
            )

            # 当前事件完成后更新近期效率
            median_duration = activity_median_duration.get(activity, duration_min)
            if median_duration > 0:
                efficiency_observation = median_duration / duration_min
                efficiency_observation = float(
                    np.clip(efficiency_observation, 0.5, 1.5)
                )
                recent_efficiency_values.append(efficiency_observation)

            # 当前事件加入资源历史区间
            previous_intervals.append((start_time, end_time))

    df_state = pd.DataFrame(state_rows)

    if df_state.empty:
        return _empty_dbn_feature_table()

    # =========================
    # 四、离散化连续状态
    # =========================

    df_state["fatigue_state"] = _discretize_to_low_medium_high(
        df_state["fatigue_value"]
    )
    df_state["workload_state"] = _discretize_to_low_medium_high(
        df_state["workload_value"]
    )
    df_state["skill_state"] = _discretize_to_low_medium_high(
        df_state["skill_value"]
    )
    df_state["efficiency_state"] = _discretize_to_low_medium_high(
        df_state["efficiency_value"]
    )

    df_state["idle_gap_state"] = _discretize_to_low_medium_high(
        df_state["idle_gap_min"]
    )
    df_state["work_duration_state"] = _discretize_to_low_medium_high(
        df_state["work_duration_min"]
    )

    df_state["availability_state"] = df_state["availability_value"].map(
        lambda x: "available" if int(x) == 1 else "unavailable"
    )

    # =========================
    # 五、构造二时间片 DBN 样本
    # =========================

    transition_rows = []

    for resource, df_r_state in df_state.groupby("resource"):
        df_r_state = df_r_state.sort_values(
            by=["resource_event_index"]
        ).reset_index(drop=True)

        # 最后一条事件没有下一状态，因此不能构成 t -> t+1 样本
        for i in range(len(df_r_state) - 1):
            cur = df_r_state.iloc[i]
            nxt = df_r_state.iloc[i + 1]

            transition_rows.append(
                {
                    # 样本标识
                    "resource": resource,
                    "transition_index": i,
                    "case_id_t": cur["case_id"],
                    "activity_t": cur["activity"],
                    "start_time_t": cur["start_time"],
                    "end_time_t": cur["end_time"],

                    # 时间上下文
                    "weekday_t": cur["weekday"],
                    "hour_t": cur["hour"],

                    # 当前时间片状态 S_t
                    "fatigue_t": cur["fatigue_state"],
                    "workload_t": cur["workload_state"],
                    "skill_t": cur["skill_state"],
                    "efficiency_t": cur["efficiency_state"],
                    "availability_t": cur["availability_state"],

                    # 当前上下文 C_t
                    "idle_gap_t": cur["idle_gap_state"],
                    "work_duration_t": cur["work_duration_state"],

                    # 下一时间片状态 S_{t+1}
                    "fatigue_t1": nxt["fatigue_state"],
                    "workload_t1": nxt["workload_state"],
                    "skill_t1": nxt["skill_state"],
                    "efficiency_t1": nxt["efficiency_state"],
                    "availability_t1": nxt["availability_state"],

                    # 连续值保留，便于调试和后续改进
                    "fatigue_value_t": cur["fatigue_value"],
                    "workload_value_t": cur["workload_value"],
                    "skill_value_t": cur["skill_value"],
                    "efficiency_value_t": cur["efficiency_value"],
                    "idle_gap_min_t": cur["idle_gap_min"],
                    "work_duration_min_t": cur["work_duration_min"],
                    "waiting_time_min_t": float(cur.get("waiting_time_min", 0.0)),

                    "fatigue_value_t1": nxt["fatigue_value"],
                    "workload_value_t1": nxt["workload_value"],
                    "skill_value_t1": nxt["skill_value"],
                    "efficiency_value_t1": nxt["efficiency_value"],
                }
            )

    dbn_features = pd.DataFrame(transition_rows)

    if dbn_features.empty:
        return _empty_dbn_feature_table()

    dbn_features.reset_index(drop=True, inplace=True)

    return dbn_features


def _to_dataframe(log: Union[EventLog, pd.DataFrame]) -> pd.DataFrame:
    """
    将 pm4py EventLog 或 DataFrame 统一转换为 DataFrame。
    """

    if isinstance(log, pd.DataFrame):
        return log.copy()

    return pm4py.convert_to_dataframe(log).copy()


def _check_required_columns(df: pd.DataFrame, required_cols: list[str]):
    """
    检查 DataFrame 是否包含必要字段。
    """

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"构建 DBN 资源状态特征失败，事件日志缺少字段：{missing_cols}"
        )


def _count_active_intervals(
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
        current_time: pd.Timestamp
) -> int:
    """
    统计当前时间点资源已有的并发任务数。

    只统计已经开始但尚未结束的历史任务。
    """

    count = 0

    for start, end in intervals:
        if start <= current_time < end:
            count += 1

    return count


def _compute_idle_gap(
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
        current_start: pd.Timestamp
) -> float:
    """
    计算当前任务开始前的空闲时间，单位为分钟。

    如果此前没有完成任务，或者此前任务与当前任务重叠，则返回 0。
    """

    if not intervals:
        return 0.0

    previous_finished_ends = [
        end
        for _, end in intervals
        if end <= current_start
    ]

    if not previous_finished_ends:
        return 0.0

    last_finished_end = max(previous_finished_ends)

    idle_gap_min = (
        current_start - last_finished_end
    ).total_seconds() / 60.0

    return max(0.0, float(idle_gap_min))


def _discretize_to_low_medium_high(series: pd.Series) -> pd.Series:
    """
    将连续变量离散化为 low / medium / high。

    采用 33% 和 66% 分位数作为切分点。
    如果变量几乎没有变化，则统一设为 medium。
    """

    s = pd.to_numeric(series, errors="coerce").fillna(0.0)

    if s.nunique() <= 1:
        return pd.Series(
            ["medium"] * len(s),
            index=s.index
        )

    q1 = s.quantile(0.33)
    q2 = s.quantile(0.66)

    # 如果分位数重合，说明数据分布过于集中
    if q1 == q2:
        median_value = s.median()

        return s.map(
            lambda x: "low" if x < median_value
            else ("high" if x > median_value else "medium")
        )

    return s.map(
        lambda x: "low" if x <= q1
        else ("medium" if x <= q2 else "high")
    )


def _is_resource_available(
        calendars: Optional[dict],
        resource: str,
        timestamp: pd.Timestamp
) -> bool:
    """
    根据资源日历判断资源在当前时间是否可用。

    兼容两类日历格式：
    1. weekday 为数字 0-6；
    2. weekday 为英文名称 Monday、Tuesday 等；
    3. hour 为 int 或 str。
    """

    if calendars is None:
        return True

    if resource not in calendars:
        return True

    calendar = calendars.get(resource, {})

    weekday_num = timestamp.weekday()
    weekday_name = timestamp.strftime("%A")
    hour_num = timestamp.hour
    hour_str = str(hour_num)

    day_calendar = None

    for key in [
        weekday_num,
        str(weekday_num),
        weekday_name
    ]:
        if key in calendar:
            day_calendar = calendar[key]
            break

    if day_calendar is None:
        return True

    if hour_num in day_calendar:
        return bool(day_calendar[hour_num])

    if hour_str in day_calendar:
        return bool(day_calendar[hour_str])

    return True


def _empty_dbn_feature_table() -> pd.DataFrame:
    """
    返回一个空的 DBN 特征表，保证列名稳定。
    """

    columns = [
        "resource",
        "transition_index",
        "case_id_t",
        "activity_t",
        "start_time_t",
        "end_time_t",
        "weekday_t",
        "hour_t",

        "fatigue_t",
        "workload_t",
        "skill_t",
        "efficiency_t",
        "availability_t",

        "idle_gap_min_t",
        "work_duration_min_t",
        "waiting_time_min_t",

        "fatigue_t1",
        "workload_t1",
        "skill_t1",
        "efficiency_t1",
        "availability_t1",

        "fatigue_value_t",
        "workload_value_t",
        "skill_value_t",
        "efficiency_value_t",
        "idle_gap_min_t",
        "work_duration_min_t",

        "fatigue_value_t1",
        "workload_value_t1",
        "skill_value_t1",
        "efficiency_value_t1"
    ]

    return pd.DataFrame(columns=columns)
def _timestamp_key(value) -> str | None:
    """
    将时间戳规范化为秒级字符串，用于跨 DataFrame 匹配事件。
    """

    ts = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(ts):
        return None

    return pd.Timestamp(ts).round("s").isoformat()


def _build_waiting_time_lookup_from_df_features(
        df_features: Optional[pd.DataFrame]
) -> dict:
    """
    从 drq 的事件级特征表中构造真实等待时间查找表。

    key:
        case_id, activity, resource, start_time, end_time

    value:
        waiting_time_min = start_t - enabled_t
    """

    if df_features is None or df_features.empty:
        return {}

    df = df_features.copy()

    activity_col = None
    for candidate in ["transition_label", "concept:name", "activity"]:
        if candidate in df.columns:
            activity_col = candidate
            break

    required_cols = [
        "case_id",
        "resource",
        "enabled_t",
        "start_t",
        "end_t"
    ]

    if activity_col is None:
        return {}

    for col in required_cols:
        if col not in df.columns:
            return {}

    df["enabled_t_tmp"] = pd.to_datetime(
        df["enabled_t"],
        utc=True,
        errors="coerce"
    )

    df["start_t_tmp"] = pd.to_datetime(
        df["start_t"],
        utc=True,
        errors="coerce"
    )

    df["end_t_tmp"] = pd.to_datetime(
        df["end_t"],
        utc=True,
        errors="coerce"
    )

    df.dropna(
        subset=[
            "case_id",
            activity_col,
            "resource",
            "enabled_t_tmp",
            "start_t_tmp",
            "end_t_tmp"
        ],
        inplace=True
    )

    if df.empty:
        return {}

    df["waiting_time_min_tmp"] = (
        df["start_t_tmp"] - df["enabled_t_tmp"]
    ).dt.total_seconds() / 60.0

    df["waiting_time_min_tmp"] = (
        df["waiting_time_min_tmp"]
        .clip(lower=0.0)
        .fillna(0.0)
    )

    lookup = {}

    for _, row in df.iterrows():
        key = (
            str(row["case_id"]),
            str(row[activity_col]),
            str(row["resource"]),
            _timestamp_key(row["start_t_tmp"]),
            _timestamp_key(row["end_t_tmp"])
        )

        lookup[key] = float(row["waiting_time_min_tmp"])

    return lookup


def _lookup_waiting_time_min(
        waiting_time_lookup: dict,
        row: pd.Series
) -> float:
    """
    根据当前日志事件查找真实等待时间。
    如果无法匹配，则返回 0，避免用 idle gap 误作等待时间。
    """

    if not waiting_time_lookup:
        return 0.0

    key = (
        str(row["case_id"]),
        str(row["activity"]),
        str(row["resource"]),
        _timestamp_key(row["start_time"]),
        _timestamp_key(row["end_time"])
    )

    return float(waiting_time_lookup.get(key, 0.0))