"""

真正的离散二时间片 DBN 动态资源状态发现模块。

该模块用于从事件日志中学习资源状态动态贝叶斯网络，
包括：
1. 构造资源级二时间片状态转移样本；
2. 定义 DBN 结构；
3. 估计条件概率表 CPT；
4. 输出可序列化的 DBN 参数，供 SimulatorParameters 保存，
   并供 SimulatorEngine 运行时更新资源状态使用。

注意：
当前实现不依赖 pgmpy，而是直接学习并保存离散 DBN 的结构和 CPT。
这样更容易与现有 drq JSON 参数体系集成。
"""

from __future__ import annotations

import itertools
import json
from typing import Optional

import numpy as np
import pandas as pd
from pm4py.objects.log.obj import EventLog

from drqsim.discovery.resource_state_features import build_resource_state_dbn_features


# =========================
# 一、状态空间定义
# =========================

THREE_LEVEL_STATES = ["low", "medium", "high"]
AVAILABILITY_STATES = ["available", "unavailable"]


def discover_resource_state_params(
        log: EventLog,
        resources: list,
        activities: list,
        df_features: Optional[pd.DataFrame] = None,
        calendars: Optional[dict] = None,
        dbn_features: Optional[pd.DataFrame] = None,
        smoothing: float = 1.0
) -> dict:
    """
    从事件日志中发现真正的 DBN 动态资源状态模型。

    参数
    ----
    log:
        输入事件日志，通常为训练集日志。

    resources:
        资源列表。

    activities:
        活动列表。

    df_features:
        原 drq 的事件级特征表。
        当前函数不强依赖它，但可以传入以便 build_resource_state_dbn_features 复用。

    calendars:
        资源日历，用于构造 availability 状态。

    dbn_features:
        已经构建好的 DBN 二时间片特征表。
        如果传入，则不再重复调用 build_resource_state_dbn_features。

    smoothing:
        CPT 的拉普拉斯平滑系数。
        默认为 1.0，避免某些条件组合未出现时概率为 0。

    返回
    ----
    resource_state_params:
        动态资源状态 DBN 参数字典。
    """

    try:
        # =========================
        # 1. 构建 DBN 二时间片样本
        # =========================
        if dbn_features is None:
            dbn_features = build_resource_state_dbn_features(
                log=log,
                df_features=df_features,
                resources=resources,
                activities=activities,
                calendars=calendars
            )

        if dbn_features is None or dbn_features.empty:
            return _disabled_params(
                reason="Empty DBN feature table. Cannot learn resource-state DBN."
            )

        # 统一转成字符串，保证 JSON key 和 CPT 查询稳定
        dbn_features = _normalize_dbn_feature_types(dbn_features)

        # 检查 DBN 特征表是否包含必要字段
        required_cols = [
            "resource",
            "activity_t",

            "fatigue_t",
            "workload_t",
            "skill_t",
            "efficiency_t",
            "availability_t",

            "idle_gap_t",
            "work_duration_t",
            "hour_t",
            "weekday_t",

            "fatigue_t1",
            "workload_t1",
            "skill_t1",
            "efficiency_t1",
            "availability_t1"
        ]

        missing_cols = [
            col for col in required_cols
            if col not in dbn_features.columns
        ]

        if missing_cols:
            return _disabled_params(
                reason=f"DBN feature table misses required columns: {missing_cols}"
            )

        # =========================
        # 2. 定义 DBN 变量状态空间
        # =========================
        activity_states = _infer_states(
            dbn_features["activity_t"],
            fallback=[str(a) for a in activities]
        )

        hour_states = _infer_states(
            dbn_features["hour_t"],
            fallback=[str(i) for i in range(24)]
        )

        weekday_states = _infer_states(
            dbn_features["weekday_t"],
            fallback=[str(i) for i in range(7)]
        )

        variable_states = {
            # 当前时间片状态变量
            "fatigue_t": THREE_LEVEL_STATES,
            "workload_t": THREE_LEVEL_STATES,
            "skill_t": THREE_LEVEL_STATES,
            "efficiency_t": THREE_LEVEL_STATES,
            "availability_t": AVAILABILITY_STATES,

            # 当前上下文变量
            "activity_t": activity_states,
            "idle_gap_t": THREE_LEVEL_STATES,
            "work_duration_t": THREE_LEVEL_STATES,
            "hour_t": hour_states,
            "weekday_t": weekday_states,

            # 下一时间片状态变量
            "fatigue_t1": THREE_LEVEL_STATES,
            "workload_t1": THREE_LEVEL_STATES,
            "skill_t1": THREE_LEVEL_STATES,
            "efficiency_t1": THREE_LEVEL_STATES,
            "availability_t1": AVAILABILITY_STATES
        }

        # =========================
        # 3. 定义二时间片 DBN 结构
        # =========================
        dbn_structure = _build_default_resource_state_dbn_structure()

        # =========================
        # 4. 学习 CPT
        # =========================
        cpds = {}

        # P(Fatigue_{t+1} | Fatigue_t, Workload_t, IdleGap_t)
        cpds["fatigue"] = _learn_cpt(
            df=dbn_features,
            target="fatigue_t1",
            parents=["fatigue_t", "workload_t", "idle_gap_t"],
            variable_states=variable_states,
            smoothing=smoothing
        )

        # P(Workload_{t+1} | Workload_t, Activity_t, Hour_t)
        cpds["workload"] = _learn_cpt(
            df=dbn_features,
            target="workload_t1",
            parents=["workload_t", "activity_t", "hour_t"],
            variable_states=variable_states,
            smoothing=smoothing
        )

        # P(Skill_{t+1} | Skill_t, Activity_t)
        cpds["skill"] = _learn_cpt(
            df=dbn_features,
            target="skill_t1",
            parents=["skill_t", "activity_t"],
            variable_states=variable_states,
            smoothing=smoothing
        )

        # P(Efficiency_{t+1} | Fatigue_{t+1}, Skill_{t+1}, Workload_{t+1})
        cpds["efficiency"] = _learn_cpt(
            df=dbn_features,
            target="efficiency_t1",
            parents=["fatigue_t1", "skill_t1", "workload_t1"],
            variable_states=variable_states,
            smoothing=smoothing
        )

        # P(Availability_{t+1} | Availability_t, Hour_t, Weekday_t)
        cpds["availability"] = _learn_cpt(
            df=dbn_features,
            target="availability_t1",
            parents=["availability_t", "hour_t", "weekday_t"],
            variable_states=variable_states,
            smoothing=smoothing
        )

        # =========================
        # 5. 估计每个资源的初始离散状态
        # =========================
        initial_state = _estimate_initial_state(
            dbn_features=dbn_features,
            resources=resources,
            activities=activities
        )

        # =========================
        # 6. 从日志中学习资源状态对仿真行为的作用参数
        # =========================
        # 注意：
        # DBN 的核心是 structure + cpds，用于学习状态如何演化；
        # state_effect_mapping 则学习 DBN 状态如何影响执行时间、等待时间、
        # 资源选择权重和动态容量。
        effect_params = _learn_state_effect_mapping(
            dbn_features=dbn_features,
            df_features=df_features,
            smoothing=smoothing
        )

        # 新增：对 DBN 的运行时作用强度进行稳定化约束
        # 目的：保留动态资源状态的相对趋势，但避免等待时间和执行时间被连续放大。
        effect_params = _stabilize_runtime_effect_params(effect_params)

        return {
            "enabled": True,
            "model_type": "discrete_two_slice_dbn",

            "description": (
                "A discrete two-slice dynamic Bayesian network for resource-state "
                "evolution. The DBN learns conditional transition probabilities "
                "from resource-level state-transition samples."
            ),

            # DBN 状态变量
            "state_variables": [
                "fatigue",
                "workload",
                "skill",
                "efficiency",
                "availability"
            ],

            # 上下文变量
            "context_variables": [
                "activity",
                "idle_gap",
                "work_duration",
                "hour",
                "weekday"
            ],

            # 每个变量的离散状态空间
            "variable_states": variable_states,

            # 二时间片 DBN 结构
            "structure": dbn_structure,

            # 条件概率表
            "cpds": cpds,

            # 初始资源状态
            "initial_state": initial_state,

            # CPT 平滑系数
            "smoothing": smoothing,
            # 新增：运行时离散化阈值
            "discretization": _learn_discretization_metadata(dbn_features),

            # 状态对仿真行为的影响映射
            "effect_params": effect_params,
            "runtime_options": {
                "state_transition_mode": "tempered_sample",
                "cpt_temperature": 1.5
            },
            # 训练摘要
            "training_summary": {
                "num_transition_samples": int(len(dbn_features)),
                "num_resources": int(len(resources)),
                "num_activities": int(len(activities)),
                "resources_covered": sorted(
                    dbn_features["resource"].dropna().unique().tolist()
                )
            }
        }

    except Exception as e:
        return _disabled_params(
            reason=f"Failed to learn resource-state DBN: {str(e)}"
        )


def _build_default_resource_state_dbn_structure() -> dict:
    """
    定义资源状态二时间片 DBN 结构。

    结构含义：
    1. 疲劳状态由上一时刻疲劳、当前负载和空闲间隔决定；
    2. 负载状态由上一时刻负载、当前活动和小时段决定；
    3. 技能状态由上一时刻技能和当前活动决定；
    4. 效率状态由下一时刻疲劳、下一时刻技能和下一时刻负载决定；
    5. 可用性由上一时刻可用性、小时段和星期决定。
    """

    edges = [
        # Fatigue transition
        ["fatigue_t", "fatigue_t1"],
        ["workload_t", "fatigue_t1"],
        ["idle_gap_t", "fatigue_t1"],

        # Workload transition
        ["workload_t", "workload_t1"],
        ["activity_t", "workload_t1"],
        ["hour_t", "workload_t1"],

        # Skill transition
        ["skill_t", "skill_t1"],
        ["activity_t", "skill_t1"],

        # Efficiency transition
        ["fatigue_t1", "efficiency_t1"],
        ["skill_t1", "efficiency_t1"],
        ["workload_t1", "efficiency_t1"],

        # Availability transition
        ["availability_t", "availability_t1"],
        ["hour_t", "availability_t1"],
        ["weekday_t", "availability_t1"]
    ]

    return {
        "time_slices": 2,
        "edges": edges,
        "targets": {
            "fatigue": "fatigue_t1",
            "workload": "workload_t1",
            "skill": "skill_t1",
            "efficiency": "efficiency_t1",
            "availability": "availability_t1"
        }
    }


def _learn_cpt(
        df: pd.DataFrame,
        target: str,
        parents: list[str],
        variable_states: dict,
        smoothing: float = 1.0
) -> dict:
    """
    根据二时间片样本学习一个条件概率表 CPT。

    形式：
        P(target | parents)

    返回结构：
        {
            "target": target,
            "parents": parents,
            "target_states": [...],
            "parent_states": {...},
            "table": {
                parent_key: {
                    target_state_1: prob,
                    target_state_2: prob,
                    ...
                }
            },
            "default": {...}
        }
    """

    target_states = variable_states[target]
    parent_states = {
        parent: variable_states[parent]
        for parent in parents
    }

    # 全局先验分布，用于未出现过的父状态组合
    default_distribution = _estimate_distribution(
        values=df[target],
        states=target_states,
        smoothing=smoothing
    )

    # 收集观察到的条件计数
    observed_counts = {}

    for _, row in df.iterrows():
        parent_assignment = {
            parent: str(row[parent])
            for parent in parents
        }

        key = _parent_key(parent_assignment)

        if key not in observed_counts:
            observed_counts[key] = {
                state: 0.0
                for state in target_states
            }

        target_value = str(row[target])

        if target_value not in observed_counts[key]:
            # 如果出现不在预设状态空间中的值，跳过
            continue

        observed_counts[key][target_value] += 1.0

    # 为所有父状态组合生成 CPT，保证运行时不会查不到
    table = {}

    all_parent_assignments = _all_parent_assignments(parent_states)

    for parent_assignment in all_parent_assignments:
        key = _parent_key(parent_assignment)

        if key in observed_counts:
            table[key] = _normalize_counts(
                counts=observed_counts[key],
                states=target_states,
                smoothing=smoothing
            )
        else:
            table[key] = default_distribution

    return {
        "target": target,
        "parents": parents,
        "target_states": target_states,
        "parent_states": parent_states,
        "table": table,
        "default": default_distribution
    }

def _learn_discretization_metadata(
        dbn_features: pd.DataFrame
) -> dict:
    """
    保存训练阶段连续变量的离散化阈值，供仿真运行时使用。
    """

    metadata = {}

    continuous_cols = [
        "workload_value_t",
        "idle_gap_min_t",
        "work_duration_min_t",
        "fatigue_value_t",
        "skill_value_t",
        "efficiency_value_t"
    ]

    for col in continuous_cols:
        if col not in dbn_features.columns:
            continue

        values = pd.to_numeric(
            dbn_features[col],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        if values.nunique() <= 1:
            q1 = float(values.iloc[0])
            q2 = float(values.iloc[0])
        else:
            q1 = float(values.quantile(0.33))
            q2 = float(values.quantile(0.66))

        metadata[col] = {
            "q1": q1,
            "q2": q2,
            "states": ["low", "medium", "high"]
        }

    return metadata
def _estimate_initial_state(
        dbn_features: pd.DataFrame,
        resources: list,
        activities: list
) -> dict:
    """
    根据每个资源在 DBN 特征表中的第一条样本估计初始状态。

    注意：
    skill 是活动相关状态，因此为每个资源保存 activity -> skill_state。
    """

    initial_state = {}

    for resource in resources:
        resource = str(resource)
        df_r = dbn_features[dbn_features["resource"] == resource].copy()

        if df_r.empty:
            initial_state[resource] = _default_initial_state(activities)
            continue

        df_r.sort_values(
            by=["start_time_t"] if "start_time_t" in df_r.columns else ["transition_index"],
            inplace=True
        )

        first = df_r.iloc[0]

        skill_by_activity = {}

        for activity in activities:
            activity = str(activity)

            df_ra = df_r[df_r["activity_t"] == activity]

            if df_ra.empty:
                skill_by_activity[activity] = "medium"
            else:
                skill_by_activity[activity] = _mode_or_default(
                    df_ra["skill_t"],
                    default="medium"
                )

        initial_state[resource] = {
            "fatigue": str(first.get("fatigue_t", "medium")),
            "workload": str(first.get("workload_t", "low")),
            "efficiency": str(first.get("efficiency_t", "medium")),
            "availability": str(first.get("availability_t", "available")),
            "skill": skill_by_activity,
            "last_end_time": None
        }

    return initial_state


def _default_initial_state(activities: list) -> dict:
    """
    资源没有足够样本时的默认初始状态。
    """

    return {
        "fatigue": "low",
        "workload": "low",
        "efficiency": "medium",
        "availability": "available",
        "skill": {
            str(activity): "medium"
            for activity in activities
        },
        "last_end_time": None
    }


def _learn_resource_selection_effect(
        dbn_features: pd.DataFrame
) -> dict:
    """
    从日志中学习资源状态对资源选择倾向的影响。

    注意：
    如果没有候选资源集合和未被选择资源的负样本，资源选择影响只能从
    “被实际选择资源的状态分布”中学习一个弱选择因子。
    更严格的做法是基于 df_features 中的 enabled_resources 构造
    chosen / not chosen 样本，然后学习多项选择模型。
    """

    df = dbn_features.copy()

    if df.empty:
        return _neutral_resource_selection_effect()

    factor_by_state = {}

    for state_var in [
        "fatigue_t",
        "skill_t",
        "efficiency_t",
        "availability_t"
    ]:
        logical_name = state_var.replace("_t", "")

        factor_by_state[logical_name] = _learn_frequency_factor_by_state(
            df=df,
            state_col=state_var,
            states=_states_for_variable(state_var),
            min_factor=0.50,
            max_factor=1.50
        )

    joint_factor = _learn_joint_frequency_factor(
        df=df,
        state_cols=[
            "fatigue_t",
            "skill_t",
            "efficiency_t",
            "availability_t"
        ],
        min_count=3,
        default_factor=1.0,
        min_factor=0.50,
        max_factor=1.50
    )

    return {
        "source": "event_log_selected_resource_distribution",
        "target": "resource_selection",
        "factor_by_state": factor_by_state,
        "joint_factor": joint_factor,
        "default_factor": 1.0,
        "min_factor": 0.50,
        "max_factor": 1.50,
        "note": (
            "This factor is learned from the states of actually selected "
            "resources. For a stricter resource-choice model, construct "
            "chosen/not-chosen samples from enabled resource sets."
        )
    }


def _learn_capacity_effect(
        dbn_features: pd.DataFrame
) -> dict:
    """
    从日志中学习不同疲劳状态下的动态容量。

    使用 workload_value_t 的高分位数估计资源在对应状态下的可承载并发量。
    """

    df = dbn_features.copy()

    if "workload_value_t" not in df.columns:
        return _neutral_capacity_effect()

    df["workload_value_t"] = pd.to_numeric(
        df["workload_value_t"],
        errors="coerce"
    )

    df.dropna(subset=["fatigue_t", "workload_value_t"], inplace=True)

    if df.empty:
        return _neutral_capacity_effect()

    capacity_by_fatigue = {}

    for fatigue_state in ["low", "medium", "high"]:
        df_s = df[df["fatigue_t"] == fatigue_state]

        if df_s.empty:
            capacity_by_fatigue[fatigue_state] = 1
            continue

        # workload_value_t 表示当前已有并发任务数；
        # 资源可接收当前任务，因此容量至少为 workload + 1。
        q = float(df_s["workload_value_t"].quantile(0.90))
        capacity_by_fatigue[fatigue_state] = max(1, int(round(q + 1)))

    return {
        "source": "event_log",
        "target": "dynamic_capacity",
        "default_capacity": 1,
        "capacity_by_fatigue": capacity_by_fatigue,
        "max_capacity": max(capacity_by_fatigue.values())
    }


def _learn_factor_by_state(
        df: pd.DataFrame,
        state_col: str,
        value_col: str,
        default_factor: float,
        min_factor: float,
        max_factor: float
) -> dict:
    """
    学习单个状态变量对某个连续比例变量的影响因子。
    """

    result = {}

    states = _states_for_variable(state_col)

    global_median = float(df[value_col].median())

    if global_median <= 0:
        global_median = 1.0

    for state in states:
        df_s = df[df[state_col] == state]

        if df_s.empty:
            result[state] = default_factor
            continue

        state_median = float(df_s[value_col].median())

        factor = state_median / global_median

        result[state] = float(
            np.clip(factor, min_factor, max_factor)
        )

    return result


def _learn_joint_factor(
        df: pd.DataFrame,
        state_cols: list[str],
        value_col: str,
        min_count: int,
        default_factor: float,
        min_factor: float,
        max_factor: float
) -> dict:
    """
    学习联合状态组合下的影响因子。
    """

    global_median = float(df[value_col].median())

    if global_median <= 0:
        global_median = 1.0

    result = {}

    grouped = df.groupby(state_cols)

    for state_tuple, df_g in grouped:
        if len(df_g) < min_count:
            continue

        if not isinstance(state_tuple, tuple):
            state_tuple = (state_tuple,)

        assignment = {
            col: str(value)
            for col, value in zip(state_cols, state_tuple)
        }

        key = _parent_key(assignment)

        factor = float(df_g[value_col].median()) / global_median

        result[key] = float(
            np.clip(factor, min_factor, max_factor)
        )

    return {
        "state_cols": state_cols,
        "table": result,
        "default_factor": default_factor
    }


def _learn_frequency_factor_by_state(
        df: pd.DataFrame,
        state_col: str,
        states: list[str],
        min_factor: float,
        max_factor: float
) -> dict:
    """
    根据被选择资源的状态频率学习选择倾向因子。

    如果某个状态在被选择资源中出现频率更高，则因子略大。
    """

    counts = df[state_col].value_counts(normalize=True).to_dict()

    uniform_prob = 1.0 / max(len(states), 1)

    result = {}

    for state in states:
        p = float(counts.get(state, 0.0))

        if uniform_prob <= 0:
            factor = 1.0
        else:
            factor = p / uniform_prob

        result[state] = float(
            np.clip(factor, min_factor, max_factor)
        )

    return result


def _learn_joint_frequency_factor(
        df: pd.DataFrame,
        state_cols: list[str],
        min_count: int,
        default_factor: float,
        min_factor: float,
        max_factor: float
) -> dict:
    """
    根据联合状态出现频率学习资源选择倾向因子。
    """

    result = {}

    total = len(df)

    if total <= 0:
        return {
            "state_cols": state_cols,
            "table": result,
            "default_factor": default_factor
        }

    grouped = df.groupby(state_cols)

    expected_uniform = 1.0

    for state_tuple, df_g in grouped:
        if len(df_g) < min_count:
            continue

        if not isinstance(state_tuple, tuple):
            state_tuple = (state_tuple,)

        assignment = {
            col: str(value)
            for col, value in zip(state_cols, state_tuple)
        }

        key = _parent_key(assignment)

        observed = len(df_g) / total

        factor = observed / expected_uniform

        result[key] = float(
            np.clip(factor, min_factor, max_factor)
        )

    return {
        "state_cols": state_cols,
        "table": result,
        "default_factor": default_factor
    }


def _states_for_variable(state_col: str) -> list[str]:
    """
    根据变量名返回离散状态空间。
    """

    if "availability" in state_col:
        return ["available", "unavailable"]

    return ["low", "medium", "high"]


def _neutral_execution_time_effect() -> dict:
    return {
        "source": "neutral",
        "target": "execution_time",
        "factor_by_state": {},
        "joint_factor": {
            "state_cols": [],
            "table": {},
            "default_factor": 1.0
        },
        "default_factor": 1.0,
        "min_factor": 0.50,
        "max_factor": 2.00
    }


def _neutral_waiting_time_effect() -> dict:
    return {
        "source": "neutral",
        "target": "waiting_time",
        "factor_by_state": {},
        "joint_factor": {
            "state_cols": [],
            "table": {},
            "default_factor": 1.0
        },
        "default_factor": 1.0,
        "min_factor": 0.50,
        "max_factor": 2.00
    }


def _neutral_resource_selection_effect() -> dict:
    return {
        "source": "neutral",
        "target": "resource_selection",
        "factor_by_state": {},
        "joint_factor": {
            "state_cols": [],
            "table": {},
            "default_factor": 1.0
        },
        "default_factor": 1.0,
        "min_factor": 0.50,
        "max_factor": 1.50
    }


def _neutral_capacity_effect() -> dict:
    return {
        "source": "neutral",
        "target": "dynamic_capacity",
        "default_capacity": 1,
        "capacity_by_fatigue": {
            "low": 1,
            "medium": 1,
            "high": 1
        },
        "max_capacity": 1
    }

def _learn_state_effect_mapping(
        dbn_features: pd.DataFrame,
        df_features: Optional[pd.DataFrame] = None,
        smoothing: float = 1.0
) -> dict:
    """
    从日志中学习资源状态到仿真行为的映射参数。

    DBN 本身学习的是资源状态转移：
        P(S_{t+1} | S_t, C_t)

    本函数学习的是：
        1. 状态对执行时间的影响；
        2. 状态对等待时间的影响；
        3. 状态对资源选择权重的影响；
        4. 状态对多任务容量的影响。

    这些参数不是人工指定，而是从 DBN 特征表和事件级特征表中估计。
    """

    effect_mapping = {
        "state_score": {
            "low": 0.0,
            "medium": 0.5,
            "high": 1.0,
            "available": 1.0,
            "unavailable": 0.0
        },

        "execution_time": _learn_execution_time_effect(
            dbn_features=dbn_features
        ),

        "waiting_time": _learn_waiting_time_effect(
            dbn_features=dbn_features,
            df_features=df_features
        ),

        "resource_selection": _learn_resource_selection_effect(
            dbn_features=dbn_features
        ),

        "capacity": _learn_capacity_effect(
            dbn_features=dbn_features
        )
    }

    return effect_mapping


def _learn_execution_time_effect(
        dbn_features: pd.DataFrame
) -> dict:
    """
    从日志中学习资源状态对执行时间的影响。

    学习思想：
    1. 先计算每个活动的中位执行时间；
    2. 对每个事件计算 duration_ratio：
           当前执行时间 / 当前活动中位执行时间
    3. 按 fatigue、workload、skill、efficiency 的离散状态分组；
    4. 学习不同状态下的执行时间放大或缩小因子。
    """

    df = dbn_features.copy()

    if "work_duration_min_t" not in df.columns:
        return _neutral_execution_time_effect()

    df["work_duration_min_t"] = pd.to_numeric(
        df["work_duration_min_t"],
        errors="coerce"
    )

    df.dropna(
        subset=["activity_t", "work_duration_min_t"],
        inplace=True
    )

    df = df[df["work_duration_min_t"] > 0].copy()

    if df.empty:
        return _neutral_execution_time_effect()

    activity_median = (
        df.groupby("activity_t")["work_duration_min_t"]
        .median()
        .to_dict()
    )

    df["activity_median_duration"] = df["activity_t"].map(activity_median)

    df = df[df["activity_median_duration"] > 0].copy()

    df["duration_ratio"] = (
        df["work_duration_min_t"] / df["activity_median_duration"]
    )

    df["duration_ratio"] = df["duration_ratio"].clip(0.25, 4.0)

    factor_by_state = {}

    for state_var in [
        "fatigue_t",
        "workload_t",
        "skill_t",
        "efficiency_t"
    ]:
        logical_name = state_var.replace("_t", "")

        factor_by_state[logical_name] = _learn_factor_by_state(
            df=df,
            state_col=state_var,
            value_col="duration_ratio",
            default_factor=1.0,
            min_factor=0.50,
            max_factor=2.00
        )

    # 学习联合状态因子，运行时优先使用联合状态，查不到时退化到单变量因子
    joint_cols = [
        "fatigue_t",
        "workload_t",
        "skill_t",
        "efficiency_t"
    ]

    joint_factor = _learn_joint_factor(
        df=df,
        state_cols=joint_cols,
        value_col="duration_ratio",
        min_count=3,
        default_factor=1.0,
        min_factor=0.50,
        max_factor=2.00
    )

    return {
        "source": "event_log",
        "target": "execution_time",
        "factor_by_state": factor_by_state,
        "joint_factor": joint_factor,
        "default_factor": 1.0,
        "min_factor": 0.50,
        "max_factor": 2.00
    }


def _learn_waiting_time_effect(
        dbn_features: pd.DataFrame,
        df_features: Optional[pd.DataFrame] = None
) -> dict:
    """
    从日志中学习资源状态对等待时间的影响。

    优先使用真实 waiting_time。
    如果当前特征表没有 waiting_time，则退化使用 idle_gap_min_t 作为弱代理。
    注意：idle gap 并不等价于业务等待时间，因此建议后续在
    build_resource_state_dbn_features() 中显式加入 waiting_time_min_t。
    """

    df = dbn_features.copy()

    waiting_col = None

    for candidate in [
        "waiting_time_min_t",
        "waiting_time_min",
        "waiting_time"
    ]:
        if candidate in df.columns:
            waiting_col = candidate
            break

    if waiting_col is None:
        # 关键修改：
        # 不再使用 idle_gap_min_t 作为等待时间代理。
        # idle gap 表示资源空闲间隔，不等价于业务等待时间。
        # 用它学习 waiting effect 容易把“资源空闲”误判成“等待变长”，
        # 从而系统性拉高 CTD。
        return _neutral_waiting_time_effect()
    else:
        source = "event_log"

    df[waiting_col] = pd.to_numeric(
        df[waiting_col],
        errors="coerce"
    )

    df.dropna(subset=[waiting_col], inplace=True)
    df = df[df[waiting_col] >= 0].copy()

    if df.empty:
        return _neutral_waiting_time_effect()

    median_wait = float(df[waiting_col].median())

    if median_wait <= 0:
        # 避免 0 等待时间导致比例不稳定
        median_wait = float(df[waiting_col].mean())

    if median_wait <= 0:
        return _neutral_waiting_time_effect()

    df["waiting_ratio"] = (df[waiting_col] / median_wait).clip(0.25, 4.0)

    factor_by_state = {}

    for state_var in [
        "fatigue_t",
        "workload_t",
        "availability_t"
    ]:
        logical_name = state_var.replace("_t", "")

        factor_by_state[logical_name] = _learn_factor_by_state(
            df=df,
            state_col=state_var,
            value_col="waiting_ratio",
            default_factor=1.0,
            min_factor=0.50,
            max_factor=2.00
        )

    joint_factor = _learn_joint_factor(
        df=df,
        state_cols=["fatigue_t", "workload_t", "availability_t"],
        value_col="waiting_ratio",
        min_count=3,
        default_factor=1.0,
        min_factor=0.50,
        max_factor=2.00
    )

    return {
        "source": source,
        "target": "waiting_time",
        "factor_by_state": factor_by_state,
        "joint_factor": joint_factor,
        "default_factor": 1.0,
        "min_factor": 0.50,
        "max_factor": 2.00
    }
def _stabilize_runtime_effect_params(effect_params: dict) -> dict:
    """
    对 DBN 状态影响因子进行节律保持型稳定化。

    修改目的：
    1. 避免 DBN 过度压缩等待时间和执行时间；
    2. 保留 DBN 对资源状态的动态表达能力；
    3. 降低 AED、CED、RED 的时间分布偏移；
    4. 保持 CAR 和 CTD 不再明显恶化。
    """

    import copy

    effect_params = copy.deepcopy(effect_params or {})

    # =====================================================
    # 1. 执行时间影响：允许轻微动态变化，但禁止明显压缩
    # =====================================================
    execution_time = effect_params.get("execution_time", {})

    execution_time.update({
        # 原始 factor > 1 时，允许较弱放大
        "upper_effect_strength": 0.18,

        # 原始 factor < 1 时，只允许非常弱的缩短
        # 这是降低 AED/RED 上升的关键
        "lower_effect_strength": 0.06,

        # 执行时间运行时最多缩短 3%，最多放大 10%
        "runtime_min_factor": 0.97,
        "runtime_max_factor": 1.10,

        # 单事件执行时间修正保护
        "max_relative_delta": 0.12,
        "max_abs_delta_min": 20.0,

        # 第一个活动不做执行时间压缩，避免影响 AED 的首段分布
        "protect_first_activity": True
    })

    effect_params["execution_time"] = execution_time

    # =====================================================
    # 2. 等待时间影响：只允许极弱修正
    # =====================================================
    waiting_time = effect_params.get("waiting_time", {})

    waiting_time.update({
        # 等待时间对 AED/RED/CTD 极其敏感，因此只保留弱动态修正
        "upper_effect_strength": 0.08,
        "lower_effect_strength": 0.04,

        # 等待时间最多缩短 2%，最多放大 6%
        "runtime_min_factor": 0.98,
        "runtime_max_factor": 1.06,

        # 单事件等待时间修正保护
        "max_relative_delta": 0.08,
        "max_abs_delta_min": 15.0,

        "protect_first_activity": True
    })

    effect_params["waiting_time"] = waiting_time

    # =====================================================
    # 3. 资源选择影响：进一步减弱，避免事件过度集中到少数高效资源
    # =====================================================
    resource_selection = effect_params.get("resource_selection", {})

    resource_selection.update({
        "upper_effect_strength": 0.15,
        "lower_effect_strength": 0.10,
        "runtime_min_factor": 0.90,
        "runtime_max_factor": 1.10
    })

    effect_params["resource_selection"] = resource_selection

    # =====================================================
    # 4. 动态容量：允许非常保守的容量下降，但保护第一个活动
    # =====================================================
    capacity = effect_params.get("capacity", {})

    capacity.update({
        # 不再完全禁止容量下降，否则会破坏真实拥塞节律
        "allow_capacity_reduction": True,

        # 只允许高疲劳时容量下降
        "capacity_reduction_policy": "high_fatigue_only",

        # 第一个活动不受容量下降影响，保护 CAR
        "protect_first_activity": True
    })

    effect_params["capacity"] = capacity

    return effect_params

def _learn_resource_selection_effect(
        dbn_features: pd.DataFrame
) -> dict:
    """
    从日志中学习资源状态对资源选择倾向的影响。

    注意：
    如果没有候选资源集合和未被选择资源的负样本，资源选择影响只能从
    “被实际选择资源的状态分布”中学习一个弱选择因子。
    更严格的做法是基于 df_features 中的 enabled_resources 构造
    chosen / not chosen 样本，然后学习多项选择模型。
    """

    df = dbn_features.copy()

    if df.empty:
        return _neutral_resource_selection_effect()

    factor_by_state = {}

    for state_var in [
        "fatigue_t",
        "skill_t",
        "efficiency_t",
        "availability_t"
    ]:
        logical_name = state_var.replace("_t", "")

        factor_by_state[logical_name] = _learn_frequency_factor_by_state(
            df=df,
            state_col=state_var,
            states=_states_for_variable(state_var),
            min_factor=0.50,
            max_factor=1.50
        )

    joint_factor = _learn_joint_frequency_factor(
        df=df,
        state_cols=[
            "fatigue_t",
            "skill_t",
            "efficiency_t",
            "availability_t"
        ],
        min_count=3,
        default_factor=1.0,
        min_factor=0.50,
        max_factor=1.50
    )

    return {
        "source": "event_log_selected_resource_distribution",
        "target": "resource_selection",
        "factor_by_state": factor_by_state,
        "joint_factor": joint_factor,
        "default_factor": 1.0,
        "min_factor": 0.50,
        "max_factor": 1.50,
        "note": (
            "This factor is learned from the states of actually selected "
            "resources. For a stricter resource-choice model, construct "
            "chosen/not-chosen samples from enabled resource sets."
        )
    }


def _learn_capacity_effect(
        dbn_features: pd.DataFrame
) -> dict:
    """
    从日志中学习不同疲劳状态下的动态容量。

    使用 workload_value_t 的高分位数估计资源在对应状态下的可承载并发量。
    """

    df = dbn_features.copy()

    if "workload_value_t" not in df.columns:
        return _neutral_capacity_effect()

    df["workload_value_t"] = pd.to_numeric(
        df["workload_value_t"],
        errors="coerce"
    )

    df.dropna(subset=["fatigue_t", "workload_value_t"], inplace=True)

    if df.empty:
        return _neutral_capacity_effect()

    capacity_by_fatigue = {}

    for fatigue_state in ["low", "medium", "high"]:
        df_s = df[df["fatigue_t"] == fatigue_state]

        if df_s.empty:
            capacity_by_fatigue[fatigue_state] = 1
            continue

        # workload_value_t 表示当前已有并发任务数；
        # 资源可接收当前任务，因此容量至少为 workload + 1。
        q = float(df_s["workload_value_t"].quantile(0.90))
        capacity_by_fatigue[fatigue_state] = max(1, int(round(q + 1)))

    return {
        "source": "event_log",
        "target": "dynamic_capacity",
        "default_capacity": 1,
        "capacity_by_fatigue": capacity_by_fatigue,
        "max_capacity": max(capacity_by_fatigue.values())
    }


def _learn_factor_by_state(
        df: pd.DataFrame,
        state_col: str,
        value_col: str,
        default_factor: float,
        min_factor: float,
        max_factor: float
) -> dict:
    """
    学习单个状态变量对某个连续比例变量的影响因子。
    """

    result = {}

    states = _states_for_variable(state_col)

    global_median = float(df[value_col].median())

    if global_median <= 0:
        global_median = 1.0

    for state in states:
        df_s = df[df[state_col] == state]

        if df_s.empty:
            result[state] = default_factor
            continue

        state_median = float(df_s[value_col].median())

        factor = state_median / global_median

        result[state] = float(
            np.clip(factor, min_factor, max_factor)
        )

    return result


def _learn_joint_factor(
        df: pd.DataFrame,
        state_cols: list[str],
        value_col: str,
        min_count: int,
        default_factor: float,
        min_factor: float,
        max_factor: float
) -> dict:
    """
    学习联合状态组合下的影响因子。
    """

    global_median = float(df[value_col].median())

    if global_median <= 0:
        global_median = 1.0

    result = {}

    grouped = df.groupby(state_cols)

    for state_tuple, df_g in grouped:
        if len(df_g) < min_count:
            continue

        if not isinstance(state_tuple, tuple):
            state_tuple = (state_tuple,)

        assignment = {
            col: str(value)
            for col, value in zip(state_cols, state_tuple)
        }

        key = _parent_key(assignment)

        factor = float(df_g[value_col].median()) / global_median

        result[key] = float(
            np.clip(factor, min_factor, max_factor)
        )

    return {
        "state_cols": state_cols,
        "table": result,
        "default_factor": default_factor
    }


def _learn_frequency_factor_by_state(
        df: pd.DataFrame,
        state_col: str,
        states: list[str],
        min_factor: float,
        max_factor: float
) -> dict:
    """
    根据被选择资源的状态频率学习选择倾向因子。

    如果某个状态在被选择资源中出现频率更高，则因子略大。
    """

    counts = df[state_col].value_counts(normalize=True).to_dict()

    uniform_prob = 1.0 / max(len(states), 1)

    result = {}

    for state in states:
        p = float(counts.get(state, 0.0))

        if uniform_prob <= 0:
            factor = 1.0
        else:
            factor = p / uniform_prob

        result[state] = float(
            np.clip(factor, min_factor, max_factor)
        )

    return result


def _learn_joint_frequency_factor(
        df: pd.DataFrame,
        state_cols: list[str],
        min_count: int,
        default_factor: float,
        min_factor: float,
        max_factor: float
) -> dict:
    """
    根据联合状态出现频率学习资源选择倾向因子。
    """

    result = {}

    total = len(df)

    if total <= 0:
        return {
            "state_cols": state_cols,
            "table": result,
            "default_factor": default_factor
        }

    grouped = df.groupby(state_cols)

    expected_uniform = 1.0

    for state_tuple, df_g in grouped:
        if len(df_g) < min_count:
            continue

        if not isinstance(state_tuple, tuple):
            state_tuple = (state_tuple,)

        assignment = {
            col: str(value)
            for col, value in zip(state_cols, state_tuple)
        }

        key = _parent_key(assignment)

        observed = len(df_g) / total

        factor = observed / expected_uniform

        result[key] = float(
            np.clip(factor, min_factor, max_factor)
        )

    return {
        "state_cols": state_cols,
        "table": result,
        "default_factor": default_factor
    }


def _states_for_variable(state_col: str) -> list[str]:
    """
    根据变量名返回离散状态空间。
    """

    if "availability" in state_col:
        return ["available", "unavailable"]

    return ["low", "medium", "high"]


def _neutral_execution_time_effect() -> dict:
    return {
        "source": "neutral",
        "target": "execution_time",
        "factor_by_state": {},
        "joint_factor": {
            "state_cols": [],
            "table": {},
            "default_factor": 1.0
        },
        "default_factor": 1.0,
        "min_factor": 0.50,
        "max_factor": 2.00
    }


def _neutral_waiting_time_effect() -> dict:
    return {
        "source": "neutral",
        "target": "waiting_time",
        "factor_by_state": {},
        "joint_factor": {
            "state_cols": [],
            "table": {},
            "default_factor": 1.0
        },
        "default_factor": 1.0,
        "min_factor": 0.50,
        "max_factor": 2.00
    }


def _neutral_resource_selection_effect() -> dict:
    return {
        "source": "neutral",
        "target": "resource_selection",
        "factor_by_state": {},
        "joint_factor": {
            "state_cols": [],
            "table": {},
            "default_factor": 1.0
        },
        "default_factor": 1.0,
        "min_factor": 0.50,
        "max_factor": 1.50
    }


def _neutral_capacity_effect() -> dict:
    return {
        "source": "neutral",
        "target": "dynamic_capacity",
        "default_capacity": 1,
        "capacity_by_fatigue": {
            "low": 1,
            "medium": 1,
            "high": 1
        },
        "max_capacity": 1
    }

def _default_state_effect_mapping() -> dict:
    """
    返回资源状态到仿真行为的默认映射参数。

    注意：
    这些参数不是 DBN 的 CPT，也不是资源状态转移概率。
    DBN 的核心参数是 structure 和 cpds。
    本函数仅定义 DBN 推断得到的离散资源状态
    如何影响资源选择、等待时间和执行时间。

    默认值只表示合理的方向性先验，
    后续应通过验证集调参、贝叶斯优化或日志统计学习进行校准。
    """

    return {
        # 将离散状态映射为数值强度，供运行时调整因子使用
        "state_score": {
            "low": 0.0,
            "medium": 0.5,
            "high": 1.0,
            "available": 1.0,
            "unavailable": 0.0
        },

        "execution_time": {
            "fatigue_weight": 0.35,
            "skill_weight": 0.20,
            "efficiency_weight": 0.25,
            "workload_weight": 0.10,
            "min_factor": 0.50,
            "max_factor": 2.00
        },

        "waiting_time": {
            "fatigue_weight": 0.20,
            "workload_weight": 0.15,
            "min_factor": 0.50,
            "max_factor": 2.00
        },

        "resource_selection": {
            "fatigue_penalty": 0.40,
            "skill_bonus": 0.30,
            "efficiency_bonus": 0.20,
            "availability_penalty": 0.80,
            "min_factor": 0.10,
            "max_factor": 3.00
        },

        "capacity": {
            "default_capacity": 1,
            "multitasking_capacity": 2,
            "high_fatigue_capacity": 1,
            "high_fatigue_threshold": "high"
        }
    }


def _normalize_dbn_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DBN 特征表中用于 CPT 的离散变量统一转为字符串。
    """

    df = df.copy()

    discrete_cols = [
        "resource",
        "activity_t",

        "fatigue_t",
        "workload_t",
        "skill_t",
        "efficiency_t",
        "availability_t",

        "idle_gap_t",
        "work_duration_t",
        "hour_t",
        "weekday_t",

        "fatigue_t1",
        "workload_t1",
        "skill_t1",
        "efficiency_t1",
        "availability_t1"
    ]

    for col in discrete_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def _infer_states(values: pd.Series, fallback: list[str]) -> list[str]:
    """
    从数据中推断离散状态空间。
    如果数据为空，则使用 fallback。
    """

    states = sorted(
        [
            str(v)
            for v in values.dropna().unique().tolist()
        ]
    )

    if not states:
        states = [str(v) for v in fallback]

    return states


def _estimate_distribution(
        values: pd.Series,
        states: list[str],
        smoothing: float
) -> dict:
    """
    估计一个离散变量的平滑概率分布。
    """

    counts = {
        state: smoothing
        for state in states
    }

    for v in values.dropna().astype(str):
        if v in counts:
            counts[v] += 1.0

    total = sum(counts.values())

    if total <= 0:
        return {
            state: 1.0 / len(states)
            for state in states
        }

    return {
        state: counts[state] / total
        for state in states
    }


def _normalize_counts(
        counts: dict,
        states: list[str],
        smoothing: float
) -> dict:
    """
    将条件计数转为概率分布，并进行拉普拉斯平滑。
    """

    smoothed_counts = {
        state: float(counts.get(state, 0.0)) + smoothing
        for state in states
    }

    total = sum(smoothed_counts.values())

    if total <= 0:
        return {
            state: 1.0 / len(states)
            for state in states
        }

    return {
        state: smoothed_counts[state] / total
        for state in states
    }


def _all_parent_assignments(parent_states: dict) -> list[dict]:
    """
    生成所有父节点状态组合。
    """

    parents = list(parent_states.keys())
    states_product = itertools.product(
        *[
            parent_states[parent]
            for parent in parents
        ]
    )

    assignments = []

    for state_tuple in states_product:
        assignment = {
            parent: str(state)
            for parent, state in zip(parents, state_tuple)
        }
        assignments.append(assignment)

    return assignments


def _parent_key(parent_assignment: dict) -> str:
    """
    将父节点状态组合转换为稳定 JSON 字符串 key。

    使用 JSON 字符串作为 key 的好处是：
    1. 可以安全保存到 JSON；
    2. 运行时只要用同样方法构造 key，就能查询 CPT；
    3. 可以避免活动名中包含特殊分隔符时出错。
    """

    normalized = {
        str(k): str(v)
        for k, v in parent_assignment.items()
    }

    return json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False
    )


def _mode_or_default(values: pd.Series, default: str) -> str:
    """
    返回众数；如果为空则返回默认值。
    """

    if values is None or values.empty:
        return default

    mode_values = values.dropna().astype(str).mode()

    if mode_values.empty:
        return default

    return str(mode_values.iloc[0])


def _disabled_params(reason: str) -> dict:
    """
    返回一个关闭状态的动态资源状态 DBN 参数。
    """

    return {
        "enabled": False,
        "model_type": "discrete_two_slice_dbn",
        "reason": reason,
        "state_variables": [
            "fatigue",
            "workload",
            "skill",
            "efficiency",
            "availability"
        ],
        "context_variables": [
            "activity",
            "idle_gap",
            "work_duration",
            "hour",
            "weekday"
        ],
        "variable_states": {},
        "structure": {},
        "cpds": {},
        "initial_state": {},
        "effect_params": {}
    }