"""

DBN 动态资源状态运行时模块。

该模块在仿真过程中维护资源状态，并根据：
1. DBN 条件概率表 cpds 更新资源状态；
2. 日志学习得到的 state-effect mapping 调整资源选择、等待时间、执行时间和动态容量。
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np


class DynamicResourceStateRuntime:
    """
    DBN 动态资源状态运行时。

    与旧版本不同：
    旧版本使用疲劳增长率、恢复率、技能增长率等公式更新状态；
    当前版本使用离散二时间片 DBN 的 CPT 进行状态采样更新。
    """

    def __init__(
            self,
            resource_state_params: dict,
            resources: list,
            activities: list,
            multitasking_resources: list = None,
            random_seed: Optional[int] = None
    ):
        self.params = resource_state_params or {}

        self.resources = [str(r) for r in resources]
        self.activities = [str(a) for a in activities]
        self.multitasking_resources = [
            str(r)
            for r in (multitasking_resources or [])
        ]

        self.model_type = self.params.get("model_type", "")

        self.cpds = self.params.get("cpds", {})
        self.effect_params = self.params.get("effect_params", {})
        self.variable_states = self.params.get("variable_states", {})
        self.discretization = self.params.get("discretization", {})

        # =====================================================
        # 运行时稳定化选项
        # =====================================================
        self.runtime_options = self.params.get("runtime_options", {})

        self.state_transition_mode = self.runtime_options.get(
            "state_transition_mode",
            "tempered_sample"
        )

        self.cpt_temperature = float(
            self.runtime_options.get("cpt_temperature", 1.5)
        )

        self.rng = np.random.default_rng(random_seed)

        self.states = {}

        initial_state = self.params.get("initial_state", {})

        for resource in self.resources:
            init_r = initial_state.get(resource, {})

            self.states[resource] = {
                "fatigue": str(init_r.get("fatigue", "low")),
                "workload": str(init_r.get("workload", "low")),
                "efficiency": str(init_r.get("efficiency", "medium")),
                "availability": str(init_r.get("availability", "available")),
                "skill": {
                    str(activity): str(
                        init_r.get("skill", {}).get(activity, "medium")
                    )
                    for activity in self.activities
                },
                "last_end_time": None,
                "last_recovery_time": None
            }

    def guard_temporal_adjustment(
            self,
            base_value: float,
            adjusted_value: float,
            section_name: str,
            is_first_activity: bool = False
    ) -> float:
        """
        对单个事件的 DBN 时间修正进行保护。

        目的：
        1. 防止等待时间或执行时间被 DBN 一次性压缩过多；
        2. 防止事件在绝对时间轴和 case 内部时间轴上发生大幅偏移；
        3. 降低 AED、CED、RED。
        """

        section = self.effect_params.get(section_name, {})

        if is_first_activity and bool(section.get("protect_first_activity", False)):
            return float(base_value)

        try:
            base_value = float(base_value)
            adjusted_value = float(adjusted_value)
        except Exception:
            return adjusted_value

        if base_value <= 0:
            return max(0.0, adjusted_value)

        max_relative_delta = float(
            section.get("max_relative_delta", 0.15)
        )

        max_abs_delta_min = float(
            section.get("max_abs_delta_min", 30.0)
        )

        lower_bound = max(
            0.0,
            base_value * (1.0 - max_relative_delta) - max_abs_delta_min
        )

        upper_bound = (
                base_value * (1.0 + max_relative_delta) + max_abs_delta_min
        )

        return float(
            np.clip(
                adjusted_value,
                lower_bound,
                upper_bound
            )
        )
    def recover_by_idle_time(self, resource: str, current_time):
        """
        根据资源空闲间隔用 DBN CPT 更新疲劳状态。

        注意：
        这里不再使用疲劳恢复公式，而是查询：
            P(Fatigue_{t+1} | Fatigue_t, Workload_t, IdleGap_t)
        """

        resource = str(resource)

        state = self.states.get(resource)
        if state is None:
            return

        last_end = state.get("last_end_time")

        if last_end is None:
            return

        # 避免同一时刻重复恢复
        last_recovery_time = state.get("last_recovery_time")
        if last_recovery_time is not None and current_time <= last_recovery_time:
            return

        idle_min = (current_time - last_end).total_seconds() / 60.0

        if idle_min <= 0:
            return

        idle_gap_state = self._discretize_runtime_value(
            variable_name="idle_gap_min_t",
            value=idle_min,
            default_state="low"
        )

        parent_assignment = {
            "fatigue_t": state.get("fatigue", "low"),
            "workload_t": state.get("workload", "low"),
            "idle_gap_t": idle_gap_state
        }

        old_fatigue = state.get("fatigue", "low")

        sampled_fatigue = self._sample_from_cpd(
            cpd_name="fatigue",
            parent_assignment=parent_assignment,
            default_state=old_fatigue
        )

        # 空闲恢复不应使疲劳恶化。
        # 如果 CPT 采样导致空闲后疲劳升高，则回退到原疲劳状态。
        state["fatigue"] = self._monotone_idle_recovery(
            old_fatigue=old_fatigue,
            sampled_fatigue=sampled_fatigue,
            idle_gap_state=idle_gap_state
        )

        state["last_recovery_time"] = current_time

    def get_capacity(
            self,
            resource: str,
            base_multitasking: bool,
            current_workload: int = 0,
            is_first_activity: bool = False
    ) -> int:
        """
        根据 DBN 当前疲劳状态返回动态容量。

        当前策略：
        1. 第一个活动不降低容量，保护 CAR；
        2. 非多任务资源容量保持 1；
        3. 多任务资源只在 high fatigue 且已有负载时允许容量下降；
        4. 避免完全取消资源竞争，从而降低 AED/CED/RED 上升。
        """

        resource = str(resource)

        base_capacity = 2 if base_multitasking else 1

        if not base_multitasking:
            return 1

        capacity_params = self.effect_params.get("capacity", {})

        if is_first_activity and bool(
                capacity_params.get("protect_first_activity", True)
        ):
            return base_capacity

        fatigue_state = self.states.get(resource, {}).get("fatigue", "low")

        allow_reduction = bool(
            capacity_params.get("allow_capacity_reduction", True)
        )

        policy = capacity_params.get(
            "capacity_reduction_policy",
            "high_fatigue_only"
        )

        if not allow_reduction:
            return base_capacity

        if policy == "high_fatigue_only":
            if fatigue_state == "high" and int(current_workload) >= base_capacity:
                return max(1, base_capacity - 1)

            return base_capacity

        capacity_by_fatigue = capacity_params.get("capacity_by_fatigue", {})

        learned_capacity = int(
            capacity_by_fatigue.get(
                fatigue_state,
                base_capacity
            )
        )

        return max(1, learned_capacity)

    def adjust_resource_weights(
            self,
            resource_weights: dict,
            enabled_resources: list,
            activity: str
    ) -> dict:
        """
        使用日志学习得到的资源选择 effect mapping 调整资源权重。
        """

        activity = str(activity)

        selection_params = self.effect_params.get("resource_selection", {})

        adjusted_weights = {}

        for resource in enabled_resources:
            resource = str(resource)

            base_weight = float(resource_weights.get(resource, 0.0))

            state = self.states.get(resource, {})

            raw_factor = self._state_effect_factor(
                section=selection_params,
                state_assignment={
                    "fatigue_t": state.get("fatigue", "low"),
                    "skill_t": state.get("skill", {}).get(activity, "medium"),
                    "efficiency_t": state.get("efficiency", "medium"),
                    "availability_t": state.get("availability", "available")
                }
            )

            factor = self._shrink_effect_factor(
                raw_factor=raw_factor,
                section=selection_params
            )

            adjusted_weights[resource] = max(0.0, base_weight * factor)

        if sum(adjusted_weights.values()) <= 0:
            return {
                str(r): float(resource_weights.get(r, 1.0))
                for r in enabled_resources
            }

        return adjusted_weights

    def adjust_waiting_time(
            self,
            resource: str,
            base_waiting_time: float,
            workload: int
    ) -> float:
        """
        使用日志学习得到的等待时间 effect mapping 调整等待时间。
        """

        resource = str(resource)

        state = self.states.get(resource, {})

        workload_state = self._discretize_runtime_value(
            variable_name="workload_value_t",
            value=workload,
            default_state=state.get("workload", "low")
        )

        waiting_params = self.effect_params.get("waiting_time", {})

        if float(base_waiting_time) <= 0:
            return 0.0

        raw_factor = self._state_effect_factor(
            section=waiting_params,
            state_assignment={
                "fatigue_t": state.get("fatigue", "low"),
                "workload_t": workload_state,
                "availability_t": state.get("availability", "available")
            }
        )

        factor = self._shrink_effect_factor(
            raw_factor=raw_factor,
            section=waiting_params
        )

        return max(0.0, float(base_waiting_time) * factor)
    def adjust_execution_time(
            self,
            resource: str,
            activity: str,
            base_execution_time: float,
            workload: int
    ) -> float:
        """
        使用日志学习得到的执行时间 effect mapping 调整执行时间。
        """

        resource = str(resource)
        activity = str(activity)

        state = self.states.get(resource, {})

        workload_state = self._discretize_runtime_value(
            variable_name="workload_value_t",
            value=workload,
            default_state=state.get("workload", "low")
        )

        execution_params = self.effect_params.get("execution_time", {})

        raw_factor = self._state_effect_factor(
            section=execution_params,
            state_assignment={
                "fatigue_t": state.get("fatigue", "low"),
                "workload_t": workload_state,
                "skill_t": state.get("skill", {}).get(activity, "medium"),
                "efficiency_t": state.get("efficiency", "medium")
            }
        )

        factor = self._shrink_effect_factor(
            raw_factor=raw_factor,
            section=execution_params
        )

        return max(1.0, float(base_execution_time) * factor)

    def update_after_execution(
            self,
            resource: str,
            activity: str,
            start_time,
            end_time,
            workload: int = 0
    ):
        """
        活动完成后使用 DBN CPT 更新资源状态。

        更新顺序：
        1. 根据当前状态和上下文采样 fatigue_{t+1}；
        2. 采样 workload_{t+1}；
        3. 采样 skill_{t+1}；
        4. 基于 fatigue_{t+1}, skill_{t+1}, workload_{t+1} 采样 efficiency_{t+1}；
        5. 采样 availability_{t+1}。
        """

        resource = str(resource)
        activity = str(activity)

        state = self.states.get(resource)
        if state is None:
            return

        last_end = state.get("last_end_time")

        if last_end is None:
            idle_min = 0.0
        else:
            idle_min = max(
                0.0,
                (start_time - last_end).total_seconds() / 60.0
            )

        duration_min = max(
            0.0,
            (end_time - start_time).total_seconds() / 60.0
        )

        idle_gap_state = self._discretize_runtime_value(
            variable_name="idle_gap_min_t",
            value=idle_min,
            default_state="low"
        )

        workload_state = self._discretize_runtime_value(
            variable_name="workload_value_t",
            value=workload,
            default_state=state.get("workload", "low")
        )

        work_duration_state = self._discretize_runtime_value(
            variable_name="work_duration_min_t",
            value=duration_min,
            default_state="medium"
        )

        hour_state = str(start_time.hour)
        weekday_state = str(start_time.weekday())

        fatigue_t = state.get("fatigue", "low")
        workload_t = workload_state
        skill_t = state.get("skill", {}).get(activity, "medium")
        efficiency_t = state.get("efficiency", "medium")
        availability_t = state.get("availability", "available")

        # 1. Fatigue_{t+1}
        fatigue_t1 = self._sample_from_cpd(
            cpd_name="fatigue",
            parent_assignment={
                "fatigue_t": fatigue_t,
                "workload_t": workload_t,
                "idle_gap_t": idle_gap_state
            },
            default_state=fatigue_t
        )

        # 2. Workload_{t+1}
        workload_t1 = self._sample_from_cpd(
            cpd_name="workload",
            parent_assignment={
                "workload_t": workload_t,
                "activity_t": activity,
                "hour_t": hour_state
            },
            default_state=workload_t
        )

        # 3. Skill_{t+1}
        skill_t1 = self._sample_from_cpd(
            cpd_name="skill",
            parent_assignment={
                "skill_t": skill_t,
                "activity_t": activity
            },
            default_state=skill_t
        )

        # 4. Efficiency_{t+1}
        efficiency_t1 = self._sample_from_cpd(
            cpd_name="efficiency",
            parent_assignment={
                "fatigue_t1": fatigue_t1,
                "skill_t1": skill_t1,
                "workload_t1": workload_t1
            },
            default_state=efficiency_t
        )

        # 5. Availability_{t+1}
        availability_t1 = self._sample_from_cpd(
            cpd_name="availability",
            parent_assignment={
                "availability_t": availability_t,
                "hour_t": hour_state,
                "weekday_t": weekday_state
            },
            default_state=availability_t
        )

        state["fatigue"] = fatigue_t1
        state["workload"] = workload_t1
        state["efficiency"] = efficiency_t1
        state["availability"] = availability_t1

        if "skill" not in state:
            state["skill"] = {}

        state["skill"][activity] = skill_t1

        state["last_end_time"] = end_time
        state["last_recovery_time"] = None

    def _sample_from_cpd(
            self,
            cpd_name: str,
            parent_assignment: dict,
            default_state: str
    ) -> str:
        """
        根据 CPT 和父节点状态采样目标状态。
        """

        cpd = self.cpds.get(cpd_name)

        if not cpd:
            return default_state

        key = self._parent_key(parent_assignment)

        distribution = cpd.get("table", {}).get(key)

        if distribution is None:
            distribution = cpd.get("default", None)

        if not distribution:
            return default_state

        states = list(distribution.keys())
        probs = np.array(
            [float(distribution[s]) for s in states],
            dtype=float
        )

        total = probs.sum()

        if total <= 0:
            return default_state

        probs = probs / total

        if self.state_transition_mode == "map":
            return str(states[int(np.argmax(probs))])

        if self.state_transition_mode == "tempered_sample":
            temperature = max(0.1, float(self.cpt_temperature))

            # temperature > 1 时，概率分布更平滑；
            # temperature < 1 时，概率分布更尖锐。
            adjusted_probs = np.power(probs, 1.0 / temperature)

            total_adjusted = adjusted_probs.sum()

            if total_adjusted <= 0:
                return default_state

            adjusted_probs = adjusted_probs / total_adjusted

            return str(
                self.rng.choice(states, p=adjusted_probs)
            )

        return str(
            self.rng.choice(states, p=probs)
        )

    def _shrink_effect_factor(
            self,
            raw_factor: float,
            section: dict
    ) -> float:
        """
        对 DBN 原始影响因子进行非对称收缩。

        关键思想：
        1. factor > 1 表示放大等待/执行时间；
        2. factor < 1 表示缩短等待/执行时间；
        3. 当前 AED、CED、RED 上升说明时间链被压缩或前移，
           因此对 factor < 1 的缩短作用要更弱。
        """

        try:
            raw_factor = float(raw_factor)
        except Exception:
            raw_factor = 1.0

        if raw_factor >= 1.0:
            effect_strength = float(
                section.get(
                    "upper_effect_strength",
                    section.get("effect_strength", 1.0)
                )
            )
        else:
            effect_strength = float(
                section.get(
                    "lower_effect_strength",
                    section.get("effect_strength", 1.0)
                )
            )

        runtime_min_factor = float(
            section.get(
                "runtime_min_factor",
                section.get("min_factor", 0.25)
            )
        )

        runtime_max_factor = float(
            section.get(
                "runtime_max_factor",
                section.get("max_factor", 4.0)
            )
        )

        factor = 1.0 + effect_strength * (raw_factor - 1.0)

        return float(
            np.clip(
                factor,
                runtime_min_factor,
                runtime_max_factor
            )
        )

    def _fatigue_level(self, fatigue_state: str) -> int:
        """
        将疲劳状态映射为有序等级。
        """

        mapping = {
            "low": 0,
            "medium": 1,
            "high": 2
        }

        return int(mapping.get(str(fatigue_state), 1))

    def _fatigue_from_level(self, level: int) -> str:
        """
        将疲劳等级映射回状态。
        """

        level = int(np.clip(level, 0, 2))

        if level == 0:
            return "low"

        if level == 1:
            return "medium"

        return "high"

    def _lower_fatigue_one_level(self, fatigue_state: str) -> str:
        """
        疲劳降低一级。
        """

        level = self._fatigue_level(fatigue_state)

        return self._fatigue_from_level(level - 1)

    def _monotone_idle_recovery(
            self,
            old_fatigue: str,
            sampled_fatigue: str,
            idle_gap_state: str
    ) -> str:
        """
        空闲恢复的单调约束。

        资源空闲后，疲劳不应变得更严重。
        如果 idle_gap 为 high，则额外允许疲劳降低一级。
        """

        old_level = self._fatigue_level(old_fatigue)
        sampled_level = self._fatigue_level(sampled_fatigue)

        # 空闲后不能更疲劳
        if sampled_level > old_level:
            sampled_fatigue = old_fatigue

        # 长时间空闲时，疲劳至少尝试恢复一级
        if str(idle_gap_state) == "high":
            sampled_fatigue = self._lower_fatigue_one_level(sampled_fatigue)

        return sampled_fatigue
    def _state_effect_factor(
            self,
            section: dict,
            state_assignment: dict
    ) -> float:
        """
        根据日志学习得到的 effect mapping 查询状态影响因子。

        优先使用 joint_factor；
        如果联合状态查不到，则退化为单变量因子的乘积。
        """

        default_factor = float(section.get("default_factor", 1.0))
        min_factor = float(section.get("min_factor", 0.25))
        max_factor = float(section.get("max_factor", 4.0))

        joint_factor = section.get("joint_factor", {})
        joint_cols = joint_factor.get("state_cols", [])
        joint_table = joint_factor.get("table", {})

        if joint_cols:
            joint_assignment = {
                col: str(state_assignment.get(col, "medium"))
                for col in joint_cols
            }

            key = self._parent_key(joint_assignment)

            if key in joint_table:
                return float(
                    np.clip(
                        joint_table[key],
                        min_factor,
                        max_factor
                    )
                )

        factor_by_state = section.get("factor_by_state", {})

        factor = default_factor

        for col, state_value in state_assignment.items():
            logical_name = col.replace("_t", "").replace("_t1", "")

            factor_table = factor_by_state.get(logical_name, {})

            factor *= float(
                factor_table.get(
                    str(state_value),
                    1.0
                )
            )

        return float(
            np.clip(
                factor,
                min_factor,
                max_factor
            )
        )

    def _discretize_runtime_value(
            self,
            variable_name: str,
            value: float,
            default_state: str = "medium"
    ) -> str:
        """
        将运行时连续值离散化为 low / medium / high。

        如果 discovery 阶段保存了阈值，则使用阈值；
        否则采用简单兜底规则。
        """

        metadata = self.discretization.get(variable_name, {})

        q1 = metadata.get("q1", None)
        q2 = metadata.get("q2", None)

        try:
            value = float(value)
        except Exception:
            return default_state

        if q1 is None or q2 is None:
            if value <= 0:
                return "low"
            if value <= 1:
                return "medium"
            return "high"

        q1 = float(q1)
        q2 = float(q2)

        if value <= q1:
            return "low"

        if value <= q2:
            return "medium"

        return "high"

    def _parent_key(self, parent_assignment: dict) -> str:
        normalized = {
            str(k): str(v)
            for k, v in parent_assignment.items()
        }

        return json.dumps(
            normalized,
            sort_keys=True,
            ensure_ascii=False
        )