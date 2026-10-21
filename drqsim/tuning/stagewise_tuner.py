# drq/tuning/stagewise_tuner.py

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import os
import re
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)
STAGEWISE_TUNER_BUILD_TAG = "2026-09-21-triggered-arrival-recency-stage3b-recent-confirm-v1"

# ------------------------------------------------------------------
# 基础配置
# ------------------------------------------------------------------
CONTROL_FLOW_STAGE_METRICS = ("cfld", "ngd", "3gd")
FULL_DISCOVERY_PROFILE = "full"
CONTROL_FLOW_DISCOVERY_PROFILE = "control_flow"

# 队列调度策略：队列决策树深度优化时必须启用队列调度
QUEUE_STRATEGIES = (
    {
        "name": "balanced",
        "weights": {"tree": 0.3, "frequency": 0.5, "availability": 0.2},
    },
)

# ------------------------------------------------------------------
# 分阶段模型复杂度候选
# ------------------------------------------------------------------
# 控制流允许稍深一些；运行时三棵树统一限制在 1~4，避免小日志过拟合。
CONTROL_FLOW_TREE_DEPTH_CANDIDATES = (1, 2, 3, 4)
RUNTIME_TREE_DEPTH_CANDIDATES = (1, 2, 3, 4)
PRIORITY_TREE_DEPTH_CANDIDATES = (1, 2, 3, 4)
# Backward-compatible alias for older callers/output readers.
CPT_DEPTH_CANDIDATES = PRIORITY_TREE_DEPTH_CANDIDATES

# 向后兼容旧调用：tree_depth_candidates 仍表示控制流树候选。
TREE_DEPTH_CANDIDATES = CONTROL_FLOW_TREE_DEPTH_CANDIDATES

# ------------------------------------------------------------------
# Stage 1：控制流结构搜索（Petri 网噪声阈值 + 决策点树深度）
# ------------------------------------------------------------------
# 轻量联合搜索：每个 Optuna trial 同时搜索 IM noise_threshold + control-flow tree_depth
# 默认只搜索低噪声区间，降低 BPI12/CVS 上重复 Petri 结构评估的成本。
NOISE_THRESHOLD_RANGE = (0.0, 0.50)
STRUCTURE_OPTUNA_N_TRIALS = 50
STRUCTURE_OPTUNA_SAMPLER = "tpe"
STRUCTURE_OPTUNA_SEED = 72
STRUCTURE_VALIDATION_BUDGETS = (1, 2)
STRUCTURE_REDUCTION_FACTOR = 2

# 控制流结构最终选择正则化
ACTIVITY_COVERAGE_PENALTY_WEIGHT = 0.25
NOISE_PENALTY_WEIGHT = 0.02
LOW_NOISE_PREFERENCE_ENABLED = True
LOW_NOISE_PREFERENCE_TOL = 0.05
STRUCTURE_CONFIRMATION_ENABLED = True
STRUCTURE_CONFIRMATION_TOP_K = 3
STRUCTURE_CONFIRMATION_RUNS = 2

# 在 Stage 1 结束时固定控制流结构，Stage 2 不再切换结构
FIX_STRUCTURE_BEFORE_RUNTIME = True
TOP_K_STRUCTURES = 1

# ------------------------------------------------------------------
# Stage 2：运行时指标与 guard（用于评估到达/队列/延迟器参数）
# ------------------------------------------------------------------
RUNTIME_METRICS = ("aed", "ced", "red", "car", "ctd", "cwd")

# 分阶段目标：先让各白盒子模型用与其职责最直接相关的指标选复杂度。
ARRIVAL_STAGE_METRICS = ("car",)
ARRIVAL_STAGE_METRIC_WEIGHTS = {"car": 1.0}
RESOURCE_ALLOCATION_STAGE_METRICS = ("red", "cwd", "aed", "ctd")
RESOURCE_ALLOCATION_STAGE_METRIC_WEIGHTS = {
    "red": 2.0,
    "cwd": 1.5,
    "aed": 0.5,
    "ctd": 0.5,
}

# Queue dispatch 与 External Delay 共享同一个运行时 ready-queue 状态，因此最终联合阶段使用完整运行时指标。
DISPATCH_EXTERNAL_STAGE_METRICS = RUNTIME_METRICS
# Backward-compatible alias.
CPT_EXTERNAL_STAGE_METRICS = DISPATCH_EXTERNAL_STAGE_METRICS

RUNTIME_METRIC_WEIGHTS = {
    "aed": 0.8,
    "ced": 0.6,
    "red": 1.8,
    "car": 1.0,
    "ctd": 2.2,
    "cwd": 0.6,
}

# duration/arrival guard 仅作诊断/软惩罚，不硬剪枝
CASE_DURATION_MIN_RATIO = 0.45
CASE_DURATION_MAX_RATIO = 2.00
RUNTIME_GUARD_SOFT_PENALTY_WEIGHT = 0.25

ARRIVAL_GUARD_ENABLED = True
ARRIVAL_SPAN_MIN_RATIO = 0.70
ARRIVAL_SPAN_MAX_RATIO = 1.50
ARRIVAL_CASE_COUNT_MIN_RATIO = 0.85

# ------------------------------------------------------------------
# Stage 2e：Validation-controlled Arrival Time Scale
# ------------------------------------------------------------------
# The Arrival decision tree itself is left unchanged.  After Stage 2d has
# selected the runtime-recency horizon, validation may apply one coarse global
# multiplicative correction to sampled inter-arrival times.  Scale=1.0 is the
# mandatory identity baseline for every dataset; test data are never used.
ARRIVAL_TIME_SCALE_CALIBRATION_ENABLED = True
ARRIVAL_TIME_SCALE_CANDIDATES = (0.75, 0.875, 1.00, 1.125, 1.25)
ARRIVAL_TIME_SCALE_EVAL_RUNS = 3
ARRIVAL_TIME_SCALE_MIN_IMPROVEMENT = 0.05
ARRIVAL_TIME_SCALE_TIE_TOLERANCE = 0.02

# ------------------------------------------------------------------
# Stage 2e0：Triggered Arrival Recency
# ------------------------------------------------------------------
# 为尽量不影响已经稳定的日志，Arrival recent-window 不是默认全局启用：
# 先用 train -> chronological validation 的真实 case-arrival 统计做零仿真 drift 检测；
# 只有触发明显漂移时，才评估 recent Arrival 候选。test_log 永不参与。
ARRIVAL_RECENCY_ENABLED = True
ARRIVAL_RECENCY_RATIO_CANDIDATES = (0.25, 0.40, 0.60, 1.00)
ARRIVAL_RECENCY_MIN_CASES = 30
ARRIVAL_RECENCY_EVAL_RUNS = 2
ARRIVAL_RECENCY_MIN_IMPROVEMENT = 0.10
ARRIVAL_RECENCY_TIE_TOLERANCE = 0.02

# 保守 trigger：一般需要 >=2 个 moderate signals；任一 strong signal 也可触发。
ARRIVAL_RECENCY_TRIGGER_MEAN_SHIFT = 0.15
ARRIVAL_RECENCY_TRIGGER_P90_SHIFT = 0.20
ARRIVAL_RECENCY_TRIGGER_WEEKDAY_HOUR_JACCARD = 0.90
ARRIVAL_RECENCY_TRIGGER_STRONG_MEAN_SHIFT = 0.25
ARRIVAL_RECENCY_TRIGGER_STRONG_P90_SHIFT = 0.35
ARRIVAL_RECENCY_TRIGGER_MIN_SIGNALS = 2
# 小日志的 train/validation arrival 统计方差太大，直接保持旧路径。
ARRIVAL_RECENCY_TRIGGER_MIN_TRAIN_CASES = 1000
ARRIVAL_RECENCY_TRIGGER_MIN_VALIDATION_CASES = 200

# Drift 触发后先用真实 arrival 统计预筛选 recent ratios，再只对少数候选跑 CAR 仿真。
# 3 = identity baseline 1.0 + 最多 2 个 recent ratios。
ARRIVAL_RECENCY_MAX_SIM_CANDIDATES = 3

# ------------------------------------------------------------------
# 大日志调参：sample search + full confirmation
# ------------------------------------------------------------------
# 只对 train_validation 总 case 数 > 5000 的日志启用。
# 小/中日志完全保持原始完整日志调参流程。
TUNING_SAMPLING_ENABLED = True
TUNING_LARGE_LOG_THRESHOLD = 5000

# BPIC2017 约 2.45 万 cases：搜索阶段总共使用约 5000 cases。
# 按原 80/20 train-validation 比例分配，约为 4000 train + 1000 validation。
TUNING_SAMPLE_TOTAL_CASES = 5000
TUNING_SAMPLE_MIN_TRAIN_CASES = 1000
TUNING_SAMPLE_MIN_VALIDATION_CASES = 500

# 为降低采样改变最优参数的风险，sample search 只负责缩小候选集；
# 最终候选必须在完整 train + 完整 validation 上重新发现/确认。
TUNING_STRUCTURE_FULL_CONFIRM_TOP_K = 3
TUNING_RUNTIME_FULL_CONFIRM_TOP_K = 3
TUNING_DISPATCH_EXTERNAL_FULL_CONFIRM_TOP_K = 6
# Backward-compatible alias.
TUNING_CPT_EXTERNAL_FULL_CONFIRM_TOP_K = TUNING_DISPATCH_EXTERNAL_FULL_CONFIRM_TOP_K

# 采样采用相邻时间窗口，而不是跨全年随机抽 case：
# train 取尾部窗口，validation 取头部窗口，尽量保持真实 inter-arrival / workload 语义。
TUNING_SAMPLE_POLICY = "adjacent_temporal_windows"

# ------------------------------------------------------------------
# External Delay：Stage 2c 采用 data-driven dispatch-aware 搜索。
# 若数据选择 CPT，则联合搜索 priority-tree depth × quantile；若数据选择 FCFS，则只搜索 quantile。
# Stage 3 固定该 quantile，仅恢复 Top-K activity-level scale 校准。
# ------------------------------------------------------------------
# legacy quantile-calibration helper 保留用于兼容；新 run() 不会二次搜索 quantile。
FINAL_REDISCOVERY_CALIBRATION_ENABLED = False
FINAL_CALIBRATION_METRICS = RUNTIME_METRICS

# External Delay 长尾候选：保留 0.90~0.99，不再允许 1.0 完全不截尾。
EXTERNAL_DELAY_QUANTILE_CANDIDATES = (0.98,0.99,1.0,)

# Legacy final-calibration helper 保留的 duration 权重。
FINAL_CALIBRATION_DURATION_LOG_WEIGHT = 2.0

# 恢复 activity-level scale 校准：与 SIMOD 的 activity-specific delay scaling 思路一致。
# 本实现采用自适应 Top-K + coordinate search 控制计算成本。
TOP_ACTIVITY_SCALE_CALIBRATION_ENABLED = True
TOP_ACTIVITY_SCALE_RATIO = 0.5
TOP_ACTIVITY_SCALE_MIN_K = 5
TOP_ACTIVITY_SCALE_MAX_K = 8
# MP{0, 0.25, 0.5, 0.75, 1.0}，P2P：{0.25, 0.5, 0.75, 1.0, 1.25, 1.5,2,2.5,3},ACR :{0, 0.25, 0.5, 0.75, 1.0},
# CVS:{0.5, 0.75, 1.0,1.25,1.5,2.0,3.0,4.0,5.0}, BPIC12:{0.25, 0.5, 0.75, 1.0, 1.25}, BPIC17:{0.25, 0.5, 0.75, 1.0, 1.25}
TOP_ACTIVITY_SCALE_CANDIDATES = (0, 0.25, 0.5, 0.75, 1.0,1.125,1.25)
TOP_ACTIVITY_SCALE_MIN_IMPROVEMENT = 0.005

# ------------------------------------------------------------------
# Stage 2d：Adaptive Runtime Recency
# ------------------------------------------------------------------
# Full history remains the default. A shorter recent window is accepted only
# when chronological validation shows a meaningful improvement. Control flow,
# resource universe, DBN, queue allocation, and dispatch stay full-history.
RUNTIME_RECENCY_ENABLED = True
RUNTIME_RECENCY_RATIO_CANDIDATES = (0.25, 0.40, 1.00)
RUNTIME_RECENCY_MIN_CASES = 30
RUNTIME_RECENCY_EVAL_RUNS = 2
RUNTIME_RECENCY_MIN_IMPROVEMENT = 0.03
RUNTIME_RECENCY_TIE_TOLERANCE = 0.01
RUNTIME_RECENCY_MODULES = ("service_time", "external_delay")
# Recency selection uses a two-level duration guard.  The soft band is the
# preferred temporal scale and is already penalized by the Stage-2d scalar
# objective when a candidate falls outside it.  A candidate is hard-rejected
# only when it leaves the wider hard band.  This avoids discarding otherwise
# strong candidates because of tiny Monte-Carlo deviations such as 1.202 > 1.20.
# These limits apply only to Stage 2d; Stage 2a/2b/2c keep their existing policy.
RUNTIME_RECENCY_DURATION_MIN_RATIO = 0.5          # soft / preferred lower bound
RUNTIME_RECENCY_DURATION_MAX_RATIO = 1.20          # soft / preferred upper bound
RUNTIME_RECENCY_DURATION_HARD_MIN_RATIO = 0.45     # true rejection lower bound
RUNTIME_RECENCY_DURATION_HARD_MAX_RATIO = 1.25     # true rejection upper bound

# ------------------------------------------------------------------
# Stage 3b：大日志统一 case-level 校准抽样
# ------------------------------------------------------------------
# 所有日志使用完全相同的规则，不允许针对某个日志名称单独设置：
#
# N_cases <= 5000：
#     Stage 3b 使用完整 train_validation_log
#
# N_cases > 5000：
#     Stage 3b 使用固定连续时间窗口 case-level 子集
#
# 样本量采用 DSIM 风格有限总体样本量公式。
FINAL_CALIBRATION_SAMPLING_ENABLED = True

# 统一的大日志阈值。
# 5000-case 阈值让 CVS/BPI12 等中大型日志也进入 sample-search；最终仍做 full confirmation。
FINAL_CALIBRATION_LARGE_LOG_THRESHOLD = 5000

# 防御性上限。
FINAL_CALIBRATION_MAX_CASES = 5000

# DSIM 风格有限总体抽样参数
FINAL_CALIBRATION_CONFIDENCE_LEVEL = 95.0
FINAL_CALIBRATION_MARGIN_PERCENT = 3.0

# 固定种子保留用于兼容/可重复性；连续窗口本身是确定性的。
# baseline、所有 activity、所有 scale 必须使用同一批 cases。
FINAL_CALIBRATION_SAMPLE_SEED = 72

# Stage 3b 最终确认协议：
# - 候选搜索与最终稳定确认都使用“已选 runtime-recency/recent window”；
# - 不再回到完整 train_validation_log 做 veto，避免旧历史分布把 recent window
#   上选出的较大 activity scale（例如 CVS 的 4x/5x）回滚掉；
# - baseline 与最终候选仍重复评估，降低单次随机仿真造成的误选；
# - test_log 从不参与参数选择。
FINAL_CALIBRATION_RECENT_CONFIRMATION_ENABLED = True
FINAL_CALIBRATION_CONFIRMATION_RUNS = 2

# 旧 full-history confirmation 明确关闭。保留常量仅兼容 legacy helper；
# 当前 Stage 3b 主路径不会再使用完整历史做最终 veto。
FINAL_CALIBRATION_FULL_CONFIRMATION_ENABLED = False
# ------------------------------------------------------------------
# 资源候选过滤（发现模型时使用）
# ------------------------------------------------------------------
RESOURCE_ADAPTIVE_FILTER_ENABLED = True
RESOURCE_ADAPTIVE_THRESHOLD = 50

SMALL_RESOURCE_ENABLE_CANDIDATE_FILTER = False
SMALL_RESOURCE_CANDIDATE_COVERAGE = 1.00
SMALL_RESOURCE_CANDIDATE_MAX_K = None
SMALL_RESOURCE_CANDIDATE_MIN_K = 1

LARGE_RESOURCE_ENABLE_CANDIDATE_FILTER = True
LARGE_RESOURCE_CANDIDATE_COVERAGE = 0.9
LARGE_RESOURCE_CANDIDATE_MAX_K = 55
LARGE_RESOURCE_CANDIDATE_MIN_K = 3

# ------------------------------------------------------------------
# 对齐超时保护
# ------------------------------------------------------------------
ALIGNMENT_TIMEOUT_SECONDS = 1800  # 30 分钟

# ------------------------------------------------------------------
# 运行效率优化（不缩减候选空间）
# ------------------------------------------------------------------
# 1) 相同日志 + 相同 Petri 结构 + 相同子模型超参数时，复用已发现模型状态。
#    对 Stage 1 的“不同 noise 但得到同一 Petri 网”尤其有效。
DISCOVERY_STATE_CACHE_ENABLED = True
DISCOVERY_STATE_CACHE_MAX_ENTRIES = 64

# 2) 如果 drqIntegration 在 discover_parameters() 后暴露特征 DataFrame，
#    则在同一日志 + 同一 Petri 结构下复用，避免重复 alignment / feature building。
PRECOMPUTED_FEATURE_CACHE_ENABLED = False
PRECOMPUTED_FEATURE_CACHE_MAX_ENTRIES = 32

# 3) 候选评估统一串行执行。
#    Stage 2c 的 External Delay quantile 与 Stage 3b 的 activity scale 都按候选顺序逐个评估。
#    不再使用 ThreadPoolExecutor，避免 Python GIL、多个完整仿真同时争用 CPU/内存，
#    同时保证每个候选使用同一个 integration 实例按确定顺序更新运行时配置。
CANDIDATE_EVAL_EXECUTION_MODE = "serial"

# 4) Stage 3b 中候选 scale 等于当前 scale（通常为 1.0）时，直接复用 current_eval。
#    这不会删除候选，只避免重复跑完全相同的仿真状态。
REUSE_IDENTITY_SCALE_BASELINE = True

class StagewiseTuner:
    """
    分阶段超参数优化器。

    Stage 1：控制流结构搜索，目标指标为 CFLD / NGD / 3GD。
    自动模式使用轻量联合搜索：每个 Optuna trial 同时选择
    Petri-net noise_threshold 与 decision-point tree_depth；noise=0 的每个 depth
    都作为 anchor 强制评估。固定 noise / PNML 模式仍保留原 tree-depth grid。

    Stage 2a：固定控制流结构，独立搜索 Case Arrival Tree 深度（CAR + arrival diagnostics）。

    Stage 2b：固定 Arrival，独立搜索 Resource Allocation Tree 深度。
    Ready Queue 始终存在，Dispatch policy 始终由数据自动判定为 CPT 或 FCFS；Stage 2a/2b 不再人工开关 CPT。

    Stage 2c：固定 Arrival / Resource Allocation，进行 dispatch-aware External Delay 搜索：
    - 若数据选择 CPT：联合搜索 Priority Tree depth × External Delay runtime tail quantile；
    - 若数据选择 FCFS：Priority Tree depth 不适用，仅搜索 External Delay quantile。

    Stage 2d：在训练集内部 chronological validation 上选择 runtime recency ratio。
    1.0 始终是默认基线；recent window 只有在 validation 上显著改善且通过严格时间 guard 时才启用。

    Stage 2e0：先做零仿真的 Arrival drift trigger。仅在 train -> validation 到达模式明显变化时，
    才独立搜索 Arrival recent-window；未触发时保持 full-history Arrival，避免影响其他日志。

    Stage 2e：固定已选 Arrival recency，仅在 validation 上校准 arrival_time_scale。
    1.0 始终保留为 identity baseline；非 1.0 scale 必须取得有意义的 CAR 改善才会启用。

    Stage 3：完整 train_validation_log 负责结构与稳定行为；选中的 recent window 仅负责
    service time / external delay。随后固定 Stage 2c quantile 做 Top-K activity scale 校准，
    并在完整 train_validation_log 上重复确认最终 activity scales。

    大日志（>5000 cases）：各调参阶段采用 sample search + full confirmation；
    小/中日志完全使用完整 train/validation。

    Stage 4：test_log 只做最终评估，不参与任何调参。
    """

    def __init__(self, integration):
        self.integration = integration
        # run-local caches: key 中包含 event_log 对象 identity，因此不会跨日志误复用。
        self._discovery_state_cache: dict[tuple, dict] = {}
        self._precomputed_feature_cache: dict[tuple, Any] = {}
        self._cache_stats = {
            "discovery_hits": 0,
            "discovery_misses": 0,
            "feature_hits": 0,
            "feature_stores": 0,
        }
        logger.warning(
            "[StagewiseTuner Loaded] file=%s, build=%s, adaptive_resource_filter=%s",
            __file__,
            STAGEWISE_TUNER_BUILD_TAG,
            bool(RESOURCE_ADAPTIVE_FILTER_ENABLED),
        )

    @staticmethod
    def _cache_log_key(event_log) -> tuple:
        """进程内日志缓存键；只用于当前 StagewiseTuner.run 生命周期。"""
        try:
            n = len(event_log)
        except Exception:
            n = None
        return (id(event_log), n)

    @staticmethod
    def _stable_node_token(obj) -> tuple:
        if obj is None:
            return ("None",)
        return (
            obj.__class__.__module__,
            obj.__class__.__name__,
            str(getattr(obj, "name", "")),
            str(getattr(obj, "label", "")),
        )

    def _current_process_structure_signature(self) -> str:
        """为当前 Petri/process model 构造尽量稳定的结构 hash。

        优先读取 PetriNet 的 places/transitions/arcs；若集成层只暴露 dict，
        则对其中与结构相关的字段做稳定化序列化。hash 只用于本次调参缓存。
        """
        integration = self.integration
        process_model = getattr(integration, "process_model", None)

        net = None
        initial_marking = None
        final_marking = None
        candidates = []
        if isinstance(process_model, dict):
            for key in ("petri_net", "net", "model"):
                if process_model.get(key) is not None:
                    candidates.append(process_model.get(key))
            initial_marking = process_model.get("initial_marking")
            final_marking = process_model.get("final_marking")
        for attr in ("petri_net", "net"):
            value = getattr(integration, attr, None)
            if value is not None:
                candidates.append(value)
        for candidate in candidates:
            if all(hasattr(candidate, name) for name in ("places", "transitions", "arcs")):
                net = candidate
                break

        if initial_marking is None:
            initial_marking = getattr(integration, "initial_marking", None)
        if final_marking is None:
            final_marking = getattr(integration, "final_marking", None)

        payload = {}
        if net is not None:
            payload["places"] = sorted(self._stable_node_token(x) for x in list(net.places))
            payload["transitions"] = sorted(self._stable_node_token(x) for x in list(net.transitions))
            arcs = []
            for arc in list(net.arcs):
                arcs.append((self._stable_node_token(getattr(arc, "source", None)),
                             self._stable_node_token(getattr(arc, "target", None))))
            payload["arcs"] = sorted(arcs)

            def _marking_token(marking):
                if not isinstance(marking, dict):
                    try:
                        items = marking.items()
                    except Exception:
                        return []
                else:
                    items = marking.items()
                out = []
                for place, tokens in items:
                    try:
                        count = int(tokens)
                    except Exception:
                        count = str(tokens)
                    out.append((self._stable_node_token(place), count))
                return sorted(out)

            payload["initial_marking"] = _marking_token(initial_marking)
            payload["final_marking"] = _marking_token(final_marking)
        else:
            # Fallback: retain only semantically stable structural fields when possible.
            if isinstance(process_model, dict):
                for key in (
                    "activities", "places", "transitions", "arcs", "source", "sink",
                    "decision_points", "process_tree", "petri_source",
                ):
                    if key in process_model:
                        value = process_model.get(key)
                        try:
                            payload[key] = self._stable_cache_value(value)
                        except Exception:
                            payload[key] = str(value)
            else:
                payload["model"] = self._stable_cache_value(process_model)

        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return ("petri:" if net is not None else "fallback:") + hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()

    @classmethod
    def _stable_cache_value(cls, value, depth: int = 0):
        if depth > 5:
            return str(type(value).__name__)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            ignored = {"visualization", "svg", "png", "image", "metrics", "quality"}
            return {
                str(k): cls._stable_cache_value(v, depth + 1)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
                if str(k).lower() not in ignored
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            items = [cls._stable_cache_value(v, depth + 1) for v in value]
            try:
                return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, default=str))
            except Exception:
                return items
        if all(hasattr(value, name) for name in ("places", "transitions", "arcs")):
            return {
                "places": sorted(cls._stable_node_token(x) for x in list(value.places)),
                "transitions": sorted(cls._stable_node_token(x) for x in list(value.transitions)),
                "arcs": sorted((cls._stable_node_token(getattr(a, "source", None)),
                                cls._stable_node_token(getattr(a, "target", None))) for a in list(value.arcs)),
            }
        name = getattr(value, "name", None)
        label = getattr(value, "label", None)
        if name is not None or label is not None:
            return cls._stable_node_token(value)
        # strip memory addresses from repr so equivalent objects do not miss cache solely due to id
        return re.sub(r"0x[0-9a-fA-F]+", "0x*", repr(value))

    def _capture_model_state(self) -> dict:
        """捕获仿真真正依赖的已发现模型状态；失败时返回空 dict。"""
        state = {}
        # 这些字段覆盖当前 tuner/Integration 已知的模型状态；诊断字段不缓存。
        for attr in (
            "process_model", "drq_params", "petri_net", "net",
            "initial_marking", "final_marking", "process_tree",
        ):
            if hasattr(self.integration, attr):
                try:
                    state[attr] = copy.deepcopy(getattr(self.integration, attr))
                except Exception:
                    pass
        return state

    def _restore_model_state(self, state: dict, event_log=None) -> bool:
        if not state:
            return False
        try:
            for attr, value in state.items():
                setattr(self.integration, attr, copy.deepcopy(value))
            if event_log is not None:
                self.integration.event_log = event_log
            return True
        except Exception as exc:
            logger.warning("Cached model-state restore failed; falling back to discovery: %s", exc)
            return False

    def _capture_precomputed_features(self):
        """兼容不同 Integration 命名；只有真实存在的 DataFrame/feature object 才缓存。"""
        for attr in (
            "precomputed_df_features", "df_features", "_df_features",
            "feature_df", "features_df", "_feature_df", "_features_df",
        ):
            if hasattr(self.integration, attr):
                value = getattr(self.integration, attr)
                if value is not None:
                    try:
                        return copy.deepcopy(value)
                    except Exception:
                        return value
        return None

    @staticmethod
    def _extract_arrival_guard_from_integration(integration) -> dict:
        diagnostics = getattr(integration, "_last_arrival_diagnostics", {}) or {}
        guard = {
            "enabled": bool(diagnostics),
            "arrival_span_ratio": None,
            "case_count_ratio": None,
            "real_cases": diagnostics.get("real_cases"),
            "sim_cases": diagnostics.get("sim_cases"),
            "weekday_hour_jaccard": diagnostics.get("weekday_hour_jaccard"),
        }
        try:
            span_ratio = float(diagnostics.get("span_ratio_sim_over_real"))
            if np.isfinite(span_ratio):
                guard["arrival_span_ratio"] = span_ratio
        except Exception:
            pass
        try:
            real_cases = float(diagnostics.get("real_cases"))
            sim_cases = float(diagnostics.get("sim_cases"))
            if np.isfinite(real_cases) and np.isfinite(sim_cases) and real_cases > 0:
                guard["case_count_ratio"] = sim_cases / real_cases
        except Exception:
            pass
        span = guard.get("arrival_span_ratio")
        case_ratio = guard.get("case_count_ratio")
        passed = True
        if isinstance(span, (int, float, np.integer, np.floating)) and np.isfinite(float(span)):
            passed = passed and (float(ARRIVAL_SPAN_MIN_RATIO) <= float(span) <= float(ARRIVAL_SPAN_MAX_RATIO))
        if isinstance(case_ratio, (int, float, np.integer, np.floating)) and np.isfinite(float(case_ratio)):
            passed = passed and (float(case_ratio) >= float(ARRIVAL_CASE_COUNT_MIN_RATIO))
        guard["passed"] = bool(passed)
        return guard

    @staticmethod
    def _extract_dispatch_summary_from_integration(integration) -> dict:
        """Read the queue-dispatch discovery result without treating CPT as a manual on/off switch."""
        summary = {}
        try:
            summary = (
                (getattr(integration, "drq_params", None) or {})
                .get("queue_scheduling_params", {})
                .get("conditional_prioritization", {})
            ) or {}
        except Exception:
            summary = {}

        model_kind = summary.get("model_kind")
        normalized_kind = str(model_kind or "").strip().lower()
        priority_tree_active = normalized_kind == "pairwise_decision_tree"
        return {
            "model_kind": model_kind,
            "dispatch_policy": "conditional_prioritization_tree" if priority_tree_active else "fcfs",
            "priority_tree_active": bool(priority_tree_active),
            "n_competition_pairs": summary.get("n_competition_pairs"),
            "overtaking_rate": summary.get("overtaking_rate"),
            "raw": summary,
        }

    def run(
            self,
            full_event_log,
            pnml_filepath=None,
            noise_threshold_range=None,
            tree_depth_candidates=TREE_DEPTH_CANDIDATES,
            metrics_labels=None,
            test_evaluation_runs: int = 5,
            save_test_simulation_logs: bool = False,
            test_simulation_log_dir: str | None = None,
            test_simulation_log_prefix: str = "test_evaluation",
            test_event_log=None,
            **legacy_kwargs,
    ) -> dict:

        # ----------------------------------------------------------
        # 与旧版 routes.py 的向后兼容
        # ----------------------------------------------------------
        # 一些旧的 routes.py 仍可能使用 legacy 关键字调用 StagewiseTuner.run()。
        # 清理后的调优器通过 **legacy_kwargs 接受这些参数，以避免旧前端报错，
        # 但只有有用的控制流/运行时深度输入才会被映射。
        if legacy_kwargs:
            logger.info(
                "Ignoring/mapping legacy StagewiseTuner.run kwargs: %s",
                sorted(legacy_kwargs.keys()),
            )
        legacy_structure_metrics = legacy_kwargs.get("structure_metrics")
        legacy_runtime_metrics = legacy_kwargs.get("runtime_metrics")

        if legacy_kwargs.get("max_depth_candidates") is not None:
            # 保持与旧 routes.py 的兼容，但将有效控制流深度网格统一为 1、2、3、4。
            # 这避免了之前 routes.py 只传 (1, 2, 3)，而到达树和队列树使用 (1, 2, 3, 4) 的情况。
            tree_depth_candidates = TREE_DEPTH_CANDIDATES

        if legacy_kwargs.get("top_k_structures") is not None:
            # 新协议在 Stage 1 后固定结构。旧 routes.py 可能仍传 top_k_structures=2/3；
            # 忽略该参数，以避免将 structure_idx 重新作为 Stage-2 的运行时参数。
            logger.info(
                "Ignoring legacy top_k_structures=%s; using fixed Stage-1 structure with top_k=1",
                legacy_kwargs.get("top_k_structures"),
            )

        if legacy_kwargs.get("case_duration_min_ratio") is not None:
            # 保持统一的 duration guard。旧 routes.py 版本可能仍传 0.30，
            # 这会重新打开 duration 被压缩的失败模式。我们在此刻意忽略更低的 legacy 值。
            logger.info(
                "Ignoring legacy case_duration_min_ratio=%s; using unified value %.3f",
                legacy_kwargs.get("case_duration_min_ratio"),
                float(CASE_DURATION_MIN_RATIO),
            )

        if legacy_kwargs.get("case_duration_max_ratio") is not None:
            # 忽略 legacy 最大值，使用统一的 duration guard。
            logger.info(
                "Ignoring legacy case_duration_max_ratio=%s; using unified value %.3f",
                legacy_kwargs.get("case_duration_max_ratio"),
                float(CASE_DURATION_MAX_RATIO),
            )

        resource_legacy_keys = (
            "enable_candidate_resource_filter",
            "candidate_resource_coverage",
            "candidate_resource_max_k",
            "candidate_resource_min_k",
        )
        if any(legacy_kwargs.get(key) is not None for key in resource_legacy_keys):
            logger.info(
                "忽略 legacy 候选资源过滤参数；使用统一自适应资源过滤："
                "enabled=%s, threshold=%s, large_policy=(coverage=%s, max_k=%s, min_k=%s)",
                bool(RESOURCE_ADAPTIVE_FILTER_ENABLED),
                int(RESOURCE_ADAPTIVE_THRESHOLD),
                float(LARGE_RESOURCE_CANDIDATE_COVERAGE),
                LARGE_RESOURCE_CANDIDATE_MAX_K,
                int(LARGE_RESOURCE_CANDIDATE_MIN_K),
            )
        if metrics_labels is None:
            metrics_labels = [
                "cfld", "3gd", "ngd",
                "aed", "ced", "red",
                "car", "ctd", "cwd",
            ]

        structure_metrics = (
            tuple(legacy_structure_metrics)
            if legacy_structure_metrics is not None
            else CONTROL_FLOW_STAGE_METRICS
        )

        runtime_metrics = (
            tuple(legacy_runtime_metrics)
            if legacy_runtime_metrics is not None
            else RUNTIME_METRICS
        )

        top_k_structures = 1 if FIX_STRUCTURE_BEFORE_RUNTIME else TOP_K_STRUCTURES
        queue_strategy_candidates = QUEUE_STRATEGIES
        case_duration_min_ratio = CASE_DURATION_MIN_RATIO
        case_duration_max_ratio = CASE_DURATION_MAX_RATIO
        # 外部延迟比例与 CTD 容差已移除；外部延迟由 Simod-lite 活动级延迟器处理。
        max_external_delay_ratio = None
        ctd_tolerance = None
        # 候选资源过滤在 _discover_candidate_model() 内部根据实际发现分区的资源数量自适应解析。
        enable_candidate_resource_filter = None
        candidate_resource_coverage = None
        candidate_resource_max_k = None
        candidate_resource_min_k = None

        # ----------------------------------------------------------
        # Petri-net 噪声阈值模式
        # ----------------------------------------------------------
        # routes.py 可传入以下两种形式之一：
        #   1) 真正的区间，例如 (0.0, 0.50)，用于自动优化；
        #   2) 退化区间，例如 (0.0, 0.0) 或 (0.2, 0.2)，用于手动固定阈值。
        # 不要扩展退化区间；否则手动模式仍会在用户指定值附近进行小规模搜索。
        caller_supplied_noise_range = noise_threshold_range is not None
        if noise_threshold_range is None:
            noise_threshold_range = NOISE_THRESHOLD_RANGE
        try:
            noise_low, noise_high = noise_threshold_range
            noise_low = float(noise_low)
            noise_high = float(noise_high)
        except Exception:
            noise_low, noise_high = NOISE_THRESHOLD_RANGE
            caller_supplied_noise_range = False

        if noise_low > noise_high:
            noise_low, noise_high = noise_high, noise_low

        noise_low = float(np.clip(noise_low, 0.0, 1.0))
        noise_high = float(np.clip(noise_high, 0.0, 1.0))

        noise_threshold_is_fixed = abs(noise_high - noise_low) <= 1e-12
        if not noise_threshold_is_fixed:
            # 轻量联合搜索统一限制在配置的默认上界内。旧 routes.py / UI 可能仍传
            # (0.0, 0.5)；若直接尊重该旧范围，会悄悄抵消这里的轻量化配置。
            # 手动固定阈值（low == high）不受此限制。
            configured_high = float(np.clip(float(NOISE_THRESHOLD_RANGE[1]), 0.0, 1.0))
            if noise_low <= configured_high < noise_high:
                logger.info(
                    "Capping automatic noise_threshold upper bound from %.6f to configured %.6f "
                    "for lightweight joint structure search.",
                    noise_high, configured_high,
                )
                noise_high = configured_high
            if noise_high <= noise_low:
                noise_high = min(1.0, noise_low + 0.01)
            noise_threshold_is_fixed = abs(noise_high - noise_low) <= 1e-12

        # Stage 1 控制流深度候选与运行时树深度候选分开。
        tree_depth_candidates = list(CONTROL_FLOW_TREE_DEPTH_CANDIDATES)

        arrival_tree_depth_candidates = self._normalize_depth_candidates(
            legacy_kwargs.get("arrival_decision_tree_max_depth_candidates")
            or legacy_kwargs.get("arrival_tree_depth_candidates")
            or RUNTIME_TREE_DEPTH_CANDIDATES,
            default=RUNTIME_TREE_DEPTH_CANDIDATES,
            min_depth=1,
            max_depth=4,
        )
        queue_runtime_tree_depth_candidates = self._normalize_depth_candidates(
            legacy_kwargs.get("queue_scheduling_max_depth_candidates")
            or legacy_kwargs.get("queue_tree_depth_candidates")
            or RUNTIME_TREE_DEPTH_CANDIDATES,
            default=RUNTIME_TREE_DEPTH_CANDIDATES,
            min_depth=1,
            max_depth=4,
        )
        priority_tree_depth_candidates = self._normalize_depth_candidates(
            legacy_kwargs.get("conditional_prioritization_max_depth_candidates")
            or legacy_kwargs.get("cpt_tree_depth_candidates")
            or PRIORITY_TREE_DEPTH_CANDIDATES,
            default=PRIORITY_TREE_DEPTH_CANDIDATES,
            min_depth=1,
            max_depth=4,
        )
        external_delay_quantile_candidates = self._normalize_quantile_candidates(
            legacy_kwargs.get("external_delay_quantile_candidates")
            or legacy_kwargs.get("external_delay_max_delay_quantile_candidates")
            or EXTERNAL_DELAY_QUANTILE_CANDIDATES
        )

        if pnml_filepath:
            structure_search_description = "fixed_pnml_tree_depth_grid"
        elif noise_threshold_is_fixed:
            structure_search_description = f"fixed_petri_noise_threshold[{noise_low}] + tree_depth_grid"
        else:
            structure_search_description = (
                f"joint_petri_noise_tree_depth_tpe[noise={noise_low}:{noise_high}, "
                f"depths={tree_depth_candidates}]"
            )

        logger.info(
            "Strict staged tuning started. "
            f"structure_metrics={structure_metrics}, runtime_metrics={runtime_metrics}, "
            f"structure_search={structure_search_description}, "
            f"control_flow_tree_depth_candidates={tree_depth_candidates}, "
            f"arrival_tree_depth_candidates={arrival_tree_depth_candidates}, "
            f"queue_tree_depth_candidates={queue_runtime_tree_depth_candidates}, "
            f"priority_tree_depth_candidates={priority_tree_depth_candidates}, "
            f"external_delay_quantile_candidates={external_delay_quantile_candidates}, "
            f"arrival_time_scale_candidates={ARRIVAL_TIME_SCALE_CANDIDATES}, "
            f"runtime_search=(Stage2a arrival depth; Stage2b resource-allocation depth; "
            f"Stage2c data-driven dispatch + external-delay quantile; Stage2d runtime recency; "
            f"Stage2e0 triggered arrival recency; Stage2e validation-controlled arrival scale; "
            f"Stage3 fixed rediscovery)"
        )

        # ===== 日志划分已禁用 =====
        # 新流程：用户直接上传测试集日志，不再进行 80/20 train/test 划分。
        # 上传的完整日志同时用于模型发现（Stage 1-3）、参数校准和最终仿真评估（Stage 4）。
        #
        # partitions = self.integration.split_event_log(
        #     full_event_log,
        #     train_validation_ratio=0.8,
        #     train_ratio_within_train_validation=0.8,
        #     sort_temporally=True,
        # )

        from types import SimpleNamespace

        # Stage 4 评估日志：优先使用外部传入的 test_event_log（_test 后缀文件），
        # 如果未提供则回退到 full_event_log。
        evaluation_test_log = test_event_log if test_event_log is not None else full_event_log

        # ===== 训练集内部 80/20 划分 =====
        # 上传的训练集日志按 case 时间顺序分为 train (80%) 和 validation (20%)，
        # train_validation_log 保持为完整的上传日志，test_log 使用外部测试集。
        try:
            from drqsim.utils.log_partitioner import (
                _to_dataframe, _to_event_log, _get_ordered_case_ids,
                _split_case_ids, _filter_by_case_ids,
            )
            import pandas as _pd

            _CASE_ID_COL = "case:concept:name"
            _END_TIME_COL = "time:timestamp"
            _START_TIME_COL = "start:timestamp"

            _split_df = _to_dataframe(full_event_log)

            # 如果缺少 start:timestamp，用 time:timestamp 填充
            if _START_TIME_COL not in _split_df.columns:
                _split_df[_START_TIME_COL] = _split_df[_END_TIME_COL]

            _split_df[_CASE_ID_COL] = _split_df[_CASE_ID_COL].astype(str)
            _split_df[_START_TIME_COL] = _pd.to_datetime(_split_df[_START_TIME_COL], utc=True, errors="coerce")
            _split_df[_END_TIME_COL] = _pd.to_datetime(_split_df[_END_TIME_COL], utc=True, errors="coerce")
            _split_df = _split_df.dropna(subset=[_CASE_ID_COL, _START_TIME_COL, _END_TIME_COL]).copy()

            _case_ids = _get_ordered_case_ids(
                df=_split_df,
                case_id_col=_CASE_ID_COL,
                start_time_col=_START_TIME_COL,
                end_time_col=_END_TIME_COL,
                sort_temporally=True,
            )

            if len(_case_ids) >= 3:
                _train_case_ids, _validation_case_ids = _split_case_ids(_case_ids, 0.8)

                _train_split_df = _filter_by_case_ids(
                    _split_df, _train_case_ids, _CASE_ID_COL, _START_TIME_COL, _END_TIME_COL,
                )
                _validation_split_df = _filter_by_case_ids(
                    _split_df, _validation_case_ids, _CASE_ID_COL, _START_TIME_COL, _END_TIME_COL,
                )

                _train_split_log = _to_event_log(_train_split_df, _CASE_ID_COL)
                _validation_split_log = _to_event_log(_validation_split_df, _CASE_ID_COL)
            else:
                _train_split_log = full_event_log
                _validation_split_log = full_event_log
                logger.warning(
                    f"Too few cases ({len(_case_ids)}) for train/validation split, "
                    f"using full_event_log for both"
                )
        except Exception as split_error:
            logger.warning(
                f"Train/validation split failed, using full_event_log for both: {split_error}"
            )
            _train_split_log = full_event_log
            _validation_split_log = full_event_log

        partitions = SimpleNamespace(
            train_log=_train_split_log,
            validation_log=_validation_split_log,
            train_validation_log=full_event_log,
            test_log=evaluation_test_log,
        )

        # 同步设置 integration 实例属性，保持与其他模块的兼容性
        self.integration.full_event_log = full_event_log
        self.integration.train_log = _train_split_log
        self.integration.validation_log = _validation_split_log
        self.integration.train_validation_log = full_event_log
        self.integration.test_log = evaluation_test_log

        logger.info(
            "Internal train/validation split (80/20): "
            f"train={len(_train_split_log)}, validation={len(_validation_split_log)}, "
            f"train_validation={len(full_event_log)}, test={len(evaluation_test_log)} traces"
        )

        if queue_strategy_candidates is None:
            queue_strategy_candidates = QUEUE_STRATEGIES
        queue_strategy_candidates = self._normalize_queue_strategy_candidates(
            queue_strategy_candidates
        )

        # ==========================================================
        # 大日志统一调参采样：sample search + full confirmation
        # ==========================================================
        tuning_search = self._prepare_large_log_tuning_search_logs(
            train_log=partitions.train_log,
            validation_log=partitions.validation_log,
            train_validation_log=partitions.train_validation_log,
        )
        tuning_sampling_info = dict(tuning_search.get("info", {}) or {})
        tuning_sampling_applied = bool(tuning_sampling_info.get("applied", False))

        # ==========================================================
        # Stage 1：控制流结构搜索
        # 大日志：sample 上筛选 Top-K，再用完整 train/validation 确认。
        # ==========================================================
        stage1_search_top_k = (
            max(int(top_k_structures), int(TUNING_STRUCTURE_FULL_CONFIRM_TOP_K))
            if tuning_sampling_applied else int(top_k_structures)
        )
        structure_search_result = self._run_split_structure_search(
            train_log=tuning_search["search_train_log"],
            validation_log=tuning_search["search_validation_log"],
            pnml_filepath=pnml_filepath,
            structure_metrics=structure_metrics,
            tree_depth_candidates=tree_depth_candidates,
            noise_threshold_range=(noise_low, noise_high),
            top_k_structures=stage1_search_top_k,
            n_trials=STRUCTURE_OPTUNA_N_TRIALS,
            optuna_seed=STRUCTURE_OPTUNA_SEED,
            sampler_name=STRUCTURE_OPTUNA_SAMPLER,
            validation_budgets=STRUCTURE_VALIDATION_BUDGETS,
            reduction_factor=STRUCTURE_REDUCTION_FACTOR,
        )

        structure_records = list(structure_search_result["structure_records"])
        top_structures = list(structure_search_result["top_structures"])
        structure_optuna_summary = dict(structure_search_result["optuna_summary"] or {})
        structure_quality_records = []

        logger.info(
            "Split structure search completed. Top candidates: "
            f"{[(r.get('noise_threshold'), r.get('tree_depth'), r.get('structure_score')) for r in top_structures]}"
        )

        if tuning_sampling_applied:
            full_structure_confirmation = self._confirm_top_structure_candidates(
                candidate_records=top_structures,
                train_log=partitions.train_log,
                validation_log=partitions.validation_log,
                pnml_filepath=pnml_filepath,
                structure_metrics=structure_metrics,
                top_k_structures=top_k_structures,
            )
            confirmed_structures = list(full_structure_confirmation.get("top_structures") or [])
            if confirmed_structures:
                top_structures = confirmed_structures
            structure_records.extend(
                list(full_structure_confirmation.get("confirmation_records", []) or [])
            )
            structure_optuna_summary["large_log_sample_search_full_confirmation"] = {
                "sampling": tuning_sampling_info,
                "sample_top_k": int(stage1_search_top_k),
                "full_confirmation": full_structure_confirmation.get("summary", {}),
            }

        if FIX_STRUCTURE_BEFORE_RUNTIME:
            if not top_structures:
                raise RuntimeError("Stage 1 produced no structure candidate to fix before runtime search.")
            fixed_stage1_structure = top_structures[0]
            runtime_top_structures = [fixed_stage1_structure]
            logger.info(
                "Stage 1 fixed structure for Stage 2: noise_threshold=%s, tree_depth=%s, structure_score=%s. "
                "Stage 2 will not optimize structure_idx.",
                fixed_stage1_structure.get("noise_threshold"),
                fixed_stage1_structure.get("tree_depth"),
                fixed_stage1_structure.get("structure_score"),
            )
        else:
            runtime_top_structures = top_structures

        # ==========================================================
        # Stage 2a + 2b：先选 Arrival depth，再选 Resource Allocation depth。
        # Ready Queue 始终启用；Dispatch policy 不由 tuner 开关，而由日志自动判定为 CPT/FCFS。
        # 两个阶段都使用固定的浅层 priority-tree probe，避免优先级树复杂度干扰 Arrival/Resource 选择。
        # ==========================================================
        logger.info("Stage 2a/2b started with full ready-queue discovery and data-driven dispatch policy.")

        stage2_core_result = self._run_arrival_resource_depth_search(
            top_structures=runtime_top_structures,
            train_log=tuning_search["search_train_log"],
            validation_log=tuning_search["search_validation_log"],
            arrival_train_log=tuning_search["arrival_search_train_log"],
            arrival_validation_log=tuning_search["arrival_search_validation_log"],
            confirmation_train_log=tuning_search["confirmation_train_log"],
            confirmation_validation_log=tuning_search["confirmation_validation_log"],
            full_confirmation_top_k=TUNING_RUNTIME_FULL_CONFIRM_TOP_K,
            pnml_filepath=pnml_filepath,
            runtime_metrics=runtime_metrics,
            arrival_depth_candidates=arrival_tree_depth_candidates,
            resource_depth_candidates=queue_runtime_tree_depth_candidates,
            case_duration_min_ratio=case_duration_min_ratio,
            case_duration_max_ratio=case_duration_max_ratio,
            max_external_delay_ratio=max_external_delay_ratio,
            ctd_tolerance=ctd_tolerance,
            enable_candidate_resource_filter=enable_candidate_resource_filter,
            candidate_resource_coverage=candidate_resource_coverage,
            candidate_resource_max_k=candidate_resource_max_k,
            candidate_resource_min_k=candidate_resource_min_k,
        )

        selected_core_runtime = stage2_core_result["selected"]

        # ==========================================================
        # Stage 2c：Data-driven Dispatch × External Delay。
        # FCFS 数据集：只搜索 External Delay quantile。
        # CPT 数据集：搜索 Priority Tree depth × External Delay quantile。
        # ==========================================================
        dispatch_external_result = self._run_dispatch_external_search(
            base_candidate=selected_core_runtime,
            train_log=tuning_search["search_train_log"],
            validation_log=tuning_search["search_validation_log"],
            confirmation_train_log=tuning_search["confirmation_train_log"],
            confirmation_validation_log=tuning_search["confirmation_validation_log"],
            full_confirmation_top_k=TUNING_DISPATCH_EXTERNAL_FULL_CONFIRM_TOP_K,
            pnml_filepath=pnml_filepath,
            runtime_metrics=runtime_metrics,
            priority_tree_depth_candidates=priority_tree_depth_candidates,
            external_delay_quantile_candidates=external_delay_quantile_candidates,
            case_duration_min_ratio=case_duration_min_ratio,
            case_duration_max_ratio=case_duration_max_ratio,
            max_external_delay_ratio=max_external_delay_ratio,
            ctd_tolerance=ctd_tolerance,
            enable_candidate_resource_filter=enable_candidate_resource_filter,
            candidate_resource_coverage=candidate_resource_coverage,
            candidate_resource_max_k=candidate_resource_max_k,
            candidate_resource_min_k=candidate_resource_min_k,
        )

        selected = dispatch_external_result["selected"]
        runtime_records = list(stage2_core_result.get("runtime_records", [])) + list(
            dispatch_external_result.get("runtime_records", [])
        )
        optuna_summary = {
            "enabled": True,
            "search_type": "sequential_arrival_resource_then_data_driven_dispatch_external",
            "large_log_tuning_sampling": tuning_sampling_info,
            "stage2a_2b_arrival_then_resource_allocation": stage2_core_result.get("optuna_summary", {}),
            "stage2c_dispatch_external": dispatch_external_result.get("optuna_summary", {}),
        }

        # 活动级外部延迟缩放协议：
        # 保持全局外部延迟缩放固定为 1.0，仅在最终重新发现后进行 Top-K 活动级局部校准。
        duration_calibration = {
            "enabled": bool(TOP_ACTIVITY_SCALE_CALIBRATION_ENABLED),
            "reason": "top_k_activity_level_external_delay_scale_protocol",
            "global_scale_fixed": 1.0,
        }

        selected_tree_depth = int(
            selected.get(
                "tree_depth",
                selected.get(
                    "decision_point_max_depth",
                    selected.get("max_depth_tree", 1),
                ),
            )
        )
        selected_arrival_depth = int(selected.get("arrival_decision_tree_max_depth", selected_tree_depth))
        selected_queue_depth = int(selected.get("queue_scheduling_max_depth", selected_tree_depth))
        selected_priority_depth = int(
            selected.get(
                "conditional_prioritization_max_depth",
                min(PRIORITY_TREE_DEPTH_CANDIDATES),
            )
        )
        selected_dispatch_policy = str(selected.get("dispatch_policy") or "fcfs")
        priority_tree_active = selected_dispatch_policy == "conditional_prioritization_tree"
        selected_external_quantile = float(
            selected.get("external_delay_max_delay_quantile", 1.0)
        )

        selected_hyperparameters = {
            "noise_threshold": selected.get("noise_threshold"),
            "tree_depth": selected_tree_depth,
            "max_depth_tree": selected_tree_depth,
            "decision_point_max_depth": selected_tree_depth,
            "arrival_decision_tree_max_depth": selected_arrival_depth,
            "case_arrival_max_depth": selected_arrival_depth,
            "queue_scheduling_max_depth": selected_queue_depth,
            # Legacy integration field retained; it is meaningful only when priority_tree_active=True.
            "conditional_prioritization_max_depth": selected_priority_depth,
            "priority_tree_max_depth": selected_priority_depth if priority_tree_active else None,
            "priority_tree_active": bool(priority_tree_active),
            "dispatch_policy": selected_dispatch_policy,
            "conditional_prioritization_model_kind": selected.get("conditional_prioritization_model_kind"),
            "conditional_prioritization_pairs": selected.get("conditional_prioritization_pairs"),
            "conditional_prioritization_overtaking_rate": selected.get("conditional_prioritization_overtaking_rate"),
            "queue_strategy_name": selected.get("queue_strategy_name"),
            "queue_score_mixing_weights": selected.get("queue_score_mixing_weights"),
            "queue_scheduling_force_empirical": bool(selected.get("queue_scheduling_force_empirical", False)),
            "arrival_model_type": "case_arrival_decision_tree_interarrival",
            "external_delay_model_type": "activity_specific_external_delay_timer",
            "external_delay_tree_enabled": False,
            "external_delay_quantile_search": {
                "enabled": True,
                "search_mode": ("joint_with_priority_tree_depth" if priority_tree_active else "external_quantile_only_fcfs"),
                "candidate_quantiles": [float(x) for x in external_delay_quantile_candidates],
                "selected_quantile": float(selected_external_quantile),
                "global_scale_fixed": 1.0,
            },
            "external_delay_max_delay_quantile": float(selected_external_quantile),
            "fixed_runtime_controls": {
                "arrival_time_scale": 1.0,
                "arrival_demand_multiplier": 1.0,
                "external_delay_global_scale": 1.0,
                "external_delay_max_delay_quantile": float(selected_external_quantile),
                "queue_strategy_name": selected.get("queue_strategy_name"),
            },
            "external_delay_activity_scale_protocol": {
                "enabled": bool(TOP_ACTIVITY_SCALE_CALIBRATION_ENABLED),
                "top_k_policy": "adaptive_ratio_with_min_max_caps",
                "ratio": float(TOP_ACTIVITY_SCALE_RATIO),
                "min_k": int(TOP_ACTIVITY_SCALE_MIN_K),
                "max_k": int(TOP_ACTIVITY_SCALE_MAX_K),
                "candidate_scales": [float(x) for x in TOP_ACTIVITY_SCALE_CANDIDATES],
                "selection_partition": "train_validation_log_or_fixed_case_sample_if_large",
                "uses_test_log": False,
                "method": "coordinate_search_after_final_rediscovery_with_fixed_stage2c_quantile",
                "sampling_policy": {
                    "enabled": bool(FINAL_CALIBRATION_SAMPLING_ENABLED),
                    "large_log_threshold": int(FINAL_CALIBRATION_LARGE_LOG_THRESHOLD),
                    "max_cases": int(FINAL_CALIBRATION_MAX_CASES),
                    "confidence_level": float(FINAL_CALIBRATION_CONFIDENCE_LEVEL),
                    "margin_percent": float(FINAL_CALIBRATION_MARGIN_PERCENT),
                    "seed": int(FINAL_CALIBRATION_SAMPLE_SEED),
                    "sample_size_method": "finite_population_dsim_style",
                    "recent_validation_confirmation": bool(FINAL_CALIBRATION_RECENT_CONFIRMATION_ENABLED),
                    "confirmation_runs": int(FINAL_CALIBRATION_CONFIRMATION_RUNS),
                    "full_log_confirmation": False,
                },
            },
            "external_delay_activity_scales": {},
            "case_duration_calibration": duration_calibration,
            "structure_score": selected.get("structure_score"),
            "optuna_objective_score": selected.get("optuna_objective_score"),
            "runtime_score": selected.get("runtime_score"),
            "validation_metrics": selected.get("metrics", {}),
            "case_duration_guard": selected.get("case_duration_guard", {}),
            "arrival_guard": selected.get("arrival_guard", {}),
            "ctd_tolerance": float(ctd_tolerance) if ctd_tolerance is not None else None,
            "ctd_guard": selected.get("ctd_guard", {}),
            "runtime_guard_selection_mode": selected.get("runtime_guard_selection_mode"),
            "runtime_search_stage": selected.get("runtime_search_stage"),
            "fixed_structure_before_runtime": bool(FIX_STRUCTURE_BEFORE_RUNTIME),
            "fixed_stage1_structure": {
                "noise_threshold": (runtime_top_structures[0] or {}).get("noise_threshold") if runtime_top_structures else None,
                "tree_depth": (runtime_top_structures[0] or {}).get("tree_depth") if runtime_top_structures else None,
                "structure_score": (runtime_top_structures[0] or {}).get("structure_score") if runtime_top_structures else None,
            },
            "external_delay_ratio_guard": {
                "max_external_delay_ratio": max_external_delay_ratio,
                "observed_external_delay_ratio": (selected.get("case_duration_guard", {}) or {}).get("external_delay_ratio"),
            },
            "optuna_trial_number": selected.get("trial_number"),
            "optuna_resource_budget": selected.get("resource_budget"),
            "candidate_resource_filter": self._resource_filter_policy_summary(),
        }

        # ==========================================================
        # Stage 2d：Adaptive Runtime Recency
        # Full history remains the default. Only time-related modules may use
        # a recent tail window, selected solely on chronological validation.
        # ==========================================================
        runtime_recency = self._select_runtime_recency(
            structure_train_log=partitions.train_log,
            validation_log=partitions.validation_log,
            pnml_filepath=pnml_filepath,
            selected_hyperparameters=selected_hyperparameters,
            runtime_metrics=runtime_metrics,
            enable_candidate_resource_filter=enable_candidate_resource_filter,
            candidate_resource_coverage=candidate_resource_coverage,
            candidate_resource_max_k=candidate_resource_max_k,
            candidate_resource_min_k=candidate_resource_min_k,
        )
        selected_runtime_recency_ratio = float(runtime_recency.get("selected_ratio", 1.0))
        selected_hyperparameters["runtime_recency"] = runtime_recency
        selected_hyperparameters["runtime_recency_ratio"] = float(selected_runtime_recency_ratio)
        selected_hyperparameters["runtime_recency_modules"] = list(RUNTIME_RECENCY_MODULES)
        optuna_summary["stage2d_adaptive_runtime_recency"] = runtime_recency

        # ==========================================================
        # Stage 2e0：Triggered Arrival Recency
        # 先做零仿真 drift trigger；未触发时完全保持旧的 full-history Arrival。
        # 触发后，Arrival recent-window 独立于 service_time/external_delay recency 选择。
        # ==========================================================
        arrival_recency = self._select_arrival_recency(
            structure_train_log=partitions.train_log,
            validation_log=partitions.validation_log,
            pnml_filepath=pnml_filepath,
            selected_hyperparameters=selected_hyperparameters,
            runtime_recency_ratio=selected_runtime_recency_ratio,
            enable_candidate_resource_filter=enable_candidate_resource_filter,
            candidate_resource_coverage=candidate_resource_coverage,
            candidate_resource_max_k=candidate_resource_max_k,
            candidate_resource_min_k=candidate_resource_min_k,
        )
        selected_arrival_recency_ratio = float(arrival_recency.get("selected_ratio", 1.0))
        selected_hyperparameters["arrival_recency"] = arrival_recency
        selected_hyperparameters["arrival_recency_ratio"] = float(selected_arrival_recency_ratio)
        selected_hyperparameters["arrival_recency_modules"] = ["arrival"]
        optuna_summary["stage2e0_triggered_arrival_recency"] = arrival_recency

        # ==========================================================
        # Stage 2e：Validation-controlled Arrival Time Scale
        # Keep the selected Arrival tree and recency horizon fixed.  Only the
        # multiplicative runtime scale is calibrated on chronological validation.
        # Identity scale=1.0 is mandatory and is preferred unless validation
        # shows a meaningful CAR improvement (or the identity guard is unsafe).
        # ==========================================================
        arrival_scale_calibration = self._select_arrival_time_scale(
            structure_train_log=partitions.train_log,
            validation_log=partitions.validation_log,
            pnml_filepath=pnml_filepath,
            selected_hyperparameters=selected_hyperparameters,
            runtime_recency_ratio=selected_runtime_recency_ratio,
            arrival_recency_ratio=selected_arrival_recency_ratio,
            enable_candidate_resource_filter=enable_candidate_resource_filter,
            candidate_resource_coverage=candidate_resource_coverage,
            candidate_resource_max_k=candidate_resource_max_k,
            candidate_resource_min_k=candidate_resource_min_k,
        )
        selected_arrival_time_scale = float(
            arrival_scale_calibration.get("selected_scale", 1.0)
        )
        selected_hyperparameters["arrival_time_scale"] = float(selected_arrival_time_scale)
        selected_hyperparameters["arrival_demand_multiplier"] = 1.0
        selected_hyperparameters["arrival_time_scale_calibration"] = arrival_scale_calibration
        selected_hyperparameters.setdefault("fixed_runtime_controls", {})[
            "arrival_time_scale"
        ] = float(selected_arrival_time_scale)
        selected_hyperparameters["fixed_runtime_controls"][
            "arrival_demand_multiplier"
        ] = 1.0
        optuna_summary["stage2e_validation_controlled_arrival_scale"] = arrival_scale_calibration

        logger.info("Selected hyperparameters: %s", selected_hyperparameters)

        # ==========================================================
        # Stage 3：完整历史负责结构/稳定行为。
        # service_time/external_delay 使用 Stage2d recent window；Arrival 仅在 drift trigger
        # 已触发且 validation 确认后，使用自己独立的 recent window。calendar 仍保持完整历史。
        # ==========================================================
        final_runtime_log, final_runtime_cases = self._build_runtime_recency_log(
            partitions.train_validation_log,
            selected_runtime_recency_ratio,
        )
        if final_runtime_log is None:
            logger.warning(
                "Final runtime-recency window became too small; falling back to full history."
            )
            final_runtime_log = partitions.train_validation_log
            final_runtime_cases = self._count_log_cases(partitions.train_validation_log)
            selected_runtime_recency_ratio = 1.0
            selected_hyperparameters["runtime_recency_ratio"] = 1.0
            selected_hyperparameters["runtime_recency"]["final_fallback_to_full_history"] = True

        final_arrival_log, final_arrival_cases = self._build_arrival_recency_log(
            partitions.train_validation_log,
            selected_arrival_recency_ratio,
        )
        if final_arrival_log is None:
            logger.warning(
                "Final Arrival-recency window became too small; falling back to full-history Arrival."
            )
            final_arrival_log = partitions.train_validation_log
            final_arrival_cases = self._count_log_cases(partitions.train_validation_log)
            selected_arrival_recency_ratio = 1.0
            selected_hyperparameters["arrival_recency_ratio"] = 1.0
            selected_hyperparameters["arrival_recency"]["final_fallback_to_full_history"] = True

        logger.info(
            "Final rediscovery started: "
            f"noise_threshold={selected_hyperparameters['noise_threshold']}, "
            f"control_flow_depth={selected_tree_depth}, "
            f"arrival_depth={selected_arrival_depth}, resource_depth={selected_queue_depth}, "
            f"dispatch_policy={selected_dispatch_policy}, "
            f"priority_tree_depth={selected_priority_depth if priority_tree_active else 'n/a'}, "
            f"runtime_recency_ratio={selected_runtime_recency_ratio:.3f}, "
            f"arrival_recency_ratio={selected_arrival_recency_ratio:.3f}, "
            f"arrival_time_scale={selected_arrival_time_scale:.3f}, "
            f"runtime_cases={final_runtime_cases}, arrival_cases={final_arrival_cases}"
        )

        self._discover_candidate_model(
            event_log=partitions.train_validation_log,
            runtime_event_log=(
                final_runtime_log if selected_runtime_recency_ratio < 1.0 - 1e-12 else None
            ),
            pnml_filepath=pnml_filepath,
            noise_threshold=selected_hyperparameters["noise_threshold"],
            max_depth_tree=selected_tree_depth,
            decision_point_max_depth=selected_tree_depth,
            arrival_decision_tree_max_depth=selected_arrival_depth,
            queue_scheduling_max_depth=selected_queue_depth,
            conditional_prioritization_max_depth=selected_priority_depth,
            queue_scheduling_force_empirical=bool(selected_hyperparameters.get("queue_scheduling_force_empirical", False)),
            discovery_profile=FULL_DISCOVERY_PROFILE,
            build_process_artifacts=True,
            enable_candidate_resource_filter=enable_candidate_resource_filter,
            candidate_resource_coverage=candidate_resource_coverage,
            candidate_resource_max_k=candidate_resource_max_k,
            candidate_resource_min_k=candidate_resource_min_k,
            alignment_timeout_seconds=ALIGNMENT_TIMEOUT_SECONDS,
        )

        final_arrival_overlay = self._apply_arrival_recency_overlay(
            partitions.train_validation_log,
            selected_arrival_recency_ratio,
            selected_hyperparameters,
        )
        if selected_arrival_recency_ratio < 1.0 - 1e-12 and not bool(final_arrival_overlay.get("applied", False)):
            logger.warning("Selected Arrival recency could not be applied during final rediscovery; using full history.")
            selected_arrival_recency_ratio = 1.0
            selected_hyperparameters["arrival_recency_ratio"] = 1.0
        selected_hyperparameters["arrival_recency_final_overlay"] = final_arrival_overlay

        final_dispatch_summary = self._extract_dispatch_summary_from_integration(self.integration)
        final_dispatch_policy = str(final_dispatch_summary.get("dispatch_policy") or "fcfs")
        final_priority_tree_active = bool(final_dispatch_summary.get("priority_tree_active"))
        if final_dispatch_policy != selected_hyperparameters.get("dispatch_policy"):
            logger.info(
                "Final rediscovery updated data-driven dispatch policy: stage2c=%s -> final=%s",
                selected_hyperparameters.get("dispatch_policy"),
                final_dispatch_policy,
            )
        selected_hyperparameters["dispatch_policy"] = final_dispatch_policy
        selected_hyperparameters["priority_tree_active"] = final_priority_tree_active
        selected_hyperparameters["priority_tree_max_depth"] = (
            int(selected_priority_depth) if final_priority_tree_active else None
        )
        selected_hyperparameters["conditional_prioritization_model_kind"] = final_dispatch_summary.get("model_kind")
        selected_hyperparameters["conditional_prioritization_pairs"] = final_dispatch_summary.get("n_competition_pairs")
        selected_hyperparameters["conditional_prioritization_overtaking_rate"] = final_dispatch_summary.get("overtaking_rate")

        self._apply_selected_runtime_parameters(selected_hyperparameters)

        # Stage 3：Dispatch policy 由数据决定；Stage 2c 已选定 External quantile。
        # 若使用 CPT，priority-tree depth 也已在 Stage 2c 选定；若使用 FCFS，则该深度不参与选择。
        # final rediscovery 后固定 quantile，不再二次搜索；
        # 仅恢复 Top-K activity-specific External Delay scale 校准。
        try:
            self.integration._set_external_delay_activity_scales({}, reset_others=True)
        except Exception:
            pass
        self.integration._set_external_delay_runtime_config(
            global_scale=1.0,
            max_delay_quantile=float(selected_external_quantile),
        )

        top_k_scale_calibration = self._calibrate_top_activity_scales_after_rediscovery(
            search_log=final_runtime_log,
            # Stage 3b confirmation intentionally stays on the selected recent/runtime
            # window. Full-history veto is disabled because it can favor stale temporal
            # behavior and reject large scales that are beneficial on the recent window.
            confirmation_log=final_runtime_log,
            selected_quantile=float(selected_external_quantile),
        )
        selected_activity_scales = dict(
            (top_k_scale_calibration or {}).get("selected_activity_scales", {}) or {}
        )
        selected_hyperparameters["external_delay_activity_scales"] = selected_activity_scales
        selected_hyperparameters["final_rediscovery_calibration"] = {
            "enabled": bool(TOP_ACTIVITY_SCALE_CALIBRATION_ENABLED),
            "reason": "fixed_stage2c_quantile_then_top_k_activity_scale_calibration",
            "selected_quantile": float(selected_external_quantile),
            "selected_activity_scales": selected_activity_scales,
            "top_activity_scale_calibration": top_k_scale_calibration,
            "global_scale_fixed": 1.0,
            "uses_test_log": False,
            "runtime_recency_ratio": float(selected_runtime_recency_ratio),
            "runtime_recency_cases": int(final_runtime_cases),
            "runtime_recency_modules": list(RUNTIME_RECENCY_MODULES),
            "arrival_recency_ratio": float(selected_arrival_recency_ratio),
            "arrival_recency_cases": int(final_arrival_cases),
            "arrival_recency_triggered": bool((selected_hyperparameters.get("arrival_recency") or {}).get("trigger", {}).get("triggered", False)),
        }

        parameters = self.integration._convert_drq_to_app_format()
        process_model = self.integration.process_model

        # ==========================================================
        # Stage 4：test_log 最终评估
        # ==========================================================
        logger.info("Final test evaluation started.")

        if test_evaluation_runs > 1:
            repeated_kwargs = {
                "n_runs": test_evaluation_runs,
                "set_as_current_metrics": True,
                "metrics_labels": metrics_labels,
            }

            try:
                repeated_signature = inspect.signature(
                    self.integration.evaluate_on_event_log_repeated
                )
                repeated_supported = set(repeated_signature.parameters.keys())
            except Exception:
                repeated_supported = set(repeated_kwargs.keys())

            if "save_simulated_logs" in repeated_supported:
                repeated_kwargs["save_simulated_logs"] = bool(save_test_simulation_logs)
            elif save_test_simulation_logs:
                logger.warning(
                    "当前 drqIntegration.evaluate_on_event_log_repeated() "
                    "不支持 save_simulated_logs；重复测试日志不会被保存。"
                    "请将 drq_integration.py 替换为 context-arrival 版本以启用该功能。"
                )

            if "simulation_log_dir" in repeated_supported:
                repeated_kwargs["simulation_log_dir"] = test_simulation_log_dir
            if "simulation_log_prefix" in repeated_supported:
                repeated_kwargs["simulation_log_prefix"] = test_simulation_log_prefix

            test_metrics = self.integration.evaluate_on_event_log_repeated(
                partitions.test_log,
                **repeated_kwargs,
            )
        else:
            test_metrics = self.integration.evaluate_on_event_log(
                partitions.test_log,
                set_as_current_metrics=True,
                metrics_labels=metrics_labels,
            )

        parameters = self.integration._convert_drq_to_app_format()
        parameters["drq_metrics"] = test_metrics
        parameters["time_diagnostics"] = getattr(self.integration, "time_diagnostics", {})

        return {
            "parameters": parameters,
            "process_model": process_model,
            "log_partitions": partitions,
            "test_metrics": test_metrics,
            "test_metric_details": getattr(self.integration, "repeated_metric_details", {}),
            "test_simulation_logs": getattr(self.integration, "repeated_simulation_log_files", []),
            "time_diagnostics": getattr(self.integration, "time_diagnostics", {}),
            "evaluation_runs": test_evaluation_runs,
            "hyperparameter_search": {
                "protocol": (
                    "manual_or_automatic_petri_noise_threshold_then_tree_depth_grid_validation_cfld_ngd_3gd_"
                    "stable_stage1_final_selection_low_noise_activity_coverage_confirmation_"
                    "stage2a_arrival_depth_grid_car_arrival_guard_"
                    "stage2b_resource_allocation_depth_grid_data_driven_dispatch_"
                    "stage2c_dispatch_aware_external_quantile_search_"
                    "stage2d_adaptive_runtime_recency_chronological_validation_"
                    "stage2e_validation_controlled_arrival_time_scale_"
                    "stage3_full_history_structure_recent_runtime_rediscovery_topk_activity_scale_calibration_"
                    "large_log_sample_search_topk_full_validation_confirmation_"
                    "adaptive_resource_filter_by_resource_count_"
                    "symmetric_log_duration_regularization_global_scale_fixed_"
                    "train_validation_for_final_test_for_evaluation"
                ),
                "selected_hyperparameters": selected_hyperparameters,
                "structure_quality_candidates": structure_quality_records,
                "structure_candidates": structure_records,
                "top_structure_candidates": top_structures,
                "structure_optuna_summary": structure_optuna_summary,
                "runtime_candidates": runtime_records,
                "optuna_summary": optuna_summary,
                "speed_optimization": {
                    "build_tag": STAGEWISE_TUNER_BUILD_TAG,
                    "tuning_sampling_threshold": int(TUNING_LARGE_LOG_THRESHOLD),
                    "final_calibration_sampling_threshold": int(FINAL_CALIBRATION_LARGE_LOG_THRESHOLD),
                    "discovery_state_cache_enabled": bool(DISCOVERY_STATE_CACHE_ENABLED),
                    "discovery_state_cache_max_entries": int(DISCOVERY_STATE_CACHE_MAX_ENTRIES),
                    "precomputed_feature_cache_enabled": bool(PRECOMPUTED_FEATURE_CACHE_ENABLED),
                    "precomputed_feature_cache_max_entries": int(PRECOMPUTED_FEATURE_CACHE_MAX_ENTRIES),
                    "candidate_eval_execution_mode": CANDIDATE_EVAL_EXECUTION_MODE,
                    # Backward-compatible fields: serial-only implementation always reports no parallelism.
                    "parallel_candidate_eval_enabled": False,
                    "parallel_workers": 1,
                    "stage2c_parallel_enabled": False,
                    "stage3b_parallel_enabled": False,
                    "reuse_identity_scale_baseline": bool(REUSE_IDENTITY_SCALE_BASELINE),
                    "cache_stats": dict(self._cache_stats),
                },
            },
        }

    def _run_split_structure_search(
            self,
            train_log,
            validation_log,
            pnml_filepath,
            structure_metrics: tuple[str, ...],
            tree_depth_candidates: list[int],
            noise_threshold_range: tuple[float, float],
            top_k_structures: int,
            n_trials: int,
            optuna_seed: int,
            sampler_name: str,
            validation_budgets=(1, 2),
            reduction_factor: int = 2,
    ) -> dict:
        """Stage-1 structure search.

        Automatic IM mode performs a lightweight JOINT search over
        ``(noise_threshold, tree_depth)``.  This avoids the old sequential
        approximation (noise searched only at one probe depth, then depth grid)
        while keeping the evaluation budget bounded.

        Fixed-noise and fixed-PNML modes intentionally retain the depth grid,
        because only one structural dimension remains free in those modes.
        """
        tree_depth_candidates = sorted({max(1, int(d)) for d in tree_depth_candidates})
        if not tree_depth_candidates:
            tree_depth_candidates = list(TREE_DEPTH_CANDIDATES)

        budgets = sorted({max(1, int(b)) for b in validation_budgets})
        if not budgets:
            budgets = [1]

        confirmation_top_k = max(1, int(STRUCTURE_CONFIRMATION_TOP_K))
        search_top_k = max(int(top_k_structures), confirmation_top_k)

        noise_low, noise_high = noise_threshold_range
        noise_low = float(np.clip(float(noise_low), 0.0, 1.0))
        noise_high = float(np.clip(float(noise_high), 0.0, 1.0))
        if noise_high < noise_low:
            noise_low, noise_high = noise_high, noise_low
        noise_is_fixed = abs(noise_high - noise_low) <= 1e-12

        # --------------------------------------------------------------
        # Automatic mode: one Optuna trial == one complete CF structure
        # candidate (IM noise threshold + decision-point tree depth).
        # No additional Stage-1B depth grid is needed.
        # --------------------------------------------------------------
        if (not pnml_filepath) and (not noise_is_fixed):
            joint_result = self._run_joint_structure_search(
                train_log=train_log,
                validation_log=validation_log,
                structure_metrics=structure_metrics,
                noise_threshold_range=(noise_low, noise_high),
                tree_depth_candidates=tree_depth_candidates,
                top_k_structures=search_top_k,
                n_trials=n_trials,
                optuna_seed=optuna_seed,
                sampler_name=sampler_name,
                validation_budgets=budgets,
                reduction_factor=reduction_factor,
            )

            confirmation_result = {
                "enabled": False,
                "top_structures": joint_result["top_structures"][:max(1, int(top_k_structures))],
                "confirmation_records": [],
                "summary": {"enabled": False},
            }
            if STRUCTURE_CONFIRMATION_ENABLED:
                confirmation_result = self._confirm_top_structure_candidates(
                    candidate_records=joint_result["top_structures"],
                    train_log=train_log,
                    validation_log=validation_log,
                    pnml_filepath=None,
                    structure_metrics=structure_metrics,
                    top_k_structures=top_k_structures,
                )

            final_top_structures = (
                confirmation_result.get("top_structures")
                or joint_result["top_structures"]
            )
            final_top_structures = final_top_structures[:max(1, int(top_k_structures))]

            return {
                "structure_records": (
                    list(joint_result["structure_records"])
                    + list(confirmation_result.get("confirmation_records", []))
                ),
                "top_structures": final_top_structures,
                "optuna_summary": {
                    "enabled": True,
                    "search_type": "joint_petri_noise_tree_depth_tpe_with_stable_final_selection",
                    "joint_structure_search": joint_result["optuna_summary"],
                    "final_confirmation": confirmation_result.get("summary", {}),
                    "metrics": list(structure_metrics),
                    "final_selection_budget": int(max(budgets)),
                    "top_k_structures": int(top_k_structures),
                    "internal_confirmation_top_k": int(search_top_k),
                    "selection_regularization": {
                        "activity_coverage_penalty_weight": float(ACTIVITY_COVERAGE_PENALTY_WEIGHT),
                        "noise_penalty_weight": float(NOISE_PENALTY_WEIGHT),
                        "low_noise_preference_enabled": bool(LOW_NOISE_PREFERENCE_ENABLED),
                        "low_noise_preference_tolerance": float(LOW_NOISE_PREFERENCE_TOL),
                        "confirmation_runs": int(STRUCTURE_CONFIRMATION_RUNS),
                    },
                },
            }

        # --------------------------------------------------------------
        # Fixed PNML / fixed noise: only depth is free, so retain grid.
        # --------------------------------------------------------------
        if pnml_filepath:
            logger.info(
                "Stage 1A skipped because PNML is provided. "
                "Petri net is fixed; only tree_depth grid search will be executed."
            )
            petri_records = []
            petri_candidates = [{
                "stage": "petri_fixed_pnml",
                "trial_number": -1,
                "resource_budget": max(budgets),
                "noise_threshold": None,
                "pnml_filepath": pnml_filepath,
                "metrics": {},
                "process_quality": {},
                "optuna_objective_score": 0.0,
                "petri_score": 0.0,
                "petri_selection_score": 0.0,
                "status": "ok",
            }]
            petri_summary = {
                "enabled": False,
                "skipped_reason": "pnml_filepath_provided",
                "noise_threshold_search": "skipped_fixed_pnml",
                "noise_threshold_range": None,
                "top_k_petri_candidates": 1,
            }
        else:
            fixed_noise_threshold = float(noise_low)
            logger.info(
                "Stage 1A joint search skipped because a fixed noise threshold was provided. "
                "Using Inductive Miner noise_threshold=%.6f; only tree_depth grid search will be executed.",
                fixed_noise_threshold,
            )
            petri_records = []
            petri_candidates = [{
                "stage": "petri_fixed_noise_threshold",
                "trial_number": -1,
                "resource_budget": max(budgets),
                "noise_threshold": fixed_noise_threshold,
                "pnml_filepath": None,
                "metrics": {},
                "process_quality": {},
                "optuna_objective_score": 0.0,
                "petri_score": 0.0,
                "petri_selection_score": 0.0,
                "status": "ok",
            }]
            petri_summary = {
                "enabled": False,
                "skipped_reason": "fixed_noise_threshold_from_ui_or_caller",
                "noise_threshold_search": "skipped_fixed_threshold",
                "noise_threshold": fixed_noise_threshold,
                "noise_threshold_range": [fixed_noise_threshold, fixed_noise_threshold],
                "top_k_petri_candidates": 1,
            }

        tree_result = self._run_tree_depth_grid_search(
            petri_candidates=petri_candidates,
            train_log=train_log,
            validation_log=validation_log,
            pnml_filepath=pnml_filepath,
            structure_metrics=structure_metrics,
            tree_depth_candidates=tree_depth_candidates,
            validation_budgets=budgets,
            top_k_structures=search_top_k,
        )

        confirmation_result = {
            "enabled": False,
            "top_structures": tree_result["top_structures"][:max(1, int(top_k_structures))],
            "confirmation_records": [],
            "summary": {"enabled": False},
        }
        if STRUCTURE_CONFIRMATION_ENABLED:
            confirmation_result = self._confirm_top_structure_candidates(
                candidate_records=tree_result["top_structures"],
                train_log=train_log,
                validation_log=validation_log,
                pnml_filepath=pnml_filepath,
                structure_metrics=structure_metrics,
                top_k_structures=top_k_structures,
            )

        final_top_structures = confirmation_result.get("top_structures") or tree_result["top_structures"]
        final_top_structures = final_top_structures[:max(1, int(top_k_structures))]

        return {
            "structure_records": (
                list(petri_records)
                + list(tree_result["tree_depth_records"])
                + list(confirmation_result.get("confirmation_records", []))
            ),
            "top_structures": final_top_structures,
            "optuna_summary": {
                "enabled": True,
                "search_type": "fixed_structure_dimension_then_tree_depth_grid",
                "petri_noise_search": petri_summary,
                "tree_depth_grid_search": tree_result["grid_summary"],
                "final_confirmation": confirmation_result.get("summary", {}),
                "metrics": list(structure_metrics),
                "final_selection_budget": int(max(budgets)),
                "top_k_structures": int(top_k_structures),
                "internal_confirmation_top_k": int(search_top_k),
                "selection_regularization": {
                    "activity_coverage_penalty_weight": float(ACTIVITY_COVERAGE_PENALTY_WEIGHT),
                    "noise_penalty_weight": float(NOISE_PENALTY_WEIGHT),
                    "low_noise_preference_enabled": bool(LOW_NOISE_PREFERENCE_ENABLED),
                    "low_noise_preference_tolerance": float(LOW_NOISE_PREFERENCE_TOL),
                    "confirmation_runs": int(STRUCTURE_CONFIRMATION_RUNS),
                },
            },
        }

    def _run_joint_structure_search(
            self,
            train_log,
            validation_log,
            structure_metrics: tuple[str, ...],
            noise_threshold_range: tuple[float, float],
            tree_depth_candidates: list[int],
            top_k_structures: int,
            n_trials: int,
            optuna_seed: int,
            sampler_name: str,
            validation_budgets=(1, 2),
            reduction_factor: int = 2,
    ) -> dict:
        """Lightweight joint TPE search for IM noise and CF tree depth."""
        try:
            import optuna
        except Exception as exc:
            raise RuntimeError(
                "Optuna is required for joint structure search. "
                "Please install it with: pip install optuna"
            ) from exc

        low, high = noise_threshold_range
        low = float(np.clip(low, 0.0, 1.0))
        high = float(np.clip(high, 0.0, 1.0))
        if high < low:
            low, high = high, low
        if abs(high - low) <= 1e-12:
            raise RuntimeError(
                "_run_joint_structure_search() received a fixed noise range; "
                "fixed thresholds should use the depth-grid branch."
            )

        depths = sorted({max(1, int(d)) for d in tree_depth_candidates})
        if not depths:
            depths = list(TREE_DEPTH_CANDIDATES)

        budgets = sorted({max(1, int(b)) for b in validation_budgets}) or [1]
        n_trials = max(1, int(n_trials))
        reduction_factor = max(2, int(reduction_factor))
        structure_records: list[dict] = []
        reference_activities = self._extract_log_activity_set(train_log)

        logger.info(
            "Stage 1AB joint structure search started: "
            "noise_threshold_range=(%.4f, %.4f), tree_depth_candidates=%s, "
            "n_trials=%s, budgets=%s, metrics=%s",
            low, high, depths, n_trials, budgets, list(structure_metrics),
        )

        sampler_request = str(sampler_name or "tpe").lower()
        if sampler_request == "random":
            sampler = optuna.samplers.RandomSampler(seed=int(optuna_seed))
            sampler_display_name = "RandomSampler"
        else:
            # With 20 total trials and four deterministic anchors, six startup
            # trials leave most of the remaining budget for adaptive TPE search.
            sampler = optuna.samplers.TPESampler(
                seed=int(optuna_seed),
                n_startup_trials=min(6, n_trials),
            )
            sampler_display_name = "TPESampler"

        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min(budgets),
            reduction_factor=reduction_factor,
            min_early_stopping_rate=0,
        )

        def objective(trial: Any) -> float:
            noise_threshold = float(trial.suggest_float("noise_threshold", low, high))
            tree_depth = int(trial.suggest_categorical("tree_depth", depths))

            try:
                self._discover_candidate_model(
                    event_log=train_log,
                    pnml_filepath=None,
                    noise_threshold=noise_threshold,
                    max_depth_tree=tree_depth,
                    decision_point_max_depth=tree_depth,
                    discovery_profile=CONTROL_FLOW_DISCOVERY_PROFILE,
                    build_process_artifacts=False,
                    precomputed_df_features=None,
                    reuse_current_process_model=False,
                    alignment_timeout_seconds=ALIGNMENT_TIMEOUT_SECONDS,
                )
                activity_coverage = self._compute_current_process_activity_coverage(reference_activities)

                last_score = float("inf")
                for budget in budgets:
                    validation_metrics = self._evaluate(
                        event_log=validation_log,
                        n_runs=budget,
                        metrics_labels=list(structure_metrics),
                        enable_time_diagnostics=False,
                    )
                    flat_metrics = self.integration._flatten_formatted_metrics(validation_metrics)
                    last_score = self._scalar_structure_objective(
                        flat_metrics,
                        metric_names=list(structure_metrics),
                    )

                    record = {
                        "stage": "joint_structure_optuna",
                        "trial_number": int(trial.number),
                        "resource_budget": int(budget),
                        "noise_threshold": float(noise_threshold),
                        "tree_depth": int(tree_depth),
                        "max_depth_tree": int(tree_depth),
                        "decision_point_max_depth": int(tree_depth),
                        "metrics": flat_metrics,
                        "process_quality": {},
                        "activity_coverage": activity_coverage,
                        "optuna_objective_score": float(last_score),
                        "status": "ok",
                    }
                    structure_records.append(record)
                    trial.set_user_attr("last_structure_record", record)
                    trial.report(float(last_score), step=int(budget))

                    if trial.should_prune():
                        record["status"] = "pruned"
                        record["pruned_at_budget"] = int(budget)
                        logger.info(
                            "Optuna pruned joint structure trial %s at budget=%s: "
                            "noise=%.6f depth=%s score=%.6f",
                            trial.number, budget, noise_threshold, tree_depth, last_score,
                        )
                        raise optuna.TrialPruned()

                return float(last_score)

            except optuna.TrialPruned:
                raise
            except Exception as exc:
                is_timeout = type(exc).__name__ == "AlignmentTimeoutError"
                logger.warning(
                    "Joint structure candidate %s%s failed: noise_threshold=%.6f, "
                    "tree_depth=%s, error=%s",
                    trial.number,
                    " TIMED OUT" if is_timeout else "",
                    noise_threshold,
                    tree_depth,
                    exc,
                )
                structure_records.append({
                    "stage": "joint_structure_optuna",
                    "trial_number": int(trial.number),
                    "noise_threshold": float(noise_threshold),
                    "tree_depth": int(tree_depth),
                    "max_depth_tree": int(tree_depth),
                    "decision_point_max_depth": int(tree_depth),
                    "metrics": {},
                    "process_quality": {},
                    "optuna_objective_score": float("inf"),
                    "status": "timeout" if is_timeout else "failed",
                    "error": str(exc),
                })
                return float("inf")

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name="drq_joint_structure_noise_depth",
        )

        # Anchor every depth at the lowest-noise/default boundary.  With the
        # default range this is noise=0, ensuring the drq-style baseline is
        # never discarded merely because TPE did not sample it.
        anchor_noise = 0.0 if low <= 0.0 <= high else float(low)
        anchor_trials = [
            {"noise_threshold": float(anchor_noise), "tree_depth": int(depth)}
            for depth in depths
        ][:n_trials]
        for anchor in anchor_trials:
            try:
                study.enqueue_trial(anchor)
                logger.info(
                    "Enqueued joint structure anchor: noise_threshold=%.6f, tree_depth=%s",
                    anchor["noise_threshold"], anchor["tree_depth"],
                )
            except Exception as exc:
                logger.warning("Failed to enqueue joint structure anchor %s: %s", anchor, exc)

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=None,
            gc_after_trial=True,
            show_progress_bar=False,
        )

        valid_records = [
            r for r in structure_records
            if r.get("status") == "ok"
            and any(self._get_numeric_metric(r, m) is not None for m in structure_metrics)
        ]
        if not valid_records:
            raise RuntimeError("Joint structure search produced no valid candidate.")

        final_budget = max(int(r.get("resource_budget", 1)) for r in valid_records)
        final_budget_records = [
            r for r in valid_records
            if int(r.get("resource_budget", 1)) == final_budget
        ] or valid_records

        self._attach_ideal_point_scores(
            final_budget_records,
            metric_names=list(structure_metrics),
            metrics_key="metrics",
            score_key="structure_score",
        )
        self._attach_structure_selection_scores(
            final_budget_records,
            base_score_key="structure_score",
            selection_score_key="structure_selection_score",
        )
        pareto_records = self._pareto_prune_records(
            final_budget_records,
            metric_names=list(structure_metrics),
        )
        top_structures = self._select_structure_records_with_low_noise_preference(
            primary_records=pareto_records,
            fallback_records=final_budget_records,
            top_k=top_k_structures,
            score_key="structure_selection_score",
        )

        logger.info(
            "Stage 1AB joint structure search completed. Top candidates: %s",
            [
                (
                    r.get("noise_threshold"),
                    r.get("tree_depth"),
                    r.get("structure_score"),
                    r.get("structure_selection_score"),
                    (r.get("activity_coverage") or {}).get("coverage"),
                )
                for r in top_structures
            ],
        )

        return {
            "structure_records": structure_records,
            "top_structures": top_structures,
            "optuna_summary": {
                "enabled": True,
                "sampler": sampler_display_name,
                "pruner": "SuccessiveHalvingPruner",
                "search": "joint_noise_threshold_and_tree_depth",
                "noise_threshold_search": "continuous",
                "noise_threshold_range": [float(low), float(high)],
                "tree_depth_candidates": list(depths),
                "forced_anchor_trials": list(anchor_trials),
                "n_anchor_trials": int(len(anchor_trials)),
                "n_trials_requested": int(n_trials),
                "n_adaptive_or_random_trials": int(max(0, n_trials - len(anchor_trials))),
                "n_trials_finished": len(study.trials),
                "n_complete_trials": len([
                    t for t in study.trials
                    if str(t.state).endswith("COMPLETE")
                ]),
                "n_pruned_trials": len([
                    t for t in study.trials
                    if str(t.state).endswith("PRUNED")
                ]),
                "validation_budgets": list(budgets),
                "reduction_factor": int(reduction_factor),
                "seed": int(optuna_seed),
                "final_selection_budget": int(final_budget),
                "top_k_structures": int(top_k_structures),
                "selection_score": "structure_selection_score",
                "activity_coverage_penalty_weight": float(ACTIVITY_COVERAGE_PENALTY_WEIGHT),
                "noise_penalty_weight": float(NOISE_PENALTY_WEIGHT),
                "low_noise_preference_tolerance": float(LOW_NOISE_PREFERENCE_TOL),
                "best_value": float(study.best_value) if study.best_trial is not None else None,
                "best_params": dict(study.best_params) if study.best_trial is not None else {},
            },
        }

    def _run_petri_noise_search(
            self,
            train_log,
            validation_log,
            structure_metrics: tuple[str, ...],
            noise_threshold_range: tuple[float, float],
            top_k_petri_candidates: int,
            n_trials: int,
            optuna_seed: int,
            sampler_name: str,
            validation_budgets=(1, 2),
            reduction_factor: int = 2,
            probe_tree_depth: int = 3,
    ) -> dict:
        try:
            import optuna
        except Exception as exc:
            raise RuntimeError(
                "Optuna is required for Petri-net noise-threshold search. "
                "Please install it with: pip install optuna"
            ) from exc

        low, high = noise_threshold_range
        low = float(np.clip(low, 0.0, 1.0))
        high = float(np.clip(high, 0.0, 1.0))
        if abs(high - low) <= 1e-12:
            raise RuntimeError(
                "_run_petri_noise_search() received a fixed noise_threshold_range. "
                "Fixed manual thresholds should be handled by _run_split_structure_search() "
                "without Optuna search."
            )
        if high < low:
            low, high = high, low
        if high <= low:
            high = min(1.0, low + 0.01)

        budgets = sorted({max(1, int(b)) for b in validation_budgets}) or [1]
        n_trials = max(1, int(n_trials))
        reduction_factor = max(2, int(reduction_factor))
        probe_tree_depth = max(1, int(probe_tree_depth))
        petri_records: list[dict] = []
        reference_activities = self._extract_log_activity_set(train_log)

        logger.info(
            "Stage 1A Petri-net noise search started: "
            f"noise_threshold_range=({low}, {high}), "
            f"probe_tree_depth={probe_tree_depth}, n_trials={n_trials}, "
            f"budgets={budgets}, metrics={list(structure_metrics)}"
        )

        sampler_request = str(sampler_name or "tpe").lower()
        if sampler_request == "random":
            sampler = optuna.samplers.RandomSampler(seed=int(optuna_seed))
            sampler_display_name = "RandomSampler"
        else:
            sampler = optuna.samplers.TPESampler(
                seed=int(optuna_seed),
                n_startup_trials=min(8, n_trials),
            )
            sampler_display_name = "TPESampler"

        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min(budgets),
            reduction_factor=reduction_factor,
            min_early_stopping_rate=0,
        )

        def objective(trial: Any) -> float:
            noise_threshold = float(trial.suggest_float("noise_threshold", low, high))

            try:
                self._discover_candidate_model(
                    event_log=train_log,
                    pnml_filepath=None,
                    noise_threshold=noise_threshold,
                    max_depth_tree=probe_tree_depth,
                    decision_point_max_depth=probe_tree_depth,
                    discovery_profile=CONTROL_FLOW_DISCOVERY_PROFILE,
                    build_process_artifacts=False,
                    precomputed_df_features=None,
                    reuse_current_process_model=False,
                    alignment_timeout_seconds=ALIGNMENT_TIMEOUT_SECONDS,
                )
                activity_coverage = self._compute_current_process_activity_coverage(reference_activities)

                last_score = float("inf")
                for budget in budgets:
                    validation_metrics = self._evaluate(
                        event_log=validation_log,
                        n_runs=budget,
                        metrics_labels=list(structure_metrics),
                        enable_time_diagnostics=False,
                    )
                    flat_metrics = self.integration._flatten_formatted_metrics(validation_metrics)
                    last_score = self._scalar_structure_objective(
                        flat_metrics,
                        metric_names=list(structure_metrics),
                    )

                    record = {
                        "stage": "petri_noise_optuna",
                        "trial_number": int(trial.number),
                        "resource_budget": int(budget),
                        "noise_threshold": float(noise_threshold),
                        "tree_depth": int(probe_tree_depth),
                        "max_depth_tree": int(probe_tree_depth),
                        "decision_point_max_depth": int(probe_tree_depth),
                        "metrics": flat_metrics,
                        "process_quality": {},
                        "activity_coverage": activity_coverage,
                        "optuna_objective_score": float(last_score),
                        "status": "ok",
                    }
                    petri_records.append(record)
                    trial.set_user_attr("last_petri_record", record)
                    trial.report(float(last_score), step=int(budget))

                    if trial.should_prune():
                        record["status"] = "pruned"
                        record["pruned_at_budget"] = int(budget)
                        logger.info(
                            "Optuna pruned Petri-net trial %s at budget=%s with score=%s",
                            trial.number,
                            budget,
                            last_score,
                        )
                        raise optuna.TrialPruned()

                return float(last_score)

            except optuna.TrialPruned:
                raise
            except Exception as exc:
                # AlignmentTimeoutError 和其他失败被当作得分为无穷的失败 trial，
                # 以便 Optuna 能继续下一个 trial。
                is_timeout = type(exc).__name__ == "AlignmentTimeoutError"
                logger.warning(
                    "Petri-net noise candidate %s%s: "
                    f"noise_threshold={noise_threshold}, error={exc}",
                    trial.number,
                    " TIMED OUT" if is_timeout else " failed",
                )
                petri_records.append({
                    "stage": "petri_noise_optuna",
                    "trial_number": int(trial.number),
                    "noise_threshold": float(noise_threshold),
                    "tree_depth": int(probe_tree_depth),
                    "max_depth_tree": int(probe_tree_depth),
                    "decision_point_max_depth": int(probe_tree_depth),
                    "metrics": {},
                    "process_quality": {},
                    "optuna_objective_score": float("inf"),
                    "status": "timeout" if is_timeout else "failed",
                    "error": str(exc),
                })
                return float("inf")

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name="drq_petri_noise_threshold",
        )

        forced_noise_thresholds = []
        if low <= 0.0 <= high:
            forced_noise_thresholds.append(0.0)

        for forced_noise in forced_noise_thresholds:
            try:
                study.enqueue_trial({"noise_threshold": float(forced_noise)})
                logger.info(
                    "Enqueued fixed Petri-net noise candidate: noise_threshold=%s",
                    forced_noise,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to enqueue fixed Petri-net noise candidate %s: %s",
                    forced_noise,
                    exc,
                )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=None,
            gc_after_trial=True,
            show_progress_bar=False,
        )

        valid_records = [
            r for r in petri_records
            if r.get("status") == "ok"
            and any(self._get_numeric_metric(r, m) is not None for m in structure_metrics)
        ]
        if not valid_records:
            raise RuntimeError("Petri-net noise search produced no valid candidate.")

        final_budget = max(int(r.get("resource_budget", 1)) for r in valid_records)
        final_budget_records = [
            r for r in valid_records
            if int(r.get("resource_budget", 1)) == final_budget
        ] or valid_records

        self._attach_ideal_point_scores(
            final_budget_records,
            metric_names=list(structure_metrics),
            metrics_key="metrics",
            score_key="petri_score",
        )
        self._attach_structure_selection_scores(
            final_budget_records,
            base_score_key="petri_score",
            selection_score_key="petri_selection_score",
        )
        pareto_records = self._pareto_prune_records(
            final_budget_records,
            metric_names=list(structure_metrics),
        )
        top_petri_candidates = self._select_structure_records_with_low_noise_preference(
            primary_records=pareto_records,
            fallback_records=final_budget_records,
            top_k=top_k_petri_candidates,
            score_key="petri_selection_score",
        )

        logger.info(
            "Stage 1A Petri-net noise search completed. Top Petri candidates: %s",
            [
                (
                    r.get("noise_threshold"),
                    r.get("petri_score"),
                    r.get("petri_selection_score"),
                    (r.get("activity_coverage") or {}).get("coverage"),
                )
                for r in top_petri_candidates
            ],
        )

        return {
            "petri_records": petri_records,
            "top_petri_candidates": top_petri_candidates,
            "optuna_summary": {
                "enabled": True,
                "sampler": sampler_display_name,
                "pruner": "SuccessiveHalvingPruner",
                "noise_threshold_search": "continuous",
                "noise_threshold_range": [float(low), float(high)],
                "forced_noise_thresholds": list(forced_noise_thresholds),
                "probe_tree_depth": int(probe_tree_depth),
                "metrics": list(structure_metrics),
                "n_trials_requested": int(n_trials),
                "n_trials_finished": len(study.trials),
                "n_complete_trials": len([
                    t for t in study.trials
                    if str(t.state).endswith("COMPLETE")
                ]),
                "n_pruned_trials": len([
                    t for t in study.trials
                    if str(t.state).endswith("PRUNED")
                ]),
                "validation_budgets": list(budgets),
                "reduction_factor": int(reduction_factor),
                "seed": int(optuna_seed),
                "final_selection_budget": int(final_budget),
                "top_k_petri_candidates": int(top_k_petri_candidates),
                "selection_score": "petri_selection_score",
                "activity_coverage_penalty_weight": float(ACTIVITY_COVERAGE_PENALTY_WEIGHT),
                "noise_penalty_weight": float(NOISE_PENALTY_WEIGHT),
                "low_noise_preference_tolerance": float(LOW_NOISE_PREFERENCE_TOL),
                "best_value": float(study.best_value) if study.best_trial is not None else None,
                "best_params": dict(study.best_params) if study.best_trial is not None else {},
            },
        }

    def _run_tree_depth_grid_search(
            self,
            petri_candidates: list[dict],
            train_log,
            validation_log,
            pnml_filepath,
            structure_metrics: tuple[str, ...],
            tree_depth_candidates: list[int],
            validation_budgets=(1, 2),
            top_k_structures: int = 3,
    ) -> dict:
        tree_depth_candidates = sorted({max(1, int(d)) for d in tree_depth_candidates})
        if not tree_depth_candidates:
            tree_depth_candidates = list(TREE_DEPTH_CANDIDATES)

        budgets = sorted({max(1, int(b)) for b in validation_budgets}) or [1]
        tree_depth_records: list[dict] = []
        candidate_number = 0
        reference_activities = self._extract_log_activity_set(train_log)

        logger.info(
            "Stage 1B tree-depth grid search started: "
            f"n_petri_candidates={len(petri_candidates)}, "
            f"tree_depth_candidates={tree_depth_candidates}, "
            f"budgets={budgets}, metrics={list(structure_metrics)}"
        )

        for petri_idx, petri_record in enumerate(petri_candidates):
            noise_threshold = petri_record.get("noise_threshold")

            for tree_depth in tree_depth_candidates:
                candidate_number += 1
                try:
                    self._discover_candidate_model(
                        event_log=train_log,
                        pnml_filepath=pnml_filepath,
                        noise_threshold=noise_threshold,
                        max_depth_tree=tree_depth,
                        decision_point_max_depth=tree_depth,
                        discovery_profile=CONTROL_FLOW_DISCOVERY_PROFILE,
                        build_process_artifacts=False,
                        precomputed_df_features=None,
                        reuse_current_process_model=False,
                        alignment_timeout_seconds=ALIGNMENT_TIMEOUT_SECONDS,
                    )
                    activity_coverage = self._compute_current_process_activity_coverage(reference_activities)

                    for budget in budgets:
                        validation_metrics = self._evaluate(
                            event_log=validation_log,
                            n_runs=budget,
                            metrics_labels=list(structure_metrics),
                            enable_time_diagnostics=False,
                        )
                        flat_metrics = self.integration._flatten_formatted_metrics(validation_metrics)
                        score = self._scalar_structure_objective(
                            flat_metrics,
                            metric_names=list(structure_metrics),
                        )

                        tree_depth_records.append({
                            "stage": "tree_depth_grid",
                            "trial_number": int(candidate_number),
                            "petri_candidate_index": int(petri_idx),
                            "resource_budget": int(budget),
                            "noise_threshold": noise_threshold,
                            "tree_depth": int(tree_depth),
                            "max_depth_tree": int(tree_depth),
                            "decision_point_max_depth": int(tree_depth),
                            "metrics": flat_metrics,
                            "process_quality": {},
                            "activity_coverage": activity_coverage,
                            "petri_score": petri_record.get("petri_score"),
                            "petri_selection_score": petri_record.get("petri_selection_score"),
                            "optuna_objective_score": float(score),
                            "status": "ok",
                        })

                except Exception as exc:
                    logger.warning(
                        "Tree-depth candidate failed: "
                        f"petri_idx={petri_idx}, noise_threshold={noise_threshold}, "
                        f"tree_depth={tree_depth}, error={exc}"
                    )
                    tree_depth_records.append({
                        "stage": "tree_depth_grid",
                        "trial_number": int(candidate_number),
                        "petri_candidate_index": int(petri_idx),
                        "noise_threshold": noise_threshold,
                        "tree_depth": int(tree_depth),
                        "max_depth_tree": int(tree_depth),
                        "decision_point_max_depth": int(tree_depth),
                        "metrics": {},
                        "process_quality": {},
                        "petri_score": petri_record.get("petri_score"),
                        "optuna_objective_score": float("inf"),
                        "status": "failed",
                        "error": str(exc),
                    })

        valid_records = [
            r for r in tree_depth_records
            if r.get("status") == "ok"
            and any(self._get_numeric_metric(r, m) is not None for m in structure_metrics)
        ]
        if not valid_records:
            raise RuntimeError("Tree-depth grid search produced no valid candidate.")

        final_budget = max(int(r.get("resource_budget", 1)) for r in valid_records)
        final_budget_records = [
            r for r in valid_records
            if int(r.get("resource_budget", 1)) == final_budget
        ] or valid_records

        self._attach_ideal_point_scores(
            final_budget_records,
            metric_names=list(structure_metrics),
            metrics_key="metrics",
            score_key="structure_score",
        )
        self._attach_structure_selection_scores(
            final_budget_records,
            base_score_key="structure_score",
            selection_score_key="structure_selection_score",
        )
        pareto_records = self._pareto_prune_records(
            final_budget_records,
            metric_names=list(structure_metrics),
        )
        top_structures = self._select_structure_records_with_low_noise_preference(
            primary_records=pareto_records,
            fallback_records=final_budget_records,
            top_k=top_k_structures,
            score_key="structure_selection_score",
        )

        logger.info(
            "Stage 1B tree-depth grid search completed. Top candidates: %s",
            [
                (
                    r.get("noise_threshold"),
                    r.get("tree_depth"),
                    r.get("structure_score"),
                    r.get("structure_selection_score"),
                    (r.get("activity_coverage") or {}).get("coverage"),
                )
                for r in top_structures
            ],
        )

        return {
            "tree_depth_records": tree_depth_records,
            "top_structures": top_structures,
            "grid_summary": {
                "enabled": True,
                "search": "grid",
                "n_petri_candidates": int(len(petri_candidates)),
                "tree_depth_candidates": list(tree_depth_candidates),
                "n_grid_candidates": int(len(petri_candidates) * len(tree_depth_candidates)),
                "n_valid_records": int(len(valid_records)),
                "validation_budgets": list(budgets),
                "final_selection_budget": int(final_budget),
                "top_k_structures": int(top_k_structures),
                "selection_score": "structure_selection_score",
                "activity_coverage_penalty_weight": float(ACTIVITY_COVERAGE_PENALTY_WEIGHT),
                "noise_penalty_weight": float(NOISE_PENALTY_WEIGHT),
                "low_noise_preference_tolerance": float(LOW_NOISE_PREFERENCE_TOL),
            },
        }

    @staticmethod
    def _scalar_structure_objective(metrics: dict, metric_names: list[str]) -> float:
        values = []
        missing_penalty = 0.0
        for metric_name in metric_names:
            value = metrics.get(metric_name)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                values.append(float(np.log1p(max(0.0, float(value)))))
            else:
                missing_penalty += 5.0

        if values:
            return float(sum(values) / len(values) + missing_penalty)
        return float(100.0 + missing_penalty)

    # ------------------------------------------------------------------
    # Stage 1 稳定最终选择辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_log_activity_set(event_log) -> set[str]:
        """从 EventLog 或 DataFrame 中提取活动全集。"""
        activities: set[str] = set()

        try:
            if hasattr(event_log, "columns") and "concept:name" in event_log.columns:
                for value in event_log["concept:name"].dropna().astype(str).tolist():
                    if value:
                        activities.add(value)
                return activities
        except Exception:
            pass

        try:
            for trace in event_log:
                try:
                    for event in trace:
                        value = None
                        try:
                            value = event.get("concept:name")
                        except Exception:
                            value = event["concept:name"] if "concept:name" in event else None
                        if value is not None:
                            value = str(value)
                            if value:
                                activities.add(value)
                except Exception:
                    continue
        except Exception:
            pass

        return activities

    def _extract_current_process_activity_set(self) -> set[str]:
        process_model = getattr(self.integration, "process_model", None) or {}
        activities = process_model.get("activities", []) if isinstance(process_model, dict) else []
        return {str(a) for a in activities if a is not None and str(a)}

    def _compute_current_process_activity_coverage(self, reference_activities: set[str]) -> dict:
        discovered = self._extract_current_process_activity_set()
        reference = {str(a) for a in (reference_activities or set()) if a is not None and str(a)}

        if not reference:
            coverage = 1.0
            missing = []
        else:
            missing = sorted(reference.difference(discovered))
            coverage = (len(reference) - len(missing)) / max(1, len(reference))

        extra = sorted(discovered.difference(reference)) if reference else []
        return {
            "coverage": float(coverage),
            "reference_activity_count": int(len(reference)),
            "discovered_activity_count": int(len(discovered)),
            "missing_activity_count": int(len(missing)),
            "extra_activity_count": int(len(extra)),
            "missing_activities": missing[:20],
            "extra_activities": extra[:20],
        }

    @staticmethod
    def _record_identity(row: dict) -> tuple:
        noise = row.get("noise_threshold")
        try:
            noise_key = round(float(noise), 12) if noise is not None else None
        except Exception:
            noise_key = str(noise)
        try:
            depth_key = int(row.get("tree_depth", row.get("max_depth_tree", 0)))
        except Exception:
            depth_key = row.get("tree_depth", row.get("max_depth_tree"))
        return noise_key, depth_key

    @staticmethod
    def _noise_value(row: dict) -> float | None:
        noise = row.get("noise_threshold")
        if noise is None:
            return None
        try:
            noise = float(noise)
            if np.isfinite(noise):
                return max(0.0, noise)
        except Exception:
            pass
        return None

    def _attach_structure_selection_scores(
            self,
            records: list[dict],
            base_score_key: str,
            selection_score_key: str,
    ):
        """
        添加保守的 Stage-1 选择分数。

        基础分数仍然是验证集控制流距离。我们只加入少量柔和惩罚，
        因此当非零噪声模型明显更优时仍能胜出，而在平局/接近平局时
        优先选择活动覆盖完整且噪声更低的结构。
        """
        for row in records or []:
            base = row.get(base_score_key, row.get("optuna_objective_score", float("inf")))
            try:
                base = float(base)
                if not np.isfinite(base):
                    base = float("inf")
            except Exception:
                base = float("inf")

            coverage_info = row.get("activity_coverage") or {}
            coverage = coverage_info.get("coverage")
            try:
                coverage = float(coverage)
                if not np.isfinite(coverage):
                    coverage = 1.0
            except Exception:
                coverage = 1.0
            coverage = float(np.clip(coverage, 0.0, 1.0))
            missing_ratio = max(0.0, 1.0 - coverage)
            coverage_penalty = float(ACTIVITY_COVERAGE_PENALTY_WEIGHT) * missing_ratio

            noise = self._noise_value(row)
            if noise is None:
                noise_penalty = 0.0
            else:
                noise_penalty = float(NOISE_PENALTY_WEIGHT) * noise

            row["activity_coverage_penalty"] = float(coverage_penalty)
            row["noise_penalty"] = float(noise_penalty)
            row[selection_score_key] = float(base + coverage_penalty + noise_penalty)

    @staticmethod
    def _activity_coverage_value(row: dict) -> float:
        try:
            cov = float((row.get("activity_coverage") or {}).get("coverage", 1.0))
            if np.isfinite(cov):
                return float(np.clip(cov, 0.0, 1.0))
        except Exception:
            pass
        return 1.0

    def _prefer_complete_activity_coverage_pool(
            self,
            primary_records: list[dict],
            fallback_records: list[dict],
            score_key: str,
    ) -> tuple[list[dict], list[dict], dict]:
        """只要存在完整活动覆盖的结构，就优先选择它们。

        这是一个通用保护规则，不是针对 P2P 的特例：如果挖掘过程能够保留训练日志中
        观察到的全部活动宇宙，Stage 1 就不应选择控制流得分略好但删除了某个有效低频活动的结构。
        如果不存在完整覆盖候选，则回退到观察到的最大覆盖率。
        """
        all_records = [
            row for row in list(primary_records or []) + list(fallback_records or [])
            if isinstance(row, dict) and row.get("status") == "ok"
        ]
        if not all_records:
            return list(primary_records or []), list(fallback_records or []), {"enabled": False, "reason": "no_ok_records"}

        max_cov = max(self._activity_coverage_value(row) for row in all_records)
        if max_cov >= 0.999:
            threshold = 0.999
            mode = "full_activity_coverage"
        else:
            threshold = max_cov - 1e-12
            mode = "max_available_activity_coverage"

        primary_filtered = [
            row for row in (primary_records or [])
            if row.get("status") == "ok" and self._activity_coverage_value(row) >= threshold
        ]
        fallback_filtered = [
            row for row in (fallback_records or [])
            if row.get("status") == "ok" and self._activity_coverage_value(row) >= threshold
        ]

        if not primary_filtered and not fallback_filtered:
            return list(primary_records or []), list(fallback_records or []), {
                "enabled": False,
                "reason": "filter_empty",
                "max_coverage": float(max_cov),
            }

        logger.info(
            "Stage 1 activity-coverage preference applied: mode=%s, max_coverage=%.6f, "
            "primary %s -> %s, fallback %s -> %s",
            mode,
            float(max_cov),
            len(primary_records or []),
            len(primary_filtered),
            len(fallback_records or []),
            len(fallback_filtered),
        )
        return primary_filtered or fallback_filtered, fallback_filtered or primary_filtered, {
            "enabled": True,
            "mode": mode,
            "max_coverage": float(max_cov),
            "threshold": float(threshold),
            "n_primary_before": int(len(primary_records or [])),
            "n_primary_after": int(len(primary_filtered)),
            "n_fallback_before": int(len(fallback_records or [])),
            "n_fallback_after": int(len(fallback_filtered)),
        }

    def _select_structure_records_with_low_noise_preference(
            self,
            primary_records: list[dict],
            fallback_records: list[dict],
            top_k: int,
            score_key: str,
    ) -> list[dict]:
        primary_records, fallback_records, coverage_pref = self._prefer_complete_activity_coverage_pool(
            primary_records=primary_records,
            fallback_records=fallback_records,
            score_key=score_key,
        )
        selected = self._select_top_records(
            primary_records=primary_records,
            fallback_records=fallback_records,
            top_k=top_k,
            score_key=score_key,
        )
        for row in selected:
            try:
                row["activity_coverage_preference"] = dict(coverage_pref)
            except Exception:
                pass
        selected_by_key = {self._record_identity(row): row for row in selected}

        pool_by_key = {}
        for row in list(primary_records or []) + list(fallback_records or []):
            if row.get("status") != "ok":
                continue
            key = self._record_identity(row)
            old = pool_by_key.get(key)
            if old is None or float(row.get(score_key, float("inf"))) < float(old.get(score_key, float("inf"))):
                pool_by_key[key] = row
        pool = list(pool_by_key.values())

        if not pool or not selected:
            return selected

        if LOW_NOISE_PREFERENCE_ENABLED:
            numeric_noise_pool = [row for row in pool if self._noise_value(row) is not None]
            if numeric_noise_pool:
                min_noise = min(self._noise_value(row) for row in numeric_noise_pool)
                low_noise_pool = [row for row in numeric_noise_pool if abs(self._noise_value(row) - min_noise) <= 1e-12]
                low_noise_best = min(
                    low_noise_pool,
                    key=lambda row: (
                        float(row.get(score_key, float("inf"))),
                        abs(int(row.get("tree_depth", row.get("max_depth_tree", 2))) - 2),
                        int(row.get("tree_depth", row.get("max_depth_tree", 2))),
                    ),
                )
                best_selected = selected[0]
                best_score = float(best_selected.get(score_key, float("inf")))
                low_score = float(low_noise_best.get(score_key, float("inf")))
                tolerance = float(LOW_NOISE_PREFERENCE_TOL)
                if np.isfinite(low_score) and np.isfinite(best_score) and low_score <= best_score * (1.0 + tolerance):
                    low_key = self._record_identity(low_noise_best)
                    if low_key not in selected_by_key:
                        if len(selected) >= max(1, int(top_k)):
                            selected[-1] = low_noise_best
                        else:
                            selected.append(low_noise_best)
                    low_noise_best["selected_by_low_noise_preference"] = True
                    logger.info(
                        "Stage 1 low-noise preference applied: low_noise=%s, low_score=%.6f, "
                        "best_score=%.6f, tolerance=%.3f",
                        low_noise_best.get("noise_threshold"),
                        low_score,
                        best_score,
                        tolerance,
                    )

        # 根据保守选择分数去重并再次排序。
        unique = {}
        for row in selected:
            key = self._record_identity(row)
            old = unique.get(key)
            if old is None or float(row.get(score_key, float("inf"))) < float(old.get(score_key, float("inf"))):
                unique[key] = row

        selected = list(unique.values())
        selected.sort(
            key=lambda row: (
                float(row.get(score_key, float("inf"))),
                float(row.get("optuna_objective_score", float("inf"))),
                self._noise_value(row) if self._noise_value(row) is not None else 0.0,
                abs(int(row.get("tree_depth", row.get("max_depth_tree", 2))) - 2),
                int(row.get("tree_depth", row.get("max_depth_tree", 2))),
            )
        )
        return selected[:max(1, int(top_k))]

    def _confirm_top_structure_candidates(
            self,
            candidate_records: list[dict],
            train_log,
            validation_log,
            pnml_filepath,
            structure_metrics: tuple[str, ...],
            top_k_structures: int,
    ) -> dict:
        confirmation_runs = max(1, int(STRUCTURE_CONFIRMATION_RUNS))
        confirmation_top_k = max(1, int(STRUCTURE_CONFIRMATION_TOP_K))
        candidate_records = list(candidate_records or [])[:confirmation_top_k]

        if not candidate_records:
            return {
                "enabled": bool(STRUCTURE_CONFIRMATION_ENABLED),
                "top_structures": [],
                "confirmation_records": [],
                "summary": {
                    "enabled": bool(STRUCTURE_CONFIRMATION_ENABLED),
                    "reason": "no_candidates",
                },
            }

        reference_activities = self._extract_log_activity_set(train_log)
        confirmation_records: list[dict] = []

        logger.info(
            "Stage 1C confirmation started: candidates=%s, runs_per_candidate=%s, metrics=%s",
            [
                (row.get("noise_threshold"), row.get("tree_depth"), row.get("structure_selection_score"))
                for row in candidate_records
            ],
            confirmation_runs,
            list(structure_metrics),
        )

        for idx, candidate in enumerate(candidate_records):
            noise_threshold = candidate.get("noise_threshold")
            tree_depth = int(candidate.get("tree_depth", candidate.get("max_depth_tree", 3)))
            try:
                self._discover_candidate_model(
                    event_log=train_log,
                    pnml_filepath=pnml_filepath,
                    noise_threshold=noise_threshold,
                    max_depth_tree=tree_depth,
                    decision_point_max_depth=tree_depth,
                    discovery_profile=CONTROL_FLOW_DISCOVERY_PROFILE,
                    build_process_artifacts=False,
                    precomputed_df_features=None,
                    reuse_current_process_model=False,
                    alignment_timeout_seconds=ALIGNMENT_TIMEOUT_SECONDS,
                )
                activity_coverage = self._compute_current_process_activity_coverage(reference_activities)
                validation_metrics = self._evaluate(
                    event_log=validation_log,
                    n_runs=confirmation_runs,
                    metrics_labels=list(structure_metrics),
                    enable_time_diagnostics=False,
                )
                flat_metrics = self.integration._flatten_formatted_metrics(validation_metrics)
                raw_score = self._scalar_structure_objective(
                    flat_metrics,
                    metric_names=list(structure_metrics),
                )
                record = copy.deepcopy(candidate)
                record.update({
                    "stage": "structure_confirmation",
                    "confirmation_index": int(idx),
                    "confirmed_from_stage": candidate.get("stage"),
                    "resource_budget": int(confirmation_runs),
                    "confirmation_runs": int(confirmation_runs),
                    "noise_threshold": noise_threshold,
                    "tree_depth": int(tree_depth),
                    "max_depth_tree": int(tree_depth),
                    "decision_point_max_depth": int(tree_depth),
                    "metrics": flat_metrics,
                    "activity_coverage": activity_coverage,
                    "pre_confirmation_structure_score": candidate.get("structure_score"),
                    "pre_confirmation_selection_score": candidate.get("structure_selection_score"),
                    "optuna_objective_score": float(raw_score),
                    "status": "ok",
                })
                confirmation_records.append(record)
            except Exception as exc:
                logger.warning(
                    "Stage 1C confirmation failed: noise_threshold=%s, tree_depth=%s, error=%s",
                    noise_threshold,
                    tree_depth,
                    exc,
                )
                failed = copy.deepcopy(candidate)
                failed.update({
                    "stage": "structure_confirmation",
                    "confirmation_index": int(idx),
                    "resource_budget": int(confirmation_runs),
                    "confirmation_runs": int(confirmation_runs),
                    "status": "failed",
                    "error": str(exc),
                })
                confirmation_records.append(failed)

        valid_records = [
            row for row in confirmation_records
            if row.get("status") == "ok"
            and any(self._get_numeric_metric(row, m) is not None for m in structure_metrics)
        ]

        if not valid_records:
            logger.warning("Stage 1C confirmation produced no valid records; using pre-confirmation candidates.")
            return {
                "enabled": True,
                "top_structures": candidate_records[:max(1, int(top_k_structures))],
                "confirmation_records": confirmation_records,
                "summary": {
                    "enabled": True,
                    "status": "failed_no_valid_records",
                    "confirmation_runs": int(confirmation_runs),
                    "n_candidates": int(len(candidate_records)),
                },
            }

        self._attach_ideal_point_scores(
            valid_records,
            metric_names=list(structure_metrics),
            metrics_key="metrics",
            score_key="confirmation_score",
        )
        self._attach_structure_selection_scores(
            valid_records,
            base_score_key="confirmation_score",
            selection_score_key="confirmation_selection_score",
        )
        pareto_records = self._pareto_prune_records(
            valid_records,
            metric_names=list(structure_metrics),
        )
        top_structures = self._select_structure_records_with_low_noise_preference(
            primary_records=pareto_records,
            fallback_records=valid_records,
            top_k=top_k_structures,
            score_key="confirmation_selection_score",
        )

        logger.info(
            "Stage 1C confirmation completed. Selected structures: %s",
            [
                (
                    row.get("noise_threshold"),
                    row.get("tree_depth"),
                    row.get("confirmation_score"),
                    row.get("confirmation_selection_score"),
                    (row.get("activity_coverage") or {}).get("coverage"),
                )
                for row in top_structures
            ],
        )

        return {
            "enabled": True,
            "top_structures": top_structures,
            "confirmation_records": confirmation_records,
            "summary": {
                "enabled": True,
                "status": "ok",
                "confirmation_runs": int(confirmation_runs),
                "n_candidates": int(len(candidate_records)),
                "n_valid_records": int(len(valid_records)),
                "metrics": list(structure_metrics),
                "selection_score": "confirmation_selection_score",
                "activity_coverage_penalty_weight": float(ACTIVITY_COVERAGE_PENALTY_WEIGHT),
                "noise_penalty_weight": float(NOISE_PENALTY_WEIGHT),
                "low_noise_preference_tolerance": float(LOW_NOISE_PREFERENCE_TOL),
                "selected": [
                    {
                        "noise_threshold": row.get("noise_threshold"),
                        "tree_depth": row.get("tree_depth"),
                        "confirmation_score": row.get("confirmation_score"),
                        "confirmation_selection_score": row.get("confirmation_selection_score"),
                        "activity_coverage": row.get("activity_coverage"),
                    }
                    for row in top_structures
                ],
            },
        }

    # ------------------------------------------------------------------
    # 运行时 GridSearchCV 辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_depth_candidates(
            candidates,
            default=TREE_DEPTH_CANDIDATES,
            min_depth: int = 1,
            max_depth: int = 4,
    ) -> list[int]:
        """为确定性 GridSearchCV 归一化决策树深度候选。"""
        try:
            values = [int(x) for x in candidates]
        except Exception:
            values = [int(x) for x in default]

        clean = []
        for value in values:
            value = int(np.clip(int(value), int(min_depth), int(max_depth)))
            clean.append(value)

        clean = sorted(set(clean))
        clean = [value for value in clean if int(min_depth) <= value <= int(max_depth)]

        # 尊重调用方显式传入的候选子集；只有输入无效/为空时才回退默认网格。
        if not clean:
            clean = sorted(set(int(np.clip(x, min_depth, max_depth)) for x in default))
        return clean

    @staticmethod
    def _normalize_quantile_candidates(candidates) -> list[float]:
        """归一化 External Delay runtime tail quantile 候选。"""
        try:
            values = [float(x) for x in candidates]
        except Exception:
            values = [float(x) for x in EXTERNAL_DELAY_QUANTILE_CANDIDATES]

        clean = []
        for value in values:
            if not np.isfinite(value):
                continue
            value = float(np.clip(value, 0.0, 1.0))
            if value <= 0.0:
                continue
            clean.append(value)

        # 尊重调用方显式传入的 quantile 子集；不再强制把默认候选补回。
        clean = sorted(set(clean))
        return clean or [1.0]

    @staticmethod
    def _count_log_cases(event_log) -> int:
        """Count complete cases without assuming a specific log representation."""
        try:
            if hasattr(event_log, "columns"):
                for key in ("case:concept:name", "case_id", "caseid"):
                    if key in event_log.columns:
                        return int(event_log[key].astype(str).nunique())
        except Exception:
            pass
        try:
            return int(len(event_log))
        except Exception:
            return 0

    def _take_temporal_case_window(self, event_log, target_cases: int, side: str = "head"):
        """Return a complete-case temporal window while preserving original timestamps.

        ``tail`` is used for the training sample and ``head`` for validation so the
        two sampled windows remain temporally adjacent to the original 80/20 split.
        This is deliberately not random case sampling: random cases scattered across
        the full year would distort inter-arrival times and queue pressure.
        """
        target_cases = max(1, int(target_cases))
        try:
            from drqsim.utils.log_partitioner import (
                _to_dataframe, _to_event_log, _get_ordered_case_ids, _filter_by_case_ids,
            )
            import pandas as _pd

            case_col = "case:concept:name"
            start_col = "start:timestamp"
            end_col = "time:timestamp"
            df = _to_dataframe(event_log).copy()
            if case_col not in df.columns:
                raise KeyError(f"missing case id column: {case_col}")
            if start_col not in df.columns:
                if end_col not in df.columns:
                    raise KeyError("missing start/end timestamp columns")
                df[start_col] = df[end_col]
            df[case_col] = df[case_col].astype(str)
            df[start_col] = _pd.to_datetime(df[start_col], utc=True, errors="coerce")
            df[end_col] = _pd.to_datetime(df[end_col], utc=True, errors="coerce")
            df = df.dropna(subset=[case_col, start_col, end_col]).copy()

            ordered_case_ids = list(_get_ordered_case_ids(
                df=df,
                case_id_col=case_col,
                start_time_col=start_col,
                end_time_col=end_col,
                sort_temporally=True,
            ))
            n_cases = len(ordered_case_ids)
            if n_cases <= target_cases:
                return event_log

            side = str(side or "head").lower()
            if side == "tail":
                selected_ids = ordered_case_ids[-target_cases:]
            elif side == "center":
                start_idx = max(0, (n_cases - target_cases) // 2)
                selected_ids = ordered_case_ids[start_idx:start_idx + target_cases]
            else:
                selected_ids = ordered_case_ids[:target_cases]

            sampled_df = _filter_by_case_ids(
                df, selected_ids, case_col, start_col, end_col,
            )
            return _to_event_log(sampled_df, case_col)
        except Exception as exc:
            logger.warning(
                "Temporal case-window sampling failed; using full log instead: %s", exc
            )
            return event_log

    def _build_runtime_recency_log(self, event_log, ratio: float):
        """Return the most recent complete-case window for runtime-only discovery."""
        try:
            ratio = float(np.clip(float(ratio), 0.0, 1.0))
        except Exception:
            ratio = 1.0
        n_cases = self._count_log_cases(event_log)
        if n_cases <= 0 or ratio >= 1.0 - 1e-12:
            return event_log, int(n_cases)

        target_cases = int(np.ceil(float(n_cases) * float(ratio)))
        if target_cases < int(RUNTIME_RECENCY_MIN_CASES):
            return None, int(target_cases)

        recent_log = self._take_temporal_case_window(
            event_log,
            target_cases=target_cases,
            side="tail",
        )
        return recent_log, int(self._count_log_cases(recent_log))

    @staticmethod
    def _extract_case_arrival_timestamps(event_log) -> list:
        """Extract one case-arrival timestamp per trace/case without simulation."""
        timestamps = []

        # pandas/DataFrame-like logs.
        try:
            if hasattr(event_log, "columns") and hasattr(event_log, "groupby"):
                import pandas as pd

                columns = set(str(c) for c in event_log.columns)
                case_col = "case:concept:name" if "case:concept:name" in columns else None
                time_col = (
                    "start:timestamp" if "start:timestamp" in columns
                    else ("time:timestamp" if "time:timestamp" in columns else None)
                )
                if case_col is not None and time_col is not None:
                    frame = event_log[[case_col, time_col]].copy()
                    frame[time_col] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
                    first = frame.dropna(subset=[time_col]).groupby(case_col, sort=False)[time_col].min()
                    timestamps = [x for x in first.tolist() if x is not None]
        except Exception:
            timestamps = []

        # pm4py EventLog / iterable traces.
        if not timestamps:
            try:
                for trace in event_log:
                    if trace is None:
                        continue
                    case_times = []
                    try:
                        iterator = iter(trace)
                    except Exception:
                        continue
                    for event in iterator:
                        if not isinstance(event, dict):
                            try:
                                event = dict(event)
                            except Exception:
                                continue
                        ts = event.get("start:timestamp", event.get("time:timestamp"))
                        if ts is not None:
                            case_times.append(ts)
                    if case_times:
                        try:
                            timestamps.append(min(case_times))
                        except Exception:
                            timestamps.append(case_times[0])
            except Exception:
                timestamps = []

        def _sort_key(value):
            try:
                return value.timestamp()
            except Exception:
                try:
                    return np.datetime64(value).astype("datetime64[ns]").astype(np.int64)
                except Exception:
                    return str(value)

        try:
            return sorted(timestamps, key=_sort_key)
        except Exception:
            return list(timestamps)

    @classmethod
    def _arrival_pattern_stats(cls, event_log) -> dict:
        """Cheap real-log arrival statistics used only by the drift trigger/prefilter."""
        timestamps = cls._extract_case_arrival_timestamps(event_log)
        if len(timestamps) < 2:
            return {
                "available": False,
                "n_cases": int(len(timestamps)),
                "n_inter_arrivals": 0,
            }

        deltas = []
        for left, right in zip(timestamps[:-1], timestamps[1:]):
            try:
                delta = (right - left).total_seconds() / 60.0
            except Exception:
                try:
                    delta = (
                        np.datetime64(right).astype("datetime64[ns]").astype(np.int64)
                        - np.datetime64(left).astype("datetime64[ns]").astype(np.int64)
                    ) / 60_000_000_000.0
                except Exception:
                    continue
            if np.isfinite(delta) and delta >= 0.0:
                deltas.append(float(delta))

        if not deltas:
            return {
                "available": False,
                "n_cases": int(len(timestamps)),
                "n_inter_arrivals": 0,
            }

        slot_counts = {}
        for ts in timestamps:
            try:
                slot = (int(ts.weekday()), int(ts.hour))
            except Exception:
                try:
                    import pandas as pd
                    converted = pd.Timestamp(ts)
                    slot = (int(converted.weekday()), int(converted.hour))
                except Exception:
                    continue
            slot_counts[slot] = int(slot_counts.get(slot, 0)) + 1

        total_slots = float(sum(slot_counts.values()) or 1.0)
        slot_prob = {slot: float(count) / total_slots for slot, count in slot_counts.items()}
        values = np.asarray(deltas, dtype=float)
        return {
            "available": True,
            "n_cases": int(len(timestamps)),
            "n_inter_arrivals": int(len(values)),
            "mean_min": float(np.mean(values)),
            "median_min": float(np.median(values)),
            "p90_min": float(np.percentile(values, 90)),
            "active_weekday_hour_slots": sorted([f"{wd}:{hour}" for wd, hour in slot_counts]),
            "weekday_hour_prob": {f"{wd}:{hour}": prob for (wd, hour), prob in slot_prob.items()},
        }

    @staticmethod
    def _arrival_stats_distance(reference: dict, target: dict) -> dict:
        """Compare two real-log arrival patterns; lower composite distance is better."""
        out = {
            "available": False,
            "mean_shift": None,
            "p90_shift": None,
            "weekday_hour_jaccard": None,
            "weekday_hour_tv": None,
            "composite_distance": float("inf"),
        }
        if not bool((reference or {}).get("available")) or not bool((target or {}).get("available")):
            return out

        def _relative_shift(a, b):
            try:
                a = float(a)
                b = float(b)
                if not (np.isfinite(a) and np.isfinite(b)) or abs(a) <= 1e-12:
                    return None
                return abs(b / a - 1.0)
            except Exception:
                return None

        mean_shift = _relative_shift(reference.get("mean_min"), target.get("mean_min"))
        p90_shift = _relative_shift(reference.get("p90_min"), target.get("p90_min"))

        ref_slots = set(reference.get("active_weekday_hour_slots") or [])
        target_slots = set(target.get("active_weekday_hour_slots") or [])
        union = ref_slots | target_slots
        jaccard = float(len(ref_slots & target_slots) / len(union)) if union else 1.0

        ref_prob = dict(reference.get("weekday_hour_prob") or {})
        target_prob = dict(target.get("weekday_hour_prob") or {})
        keys = set(ref_prob) | set(target_prob)
        tv = 0.5 * sum(abs(float(ref_prob.get(k, 0.0)) - float(target_prob.get(k, 0.0))) for k in keys)

        # Log-ratio distance is symmetric for faster/slower arrival rates.
        def _log_ratio(a, b):
            try:
                a = max(float(a), 1e-12)
                b = max(float(b), 1e-12)
                return abs(float(np.log(b / a)))
            except Exception:
                return 10.0

        composite = (
            _log_ratio(reference.get("mean_min"), target.get("mean_min"))
            + 0.5 * _log_ratio(reference.get("p90_min"), target.get("p90_min"))
            + 0.5 * float(tv)
        )
        out.update({
            "available": True,
            "mean_shift": None if mean_shift is None else float(mean_shift),
            "p90_shift": None if p90_shift is None else float(p90_shift),
            "weekday_hour_jaccard": float(jaccard),
            "weekday_hour_tv": float(tv),
            "composite_distance": float(composite),
        })
        return out

    def _arrival_recency_drift_trigger(self, structure_train_log, validation_log) -> dict:
        """Zero-simulation trigger. Conservative by design to preserve old behavior."""
        train_stats = self._arrival_pattern_stats(structure_train_log)
        validation_stats = self._arrival_pattern_stats(validation_log)
        distance = self._arrival_stats_distance(train_stats, validation_stats)

        result = {
            "enabled": bool(ARRIVAL_RECENCY_ENABLED),
            "uses_test_log": False,
            "triggered": False,
            "train_stats": train_stats,
            "validation_stats": validation_stats,
            "distance": distance,
            "moderate_signals": {},
            "strong_signals": {},
            "reason": "not_triggered",
        }
        if not bool(ARRIVAL_RECENCY_ENABLED):
            result["reason"] = "disabled_by_config"
            return result
        train_cases = int(train_stats.get("n_cases", 0) or 0)
        validation_cases = int(validation_stats.get("n_cases", 0) or 0)
        if (
            train_cases < int(ARRIVAL_RECENCY_TRIGGER_MIN_TRAIN_CASES)
            or validation_cases < int(ARRIVAL_RECENCY_TRIGGER_MIN_VALIDATION_CASES)
        ):
            result["reason"] = "insufficient_cases_for_arrival_recency_trigger"
            result["min_train_cases"] = int(ARRIVAL_RECENCY_TRIGGER_MIN_TRAIN_CASES)
            result["min_validation_cases"] = int(ARRIVAL_RECENCY_TRIGGER_MIN_VALIDATION_CASES)
            return result
        if not bool(distance.get("available")):
            result["reason"] = "insufficient_arrival_statistics"
            return result

        mean_shift = distance.get("mean_shift")
        p90_shift = distance.get("p90_shift")
        jaccard = distance.get("weekday_hour_jaccard")
        moderate = {
            "mean_shift": bool(mean_shift is not None and float(mean_shift) >= float(ARRIVAL_RECENCY_TRIGGER_MEAN_SHIFT)),
            "p90_shift": bool(p90_shift is not None and float(p90_shift) >= float(ARRIVAL_RECENCY_TRIGGER_P90_SHIFT)),
            "weekday_hour_jaccard": bool(jaccard is not None and float(jaccard) <= float(ARRIVAL_RECENCY_TRIGGER_WEEKDAY_HOUR_JACCARD)),
        }
        strong = {
            "mean_shift": bool(mean_shift is not None and float(mean_shift) >= float(ARRIVAL_RECENCY_TRIGGER_STRONG_MEAN_SHIFT)),
            "p90_shift": bool(p90_shift is not None and float(p90_shift) >= float(ARRIVAL_RECENCY_TRIGGER_STRONG_P90_SHIFT)),
        }
        moderate_count = int(sum(1 for value in moderate.values() if value))
        # Weekday-hour support can be noisy when validation is shorter than train;
        # it may contribute a moderate signal but never triggers recency by itself.
        strong_any = bool(any(strong.values()))
        triggered = bool(strong_any or moderate_count >= int(ARRIVAL_RECENCY_TRIGGER_MIN_SIGNALS))
        result.update({
            "triggered": triggered,
            "moderate_signals": moderate,
            "strong_signals": strong,
            "moderate_signal_count": moderate_count,
            "strong_signal_any": strong_any,
            "reason": "drift_triggered" if triggered else "drift_below_trigger_threshold",
        })
        logger.info(
            "[Arrival Recency Trigger] triggered=%s, mean_shift=%s, p90_shift=%s, "
            "weekday_hour_jaccard=%s, moderate_signals=%s, strong_signals=%s",
            triggered,
            mean_shift,
            p90_shift,
            jaccard,
            moderate,
            strong,
        )
        return result

    def _build_arrival_recency_log(self, event_log, ratio: float):
        try:
            ratio = float(np.clip(float(ratio), 0.0, 1.0))
        except Exception:
            ratio = 1.0
        n_cases = self._count_log_cases(event_log)
        if n_cases <= 0 or ratio >= 1.0 - 1e-12:
            return event_log, int(n_cases)
        target_cases = int(np.ceil(float(n_cases) * float(ratio)))
        if target_cases < int(ARRIVAL_RECENCY_MIN_CASES):
            return None, int(target_cases)
        recent_log = self._take_temporal_case_window(
            event_log,
            target_cases=target_cases,
            side="tail",
        )
        return recent_log, int(self._count_log_cases(recent_log))

    def _discover_arrival_params_only(self, arrival_log, selected_hyperparameters: dict) -> dict:
        """Discover only the Arrival calendar/tree from a supplied case window."""
        from drqsim.discovery.base_service_time_discovery import serialize_distribution_container
        from drqsim.discovery.calendar_discovery import discover_arrival_calendar
        from drqsim.discovery.time_discovery import discover_arrival_time
        from drqsim.utils.save_and_load_utils import convert_calendar_names

        depth = max(1, int(selected_hyperparameters.get("arrival_decision_tree_max_depth", 1)))
        calendar = discover_arrival_calendar(arrival_log)
        distribution = discover_arrival_time(
            log=arrival_log,
            calendar_arrival=calendar,
            max_depths=[int(depth)],
        )
        return {
            "arrival_calendar": convert_calendar_names(calendar),
            "arrival_time_distributions": serialize_distribution_container(distribution),
            "arrival_time_scale": 1.0,
            "arrival_demand_multiplier": 1.0,
            "arrival_time_shift_min": 0.0,
            "arrival_decision_tree_max_depth": int(depth),
            "case_arrival_max_depth": int(depth),
        }

    def _overlay_arrival_params(self, arrival_params: dict) -> None:
        """Replace only Arrival parameters; all other discovered modules remain untouched."""
        if not isinstance(arrival_params, dict) or not isinstance(self.integration.drq_params, dict):
            return
        current = self.integration.drq_params.setdefault("arrival_params", {})
        current.clear()
        current.update(copy.deepcopy(arrival_params))

        modular = self.integration.drq_params.get("modular_params", {})
        if isinstance(modular, dict):
            modules = modular.setdefault("modules", {})
            arrival_module = modules.setdefault("arrival", {})
            if isinstance(arrival_module, dict):
                arrival_module.update({
                    "model_type": "case_arrival_decision_tree_interarrival",
                    "arrival_calendar": copy.deepcopy(arrival_params.get("arrival_calendar", {})),
                    "inter_arrival_distribution": copy.deepcopy(arrival_params.get("arrival_time_distributions", {})),
                    "arrival_time_scale": float(arrival_params.get("arrival_time_scale", 1.0)),
                    "arrival_demand_multiplier": float(arrival_params.get("arrival_demand_multiplier", 1.0)),
                    "arrival_time_shift_min": float(arrival_params.get("arrival_time_shift_min", 0.0)),
                    "arrival_decision_tree_max_depth": int(arrival_params.get("arrival_decision_tree_max_depth", 1)),
                    "case_arrival_max_depth": int(arrival_params.get("case_arrival_max_depth", arrival_params.get("arrival_decision_tree_max_depth", 1))),
                    "recency_overlay": True,
                })

    def _apply_arrival_recency_overlay(
            self,
            source_log,
            ratio: float,
            selected_hyperparameters: dict,
    ) -> dict:
        """Overlay only the Arrival model learned from the selected recent window."""
        ratio = float(ratio)
        if ratio >= 1.0 - 1e-12:
            return {
                "applied": False,
                "ratio": 1.0,
                "cases": int(self._count_log_cases(source_log)),
                "reason": "full_history_identity",
            }
        arrival_log, arrival_cases = self._build_arrival_recency_log(source_log, ratio)
        if arrival_log is None:
            return {
                "applied": False,
                "ratio": 1.0,
                "cases": int(arrival_cases),
                "reason": "recent_window_too_small",
            }
        arrival_params = self._discover_arrival_params_only(arrival_log, selected_hyperparameters)
        self._overlay_arrival_params(arrival_params)
        return {
            "applied": True,
            "ratio": float(ratio),
            "cases": int(arrival_cases),
            "reason": "recent_arrival_overlay_applied",
        }

    def _select_arrival_recency(
            self,
            structure_train_log,
            validation_log,
            pnml_filepath,
            selected_hyperparameters: dict,
            runtime_recency_ratio: float,
            enable_candidate_resource_filter: bool | None,
            candidate_resource_coverage: float | None,
            candidate_resource_max_k: int | None,
            candidate_resource_min_k: int | None,
    ) -> dict:
        """Stage 2e0: trigger-gated, Arrival-only recent-window selection on validation."""
        trigger = self._arrival_recency_drift_trigger(structure_train_log, validation_log)
        result = {
            "enabled": bool(ARRIVAL_RECENCY_ENABLED),
            "uses_test_log": False,
            "trigger": trigger,
            "candidate_ratios": [float(x) for x in ARRIVAL_RECENCY_RATIO_CANDIDATES],
            "selected_ratio": 1.0,
            "selected_cases": int(self._count_log_cases(structure_train_log)),
            "eval_runs": int(ARRIVAL_RECENCY_EVAL_RUNS),
            "min_improvement": float(ARRIVAL_RECENCY_MIN_IMPROVEMENT),
            "tie_tolerance": float(ARRIVAL_RECENCY_TIE_TOLERANCE),
            "records": [],
            "reason": "full_history_default",
        }
        if not bool(ARRIVAL_RECENCY_ENABLED):
            result["reason"] = "disabled_by_config"
            return result
        if not bool(trigger.get("triggered", False)):
            result["reason"] = str(trigger.get("reason") or "drift_trigger_not_met")
            logger.info("[Arrival Recency] skipped: %s", result["reason"])
            return result

        n_train_cases = int(self._count_log_cases(structure_train_log))
        if n_train_cases < int(ARRIVAL_RECENCY_MIN_CASES):
            result["reason"] = "training_log_too_small"
            return result

        # Restore the Stage2d-selected service/external runtime model once.
        runtime_log, _runtime_cases = self._build_runtime_recency_log(
            structure_train_log,
            float(runtime_recency_ratio),
        )
        if runtime_log is None:
            runtime_log = structure_train_log
            runtime_recency_ratio = 1.0

        self._discover_candidate_model(
            event_log=structure_train_log,
            runtime_event_log=(runtime_log if float(runtime_recency_ratio) < 1.0 - 1e-12 else None),
            pnml_filepath=pnml_filepath,
            noise_threshold=selected_hyperparameters.get("noise_threshold"),
            max_depth_tree=int(selected_hyperparameters.get("tree_depth", 1)),
            decision_point_max_depth=int(selected_hyperparameters.get("decision_point_max_depth", 1)),
            arrival_decision_tree_max_depth=int(selected_hyperparameters.get("arrival_decision_tree_max_depth", 1)),
            queue_scheduling_max_depth=int(selected_hyperparameters.get("queue_scheduling_max_depth", 1)),
            conditional_prioritization_max_depth=int(selected_hyperparameters.get("conditional_prioritization_max_depth", 1)),
            queue_scheduling_force_empirical=bool(selected_hyperparameters.get("queue_scheduling_force_empirical", False)),
            discovery_profile=FULL_DISCOVERY_PROFILE,
            build_process_artifacts=False,
            enable_candidate_resource_filter=enable_candidate_resource_filter,
            candidate_resource_coverage=candidate_resource_coverage,
            candidate_resource_max_k=candidate_resource_max_k,
            candidate_resource_min_k=candidate_resource_min_k,
            alignment_timeout_seconds=ALIGNMENT_TIMEOUT_SECONDS,
        )
        self._apply_selected_runtime_parameters(selected_hyperparameters)
        self._set_arrival_runtime_config(1.0, 1.0)
        try:
            self.integration._set_external_delay_activity_scales({}, reset_others=True)
        except Exception:
            pass
        base_state = self._capture_model_state()

        validation_stats = self._arrival_pattern_stats(validation_log)
        raw_candidates = []
        ratios = sorted(
            {float(np.clip(float(x), 0.0, 1.0)) for x in ARRIVAL_RECENCY_RATIO_CANDIDATES} | {1.0},
            reverse=True,
        )
        for ratio in ratios:
            arrival_log, arrival_cases = self._build_arrival_recency_log(structure_train_log, ratio)
            if arrival_log is None:
                raw_candidates.append({
                    "ratio": float(ratio),
                    "cases": int(arrival_cases),
                    "status": "skipped_too_few_cases",
                    "raw_distance": float("inf"),
                })
                continue
            stats = self._arrival_pattern_stats(arrival_log)
            distance = self._arrival_stats_distance(stats, validation_stats)
            raw_candidates.append({
                "ratio": float(ratio),
                "cases": int(arrival_cases),
                "status": "raw_ok" if bool(distance.get("available")) else "raw_unavailable",
                "raw_distance": float(distance.get("composite_distance", float("inf"))),
                "raw_stats": stats,
                "raw_vs_validation": distance,
            })

        identity = next((row for row in raw_candidates if abs(float(row.get("ratio", 0.0)) - 1.0) <= 1e-12), None)
        recent_ranked = sorted(
            [row for row in raw_candidates if float(row.get("ratio", 1.0)) < 1.0 - 1e-12 and np.isfinite(float(row.get("raw_distance", float("inf"))))],
            key=lambda row: (float(row.get("raw_distance", float("inf"))), -float(row.get("ratio", 0.0))),
        )
        max_recent = max(0, int(ARRIVAL_RECENCY_MAX_SIM_CANDIDATES) - 1)
        simulation_rows = ([identity] if identity is not None else []) + recent_ranked[:max_recent]
        simulation_ratios = {float(row["ratio"]) for row in simulation_rows if row is not None}
        result["raw_prefilter_records"] = raw_candidates
        result["simulated_candidate_ratios"] = sorted(simulation_ratios, reverse=True)

        records = []
        for raw_row in raw_candidates:
            ratio = float(raw_row.get("ratio", 1.0))
            if ratio not in simulation_ratios:
                skipped = dict(raw_row)
                skipped["status"] = "prefiltered_no_simulation"
                records.append(skipped)
                continue
            try:
                self._restore_model_state(base_state, event_log=structure_train_log)
                overlay = self._apply_arrival_recency_overlay(
                    structure_train_log,
                    ratio,
                    selected_hyperparameters,
                )
                self._set_arrival_runtime_config(1.0, 1.0)
                metrics = self._evaluate(
                    event_log=validation_log,
                    n_runs=max(1, int(ARRIVAL_RECENCY_EVAL_RUNS)),
                    metrics_labels=list(ARRIVAL_STAGE_METRICS),
                    enable_time_diagnostics=True,
                )
                flat_metrics = self.integration._flatten_formatted_metrics(metrics)
                try:
                    car = float(flat_metrics.get("car", float("inf")))
                except Exception:
                    car = float("inf")
                arrival_guard = self._extract_arrival_guard()
                row = dict(raw_row)
                row.update({
                    "status": "ok" if np.isfinite(car) else "invalid_metric",
                    "car": float(car),
                    "metrics": flat_metrics,
                    "arrival_guard": arrival_guard,
                    "arrival_passed": bool(arrival_guard.get("passed", True)),
                    "overlay": overlay,
                })
                records.append(row)
                logger.info(
                    "[Arrival Recency] ratio=%.3f, cases=%s, CAR=%.6f, span=%s, guard_passed=%s, raw_distance=%s",
                    ratio,
                    raw_row.get("cases"),
                    car,
                    arrival_guard.get("arrival_span_ratio"),
                    bool(row["arrival_passed"]),
                    raw_row.get("raw_distance"),
                )
            except Exception as exc:
                failed = dict(raw_row)
                failed.update({
                    "status": "failed",
                    "car": float("inf"),
                    "arrival_passed": False,
                    "error": str(exc),
                })
                records.append(failed)
                logger.warning("Arrival recency candidate failed: ratio=%s, error=%s", ratio, exc)

        result["records"] = records
        valid = [
            row for row in records
            if row.get("status") == "ok" and np.isfinite(float(row.get("car", float("inf"))))
        ]
        safe = [row for row in valid if bool(row.get("arrival_passed", False))]
        baseline = next((row for row in valid if abs(float(row.get("ratio", 0.0)) - 1.0) <= 1e-12), None)

        def _ratio_closeness(row):
            try:
                return abs(float(np.log(max(float(row.get("ratio", 1.0)), 1e-12))))
            except Exception:
                return float("inf")

        selected = baseline
        pool = safe or valid
        if pool:
            best_car = min(float(row["car"]) for row in pool)
            tolerance = max(abs(best_car), 1e-9) * float(ARRIVAL_RECENCY_TIE_TOLERANCE)
            near_best = [row for row in pool if float(row["car"]) <= best_car + tolerance]
            best = min(near_best, key=lambda row: (_ratio_closeness(row), float(row["car"])))
            if baseline is None:
                selected = best
                result["reason"] = "identity_baseline_missing_best_candidate"
            elif not bool(baseline.get("arrival_passed", False)) and safe:
                selected = min(safe, key=lambda row: (float(row["car"]), _ratio_closeness(row)))
                result["reason"] = "identity_guard_failed_best_safe_recent"
            else:
                baseline_car = float(baseline["car"])
                required = baseline_car * (1.0 - float(ARRIVAL_RECENCY_MIN_IMPROVEMENT))
                result["baseline_car"] = baseline_car
                result["required_car_for_switch"] = float(required)
                result["best_candidate_car"] = float(best["car"])
                if float(best.get("ratio", 1.0)) < 1.0 - 1e-12 and float(best["car"]) <= float(required):
                    selected = best
                    result["reason"] = "triggered_recent_car_improvement_confirmed"
                else:
                    selected = baseline
                    result["reason"] = "triggered_but_identity_retained"
        else:
            result["reason"] = "triggered_no_valid_candidate_identity_fallback"

        selected_ratio = float((selected or {}).get("ratio", 1.0))
        result["selected_ratio"] = selected_ratio
        result["selected_cases"] = int((selected or {}).get("cases", n_train_cases))
        result["selected_car"] = None if selected is None else float(selected.get("car", float("inf")))

        # Leave integration in the selected Stage2d + Arrival state for consistent downstream diagnostics.
        self._restore_model_state(base_state, event_log=structure_train_log)
        if selected_ratio < 1.0 - 1e-12:
            self._apply_arrival_recency_overlay(structure_train_log, selected_ratio, selected_hyperparameters)
        self._set_arrival_runtime_config(1.0, 1.0)
        logger.info(
            "Triggered Arrival recency selected: ratio=%.3f, CAR=%s, reason=%s",
            selected_ratio,
            result.get("selected_car"),
            result.get("reason"),
        )
        return result

    def _select_runtime_recency(
            self,
            structure_train_log,
            validation_log,
            pnml_filepath,
            selected_hyperparameters: dict,
            runtime_metrics: tuple[str, ...],
            enable_candidate_resource_filter: bool | None,
            candidate_resource_coverage: float | None,
            candidate_resource_max_k: int | None,
            candidate_resource_min_k: int | None,
    ) -> dict:
        """Stage 2d: choose a recent runtime-discovery horizon by chronological validation.

        Ratio 1.0 is the mandatory baseline. A shorter window is accepted only if:
        1) its arrival guard passes and its duration ratio stays inside the wider
           hard band; the narrower soft band contributes a score penalty rather
           than immediate rejection; and
        2) when the baseline is itself safe, its score improves by at least
           RUNTIME_RECENCY_MIN_IMPROVEMENT.

        Test data are never used here.
        """
        result = {
            "enabled": bool(RUNTIME_RECENCY_ENABLED),
            "uses_test_log": False,
            "modules": list(RUNTIME_RECENCY_MODULES),
            "candidate_ratios": [float(x) for x in RUNTIME_RECENCY_RATIO_CANDIDATES],
            "selected_ratio": 1.0,
            "selected_cases": int(self._count_log_cases(structure_train_log)),
            "min_cases": int(RUNTIME_RECENCY_MIN_CASES),
            "eval_runs": int(RUNTIME_RECENCY_EVAL_RUNS),
            "min_improvement": float(RUNTIME_RECENCY_MIN_IMPROVEMENT),
            "tie_tolerance": float(RUNTIME_RECENCY_TIE_TOLERANCE),
            "duration_guard": {
                "soft_min_ratio": float(RUNTIME_RECENCY_DURATION_MIN_RATIO),
                "soft_max_ratio": float(RUNTIME_RECENCY_DURATION_MAX_RATIO),
                "hard_min_ratio": float(RUNTIME_RECENCY_DURATION_HARD_MIN_RATIO),
                "hard_max_ratio": float(RUNTIME_RECENCY_DURATION_HARD_MAX_RATIO),
            },
            "records": [],
            "reason": "full_history_default",
        }
        if not bool(RUNTIME_RECENCY_ENABLED):
            result["reason"] = "disabled_by_config"
            return result

        n_train_cases = int(self._count_log_cases(structure_train_log))
        if n_train_cases < int(RUNTIME_RECENCY_MIN_CASES):
            result["reason"] = "training_log_too_small"
            return result

        ratios = sorted(
            {float(np.clip(float(x), 0.0, 1.0)) for x in RUNTIME_RECENCY_RATIO_CANDIDATES} | {1.0},
            reverse=True,
        )
        records = []

        for ratio in ratios:
            runtime_log, runtime_cases = self._build_runtime_recency_log(
                structure_train_log, ratio
            )
            if runtime_log is None:
                records.append({
                    "status": "skipped_too_few_cases",
                    "ratio": float(ratio),
                    "target_cases": int(runtime_cases),
                })
                continue

            try:
                self._discover_candidate_model(
                    event_log=structure_train_log,
                    runtime_event_log=(runtime_log if float(ratio) < 1.0 - 1e-12 else None),
                    pnml_filepath=pnml_filepath,
                    noise_threshold=selected_hyperparameters.get("noise_threshold"),
                    max_depth_tree=int(selected_hyperparameters.get("tree_depth", 1)),
                    decision_point_max_depth=int(selected_hyperparameters.get("decision_point_max_depth", 1)),
                    arrival_decision_tree_max_depth=int(selected_hyperparameters.get("arrival_decision_tree_max_depth", 1)),
                    queue_scheduling_max_depth=int(selected_hyperparameters.get("queue_scheduling_max_depth", 1)),
                    conditional_prioritization_max_depth=int(selected_hyperparameters.get("conditional_prioritization_max_depth", 1)),
                    queue_scheduling_force_empirical=bool(
                        selected_hyperparameters.get("queue_scheduling_force_empirical", False)
                    ),
                    discovery_profile=FULL_DISCOVERY_PROFILE,
                    build_process_artifacts=False,
                    enable_candidate_resource_filter=enable_candidate_resource_filter,
                    candidate_resource_coverage=candidate_resource_coverage,
                    candidate_resource_max_k=candidate_resource_max_k,
                    candidate_resource_min_k=candidate_resource_min_k,
                    alignment_timeout_seconds=ALIGNMENT_TIMEOUT_SECONDS,
                )

                # Re-apply the Stage 2c runtime choices (especially External q).
                self._apply_selected_runtime_parameters(selected_hyperparameters)
                try:
                    self.integration._set_external_delay_activity_scales({}, reset_others=True)
                except Exception:
                    pass

                metrics = self._evaluate(
                    event_log=validation_log,
                    n_runs=max(1, int(RUNTIME_RECENCY_EVAL_RUNS)),
                    metrics_labels=list(runtime_metrics),
                    enable_time_diagnostics=True,
                )
                flat_metrics = self.integration._flatten_formatted_metrics(metrics)
                time_diagnostics = getattr(self.integration, "_last_time_diagnostics", {}) or {}
                duration_guard = self.integration._extract_case_duration_guard(time_diagnostics)
                external_delay_ratio = self._extract_external_delay_ratio(time_diagnostics)
                if external_delay_ratio is not None:
                    duration_guard["external_delay_ratio"] = float(external_delay_ratio)
                arrival_guard = self._extract_arrival_guard()
                repeated_arrivals = getattr(self.integration, "_repeated_arrival_diagnostics", []) or []
                if repeated_arrivals:
                    span_values = []
                    case_values = []
                    for diag in repeated_arrivals:
                        if not isinstance(diag, dict):
                            continue
                        try:
                            value = float(diag.get("span_ratio_sim_over_real"))
                            if np.isfinite(value):
                                span_values.append(value)
                        except Exception:
                            pass
                        try:
                            real_cases = float(diag.get("real_cases"))
                            sim_cases = float(diag.get("sim_cases"))
                            if np.isfinite(real_cases) and np.isfinite(sim_cases) and real_cases > 0:
                                case_values.append(sim_cases / real_cases)
                        except Exception:
                            pass
                    if span_values:
                        arrival_guard["arrival_span_ratio"] = float(np.mean(span_values))
                    if case_values:
                        arrival_guard["case_count_ratio"] = float(np.mean(case_values))
                    arrival_guard["passed"] = bool(
                        (
                            arrival_guard.get("arrival_span_ratio") is None
                            or float(ARRIVAL_SPAN_MIN_RATIO)
                            <= float(arrival_guard["arrival_span_ratio"])
                            <= float(ARRIVAL_SPAN_MAX_RATIO)
                        )
                        and (
                            arrival_guard.get("case_count_ratio") is None
                            or float(arrival_guard["case_count_ratio"]) >= float(ARRIVAL_CASE_COUNT_MIN_RATIO)
                        )
                    )

                score = self._scalar_runtime_objective(
                    metrics=flat_metrics,
                    duration_guard=duration_guard,
                    metric_names=list(runtime_metrics),
                    min_duration_ratio=float(RUNTIME_RECENCY_DURATION_MIN_RATIO),
                    max_duration_ratio=float(RUNTIME_RECENCY_DURATION_MAX_RATIO),
                    max_external_delay_ratio=None,
                )

                row = {
                    "status": "ok",
                    "ratio": float(ratio),
                    "target_cases": int(runtime_cases),
                    "actual_ratio": float(runtime_cases) / max(float(n_train_cases), 1.0),
                    "metrics": flat_metrics,
                    "case_duration_guard": duration_guard,
                    "arrival_guard": arrival_guard,
                    "score": float(score),
                }
                # Stage 2d uses a soft preferred band plus a wider hard rejection
                # band.  The scalar objective above already penalizes excursions
                # outside the soft band.  Only a hard-band violation removes a
                # candidate from recency selection.
                duration_ratio = duration_guard.get("duration_ratio")
                duration_soft_passed = False
                duration_hard_passed = False
                duration_guard_status = "invalid"
                try:
                    duration_ratio = float(duration_ratio)
                    if np.isfinite(duration_ratio):
                        duration_soft_passed = bool(
                            float(RUNTIME_RECENCY_DURATION_MIN_RATIO)
                            <= duration_ratio
                            <= float(RUNTIME_RECENCY_DURATION_MAX_RATIO)
                        )
                        duration_hard_passed = bool(
                            float(RUNTIME_RECENCY_DURATION_HARD_MIN_RATIO)
                            <= duration_ratio
                            <= float(RUNTIME_RECENCY_DURATION_HARD_MAX_RATIO)
                        )
                        if duration_soft_passed:
                            duration_guard_status = "soft_safe"
                        elif duration_hard_passed:
                            duration_guard_status = "soft_violation_hard_safe"
                        else:
                            duration_guard_status = "hard_fail"
                except Exception:
                    duration_soft_passed = False
                    duration_hard_passed = False
                    duration_guard_status = "invalid"

                arrival_passed = bool(arrival_guard.get("passed", True))
                # Keep the legacy key for downstream compatibility, but its Stage-2d
                # meaning is now "hard duration guard passed".
                row["duration_passed"] = bool(duration_hard_passed)
                row["duration_soft_passed"] = bool(duration_soft_passed)
                row["duration_hard_passed"] = bool(duration_hard_passed)
                row["duration_guard_status"] = str(duration_guard_status)
                row["arrival_passed"] = bool(arrival_passed)
                row["guard_passed"] = bool(duration_hard_passed and arrival_passed)
                records.append(row)

                logger.info(
                    "[Adaptive Runtime Recency] ratio=%.3f, cases=%s, score=%.6f, "
                    "duration_ratio=%s, duration_status=%s, arrival_span_ratio=%s, "
                    "guard_passed=%s",
                    float(ratio), int(runtime_cases), float(score),
                    duration_guard.get("duration_ratio"),
                    str(duration_guard_status),
                    arrival_guard.get("arrival_span_ratio"),
                    bool(row["guard_passed"]),
                )
            except Exception as exc:
                logger.warning(
                    "Adaptive runtime-recency candidate failed: ratio=%s, error=%s",
                    ratio, exc,
                )
                records.append({
                    "status": "failed",
                    "ratio": float(ratio),
                    "target_cases": int(runtime_cases),
                    "score": float("inf"),
                    "error": str(exc),
                })

        result["records"] = records
        baseline = next((
            row for row in records
            if row.get("status") == "ok" and abs(float(row.get("ratio", 0.0)) - 1.0) <= 1e-12
        ), None)
        recent_safe = [
            row for row in records
            if row.get("status") == "ok"
            and float(row.get("ratio", 1.0)) < 1.0 - 1e-12
            and bool(row.get("guard_passed", False))
            and np.isfinite(float(row.get("score", float("inf"))))
        ]

        if baseline is None:
            if recent_safe:
                selected = min(recent_safe, key=lambda row: float(row["score"]))
                result["reason"] = "baseline_failed_recent_safe_fallback"
            else:
                result["reason"] = "no_valid_candidate_fallback_full_history"
                return result
        elif not recent_safe:
            selected = baseline
            result["reason"] = "no_safe_recent_candidate"
        else:
            best_score = min(float(row["score"]) for row in recent_safe)
            near_best = [
                row for row in recent_safe
                if float(row["score"]) <= best_score * (1.0 + float(RUNTIME_RECENCY_TIE_TOLERANCE))
            ]
            # Conservative tie break: use more history when scores are effectively tied.
            best_recent = max(near_best, key=lambda row: float(row["ratio"]))
            baseline_score = float(baseline.get("score", float("inf")))
            baseline_safe = bool(baseline.get("guard_passed", False))
            required_score = baseline_score * (1.0 - float(RUNTIME_RECENCY_MIN_IMPROVEMENT))
            result["baseline_score"] = baseline_score
            result["best_recent_score"] = float(best_recent["score"])
            result["required_recent_score"] = float(required_score)

            if (not baseline_safe) or float(best_recent["score"]) <= float(required_score):
                selected = best_recent
                result["reason"] = (
                    "baseline_guard_failed_recent_safe"
                    if not baseline_safe else
                    "recent_window_validation_improvement_confirmed"
                )
            else:
                selected = baseline
                result["reason"] = "recent_improvement_below_threshold"

        result["selected_ratio"] = float(selected.get("ratio", 1.0))
        result["selected_cases"] = int(selected.get("target_cases", n_train_cases))
        result["selected_score"] = float(selected.get("score", float("inf")))
        result["selected_guard_passed"] = bool(selected.get("guard_passed", False))
        logger.info(
            "Adaptive runtime recency selected: ratio=%.3f, cases=%s, reason=%s",
            result["selected_ratio"], result["selected_cases"], result["reason"],
        )
        return result

    def _select_arrival_time_scale(
            self,
            structure_train_log,
            validation_log,
            pnml_filepath,
            selected_hyperparameters: dict,
            runtime_recency_ratio: float,
            arrival_recency_ratio: float = 1.0,
            enable_candidate_resource_filter: bool | None = None,
            candidate_resource_coverage: float | None = None,
            candidate_resource_max_k: int | None = None,
            candidate_resource_min_k: int | None = None,
    ) -> dict:
        """Stage 2e: calibrate only the global Arrival time scale on validation.

        The Arrival decision tree, leaf empirical distributions, dispatch model,
        External Delay quantile, and selected runtime-recency horizon are fixed.
        Therefore each scale candidate requires simulation only; the selected
        recency model is rediscovered/restored once (normally a discovery-cache hit).

        Selection is intentionally conservative:
        * scale=1.0 is a mandatory identity baseline for every dataset;
        * only CAR is optimized, with the Arrival span/case-count guard;
        * when the identity baseline is safe, a non-identity scale must improve
          validation CAR by ARRIVAL_TIME_SCALE_MIN_IMPROVEMENT;
        * near ties prefer the scale closest to 1.0;
        * test data are never used.
        """
        candidates = []
        for value in tuple(ARRIVAL_TIME_SCALE_CANDIDATES) + (1.0,):
            try:
                value = float(value)
            except Exception:
                continue
            if np.isfinite(value) and value > 0 and not any(abs(value - x) <= 1e-12 for x in candidates):
                candidates.append(value)
        candidates = sorted(candidates)

        result = {
            "enabled": bool(ARRIVAL_TIME_SCALE_CALIBRATION_ENABLED),
            "uses_test_log": False,
            "selection_metric": "car",
            "candidate_scales": [float(x) for x in candidates],
            "identity_baseline": 1.0,
            "selected_scale": 1.0,
            "arrival_demand_multiplier_fixed": 1.0,
            "eval_runs": int(ARRIVAL_TIME_SCALE_EVAL_RUNS),
            "min_improvement": float(ARRIVAL_TIME_SCALE_MIN_IMPROVEMENT),
            "tie_tolerance": float(ARRIVAL_TIME_SCALE_TIE_TOLERANCE),
            "runtime_recency_ratio": float(runtime_recency_ratio),
            "arrival_recency_ratio": float(arrival_recency_ratio),
            "records": [],
            "reason": "identity_default",
        }
        if not bool(ARRIVAL_TIME_SCALE_CALIBRATION_ENABLED):
            result["reason"] = "disabled_by_config"
            self._set_arrival_runtime_config(1.0, 1.0)
            return result
        if not candidates:
            result["reason"] = "no_valid_scale_candidates"
            self._set_arrival_runtime_config(1.0, 1.0)
            return result

        runtime_log, runtime_cases = self._build_runtime_recency_log(
            structure_train_log,
            float(runtime_recency_ratio),
        )
        if runtime_log is None:
            runtime_log = structure_train_log
            runtime_cases = int(self._count_log_cases(structure_train_log))
            result["runtime_recency_fallback_to_full_history"] = True
        result["runtime_cases"] = int(runtime_cases)

        # Restore/discover exactly the model selected by Stages 2a-2d.  This is
        # deliberately outside the scale loop, so scale calibration does not
        # trigger repeated alignment/model discovery.
        try:
            self._discover_candidate_model(
                event_log=structure_train_log,
                runtime_event_log=(
                    runtime_log if float(runtime_recency_ratio) < 1.0 - 1e-12 else None
                ),
                pnml_filepath=pnml_filepath,
                noise_threshold=selected_hyperparameters.get("noise_threshold"),
                max_depth_tree=int(selected_hyperparameters.get("tree_depth", 1)),
                decision_point_max_depth=int(selected_hyperparameters.get("decision_point_max_depth", 1)),
                arrival_decision_tree_max_depth=int(
                    selected_hyperparameters.get("arrival_decision_tree_max_depth", 1)
                ),
                queue_scheduling_max_depth=int(selected_hyperparameters.get("queue_scheduling_max_depth", 1)),
                conditional_prioritization_max_depth=int(
                    selected_hyperparameters.get("conditional_prioritization_max_depth", 1)
                ),
                queue_scheduling_force_empirical=bool(
                    selected_hyperparameters.get("queue_scheduling_force_empirical", False)
                ),
                discovery_profile=FULL_DISCOVERY_PROFILE,
                build_process_artifacts=False,
                enable_candidate_resource_filter=enable_candidate_resource_filter,
                candidate_resource_coverage=candidate_resource_coverage,
                candidate_resource_max_k=candidate_resource_max_k,
                candidate_resource_min_k=candidate_resource_min_k,
                alignment_timeout_seconds=ALIGNMENT_TIMEOUT_SECONDS,
            )
            # Re-apply all previously selected runtime choices, then force the
            # identity Arrival scale as the calibration baseline.
            self._apply_selected_runtime_parameters(selected_hyperparameters)
            arrival_overlay = self._apply_arrival_recency_overlay(
                structure_train_log,
                float(arrival_recency_ratio),
                selected_hyperparameters,
            )
            result["arrival_recency_overlay"] = arrival_overlay
            if float(arrival_recency_ratio) < 1.0 - 1e-12 and not bool(arrival_overlay.get("applied", False)):
                result["arrival_recency_ratio"] = 1.0
                arrival_recency_ratio = 1.0
            self._set_arrival_runtime_config(1.0, 1.0)
            try:
                self.integration._set_external_delay_activity_scales({}, reset_others=True)
            except Exception:
                pass
        except Exception as exc:
            logger.warning("Arrival time-scale calibration model restore failed: %s", exc)
            result["reason"] = "model_restore_failed_identity_fallback"
            result["error"] = str(exc)
            self._set_arrival_runtime_config(1.0, 1.0)
            return result

        def _aggregate_arrival_guard() -> dict:
            guard = self._extract_arrival_guard()
            repeated = getattr(self.integration, "_repeated_arrival_diagnostics", []) or []
            span_values = []
            case_values = []
            jaccard_values = []
            real_cases_values = []
            sim_cases_values = []
            for diag in repeated:
                if not isinstance(diag, dict):
                    continue
                try:
                    value = float(diag.get("span_ratio_sim_over_real"))
                    if np.isfinite(value) and value > 0:
                        span_values.append(value)
                except Exception:
                    pass
                try:
                    real_cases = float(diag.get("real_cases"))
                    sim_cases = float(diag.get("sim_cases"))
                    if np.isfinite(real_cases) and real_cases > 0 and np.isfinite(sim_cases):
                        case_values.append(sim_cases / real_cases)
                        real_cases_values.append(real_cases)
                        sim_cases_values.append(sim_cases)
                except Exception:
                    pass
                try:
                    value = float(diag.get("weekday_hour_jaccard"))
                    if np.isfinite(value):
                        jaccard_values.append(value)
                except Exception:
                    pass
            if span_values:
                guard["arrival_span_ratio"] = float(np.mean(span_values))
                guard["arrival_span_ratio_std"] = float(np.std(span_values))
            if case_values:
                guard["case_count_ratio"] = float(np.mean(case_values))
            if jaccard_values:
                guard["weekday_hour_jaccard"] = float(np.mean(jaccard_values))
            if real_cases_values:
                guard["real_cases"] = float(np.mean(real_cases_values))
            if sim_cases_values:
                guard["sim_cases"] = float(np.mean(sim_cases_values))
            span = guard.get("arrival_span_ratio")
            case_ratio = guard.get("case_count_ratio")
            passed = True
            try:
                if span is not None and np.isfinite(float(span)):
                    passed = passed and (
                        float(ARRIVAL_SPAN_MIN_RATIO) <= float(span) <= float(ARRIVAL_SPAN_MAX_RATIO)
                    )
            except Exception:
                pass
            try:
                if case_ratio is not None and np.isfinite(float(case_ratio)):
                    passed = passed and float(case_ratio) >= float(ARRIVAL_CASE_COUNT_MIN_RATIO)
            except Exception:
                pass
            guard["passed"] = bool(passed)
            return guard

        records = []
        for scale in candidates:
            try:
                self._set_arrival_runtime_config(
                    arrival_time_scale=float(scale),
                    arrival_demand_multiplier=1.0,
                )
                metrics = self._evaluate(
                    event_log=validation_log,
                    n_runs=max(1, int(ARRIVAL_TIME_SCALE_EVAL_RUNS)),
                    metrics_labels=list(ARRIVAL_STAGE_METRICS),
                    enable_time_diagnostics=True,
                )
                flat_metrics = self.integration._flatten_formatted_metrics(metrics)
                arrival_guard = _aggregate_arrival_guard()
                try:
                    car = float(flat_metrics.get("car", float("inf")))
                except Exception:
                    car = float("inf")
                if not np.isfinite(car):
                    car = float("inf")
                row = {
                    "status": "ok" if np.isfinite(car) else "invalid_metric",
                    "arrival_time_scale": float(scale),
                    "arrival_demand_multiplier": 1.0,
                    "metrics": flat_metrics,
                    "car": float(car),
                    "arrival_guard": arrival_guard,
                    "arrival_passed": bool(arrival_guard.get("passed", True)),
                    "runtime_recency_ratio": float(runtime_recency_ratio),
                }
                records.append(row)
                logger.info(
                    "[Arrival Time Scale] scale=%.3f, CAR=%.6f, arrival_span_ratio=%s, "
                    "arrival_span_std=%s, guard_passed=%s",
                    float(scale), float(car),
                    arrival_guard.get("arrival_span_ratio"),
                    arrival_guard.get("arrival_span_ratio_std"),
                    bool(row["arrival_passed"]),
                )
            except Exception as exc:
                logger.warning("Arrival time-scale candidate failed: scale=%s, error=%s", scale, exc)
                records.append({
                    "status": "failed",
                    "arrival_time_scale": float(scale),
                    "arrival_demand_multiplier": 1.0,
                    "car": float("inf"),
                    "arrival_guard": {},
                    "arrival_passed": False,
                    "error": str(exc),
                })

        result["records"] = records
        valid = [
            row for row in records
            if row.get("status") == "ok" and np.isfinite(float(row.get("car", float("inf"))))
        ]
        baseline = next((
            row for row in valid
            if abs(float(row.get("arrival_time_scale", 0.0)) - 1.0) <= 1e-12
        ), None)
        safe = [row for row in valid if bool(row.get("arrival_passed", False))]

        def _scale_closeness(row: dict) -> float:
            try:
                return abs(np.log(max(float(row.get("arrival_time_scale", 1.0)), 1e-12)))
            except Exception:
                return float("inf")

        def _arrival_closeness(row: dict) -> float:
            try:
                span = float((row.get("arrival_guard") or {}).get("arrival_span_ratio"))
                if np.isfinite(span) and span > 0:
                    return abs(np.log(span))
            except Exception:
                pass
            return float("inf")

        selected = baseline
        if safe:
            best_car = min(float(row["car"]) for row in safe)
            tolerance = max(abs(best_car), 1e-9) * float(ARRIVAL_TIME_SCALE_TIE_TOLERANCE)
            near_best = [row for row in safe if float(row["car"]) <= best_car + tolerance]
            best_safe = min(
                near_best,
                key=lambda row: (_scale_closeness(row), _arrival_closeness(row), float(row["car"])),
            )

            if baseline is None:
                selected = best_safe
                result["reason"] = "identity_baseline_missing_best_safe"
            elif not bool(baseline.get("arrival_passed", False)):
                selected = best_safe
                result["reason"] = "identity_guard_failed_best_safe"
            else:
                baseline_car = float(baseline["car"])
                required_car = baseline_car * (1.0 - float(ARRIVAL_TIME_SCALE_MIN_IMPROVEMENT))
                result["baseline_car"] = float(baseline_car)
                result["best_safe_car"] = float(best_safe["car"])
                result["required_car_for_switch"] = float(required_car)
                if (
                    abs(float(best_safe.get("arrival_time_scale", 1.0)) - 1.0) > 1e-12
                    and float(best_safe["car"]) <= float(required_car)
                ):
                    selected = best_safe
                    result["reason"] = "validation_car_improvement_confirmed"
                else:
                    selected = baseline
                    result["reason"] = "identity_retained_improvement_below_threshold"
        elif valid:
            # No candidate satisfies the broad Arrival guard.  Choose the one
            # whose simulated arrival span is closest to validation, with CAR as
            # the secondary criterion; this is safer than optimizing CAR alone.
            selected = min(
                valid,
                key=lambda row: (_arrival_closeness(row), float(row["car"]), _scale_closeness(row)),
            )
            result["reason"] = "no_guard_safe_candidate_closest_arrival_fallback"
        else:
            result["reason"] = "no_valid_candidate_identity_fallback"

        if selected is None:
            selected_scale = 1.0
        else:
            selected_scale = float(selected.get("arrival_time_scale", 1.0))
        result["selected_scale"] = float(selected_scale)
        result["selected_car"] = (
            None if selected is None or not np.isfinite(float(selected.get("car", float("inf"))))
            else float(selected.get("car"))
        )
        result["selected_arrival_guard"] = dict((selected or {}).get("arrival_guard") or {})
        result["selected_guard_passed"] = bool((selected or {}).get("arrival_passed", False))

        # Leave the integration in the selected runtime state so subsequent
        # diagnostics are internally consistent.  Stage 3 rediscovery will apply
        # the same selected value again through selected_hyperparameters.
        self._set_arrival_runtime_config(selected_scale, 1.0)
        logger.info(
            "Validation-controlled Arrival time scale selected: scale=%.3f, CAR=%s, reason=%s",
            float(selected_scale), result.get("selected_car"), result.get("reason"),
        )
        return result

    def _prepare_large_log_tuning_search_logs(
            self,
            train_log,
            validation_log,
            train_validation_log,
    ) -> dict:
        """Prepare sample-search logs only for genuinely large training logs.

        Search uses adjacent temporal windows. Full confirmation always receives the
        untouched original train/validation logs. Arrival-model discovery keeps the
        full training log because dropping training cases would directly change the
        learned inter-arrival process; only Arrival validation is sampled.
        """
        total_cases = self._count_log_cases(train_validation_log)
        train_cases = self._count_log_cases(train_log)
        validation_cases = self._count_log_cases(validation_log)
        info = {
            "enabled": bool(TUNING_SAMPLING_ENABLED),
            "applied": False,
            "policy": str(TUNING_SAMPLE_POLICY),
            "large_log_threshold": int(TUNING_LARGE_LOG_THRESHOLD),
            "target_total_cases": int(TUNING_SAMPLE_TOTAL_CASES),
            "population_cases": int(total_cases),
            "train_cases": int(train_cases),
            "validation_cases": int(validation_cases),
        }

        result = {
            "search_train_log": train_log,
            "search_validation_log": validation_log,
            "arrival_search_train_log": train_log,
            "arrival_search_validation_log": validation_log,
            "confirmation_train_log": None,
            "confirmation_validation_log": None,
            "info": info,
        }
        if not bool(TUNING_SAMPLING_ENABLED):
            info["reason"] = "disabled_by_config"
            return result
        if total_cases <= int(TUNING_LARGE_LOG_THRESHOLD):
            info["reason"] = "population_not_larger_than_threshold"
            return result
        if train_cases <= 0 or validation_cases <= 0:
            info["reason"] = "empty_train_or_validation_fallback_full"
            return result

        target_total = min(int(TUNING_SAMPLE_TOTAL_CASES), train_cases + validation_cases)
        validation_share = validation_cases / max(train_cases + validation_cases, 1)
        target_validation = int(round(target_total * validation_share))
        target_validation = max(int(TUNING_SAMPLE_MIN_VALIDATION_CASES), target_validation)
        target_validation = min(validation_cases, target_validation)
        target_train = max(1, target_total - target_validation)
        target_train = max(int(TUNING_SAMPLE_MIN_TRAIN_CASES), target_train)
        target_train = min(train_cases, target_train)

        # If minimums pushed the total over budget, reduce train first while keeping
        # enough data for resource/dispatch discovery.
        overflow = target_train + target_validation - target_total
        if overflow > 0:
            reducible_train = max(0, target_train - int(TUNING_SAMPLE_MIN_TRAIN_CASES))
            reduce_train = min(overflow, reducible_train)
            target_train -= reduce_train
            overflow -= reduce_train
        if overflow > 0:
            reducible_val = max(0, target_validation - int(TUNING_SAMPLE_MIN_VALIDATION_CASES))
            target_validation -= min(overflow, reducible_val)

        search_train = self._take_temporal_case_window(
            train_log, target_train, side="tail"
        )
        search_validation = self._take_temporal_case_window(
            validation_log, target_validation, side="head"
        )

        result.update({
            "search_train_log": search_train,
            "search_validation_log": search_validation,
            # Arrival discovery intentionally stays on full train to preserve the
            # original inter-arrival process. Candidate evaluation uses the sampled
            # validation window, then Top-K is confirmed on full validation.
            "arrival_search_train_log": train_log,
            "arrival_search_validation_log": search_validation,
            "confirmation_train_log": train_log,
            "confirmation_validation_log": validation_log,
        })
        info.update({
            "applied": True,
            "reason": "large_log_sample_search_with_full_confirmation",
            "sample_train_cases": int(self._count_log_cases(search_train)),
            "sample_validation_cases": int(self._count_log_cases(search_validation)),
            "arrival_discovery_uses_full_train": True,
            "full_confirmation": True,
            "structure_confirm_top_k": int(TUNING_STRUCTURE_FULL_CONFIRM_TOP_K),
            "runtime_confirm_top_k": int(TUNING_RUNTIME_FULL_CONFIRM_TOP_K),
            "dispatch_external_confirm_top_k": int(TUNING_DISPATCH_EXTERNAL_FULL_CONFIRM_TOP_K),
        })
        logger.info(
            "Large-log tuning sampling applied: total=%s, search_train=%s (tail), "
            "search_validation=%s (head), full_confirmation=True",
            total_cases, info["sample_train_cases"], info["sample_validation_cases"],
        )
        return result

    def _rank_runtime_candidates_for_confirmation(
            self,
            runtime_records: list[dict],
            runtime_metrics: tuple[str, ...],
            runtime_search_stage: str,
            case_duration_min_ratio: float,
            case_duration_max_ratio: float,
            max_external_delay_ratio: float | None,
            ctd_tolerance: float | None,
            metric_weights: dict[str, float] | None,
            require_arrival_guard: bool,
            require_duration_guard: bool,
            top_k: int,
            identity_keys: tuple[str, ...],
    ) -> list[dict]:
        """Rank Top-K sample candidates using exactly the existing selection logic.

        This performs no simulation. After each selected record, every record with the
        same hyperparameter identity is removed and the selector is run again.
        """
        remaining = [
            copy.deepcopy(row) for row in (runtime_records or [])
            if isinstance(row, dict) and row.get("status") == "ok"
        ]
        ranked: list[dict] = []
        top_k = max(1, int(top_k))
        while remaining and len(ranked) < top_k:
            try:
                result = self._select_runtime_candidate_from_records(
                    runtime_records=remaining,
                    runtime_metrics=runtime_metrics,
                    runtime_search_stage=runtime_search_stage,
                    case_duration_min_ratio=case_duration_min_ratio,
                    case_duration_max_ratio=case_duration_max_ratio,
                    max_external_delay_ratio=max_external_delay_ratio,
                    ctd_tolerance=ctd_tolerance,
                    metric_weights=metric_weights,
                    require_arrival_guard=require_arrival_guard,
                    require_duration_guard=require_duration_guard,
                )
                selected = copy.deepcopy(result["selected"])
            except Exception:
                break
            ranked.append(selected)
            identity = tuple(selected.get(key) for key in identity_keys)
            remaining = [
                row for row in remaining
                if tuple(row.get(key) for key in identity_keys) != identity
            ]
        return ranked

    def _run_arrival_resource_depth_search(
            self,
            top_structures: list[dict],
            train_log,
            validation_log,
            pnml_filepath,
            runtime_metrics: tuple[str, ...],
            arrival_depth_candidates: list[int],
            resource_depth_candidates: list[int],
            case_duration_min_ratio: float,
            case_duration_max_ratio: float,
            max_external_delay_ratio: float | None,
            ctd_tolerance: float | None,
            enable_candidate_resource_filter: bool,
            candidate_resource_coverage: float,
            candidate_resource_max_k: int | None,
            candidate_resource_min_k: int,
            arrival_train_log=None,
            arrival_validation_log=None,
            confirmation_train_log=None,
            confirmation_validation_log=None,
            full_confirmation_top_k: int = TUNING_RUNTIME_FULL_CONFIRM_TOP_K,
    ) -> dict:
        """Stage 2a/2b：先独立选择 Arrival depth，再选择 Resource Allocation depth。

        Ready Queue 与 dispatch discovery 始终运行，不再存在“Stage 2b 关闭 CPT”的控制逻辑。
        为避免优先级模型复杂度干扰 Arrival/Resource 两个白盒阶段，这里固定使用最浅
        priority-tree depth 作为 probe；实际 dispatch policy 仍由数据自动判定为 CPT 或 FCFS。

        1) Stage 2a：固定最浅 Resource probe，仅用 CAR + arrival diagnostics 选择 Arrival depth；
        2) Stage 2b：固定已选 Arrival，搜索 Resource Allocation depth。

        这样避免 Arrival 与 Resource Allocation 通过全局运行时指标互相补偿，同时把
        候选数从 |A|×|R| 降为 |A|+|R|。
        """
        if not top_structures:
            raise RuntimeError("Runtime depth search received no Stage-1 structure candidate.")

        arrival_train_log = train_log if arrival_train_log is None else arrival_train_log
        arrival_validation_log = (
            validation_log if arrival_validation_log is None else arrival_validation_log
        )
        full_confirmation_enabled = bool(
            confirmation_train_log is not None and confirmation_validation_log is not None
        )
        full_confirmation_top_k = max(1, int(full_confirmation_top_k))

        arrival_depth_candidates = self._normalize_depth_candidates(
            arrival_depth_candidates,
            default=RUNTIME_TREE_DEPTH_CANDIDATES,
            min_depth=1,
            max_depth=4,
        )
        resource_depth_candidates = self._normalize_depth_candidates(
            resource_depth_candidates,
            default=RUNTIME_TREE_DEPTH_CANDIDATES,
            min_depth=1,
            max_depth=4,
        )

        structure_record = top_structures[0]
        noise_threshold = structure_record.get("noise_threshold")
        control_flow_depth = int(
            structure_record.get(
                "decision_point_max_depth",
                structure_record.get("max_depth_tree", structure_record.get("tree_depth", 3)),
            )
        )

        queue_strategy_candidates = self._normalize_queue_strategy_candidates(QUEUE_STRATEGIES)
        if not queue_strategy_candidates:
            queue_strategy_candidates = self._normalize_queue_strategy_candidates(QUEUE_STRATEGIES[:1])
        strategy = queue_strategy_candidates[0]
        strategy_name = str(strategy.get("name", "balanced"))
        priority_tree_probe_depth = int(min(PRIORITY_TREE_DEPTH_CANDIDATES))
        tree_weight = float((strategy.get("weights") or {}).get("tree", 0.0))
        queue_scheduling_force_empirical = tree_weight <= 1e-12


        def _evaluate_depth_candidate(
                arrival_depth: int,
                queue_depth: int,
                metrics_to_use: tuple[str, ...],
                stage_name: str,
                trial_number: int,
                candidate_train_log=None,
                candidate_validation_log=None,
                record_phase: str = "search_cv",
        ) -> dict:
            candidate_train_log = train_log if candidate_train_log is None else candidate_train_log
            candidate_validation_log = (
                validation_log if candidate_validation_log is None else candidate_validation_log
            )
            self._discover_candidate_model(
                event_log=candidate_train_log,
                pnml_filepath=pnml_filepath,
                noise_threshold=noise_threshold,
                max_depth_tree=control_flow_depth,
                decision_point_max_depth=control_flow_depth,
                arrival_decision_tree_max_depth=int(arrival_depth),
                queue_scheduling_max_depth=int(queue_depth),
                conditional_prioritization_max_depth=int(priority_tree_probe_depth),
                queue_scheduling_force_empirical=bool(queue_scheduling_force_empirical),
                discovery_profile=FULL_DISCOVERY_PROFILE,
                build_process_artifacts=False,
                enable_candidate_resource_filter=enable_candidate_resource_filter,
                candidate_resource_coverage=candidate_resource_coverage,
                candidate_resource_max_k=candidate_resource_max_k,
                candidate_resource_min_k=candidate_resource_min_k,
                alignment_timeout_seconds=ALIGNMENT_TIMEOUT_SECONDS,
            )

            dispatch_summary = self._extract_dispatch_summary_from_integration(self.integration)

            self._set_arrival_runtime_config(
                arrival_time_scale=1.0,
                arrival_demand_multiplier=1.0,
            )
            self.integration._set_queue_scheduling_runtime_config(
                score_mixing_weights=strategy.get("weights"),
                strategy_name=strategy_name,
            )
            self.integration._set_external_delay_runtime_config(
                global_scale=1.0,
                max_delay_quantile=1.0,
            )

            validation_metrics = self._evaluate(
                event_log=candidate_validation_log,
                n_runs=1,
                metrics_labels=list(metrics_to_use),
                enable_time_diagnostics=True,
            )
            flat_metrics = self.integration._flatten_formatted_metrics(validation_metrics)
            time_diagnostics = getattr(self.integration, "_last_time_diagnostics", {}) or {}
            duration_guard = self.integration._extract_case_duration_guard(time_diagnostics)
            external_delay_ratio = self._extract_external_delay_ratio(time_diagnostics)
            if external_delay_ratio is not None:
                duration_guard["external_delay_ratio"] = float(external_delay_ratio)
            arrival_guard = self._extract_arrival_guard()

            # 仅用于日志记录；最终选择会在 _select_runtime_candidate_from_records 中
            # 按阶段专属 metric weights + guard 重新计算。
            score_duration_guard = duration_guard
            if stage_name == "arrival_depth_grid":
                # Arrival 阶段不允许 case-duration 反向决定 Arrival depth。
                score_duration_guard = dict(duration_guard or {})
                score_duration_guard["duration_ratio"] = 1.0

            score = self._scalar_runtime_objective(
                metrics=flat_metrics,
                duration_guard=score_duration_guard,
                metric_names=list(metrics_to_use),
                min_duration_ratio=case_duration_min_ratio,
                max_duration_ratio=case_duration_max_ratio,
                max_external_delay_ratio=max_external_delay_ratio,
            )

            return {
                "stage": f"{stage_name}_{record_phase}",
                "evaluation_phase": str(record_phase),
                "runtime_search_stage": stage_name,
                "trial_number": int(trial_number),
                "resource_budget": 1,
                "noise_threshold": noise_threshold,
                "tree_depth": int(control_flow_depth),
                "max_depth_tree": int(control_flow_depth),
                "decision_point_max_depth": int(control_flow_depth),
                "arrival_decision_tree_max_depth": int(arrival_depth),
                "case_arrival_max_depth": int(arrival_depth),
                "queue_scheduling_max_depth": int(queue_depth),
                "conditional_prioritization_max_depth": int(priority_tree_probe_depth),
                "conditional_prioritization_model_kind": dispatch_summary.get("model_kind"),
                "dispatch_policy": dispatch_summary.get("dispatch_policy"),
                "priority_tree_active": bool(dispatch_summary.get("priority_tree_active")),
                "conditional_prioritization_pairs": dispatch_summary.get("n_competition_pairs"),
                "conditional_prioritization_overtaking_rate": dispatch_summary.get("overtaking_rate"),
                "external_delay_model_type": "activity_specific_external_delay_timer",
                "external_delay_tree_enabled": False,
                "queue_strategy_name": strategy_name,
                "queue_score_mixing_weights": strategy.get("weights"),
                "queue_scheduling_force_empirical": bool(queue_scheduling_force_empirical),
                "structure_score": structure_record.get("structure_score"),
                "structure_metrics": structure_record.get("metrics", {}),
                "arrival_time_scale": 1.0,
                "arrival_demand_multiplier": 1.0,
                "arrival_model_type": "case_arrival_decision_tree_interarrival",
                "external_delay": {
                    "global_scale": 1.0,
                    "max_delay_quantile": 1.0,
                    "method": "full_tail_during_arrival_resource_selection",
                },
                "metrics": flat_metrics,
                "case_duration_guard": duration_guard,
                "arrival_guard": arrival_guard,
                "time_decomposition": {
                    "external_delay_ratio": duration_guard.get("external_delay_ratio"),
                },
                "optuna_objective_score": float(score),
                "status": "ok",
            }

        # ----------------------------------------------------------
        # Stage 2a：Arrival depth 独立搜索
        # ----------------------------------------------------------
        # Resource depth 仅作为构建完整可仿真模型的 probe；取最浅候选，减少资源模型复杂度
        # 对 Arrival 选择的干扰。Arrival 最终只由 CAR + arrival diagnostics 选择。
        probe_resource_depth = int(min(resource_depth_candidates))
        arrival_records: list[dict] = []
        logger.info(
            "Stage 2a Arrival depth search started: control_flow_depth=%s, probe_resource_depth=%s, "
            "arrival_depth_candidates=%s, metrics=%s",
            control_flow_depth,
            probe_resource_depth,
            arrival_depth_candidates,
            list(ARRIVAL_STAGE_METRICS),
        )

        for candidate_number, arrival_depth in enumerate(arrival_depth_candidates, start=1):
            try:
                arrival_records.append(
                    _evaluate_depth_candidate(
                        arrival_depth=int(arrival_depth),
                        queue_depth=probe_resource_depth,
                        metrics_to_use=ARRIVAL_STAGE_METRICS,
                        stage_name="arrival_depth_grid",
                        trial_number=candidate_number,
                        candidate_train_log=arrival_train_log,
                        candidate_validation_log=arrival_validation_log,
                        record_phase="sample_search" if full_confirmation_enabled else "search_cv",
                    )
                )
            except Exception as exc:
                logger.warning("Arrival depth candidate failed: depth=%s, error=%s", arrival_depth, exc)
                arrival_records.append({
                    "stage": "arrival_depth_grid_search_cv",
                    "runtime_search_stage": "arrival_depth_grid",
                    "trial_number": int(candidate_number),
                    "resource_budget": 1,
                    "noise_threshold": noise_threshold,
                    "tree_depth": int(control_flow_depth),
                    "decision_point_max_depth": int(control_flow_depth),
                    "arrival_decision_tree_max_depth": int(arrival_depth),
                    "queue_scheduling_max_depth": probe_resource_depth,
                    "conditional_prioritization_max_depth": int(priority_tree_probe_depth),
                    "metrics": {},
                    "case_duration_guard": {},
                    "arrival_guard": {},
                    "optuna_objective_score": float("inf"),
                    "status": "failed",
                    "error": str(exc),
                })

        arrival_result = self._select_runtime_candidate_from_records(
            runtime_records=arrival_records,
            runtime_metrics=ARRIVAL_STAGE_METRICS,
            runtime_search_stage="arrival_depth_grid",
            case_duration_min_ratio=case_duration_min_ratio,
            case_duration_max_ratio=case_duration_max_ratio,
            max_external_delay_ratio=max_external_delay_ratio,
            ctd_tolerance=None,
            metric_weights=ARRIVAL_STAGE_METRIC_WEIGHTS,
            require_arrival_guard=True,
            require_duration_guard=False,
        )
        arrival_confirmation_records: list[dict] = []
        if full_confirmation_enabled:
            ranked_arrival = self._rank_runtime_candidates_for_confirmation(
                runtime_records=arrival_records,
                runtime_metrics=ARRIVAL_STAGE_METRICS,
                runtime_search_stage="arrival_depth_grid_sample_ranking",
                case_duration_min_ratio=case_duration_min_ratio,
                case_duration_max_ratio=case_duration_max_ratio,
                max_external_delay_ratio=max_external_delay_ratio,
                ctd_tolerance=None,
                metric_weights=ARRIVAL_STAGE_METRIC_WEIGHTS,
                require_arrival_guard=True,
                require_duration_guard=False,
                top_k=min(full_confirmation_top_k, len(arrival_depth_candidates)),
                identity_keys=("arrival_decision_tree_max_depth",),
            )
            for confirm_idx, sample_candidate in enumerate(ranked_arrival, start=1):
                depth = int(sample_candidate.get("arrival_decision_tree_max_depth", arrival_depth_candidates[0]))
                try:
                    arrival_confirmation_records.append(
                        _evaluate_depth_candidate(
                            arrival_depth=depth,
                            queue_depth=probe_resource_depth,
                            metrics_to_use=ARRIVAL_STAGE_METRICS,
                            stage_name="arrival_depth_grid",
                            trial_number=confirm_idx,
                            candidate_train_log=confirmation_train_log,
                            candidate_validation_log=confirmation_validation_log,
                            record_phase="full_confirmation",
                        )
                    )
                except Exception as exc:
                    logger.warning("Arrival full confirmation failed: depth=%s, error=%s", depth, exc)
            if any(row.get("status") == "ok" for row in arrival_confirmation_records):
                arrival_result = self._select_runtime_candidate_from_records(
                    runtime_records=arrival_confirmation_records,
                    runtime_metrics=ARRIVAL_STAGE_METRICS,
                    runtime_search_stage="arrival_depth_grid_full_confirmation",
                    case_duration_min_ratio=case_duration_min_ratio,
                    case_duration_max_ratio=case_duration_max_ratio,
                    max_external_delay_ratio=max_external_delay_ratio,
                    ctd_tolerance=None,
                    metric_weights=ARRIVAL_STAGE_METRIC_WEIGHTS,
                    require_arrival_guard=True,
                    require_duration_guard=False,
                )

        selected_arrival_depth = int(
            arrival_result["selected"].get("arrival_decision_tree_max_depth", arrival_depth_candidates[0])
        )

        # ----------------------------------------------------------
        # Stage 2b：Resource Allocation depth 独立搜索
        # ----------------------------------------------------------
        resource_records: list[dict] = []
        logger.info(
            "Stage 2b Resource Allocation depth search started: fixed_arrival_depth=%s, "
            "resource_depth_candidates=%s, metrics=%s, dispatch=data_driven, priority_probe_depth=%s",
            selected_arrival_depth,
            resource_depth_candidates,
            list(RESOURCE_ALLOCATION_STAGE_METRICS),
            priority_tree_probe_depth,
        )

        for candidate_number, queue_depth in enumerate(resource_depth_candidates, start=1):
            try:
                resource_records.append(
                    _evaluate_depth_candidate(
                        arrival_depth=selected_arrival_depth,
                        queue_depth=int(queue_depth),
                        metrics_to_use=RESOURCE_ALLOCATION_STAGE_METRICS,
                        stage_name="resource_allocation_depth_grid",
                        trial_number=candidate_number,
                        candidate_train_log=train_log,
                        candidate_validation_log=validation_log,
                        record_phase="sample_search" if full_confirmation_enabled else "search_cv",
                    )
                )
            except Exception as exc:
                logger.warning("Resource Allocation depth candidate failed: depth=%s, error=%s", queue_depth, exc)
                resource_records.append({
                    "stage": "resource_allocation_depth_grid_search_cv",
                    "runtime_search_stage": "resource_allocation_depth_grid",
                    "trial_number": int(candidate_number),
                    "resource_budget": 1,
                    "noise_threshold": noise_threshold,
                    "tree_depth": int(control_flow_depth),
                    "decision_point_max_depth": int(control_flow_depth),
                    "arrival_decision_tree_max_depth": int(selected_arrival_depth),
                    "queue_scheduling_max_depth": int(queue_depth),
                    "conditional_prioritization_max_depth": int(priority_tree_probe_depth),
                    "metrics": {},
                    "case_duration_guard": {},
                    "arrival_guard": {},
                    "optuna_objective_score": float("inf"),
                    "status": "failed",
                    "error": str(exc),
                })

        resource_result = self._select_runtime_candidate_from_records(
            runtime_records=resource_records,
            runtime_metrics=RESOURCE_ALLOCATION_STAGE_METRICS,
            runtime_search_stage="resource_allocation_depth_grid",
            case_duration_min_ratio=case_duration_min_ratio,
            case_duration_max_ratio=case_duration_max_ratio,
            max_external_delay_ratio=max_external_delay_ratio,
            ctd_tolerance=ctd_tolerance,
            metric_weights=RESOURCE_ALLOCATION_STAGE_METRIC_WEIGHTS,
            require_arrival_guard=True,
            require_duration_guard=True,
        )

        resource_confirmation_records: list[dict] = []
        if full_confirmation_enabled:
            ranked_resource = self._rank_runtime_candidates_for_confirmation(
                runtime_records=resource_records,
                runtime_metrics=RESOURCE_ALLOCATION_STAGE_METRICS,
                runtime_search_stage="resource_allocation_depth_grid_sample_ranking",
                case_duration_min_ratio=case_duration_min_ratio,
                case_duration_max_ratio=case_duration_max_ratio,
                max_external_delay_ratio=max_external_delay_ratio,
                ctd_tolerance=ctd_tolerance,
                metric_weights=RESOURCE_ALLOCATION_STAGE_METRIC_WEIGHTS,
                require_arrival_guard=True,
                require_duration_guard=True,
                top_k=min(full_confirmation_top_k, len(resource_depth_candidates)),
                identity_keys=("queue_scheduling_max_depth",),
            )
            for confirm_idx, sample_candidate in enumerate(ranked_resource, start=1):
                depth = int(sample_candidate.get("queue_scheduling_max_depth", resource_depth_candidates[0]))
                try:
                    resource_confirmation_records.append(
                        _evaluate_depth_candidate(
                            arrival_depth=selected_arrival_depth,
                            queue_depth=depth,
                            metrics_to_use=RESOURCE_ALLOCATION_STAGE_METRICS,
                            stage_name="resource_allocation_depth_grid",
                            trial_number=confirm_idx,
                            candidate_train_log=confirmation_train_log,
                            candidate_validation_log=confirmation_validation_log,
                            record_phase="full_confirmation",
                        )
                    )
                except Exception as exc:
                    logger.warning("Resource Allocation full confirmation failed: depth=%s, error=%s", depth, exc)
            if any(row.get("status") == "ok" for row in resource_confirmation_records):
                resource_result = self._select_runtime_candidate_from_records(
                    runtime_records=resource_confirmation_records,
                    runtime_metrics=RESOURCE_ALLOCATION_STAGE_METRICS,
                    runtime_search_stage="resource_allocation_depth_grid_full_confirmation",
                    case_duration_min_ratio=case_duration_min_ratio,
                    case_duration_max_ratio=case_duration_max_ratio,
                    max_external_delay_ratio=max_external_delay_ratio,
                    ctd_tolerance=ctd_tolerance,
                    metric_weights=RESOURCE_ALLOCATION_STAGE_METRIC_WEIGHTS,
                    require_arrival_guard=True,
                    require_duration_guard=True,
                )

        all_records = (
            list(arrival_records)
            + list(arrival_confirmation_records)
            + list(resource_records)
            + list(resource_confirmation_records)
        )
        resource_result["runtime_records"] = all_records
        resource_summary = dict(resource_result.get("optuna_summary", {}) or {})
        resource_summary.update({
            "enabled": True,
            "search_type": "sequential_arrival_then_resource_depth_grid",
            "runtime_search_stage": "arrival_then_resource_allocation",
            "arrival_stage": arrival_result.get("optuna_summary", {}),
            "resource_allocation_stage": dict(resource_result.get("optuna_summary", {}) or {}),
            "arrival_depth_candidates": list(arrival_depth_candidates),
            "resource_depth_candidates": list(resource_depth_candidates),
            "n_grid_candidates": int(len(arrival_depth_candidates) + len(resource_depth_candidates)),
            "selected_arrival_depth": int(selected_arrival_depth),
            "selected_resource_allocation_depth": int(
                resource_result["selected"].get("queue_scheduling_max_depth", probe_resource_depth)
            ),
            "large_log_sample_search_full_confirmation": {
                "enabled": bool(full_confirmation_enabled),
                "top_k": int(full_confirmation_top_k) if full_confirmation_enabled else None,
                "arrival_confirmation_candidates": int(len(arrival_confirmation_records)),
                "resource_confirmation_candidates": int(len(resource_confirmation_records)),
                "arrival_discovery_used_full_train_during_sample_search": bool(full_confirmation_enabled),
            },
            "dispatch_policy_selection": "data_driven_cpt_or_fcfs",
            "priority_tree_probe_depth": int(priority_tree_probe_depth),
            "fixed_external_delay_global_scale": 1.0,
            "fixed_external_delay_max_delay_quantile": 1.0,
            "note": (
                "Arrival and Resource Allocation are selected sequentially to prevent cross-module compensation; "
                "ready-queue dispatch is always discovered from data, with a fixed shallow priority-tree probe."
            ),
        })
        resource_result["optuna_summary"] = resource_summary

        logger.info(
            "Stage 2a/2b completed: selected_arrival_depth=%s, selected_resource_allocation_depth=%s",
            selected_arrival_depth,
            resource_result["selected"].get("queue_scheduling_max_depth"),
        )
        return resource_result


    def _run_dispatch_external_search(
            self,
            base_candidate: dict,
            train_log,
            validation_log,
            pnml_filepath,
            runtime_metrics: tuple[str, ...],
            priority_tree_depth_candidates: list[int],
            external_delay_quantile_candidates: list[float],
            case_duration_min_ratio: float,
            case_duration_max_ratio: float,
            max_external_delay_ratio: float | None,
            ctd_tolerance: float | None,
            enable_candidate_resource_filter: bool,
            candidate_resource_coverage: float,
            candidate_resource_max_k: int | None,
            candidate_resource_min_k: int,
            confirmation_train_log=None,
            confirmation_validation_log=None,
            full_confirmation_top_k: int = TUNING_DISPATCH_EXTERNAL_FULL_CONFIRM_TOP_K,
    ) -> dict:
        """Stage 2c：data-driven Dispatch × External Delay 搜索。

        Ready Queue 始终启用，StagewiseTuner 不再负责开启/关闭 CPT。
        Dispatch policy 先由数据发现结果自动确定：
        - ``pairwise_decision_tree`` -> 使用 CPT，并搜索 Priority Tree depth × External quantile；
        - 其他 fallback/disabled 结果 -> 使用 FCFS，Priority Tree depth 不适用，只搜索 External quantile。

        大日志启用 sample search 时，dispatch policy 优先在完整 confirmation-train 上做一次浅层 probe，
        避免因为 sample 中竞争样本不足而误把完整数据集判成 FCFS。Priority depth 的复杂度搜索仍遵循
        sample search + full confirmation 协议。
        """
        if not isinstance(base_candidate, dict):
            raise RuntimeError("Dispatch/External search received no valid Resource-Allocation candidate.")

        full_confirmation_enabled = bool(
            confirmation_train_log is not None and confirmation_validation_log is not None
        )
        full_confirmation_top_k = max(1, int(full_confirmation_top_k))

        priority_tree_depth_candidates = self._normalize_depth_candidates(
            priority_tree_depth_candidates,
            default=PRIORITY_TREE_DEPTH_CANDIDATES,
            min_depth=1,
            max_depth=4,
        )
        quantile_candidates = self._normalize_quantile_candidates(
            external_delay_quantile_candidates
        )

        noise_threshold = base_candidate.get("noise_threshold")
        control_flow_depth = int(
            base_candidate.get(
                "decision_point_max_depth",
                base_candidate.get("max_depth_tree", base_candidate.get("tree_depth", 3)),
            )
        )
        arrival_depth = int(base_candidate.get("arrival_decision_tree_max_depth", control_flow_depth))
        resource_depth = int(base_candidate.get("queue_scheduling_max_depth", control_flow_depth))
        strategy_name = str(base_candidate.get("queue_strategy_name", "balanced"))
        strategy_weights = base_candidate.get("queue_score_mixing_weights") or {
            "tree": 0.3,
            "frequency": 0.5,
            "availability": 0.2,
        }
        queue_scheduling_force_empirical = bool(
            base_candidate.get("queue_scheduling_force_empirical", False)
        )

        joint_metrics = tuple(DISPATCH_EXTERNAL_STAGE_METRICS)
        if runtime_metrics:
            requested = [m for m in joint_metrics if m in tuple(runtime_metrics)]
            joint_metrics = tuple(requested) or tuple(DISPATCH_EXTERNAL_STAGE_METRICS)

        def _discover_dispatch_model(discovery_log, priority_depth: int) -> dict:
            self._discover_candidate_model(
                event_log=discovery_log,
                pnml_filepath=pnml_filepath,
                noise_threshold=noise_threshold,
                max_depth_tree=control_flow_depth,
                decision_point_max_depth=control_flow_depth,
                arrival_decision_tree_max_depth=arrival_depth,
                queue_scheduling_max_depth=resource_depth,
                conditional_prioritization_max_depth=int(priority_depth),
                queue_scheduling_force_empirical=queue_scheduling_force_empirical,
                discovery_profile=FULL_DISCOVERY_PROFILE,
                build_process_artifacts=False,
                enable_candidate_resource_filter=enable_candidate_resource_filter,
                candidate_resource_coverage=candidate_resource_coverage,
                candidate_resource_max_k=candidate_resource_max_k,
                candidate_resource_min_k=candidate_resource_min_k,
                alignment_timeout_seconds=ALIGNMENT_TIMEOUT_SECONDS,
            )
            self._set_arrival_runtime_config(
                arrival_time_scale=1.0,
                arrival_demand_multiplier=1.0,
            )
            self.integration._set_queue_scheduling_runtime_config(
                score_mixing_weights=strategy_weights,
                strategy_name=strategy_name,
            )
            return self._extract_dispatch_summary_from_integration(self.integration)

        priority_probe_depth = int(min(priority_tree_depth_candidates))
        dispatch_probe_source = "confirmation_train_log" if full_confirmation_enabled else "search_train_log"
        dispatch_probe_log = confirmation_train_log if full_confirmation_enabled else train_log
        try:
            dataset_dispatch = _discover_dispatch_model(dispatch_probe_log, priority_probe_depth)
        except Exception as exc:
            if full_confirmation_enabled:
                logger.warning(
                    "Stage 2c full-data dispatch probe failed; falling back to search-train probe: %s",
                    exc,
                )
                dispatch_probe_source = "search_train_log_fallback"
                dataset_dispatch = _discover_dispatch_model(train_log, priority_probe_depth)
            else:
                raise

        priority_tree_is_active = bool(dataset_dispatch.get("priority_tree_active"))
        searched_priority_depths = (
            list(priority_tree_depth_candidates)
            if priority_tree_is_active
            else [priority_probe_depth]
        )

        logger.info(
            "Stage 2c dispatch-aware External Delay search started: arrival_depth=%s, "
            "resource_depth=%s, dataset_dispatch=%s, model_kind=%s, "
            "priority_depths=%s, quantile_candidates=%s, metrics=%s",
            arrival_depth,
            resource_depth,
            dataset_dispatch.get("dispatch_policy"),
            dataset_dispatch.get("model_kind"),
            searched_priority_depths,
            quantile_candidates,
            list(joint_metrics),
        )
        if not priority_tree_is_active:
            logger.info(
                "Stage 2c dataset selected FCFS; skipping Priority Tree depth grid and tuning only External Delay quantile."
            )

        candidate_number = 0

        def _evaluate_quantile_batch(
                *,
                evaluation_log,
                priority_depth: int,
                dispatch_summary: dict,
                quantile_specs: list[dict],
                phase: str,
        ) -> list[dict]:
            nonlocal candidate_number
            records: list[dict] = []
            if not quantile_specs:
                return records

            logger.info(
                "[Serial Eval] Stage 2c evaluating %s quantiles sequentially: "
                "dispatch=%s, priority_depth=%s, phase=%s",
                len(quantile_specs),
                dispatch_summary.get("dispatch_policy"),
                priority_depth,
                phase,
            )

            for spec in quantile_specs:
                quantile = float(spec["quantile"])
                candidate_number += 1
                trial_no = int(candidate_number)
                try:
                    logger.info(
                        "[Serial Eval] Stage 2c candidate %s/%s: quantile=%s",
                        trial_no,
                        len(quantile_specs),
                        quantile,
                    )
                    self.integration._set_external_delay_runtime_config(
                        global_scale=1.0,
                        max_delay_quantile=quantile,
                    )
                    validation_metrics = self._evaluate(
                        event_log=evaluation_log,
                        n_runs=1,
                        metrics_labels=list(joint_metrics),
                        enable_time_diagnostics=True,
                    )
                    flat_metrics = self.integration._flatten_formatted_metrics(validation_metrics)
                    time_diagnostics = getattr(self.integration, "_last_time_diagnostics", {}) or {}
                    duration_guard = self.integration._extract_case_duration_guard(time_diagnostics)
                    external_delay_ratio = self._extract_external_delay_ratio(time_diagnostics)
                    if external_delay_ratio is not None:
                        duration_guard["external_delay_ratio"] = float(external_delay_ratio)
                    arrival_guard = self._extract_arrival_guard()

                    score = self._scalar_runtime_objective(
                        metrics=flat_metrics,
                        duration_guard=duration_guard,
                        metric_names=list(joint_metrics),
                        min_duration_ratio=case_duration_min_ratio,
                        max_duration_ratio=case_duration_max_ratio,
                        max_external_delay_ratio=max_external_delay_ratio,
                    )
                    method = (
                        "priority_tree_depth_x_external_quantile"
                        if bool(dispatch_summary.get("priority_tree_active"))
                        else "external_quantile_only_fcfs"
                    )
                    record = {
                        "stage": f"dispatch_external_{phase}",
                        "evaluation_phase": phase,
                        "runtime_search_stage": "dispatch_external_search",
                        "trial_number": trial_no,
                        "resource_budget": 1,
                        "noise_threshold": noise_threshold,
                        "tree_depth": int(control_flow_depth),
                        "max_depth_tree": int(control_flow_depth),
                        "decision_point_max_depth": int(control_flow_depth),
                        "arrival_decision_tree_max_depth": int(arrival_depth),
                        "case_arrival_max_depth": int(arrival_depth),
                        "queue_scheduling_max_depth": int(resource_depth),
                        # Integration compatibility field; only affects runtime when CPT is actually discovered.
                        "conditional_prioritization_max_depth": int(priority_depth),
                        "priority_tree_max_depth": int(priority_depth) if bool(dispatch_summary.get("priority_tree_active")) else None,
                        "priority_tree_active": bool(dispatch_summary.get("priority_tree_active")),
                        "external_delay_max_delay_quantile": float(quantile),
                        "conditional_prioritization_model_kind": dispatch_summary.get("model_kind"),
                        "dispatch_policy": dispatch_summary.get("dispatch_policy"),
                        "dataset_dispatch_policy": dataset_dispatch.get("dispatch_policy"),
                        "conditional_prioritization_pairs": dispatch_summary.get("n_competition_pairs"),
                        "conditional_prioritization_overtaking_rate": dispatch_summary.get("overtaking_rate"),
                        "queue_strategy_name": strategy_name,
                        "queue_score_mixing_weights": strategy_weights,
                        "queue_scheduling_force_empirical": queue_scheduling_force_empirical,
                        "external_delay_model_type": "activity_specific_external_delay_timer",
                        "external_delay_tree_enabled": False,
                        "external_delay": {
                            "global_scale": 1.0,
                            "max_delay_quantile": float(quantile),
                            "method": method,
                        },
                        "structure_score": base_candidate.get("structure_score"),
                        "metrics": flat_metrics,
                        "case_duration_guard": duration_guard,
                        "arrival_guard": arrival_guard,
                        "time_decomposition_overlap": self._extract_queue_external_overlap(time_diagnostics),
                        "optuna_objective_score": float(score),
                        "parallel_isolated_eval": False,
                        "candidate_eval_execution_mode": CANDIDATE_EVAL_EXECUTION_MODE,
                        "status": "ok",
                    }
                    if spec.get("sample_rank") is not None:
                        record["confirmed_from_sample_rank"] = int(spec["sample_rank"])
                    records.append(record)
                except Exception as exc:
                    logger.warning(
                        "Dispatch/External candidate failed: dispatch=%s, priority_depth=%s, quantile=%s, error=%s",
                        dispatch_summary.get("dispatch_policy"),
                        priority_depth,
                        quantile,
                        exc,
                    )
                    records.append({
                        "stage": f"dispatch_external_{phase}",
                        "evaluation_phase": phase,
                        "runtime_search_stage": "dispatch_external_search",
                        "trial_number": trial_no,
                        "resource_budget": 1,
                        "noise_threshold": noise_threshold,
                        "tree_depth": int(control_flow_depth),
                        "decision_point_max_depth": int(control_flow_depth),
                        "arrival_decision_tree_max_depth": int(arrival_depth),
                        "queue_scheduling_max_depth": int(resource_depth),
                        "conditional_prioritization_max_depth": int(priority_depth),
                        "external_delay_max_delay_quantile": float(quantile),
                        "conditional_prioritization_model_kind": dispatch_summary.get("model_kind"),
                        "dispatch_policy": dispatch_summary.get("dispatch_policy"),
                        "dataset_dispatch_policy": dataset_dispatch.get("dispatch_policy"),
                        "metrics": {},
                        "case_duration_guard": {},
                        "arrival_guard": {},
                        "optuna_objective_score": float("inf"),
                        "status": "failed",
                        "error": str(exc),
                    })
            return records

        runtime_records: list[dict] = []
        for priority_depth in searched_priority_depths:
            try:
                dispatch_summary = _discover_dispatch_model(train_log, int(priority_depth))
                if dispatch_summary.get("dispatch_policy") != dataset_dispatch.get("dispatch_policy"):
                    logger.info(
                        "Stage 2c sample/full dispatch mismatch at priority_depth=%s: sample=%s, dataset_probe=%s. "
                        "Full confirmation will use the dataset-level policy decision.",
                        priority_depth,
                        dispatch_summary.get("dispatch_policy"),
                        dataset_dispatch.get("dispatch_policy"),
                    )
                phase = "sample_search" if full_confirmation_enabled else "search_cv"
                runtime_records.extend(
                    _evaluate_quantile_batch(
                        evaluation_log=validation_log,
                        priority_depth=int(priority_depth),
                        dispatch_summary=dispatch_summary,
                        quantile_specs=[{"quantile": float(q)} for q in quantile_candidates],
                        phase=phase,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Dispatch discovery failed for priority_depth=%s: %s",
                    priority_depth,
                    exc,
                )
                for quantile in quantile_candidates:
                    candidate_number += 1
                    runtime_records.append({
                        "stage": "dispatch_external_discovery_failed",
                        "evaluation_phase": "sample_search" if full_confirmation_enabled else "search_cv",
                        "runtime_search_stage": "dispatch_external_search",
                        "trial_number": int(candidate_number),
                        "resource_budget": 1,
                        "noise_threshold": noise_threshold,
                        "tree_depth": int(control_flow_depth),
                        "decision_point_max_depth": int(control_flow_depth),
                        "arrival_decision_tree_max_depth": int(arrival_depth),
                        "queue_scheduling_max_depth": int(resource_depth),
                        "conditional_prioritization_max_depth": int(priority_depth),
                        "external_delay_max_delay_quantile": float(quantile),
                        "dataset_dispatch_policy": dataset_dispatch.get("dispatch_policy"),
                        "metrics": {},
                        "case_duration_guard": {},
                        "arrival_guard": {},
                        "optuna_objective_score": float("inf"),
                        "status": "failed",
                        "error": str(exc),
                    })

        def _prefer_dataset_dispatch(records: list[dict]) -> list[dict]:
            ok_rows = [row for row in records if row.get("status") == "ok"]
            preferred = [
                row for row in ok_rows
                if row.get("dispatch_policy") == dataset_dispatch.get("dispatch_policy")
            ]
            return preferred or ok_rows

        sample_selection_pool = _prefer_dataset_dispatch(runtime_records)
        if not sample_selection_pool:
            raise RuntimeError("Stage 2c produced no valid Dispatch/External candidate.")
        sample_selection_result = self._select_runtime_candidate_from_records(
            runtime_records=sample_selection_pool,
            runtime_metrics=joint_metrics,
            runtime_search_stage="dispatch_external_search",
            case_duration_min_ratio=case_duration_min_ratio,
            case_duration_max_ratio=case_duration_max_ratio,
            max_external_delay_ratio=max_external_delay_ratio,
            ctd_tolerance=ctd_tolerance,
            metric_weights={m: float(RUNTIME_METRIC_WEIGHTS.get(m, 1.0)) for m in joint_metrics},
            require_arrival_guard=True,
            require_duration_guard=True,
        )

        confirmation_records: list[dict] = []
        selection_result = sample_selection_result
        final_selection_records = sample_selection_pool

        if full_confirmation_enabled:
            ranked_joint = self._rank_runtime_candidates_for_confirmation(
                runtime_records=runtime_records,
                runtime_metrics=joint_metrics,
                runtime_search_stage="dispatch_external_sample_ranking",
                case_duration_min_ratio=case_duration_min_ratio,
                case_duration_max_ratio=case_duration_max_ratio,
                max_external_delay_ratio=max_external_delay_ratio,
                ctd_tolerance=ctd_tolerance,
                metric_weights={m: float(RUNTIME_METRIC_WEIGHTS.get(m, 1.0)) for m in joint_metrics},
                require_arrival_guard=True,
                require_duration_guard=True,
                top_k=min(
                    full_confirmation_top_k,
                    len(searched_priority_depths) * len(quantile_candidates),
                ),
                identity_keys=(
                    "conditional_prioritization_max_depth",
                    "external_delay_max_delay_quantile",
                ),
            )

            # When CPT is active, preserve at least one quantile candidate for every priority depth.
            # This prevents the sample window from eliminating a depth before full-data confirmation.
            if priority_tree_is_active:
                coverage_candidates: list[dict] = []
                for depth in searched_priority_depths:
                    depth_rows = [
                        row for row in runtime_records
                        if row.get("status") == "ok"
                        and int(row.get("conditional_prioritization_max_depth", -1)) == int(depth)
                    ]
                    if not depth_rows:
                        continue
                    coverage_candidates.extend(
                        self._rank_runtime_candidates_for_confirmation(
                            runtime_records=depth_rows,
                            runtime_metrics=joint_metrics,
                            runtime_search_stage="dispatch_external_priority_depth_coverage",
                            case_duration_min_ratio=case_duration_min_ratio,
                            case_duration_max_ratio=case_duration_max_ratio,
                            max_external_delay_ratio=max_external_delay_ratio,
                            ctd_tolerance=ctd_tolerance,
                            metric_weights={m: float(RUNTIME_METRIC_WEIGHTS.get(m, 1.0)) for m in joint_metrics},
                            require_arrival_guard=True,
                            require_duration_guard=True,
                            top_k=1,
                            identity_keys=(
                                "conditional_prioritization_max_depth",
                                "external_delay_max_delay_quantile",
                            ),
                        )
                    )

                seen_pairs: set[tuple[int, float]] = set()
                deduped: list[dict] = []
                for row in list(ranked_joint) + list(coverage_candidates):
                    pair = (
                        int(row.get("conditional_prioritization_max_depth", priority_probe_depth)),
                        float(row.get("external_delay_max_delay_quantile", 1.0)),
                    )
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    deduped.append(row)
                ranked_joint = deduped

            candidates_by_depth: dict[int, list[dict]] = {}
            for sample_rank, row in enumerate(ranked_joint, start=1):
                item = copy.deepcopy(row)
                item["sample_rank"] = int(sample_rank)
                depth = int(item.get("conditional_prioritization_max_depth", priority_probe_depth))
                candidates_by_depth.setdefault(depth, []).append(item)

            for priority_depth, depth_candidates in sorted(candidates_by_depth.items()):
                try:
                    dispatch_summary = _discover_dispatch_model(
                        confirmation_train_log,
                        int(priority_depth),
                    )
                    specs = [
                        {
                            "quantile": float(row.get("external_delay_max_delay_quantile", 1.0)),
                            "sample_rank": int(row.get("sample_rank", 0)),
                        }
                        for row in depth_candidates
                    ]
                    confirmation_records.extend(
                        _evaluate_quantile_batch(
                            evaluation_log=confirmation_validation_log,
                            priority_depth=int(priority_depth),
                            dispatch_summary=dispatch_summary,
                            quantile_specs=specs,
                            phase="full_confirmation",
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Dispatch/External full confirmation failed for priority_depth=%s: %s",
                        priority_depth,
                        exc,
                    )

            confirmation_selection_pool = _prefer_dataset_dispatch(confirmation_records)
            if confirmation_selection_pool:
                selection_result = self._select_runtime_candidate_from_records(
                    runtime_records=confirmation_selection_pool,
                    runtime_metrics=joint_metrics,
                    runtime_search_stage="dispatch_external_full_confirmation",
                    case_duration_min_ratio=case_duration_min_ratio,
                    case_duration_max_ratio=case_duration_max_ratio,
                    max_external_delay_ratio=max_external_delay_ratio,
                    ctd_tolerance=ctd_tolerance,
                    metric_weights={m: float(RUNTIME_METRIC_WEIGHTS.get(m, 1.0)) for m in joint_metrics},
                    require_arrival_guard=True,
                    require_duration_guard=True,
                )
                final_selection_records = confirmation_selection_pool

        # FCFS means priority-tree depth is not a tunable dimension. Keep the shallow probe value only
        # in the legacy integration field, and select the best External quantile at that single depth.
        if not priority_tree_is_active:
            fcfs_rows = [
                row for row in final_selection_records
                if row.get("status") == "ok" and row.get("dispatch_policy") == "fcfs"
            ]
            if fcfs_rows:
                selection_result = self._select_runtime_candidate_from_records(
                    runtime_records=fcfs_rows,
                    runtime_metrics=joint_metrics,
                    runtime_search_stage="dispatch_external_fcfs_quantile_only",
                    case_duration_min_ratio=case_duration_min_ratio,
                    case_duration_max_ratio=case_duration_max_ratio,
                    max_external_delay_ratio=max_external_delay_ratio,
                    ctd_tolerance=ctd_tolerance,
                    metric_weights={m: float(RUNTIME_METRIC_WEIGHTS.get(m, 1.0)) for m in joint_metrics},
                    require_arrival_guard=True,
                    require_duration_guard=True,
                )
                selection_result["selected"]["priority_tree_max_depth"] = None
                selection_result["selected"]["priority_tree_active"] = False
                selection_result["selected"]["runtime_fallback_reason"] = "dataset_selected_fcfs"

        selected_quantile = float(
            selection_result["selected"].get("external_delay_max_delay_quantile", 1.0)
        )
        self.integration._set_external_delay_runtime_config(
            global_scale=1.0,
            max_delay_quantile=selected_quantile,
        )

        selected_policy = str(
            selection_result["selected"].get("dispatch_policy")
            or dataset_dispatch.get("dispatch_policy")
            or "fcfs"
        )
        selected_priority_depth = int(
            selection_result["selected"].get(
                "conditional_prioritization_max_depth",
                priority_probe_depth,
            )
        )

        summary = dict(selection_result.get("optuna_summary", {}) or {})
        summary.update({
            "enabled": True,
            "search_type": (
                "priority_tree_depth_x_external_quantile"
                if priority_tree_is_active
                else "external_quantile_only_fcfs"
            ),
            "runtime_search_stage": "dispatch_external_search",
            "dispatch_policy_selection": "data_driven",
            "dispatch_policy_probe_source": dispatch_probe_source,
            "dispatch_policy": dataset_dispatch.get("dispatch_policy"),
            "dispatch_model_kind": dataset_dispatch.get("model_kind"),
            "dispatch_competition_pairs": dataset_dispatch.get("n_competition_pairs"),
            "dispatch_overtaking_rate": dataset_dispatch.get("overtaking_rate"),
            "priority_tree_tuned": bool(priority_tree_is_active),
            "priority_tree_depth_candidates_requested": list(priority_tree_depth_candidates),
            "priority_tree_depth_candidates_evaluated": list(searched_priority_depths),
            "external_delay_quantile_candidates": [float(x) for x in quantile_candidates],
            "n_search_candidates": int(len(runtime_records)),
            "large_log_sample_search_full_confirmation": {
                "enabled": bool(full_confirmation_enabled),
                "top_k": int(full_confirmation_top_k) if full_confirmation_enabled else None,
                "sample_candidates": int(len(runtime_records)),
                "full_confirmation_candidates": int(len(confirmation_records)),
                "priority_depth_coverage_safeguard": bool(priority_tree_is_active),
            },
            "fixed_arrival_depth": int(arrival_depth),
            "fixed_resource_allocation_depth": int(resource_depth),
            "selected_dispatch_policy": selected_policy,
            "selected_priority_tree_depth": (
                int(selected_priority_depth) if selected_policy == "conditional_prioritization_tree" else None
            ),
            "selected_external_delay_quantile": float(selected_quantile),
            "global_scale_fixed": 1.0,
            "metrics": list(joint_metrics),
            "note": (
                "Dispatch policy is inferred from the log. Priority-tree depth is tuned only when the dataset "
                "supports pairwise prioritization; FCFS datasets tune External Delay quantile only."
            ),
        })
        selection_result["optuna_summary"] = summary
        selection_result["runtime_records"] = list(runtime_records) + list(confirmation_records)

        logger.info(
            "Stage 2c completed: dispatch_policy=%s, priority_tree_depth=%s, selected_quantile=%s, model_kind=%s",
            selected_policy,
            selected_priority_depth if selected_policy == "conditional_prioritization_tree" else None,
            selected_quantile,
            selection_result["selected"].get("conditional_prioritization_model_kind"),
        )
        return selection_result


    # ------------------------------------------------------------------
    # 运行时候选选择
    # ------------------------------------------------------------------
    def _select_runtime_candidate_from_records(
            self,
            runtime_records: list[dict],
            runtime_metrics: tuple[str, ...],
            runtime_search_stage: str,
            case_duration_min_ratio: float,
            case_duration_max_ratio: float,
            max_external_delay_ratio: float | None,
            ctd_tolerance: float | None,
            metric_weights: dict[str, float] | None = None,
            require_arrival_guard: bool = True,
            require_duration_guard: bool = True,
    ) -> dict:
        """从已有记录池中选出最终运行时候选。

        在紧凑扩展搜索之后使用，使用与单个 Optuna 运行阶段相同的联合 guard 层级，
        对基础候选和扩展候选一起重新排序：
          1) 到达和 duration 都通过；
          2) 到达通过，duration 最接近 1.0；
          3) 没有到达安全候选，最小化联合违规。
        此处不再运行额外的仿真。
        """
        valid_runtime_records = [
            row for row in (runtime_records or [])
            if isinstance(row, dict) and row.get("status") == "ok"
        ]
        if not valid_runtime_records:
            raise RuntimeError(
                "Combined runtime selection received no metric-valid candidate from base or expanded search."
            )

        available_budgets = sorted({int(row.get("resource_budget", 1)) for row in valid_runtime_records})
        final_budget = max(available_budgets) if available_budgets else 1
        final_budget_records = [
            row for row in valid_runtime_records
            if int(row.get("resource_budget", 1)) == int(final_budget)
        ] or valid_runtime_records

        hard_records = self._hard_prune_runtime_records(
            final_budget_records,
            metric_names=list(runtime_metrics),
            min_duration_ratio=case_duration_min_ratio,
            max_duration_ratio=case_duration_max_ratio,
            max_external_delay_ratio=max_external_delay_ratio,
            ctd_tolerance=ctd_tolerance,
            arrival_span_min_ratio=ARRIVAL_SPAN_MIN_RATIO if ARRIVAL_GUARD_ENABLED else None,
            arrival_span_max_ratio=ARRIVAL_SPAN_MAX_RATIO if ARRIVAL_GUARD_ENABLED else None,
            arrival_case_count_min_ratio=ARRIVAL_CASE_COUNT_MIN_RATIO if ARRIVAL_GUARD_ENABLED else None,
        )
        if not hard_records:
            raise RuntimeError(
                "Combined runtime selection found records, but none contained usable runtime metrics."
            )

        selection_records = list(hard_records)
        arrival_passed_records = []
        duration_passed_records = []
        joint_passed_records = []

        for row in hard_records:
            arrival_passed = (
                self._runtime_arrival_guard_passed(row)
                if bool(ARRIVAL_GUARD_ENABLED) and bool(require_arrival_guard)
                else True
            )
            duration_passed = (
                self._runtime_duration_guard_passed(row)
                if bool(require_duration_guard)
                else True
            )
            row["arrival_duration_joint_guard"] = {
                "arrival_passed": bool(arrival_passed),
                "duration_passed": bool(duration_passed),
                "joint_passed": bool(arrival_passed and duration_passed),
                "duration_ratio": self._runtime_duration_ratio(row),
                "duration_closeness": self._runtime_duration_closeness(row),
                "arrival_closeness": self._runtime_arrival_closeness(row),
            }
            if arrival_passed:
                arrival_passed_records.append(row)
            if duration_passed:
                duration_passed_records.append(row)
            if arrival_passed and duration_passed:
                joint_passed_records.append(row)

        if joint_passed_records:
            selection_records = joint_passed_records
            guard_selection_mode = "joint_arrival_duration_guard_passed"
            fallback_reason = None
        elif arrival_passed_records:
            selection_records = arrival_passed_records
            guard_selection_mode = "arrival_guard_passed_duration_closest"
            fallback_reason = "no_joint_safe_candidate_after_compact_expanded_search"
        else:
            guard_selection_mode = "arrival_relaxed_no_passing_candidate"
            fallback_reason = "no_arrival_safe_candidate_after_compact_expanded_search"

        effective_metric_weights = metric_weights or RUNTIME_METRIC_WEIGHTS
        runtime_metric_weights = {
            metric_name: float(effective_metric_weights.get(metric_name, 1.0))
            for metric_name in list(runtime_metrics)
        }

        self._attach_ideal_point_scores(
            selection_records,
            metric_names=list(runtime_metrics),
            metrics_key="metrics",
            score_key="runtime_score",
            metric_weights=runtime_metric_weights,
        )

        guard_soft_weight = float(RUNTIME_GUARD_SOFT_PENALTY_WEIGHT)
        selection_score_key = "runtime_joint_guard_selection_score"
        for row in selection_records:
            try:
                guard_penalty = float(row.get("runtime_guard_soft_penalty", 0.0))
                if not np.isfinite(guard_penalty):
                    guard_penalty = 0.0
            except Exception:
                guard_penalty = 0.0
            if not bool(require_duration_guard):
                # Arrival 阶段禁止通用 duration/RED/CTD guard penalty 反向污染 Arrival depth。
                # Arrival 的失败回退由 arrival_closeness 单独处理。
                guard_penalty = 0.0
            try:
                base_runtime_score = float(row.get("runtime_score", float("inf")))
            except Exception:
                base_runtime_score = float("inf")

            duration_closeness = self._runtime_duration_closeness(row)
            arrival_closeness = self._runtime_arrival_closeness(row)
            row["runtime_score_with_guard_penalty"] = float(
                base_runtime_score + guard_soft_weight * guard_penalty
            )
            row["duration_closeness_to_one"] = float(duration_closeness)
            row["arrival_closeness_to_one"] = float(arrival_closeness)

            if guard_selection_mode == "arrival_guard_passed_duration_closest":
                row[selection_score_key] = float(
                    100.0 * duration_closeness
                    + guard_soft_weight * guard_penalty
                    + 0.05 * base_runtime_score
                )
            elif guard_selection_mode == "arrival_relaxed_no_passing_candidate":
                row[selection_score_key] = float(
                    20.0 * arrival_closeness
                    + 10.0 * duration_closeness
                    + guard_soft_weight * guard_penalty
                    + 0.05 * base_runtime_score
                )
            else:
                row[selection_score_key] = float(row["runtime_score_with_guard_penalty"])

        pareto_records = self._pareto_prune_records(
            selection_records,
            metric_names=list(runtime_metrics),
        )
        final_candidates = self._select_top_records(
            primary_records=pareto_records,
            fallback_records=selection_records,
            top_k=1,
            score_key=selection_score_key,
        )
        if not final_candidates:
            raise RuntimeError("Combined runtime selection failed to select a candidate.")

        selected = final_candidates[0]
        selected["runtime_guard_selection_mode"] = guard_selection_mode
        selected["runtime_guard_soft_penalty_weight"] = guard_soft_weight
        selected["runtime_selection_score_key"] = selection_score_key
        selected["runtime_fallback_reason"] = fallback_reason

        duration_ratios = []
        arrival_ratios = []
        for row in hard_records:
            ratio = self._runtime_duration_ratio(row)
            if ratio is not None:
                duration_ratios.append(float(ratio))
            arrival_diag = (row.get("arrival_diagnostic_guard") or row.get("arrival_guard") or {})
            span = arrival_diag.get("observed_span_ratio", arrival_diag.get("arrival_span_ratio"))
            try:
                span = float(span)
                if np.isfinite(span) and span > 0:
                    arrival_ratios.append(span)
            except Exception:
                pass

        best_duration_ratio = None
        if duration_ratios:
            best_duration_ratio = min(duration_ratios, key=lambda value: abs(np.log(max(value, 1e-9))))
        best_arrival_ratio = None
        if arrival_ratios:
            best_arrival_ratio = min(arrival_ratios, key=lambda value: abs(np.log(max(value, 1e-9))))

        selected_arrival_span_ratio = None
        selected_arrival_diag = (selected.get("arrival_diagnostic_guard") or selected.get("arrival_guard") or {})
        try:
            selected_arrival_span_ratio = float(
                selected_arrival_diag.get(
                    "observed_span_ratio",
                    selected_arrival_diag.get("arrival_span_ratio"),
                )
            )
            if not np.isfinite(selected_arrival_span_ratio):
                selected_arrival_span_ratio = None
        except Exception:
            selected_arrival_span_ratio = None

        optuna_summary = {
            "enabled": True,
            "runtime_search_stage": runtime_search_stage,
            "sampler": "combined_existing_records",
            "pruner": None,
            "n_trials_requested": None,
            "n_trials_finished": int(len(valid_runtime_records)),
            "n_complete_trials": int(len(valid_runtime_records)),
            "n_pruned_trials": int(len([r for r in (runtime_records or []) if r.get("status") == "pruned"])),
            "validation_budgets": [int(final_budget)],
            "final_selection_budget": int(final_budget),
            "external_delay_model_type": "activity_specific_external_delay_timer",
            "external_delay_tree_enabled": False,
            "fixed_external_delay_global_scale": 1.0,
            "fixed_external_delay_max_delay_quantile": (
                None if "external_quantile_joint" in str(runtime_search_stage) else 1.0
            ),
            "metric_weighting": "stage_specific_metric_weights_plus_guard_soft_penalty",
            "runtime_metric_weights": dict(runtime_metric_weights),
            "require_arrival_guard": bool(require_arrival_guard),
            "require_duration_guard": bool(require_duration_guard),
            "runtime_guard_selection_mode": guard_selection_mode,
            "runtime_selection_score_key": selection_score_key,
            "runtime_fallback_reason": fallback_reason,
            "runtime_selection_candidate_counts": {
                "metric_valid": int(len(hard_records)),
                "arrival_passed": int(len(arrival_passed_records)) if bool(ARRIVAL_GUARD_ENABLED) else None,
                "duration_passed": int(len(duration_passed_records)),
                "joint_arrival_duration_passed": int(len(joint_passed_records)),
                "used_for_final_selection": int(len(selection_records)),
            },
            "fallback_diagnostics": {
                "selected_arrival_span_ratio": selected_arrival_span_ratio,
                "selected_duration_ratio": self._runtime_duration_ratio(selected),
                "best_available_duration_ratio": best_duration_ratio,
                "best_available_arrival_span_ratio": best_arrival_ratio,
            },
            "arrival_guard": {
                "enabled": bool(ARRIVAL_GUARD_ENABLED),
                "span_min_ratio": float(ARRIVAL_SPAN_MIN_RATIO),
                "span_max_ratio": float(ARRIVAL_SPAN_MAX_RATIO),
                "case_count_min_ratio": float(ARRIVAL_CASE_COUNT_MIN_RATIO),
            },
            "candidate_resource_filter": self._resource_filter_policy_summary(),
        }

        logger.info(
            "Combined runtime candidate selection completed: mode=%s, counts=%s, "
            "selected_stage=%s, selected_duration_ratio=%s, fallback_reason=%s",
            guard_selection_mode,
            optuna_summary.get("runtime_selection_candidate_counts"),
            selected.get("runtime_search_stage"),
            self._runtime_duration_ratio(selected),
            fallback_reason,
        )

        return {
            "selected": selected,
            "runtime_records": list(runtime_records or []),
            "optuna_summary": optuna_summary,
        }

    # ------------------------------------------------------------------
    # 自适应活动-资源候选过滤
    # ------------------------------------------------------------------
    @staticmethod
    def _count_log_resources(event_log) -> int:
        """统计 EventLog 风格对象或 DataFrame 中的不同资源数量。"""
        resources: set[str] = set()
        resource_keys = ("org:resource", "resource", "Resource")

        try:
            if hasattr(event_log, "columns"):
                for key in resource_keys:
                    if key in event_log.columns:
                        for value in event_log[key].dropna().astype(str).tolist():
                            if value:
                                resources.add(value)
                        return int(len(resources))
        except Exception:
            pass

        try:
            for trace in event_log:
                try:
                    iterator = iter(trace)
                except Exception:
                    continue
                for event in iterator:
                    value = None
                    for key in resource_keys:
                        try:
                            if hasattr(event, "get"):
                                value = event.get(key)
                            elif key in event:
                                value = event[key]
                        except Exception:
                            value = None
                        if value is not None:
                            break
                    if value is not None and str(value):
                        resources.add(str(value))
        except Exception:
            pass

        return int(len(resources))

    @staticmethod
    def _resource_filter_policy_summary() -> dict:
        return {
            "mode": "adaptive_by_resource_count" if RESOURCE_ADAPTIVE_FILTER_ENABLED else "fixed_from_caller",
            "adaptive_enabled": bool(RESOURCE_ADAPTIVE_FILTER_ENABLED),
            "resource_threshold": int(RESOURCE_ADAPTIVE_THRESHOLD),
            "small_resource_policy": {
                "condition": f"resources <= {int(RESOURCE_ADAPTIVE_THRESHOLD)}",
                "enabled": bool(SMALL_RESOURCE_ENABLE_CANDIDATE_FILTER),
                "coverage": float(SMALL_RESOURCE_CANDIDATE_COVERAGE),
                "max_k": SMALL_RESOURCE_CANDIDATE_MAX_K,
                "min_k": int(SMALL_RESOURCE_CANDIDATE_MIN_K),
            },
            "large_resource_policy": {
                "condition": f"resources > {int(RESOURCE_ADAPTIVE_THRESHOLD)}",
                "enabled": bool(LARGE_RESOURCE_ENABLE_CANDIDATE_FILTER),
                "coverage": float(LARGE_RESOURCE_CANDIDATE_COVERAGE),
                "max_k": LARGE_RESOURCE_CANDIDATE_MAX_K,
                "min_k": int(LARGE_RESOURCE_CANDIDATE_MIN_K),
            },
        }

    def _resolve_candidate_resource_filter(
            self,
            event_log,
            enable_candidate_resource_filter: bool | None = None,
            candidate_resource_coverage: float | None = None,
            candidate_resource_max_k: int | None = None,
            candidate_resource_min_k: int | None = None,
    ) -> dict:
        resource_count = self._count_log_resources(event_log)

        if bool(RESOURCE_ADAPTIVE_FILTER_ENABLED):
            if int(resource_count) > int(RESOURCE_ADAPTIVE_THRESHOLD):
                mode = "adaptive_large_resource_filter"
                enabled = bool(LARGE_RESOURCE_ENABLE_CANDIDATE_FILTER)
                coverage = float(LARGE_RESOURCE_CANDIDATE_COVERAGE)
                max_k = LARGE_RESOURCE_CANDIDATE_MAX_K
                min_k = int(LARGE_RESOURCE_CANDIDATE_MIN_K)
            else:
                mode = "adaptive_small_resource_full"
                enabled = bool(SMALL_RESOURCE_ENABLE_CANDIDATE_FILTER)
                coverage = float(SMALL_RESOURCE_CANDIDATE_COVERAGE)
                max_k = SMALL_RESOURCE_CANDIDATE_MAX_K
                min_k = int(SMALL_RESOURCE_CANDIDATE_MIN_K)
        else:
            mode = "fixed_from_caller"
            enabled = bool(enable_candidate_resource_filter)
            coverage = 1.0 if candidate_resource_coverage is None else float(candidate_resource_coverage)
            max_k = candidate_resource_max_k
            min_k = 1 if candidate_resource_min_k is None else int(candidate_resource_min_k)

        if not np.isfinite(float(coverage)) or float(coverage) <= 0:
            coverage = 1.0
        coverage = float(np.clip(float(coverage), 0.0, 1.0))
        if max_k is not None:
            try:
                max_k = max(1, int(max_k))
            except Exception:
                max_k = None
        min_k = max(1, int(min_k))

        config = {
            "mode": mode,
            "resource_count": int(resource_count),
            "threshold": int(RESOURCE_ADAPTIVE_THRESHOLD),
            "enabled": bool(enabled),
            "coverage": float(coverage),
            "max_k": max_k,
            "min_k": int(min_k),
        }

        logger.info(
            "[Adaptive Resource Filter] mode=%s, resources=%s, threshold=%s, enabled=%s, "
            "coverage=%s, max_k=%s, min_k=%s",
            config["mode"],
            config["resource_count"],
            config["threshold"],
            config["enabled"],
            config["coverage"],
            config["max_k"],
            config["min_k"],
        )
        self._last_candidate_resource_filter_config = dict(config)
        return config

    # ------------------------------------------------------------------
    # 候选发现与评估
    # ------------------------------------------------------------------
    def _discover_candidate_model(
            self,
            event_log,
            pnml_filepath,
            noise_threshold,
            max_depth_tree,
            discovery_profile: str = FULL_DISCOVERY_PROFILE,
            build_process_artifacts: bool = False,
            decision_point_max_depth: int | None = None,
            arrival_decision_tree_max_depth: int | None = None,
            queue_scheduling_max_depth: int | None = None,
            conditional_prioritization_max_depth: int | None = None,
            queue_scheduling_force_empirical: bool = False,
            precomputed_df_features=None,
            reuse_current_process_model: bool = False,
            enable_candidate_resource_filter: bool | None = None,
            candidate_resource_coverage: float | None = None,
            candidate_resource_max_k: int | None = None,
            candidate_resource_min_k: int | None = None,
            alignment_timeout_seconds: int | None = None,
            runtime_event_log=None,
    ):
        self.integration.event_log = event_log

        if reuse_current_process_model and self.integration.process_model is not None:
            logger.info("Reusing current Petri net for candidate: noise_threshold=%s", noise_threshold)
        elif pnml_filepath:
            self.integration.import_petri_net_from_pnml(
                pnml_filepath,
                compute_quality=build_process_artifacts,
                generate_visualization=build_process_artifacts,
            )
        else:
            self.integration.discover_process_model(
                noise_threshold=float(noise_threshold),
                compute_quality=build_process_artifacts,
                generate_visualization=build_process_artifacts,
            )

        # All candidates use the complete model.  ``discovery_profile`` is kept
        # only as cache/search metadata; it no longer toggles model components.

        resource_filter_config = self._resolve_candidate_resource_filter(
            event_log=event_log,
            enable_candidate_resource_filter=enable_candidate_resource_filter,
            candidate_resource_coverage=candidate_resource_coverage,
            candidate_resource_max_k=candidate_resource_max_k,
            candidate_resource_min_k=candidate_resource_min_k,
        )

        # ----------------------------------------------------------
        # Speed cache: noise 不同但 Petri 结构相同时，发现阶段结果理论上等价。
        # key 使用当前日志 identity + Petri 结构 + 实际会影响子模型训练的超参数；
        # 不包含 noise_threshold 本身，因此可命中“不同 noise -> 同一结构”的场景。
        # ----------------------------------------------------------
        structure_signature = self._current_process_structure_signature()
        log_cache_key = self._cache_log_key(event_log)
        effective_runtime_log = runtime_event_log if runtime_event_log is not None else event_log
        runtime_log_cache_key = self._cache_log_key(effective_runtime_log)
        model_cache_key = (
            log_cache_key,
            runtime_log_cache_key,
            structure_signature,
            # fallback signature 可能只有 activity-level 信息，此时保留 noise 以避免假碰撞。
            None if str(structure_signature).startswith("petri:") else (float(noise_threshold) if noise_threshold is not None else None),
            str(discovery_profile),
            int(max_depth_tree) if max_depth_tree is not None else None,
            int(decision_point_max_depth) if decision_point_max_depth is not None else None,
            int(arrival_decision_tree_max_depth) if arrival_decision_tree_max_depth is not None else None,
            int(queue_scheduling_max_depth) if queue_scheduling_max_depth is not None else None,
            int(conditional_prioritization_max_depth) if conditional_prioritization_max_depth is not None else None,
            bool(queue_scheduling_force_empirical),
            bool(resource_filter_config["enabled"]),
            float(resource_filter_config["coverage"]),
            resource_filter_config["max_k"],
            int(resource_filter_config["min_k"]),
        )

        if bool(DISCOVERY_STATE_CACHE_ENABLED):
            cached_state = self._discovery_state_cache.get(model_cache_key)
            if cached_state and self._restore_model_state(cached_state, event_log=event_log):
                self._cache_stats["discovery_hits"] += 1
                logger.info(
                    "[Speed Cache] discovery HIT: profile=%s, structure=%s, depths=(cf=%s,arr=%s,res=%s,cpt=%s)",
                    discovery_profile, structure_signature[:10], decision_point_max_depth,
                    arrival_decision_tree_max_depth, queue_scheduling_max_depth, conditional_prioritization_max_depth,
                )
                return
            self._cache_stats["discovery_misses"] += 1

        feature_cache_key = (
            log_cache_key,
            structure_signature,
            None if str(structure_signature).startswith("petri:") else (float(noise_threshold) if noise_threshold is not None else None),
        )
        if (
            precomputed_df_features is None
            and bool(PRECOMPUTED_FEATURE_CACHE_ENABLED)
            and feature_cache_key in self._precomputed_feature_cache
        ):
            try:
                precomputed_df_features = copy.deepcopy(self._precomputed_feature_cache[feature_cache_key])
            except Exception:
                precomputed_df_features = self._precomputed_feature_cache[feature_cache_key]
            self._cache_stats["feature_hits"] += 1
            logger.info("[Speed Cache] precomputed feature HIT: structure=%s", structure_signature[:10])

        discover_kwargs = {
            "max_depth_tree": max_depth_tree,
            "decision_point_max_depth": decision_point_max_depth,
            "arrival_decision_tree_max_depth": arrival_decision_tree_max_depth,
            "case_arrival_max_depth": arrival_decision_tree_max_depth,
            "queue_scheduling_max_depth": queue_scheduling_max_depth,
            "conditional_prioritization_max_depth": conditional_prioritization_max_depth,
            "queue_scheduling_force_empirical": bool(queue_scheduling_force_empirical),
            "external_delay_discovery_max_quantile": 1.0,
            "external_delay_runtime_max_quantile": 1.0,
            "external_delay_min_samples_per_activity": 30,
            "external_delay_max_training_events": 10000,
            "external_delay_max_leaf_nodes": 32,
            "enable_candidate_resource_filter": bool(resource_filter_config["enabled"]),
            "candidate_resource_coverage": float(resource_filter_config["coverage"]),
            "candidate_resource_max_k": resource_filter_config["max_k"],
            "candidate_resource_min_k": int(resource_filter_config["min_k"]),
            "alignment_timeout_seconds": alignment_timeout_seconds,
            "precomputed_df_features": precomputed_df_features,
            "runtime_event_log": runtime_event_log,
            "runtime_recency_modules": (tuple(RUNTIME_RECENCY_MODULES) if runtime_event_log is not None else ()),
        }

        try:
            signature = inspect.signature(self.integration.discover_parameters)
            supported = set(signature.parameters.keys())
            filtered_kwargs = {
                key: value
                for key, value in discover_kwargs.items()
                if key in supported
            }
            dropped_kwargs = sorted(set(discover_kwargs.keys()) - set(filtered_kwargs.keys()))
            if dropped_kwargs:
                logger.info(
                    "Ignoring unsupported drqIntegration.discover_parameters kwargs: %s",
                    dropped_kwargs,
                )
        except Exception:
            filtered_kwargs = discover_kwargs

        self.integration.discover_parameters(**filtered_kwargs)

        if bool(PRECOMPUTED_FEATURE_CACHE_ENABLED) and feature_cache_key not in self._precomputed_feature_cache:
            captured_features = self._capture_precomputed_features()
            if captured_features is not None:
                if len(self._precomputed_feature_cache) >= int(PRECOMPUTED_FEATURE_CACHE_MAX_ENTRIES):
                    try:
                        self._precomputed_feature_cache.pop(next(iter(self._precomputed_feature_cache)))
                    except Exception:
                        self._precomputed_feature_cache.clear()
                self._precomputed_feature_cache[feature_cache_key] = captured_features
                self._cache_stats["feature_stores"] += 1
                logger.info("[Speed Cache] stored precomputed features: structure=%s", structure_signature[:10])

        if bool(DISCOVERY_STATE_CACHE_ENABLED):
            captured_state = self._capture_model_state()
            if captured_state:
                if len(self._discovery_state_cache) >= int(DISCOVERY_STATE_CACHE_MAX_ENTRIES):
                    try:
                        self._discovery_state_cache.pop(next(iter(self._discovery_state_cache)))
                    except Exception:
                        self._discovery_state_cache.clear()
                self._discovery_state_cache[model_cache_key] = captured_state
                logger.info(
                    "[Speed Cache] stored discovery state: profile=%s, structure=%s",
                    discovery_profile, structure_signature[:10],
                )

    def _evaluate(
            self,
            event_log,
            n_runs: int,
            metrics_labels: list[str],
            enable_time_diagnostics: bool = True,
    ) -> dict:
        n_runs = max(1, int(n_runs))

        if n_runs > 1:
            return self.integration.evaluate_on_event_log_repeated(
                event_log,
                n_runs=n_runs,
                set_as_current_metrics=False,
                metrics_labels=metrics_labels,
                enable_time_diagnostics=enable_time_diagnostics,
            )

        return self.integration.evaluate_on_event_log(
            event_log,
            set_as_current_metrics=False,
            metrics_labels=metrics_labels,
            enable_time_diagnostics=enable_time_diagnostics,
        )

    def _set_arrival_runtime_config(
            self,
            arrival_time_scale: float = 1.0,
            arrival_demand_multiplier: float = 1.0,
    ):
        """将运行时到达控制同时应用到 legacy 和 modular 两种参数视图。"""
        if self.integration.drq_params is None:
            return

        try:
            arrival_time_scale = float(arrival_time_scale)
            arrival_demand_multiplier = float(arrival_demand_multiplier)
        except Exception:
            arrival_time_scale = 1.0
            arrival_demand_multiplier = 1.0

        if not np.isfinite(arrival_time_scale) or arrival_time_scale <= 0:
            arrival_time_scale = 1.0
        if not np.isfinite(arrival_demand_multiplier) or arrival_demand_multiplier <= 0:
            arrival_demand_multiplier = 1.0

        arrival_params = self.integration.drq_params.setdefault("arrival_params", {})
        arrival_params["arrival_time_scale"] = float(arrival_time_scale)
        arrival_params["arrival_demand_multiplier"] = float(arrival_demand_multiplier)

        modular = self.integration.drq_params.get("modular_params", {})
        if isinstance(modular, dict):
            arrival_module = modular.get("modules", {}).get("arrival")
            if isinstance(arrival_module, dict):
                arrival_module["model_type"] = "case_arrival_decision_tree_interarrival"
                arrival_module["arrival_time_scale"] = float(arrival_time_scale)
                arrival_module["arrival_demand_multiplier"] = float(arrival_demand_multiplier)

    def _compute_case_duration_calibration(self, selected_record: dict) -> dict:
        """case-level duration 校准已移除；外部延迟由活动级缩放校准处理。"""
        return {
            "enabled": False,
            "factor": 1.0,
            "reason": "removed_in_favor_of_top_activity_scale_calibration",
        }

    @staticmethod
    def _finite_population_sample_size(
            population_size: int,
            confidence_level: float = FINAL_CALIBRATION_CONFIDENCE_LEVEL,
            margin_percent: float = FINAL_CALIBRATION_MARGIN_PERCENT,
    ) -> int:
        """
        计算 Stage 3b 的 DSIM 风格有限总体 case 样本量。

        公式：

            n0 = Z^2 * p * (1-p) / e^2

            n = n0 / (1 + (n0 - 1) / N)

        其中：
            N = 总 case 数
            Z = 置信水平对应 Z 值
            p = 0.5
            e = margin

        p=0.5 对应最大方差，因此属于比较保守的样本量设置。

        注意：
        95% confidence 和 ±3% margin
        这里只用于统一确定 Stage 3b 的计算预算。

        它不意味着 RED、CTD、CED、CWD
        本身具有“95%置信区间 ±3%”。
        """

        try:
            population_size = int(population_size)
        except Exception:
            population_size = 0

        if population_size <= 0:
            return 0

        z_map = {
            50.0: 0.67,
            68.0: 0.99,
            90.0: 1.64,
            95.0: 1.96,
            99.0: 2.57,
        }

        try:
            level = float(confidence_level)
        except Exception:
            level = 95.0

        if level not in z_map:
            level = 95.0

        z = float(z_map[level])

        try:
            e = float(margin_percent) / 100.0
        except Exception:
            e = 0.03

        if not np.isfinite(e) or e <= 0:
            e = 0.03

        # 最大方差的保守设置
        p = 0.5

        # 无限总体样本量
        n0 = (
                (z ** 2)
                * p
                * (1.0 - p)
                / (e ** 2)
        )

        # 有限总体修正
        n = n0 / (
                1.0
                + (
                        (n0 - 1.0)
                        / float(population_size)
                )
        )

        return int(
            min(
                population_size,
                max(
                    1,
                    int(np.ceil(n)),
                ),
            )
        )

    def _prepare_final_calibration_log(
            self,
            train_validation_log,
    ):
        """
        为 Stage 3b 构造一次性的固定 case-level 校准日志。

        统一协议：

        1. N_cases <= FINAL_CALIBRATION_LARGE_LOG_THRESHOLD
           → 不抽样，返回完整 train_validation_log。

        2. N_cases > FINAL_CALIBRATION_LARGE_LOG_THRESHOLD
           → 取完整 case 的连续时间窗口，避免随机散点破坏 inter-arrival。

        3. 样本量至少使用大日志 tuning 的目标规模，并保留有限总体公式作为下限参考。

        4. 窗口选择是确定性的。

        5. 这个方法只在 Stage 3b 入口调用一次。
           因此 baseline 与后面的所有 activity × scale
           都会使用完全相同的 case 子集。

        6. 抽的是完整 case，不抽单个 event。

        7. 如果抽样失败，则回退到完整 train_validation_log。
        """

        info = {
            "enabled": bool(
                FINAL_CALIBRATION_SAMPLING_ENABLED
            ),

            "applied": False,

            "policy":
                "center_temporal_window_case_sampling",

            "large_log_threshold": int(
                FINAL_CALIBRATION_LARGE_LOG_THRESHOLD
            ),

            "max_cases": int(
                FINAL_CALIBRATION_MAX_CASES
            ),

            "confidence_level": float(
                FINAL_CALIBRATION_CONFIDENCE_LEVEL
            ),

            "margin_percent": float(
                FINAL_CALIBRATION_MARGIN_PERCENT
            ),

            "seed": int(
                FINAL_CALIBRATION_SAMPLE_SEED
            ),

            "sample_size_method":
                "finite_population_dsim_style",
        }

        # ==========================================================
        # 抽样功能关闭
        # ==========================================================
        if not bool(
                FINAL_CALIBRATION_SAMPLING_ENABLED
        ):
            info["reason"] = "disabled_by_config"

            return (
                train_validation_log,
                info,
            )

        try:
            from drqsim.utils.log_partitioner import (
                _to_dataframe,
                _to_event_log,
                _get_ordered_case_ids,
                _filter_by_case_ids,
            )

            import pandas as _pd

            case_col = "case:concept:name"
            start_col = "start:timestamp"
            end_col = "time:timestamp"

            # ======================================================
            # EventLog -> DataFrame
            # ======================================================
            df = _to_dataframe(
                train_validation_log
            ).copy()

            if case_col not in df.columns:
                raise KeyError(
                    f"missing case id column: {case_col}"
                )

            # case ID 统一转字符串
            df[case_col] = (
                df[case_col]
                    .astype(str)
            )

            # ======================================================
            # 按 case 的真实时间顺序构造抽样池。
            # 随机抽散落 case 会制造虚假的长 inter-arrival，因此这里改为
            # 一个连续的中心时间窗口。
            # ======================================================
            if start_col not in df.columns:
                if end_col not in df.columns:
                    raise KeyError("missing start/end timestamp columns")
                df[start_col] = df[end_col]
            df[start_col] = _pd.to_datetime(df[start_col], utc=True, errors="coerce")
            df[end_col] = _pd.to_datetime(df[end_col], utc=True, errors="coerce")
            df = df.dropna(subset=[case_col, start_col, end_col]).copy()
            case_ids = list(_get_ordered_case_ids(
                df=df,
                case_id_col=case_col,
                start_time_col=start_col,
                end_time_col=end_col,
                sort_temporally=True,
            ))
            n_cases = int(len(case_ids))

            info[
                "population_cases"
            ] = int(n_cases)

            # ======================================================
            # 空日志保护
            # ======================================================
            if n_cases <= 0:
                info["sampled_cases"] = 0
                info["sampling_fraction"] = 1.0
                info["reason"] = "empty_log"

                return (
                    train_validation_log,
                    info,
                )

            threshold = max(
                1,
                int(
                    FINAL_CALIBRATION_LARGE_LOG_THRESHOLD
                ),
            )

            # ======================================================
            # 小/中型日志：
            #
            # 完全保持你原来的 Stage 3b 行为。
            # ======================================================
            if n_cases <= threshold:
                info[
                    "sampled_cases"
                ] = int(n_cases)

                info[
                    "sampling_fraction"
                ] = 1.0

                info[
                    "reason"
                ] = (
                    "population_not_larger_than_"
                    "uniform_threshold"
                )

                return (
                    train_validation_log,
                    info,
                )

            # ======================================================
            # 大日志：
            # 根据有限总体公式计算 case 样本量
            # ======================================================
            statistical_n = (
                self._finite_population_sample_size(
                    population_size=n_cases,

                    confidence_level=(
                        FINAL_CALIBRATION_CONFIDENCE_LEVEL
                    ),

                    margin_percent=(
                        FINAL_CALIBRATION_MARGIN_PERCENT
                    ),
                )
            )

            max_cases = max(
                1,
                int(
                    FINAL_CALIBRATION_MAX_CASES
                ),
            )

            stability_floor = min(n_cases, int(TUNING_SAMPLE_TOTAL_CASES))
            sample_n = min(
                n_cases,
                max_cases,
                max(
                    1,
                    int(statistical_n),
                    int(stability_floor),
                ),
            )

            # ======================================================
            # 连续中心时间窗口：保留完整 case 和真实相邻到达间隔。
            # ======================================================
            start_idx = max(0, (n_cases - int(sample_n)) // 2)
            sampled_case_ids = case_ids[start_idx:start_idx + int(sample_n)]
            sampled_df = _filter_by_case_ids(
                df, sampled_case_ids, case_col, start_col, end_col
            )

            # ======================================================
            # 保持时间顺序
            # ======================================================
            sort_cols = []

            if start_col in sampled_df.columns:
                sampled_df[start_col] = (
                    _pd.to_datetime(
                        sampled_df[start_col],
                        utc=True,
                        errors="coerce",
                    )
                )

                sort_cols.append(
                    start_col
                )

            if end_col in sampled_df.columns:
                sampled_df[end_col] = (
                    _pd.to_datetime(
                        sampled_df[end_col],
                        utc=True,
                        errors="coerce",
                    )
                )

                sort_cols.append(
                    end_col
                )

            if sort_cols:
                sampled_df = (
                    sampled_df.sort_values(
                        sort_cols,
                        kind="mergesort",
                    )
                )

            # ======================================================
            # DataFrame -> EventLog
            # ======================================================
            calibration_log = (
                _to_event_log(
                    sampled_df,
                    case_col,
                )
            )

            info.update({
                "applied": True,

                "statistical_sample_size":
                    int(statistical_n),

                "sampled_cases":
                    int(sample_n),

                "sampling_fraction":
                    float(
                        sample_n
                        / max(
                            n_cases,
                            1,
                        )
                    ),

                "reason":
                    "population_larger_than_threshold_temporal_window",
                "window_side": "center",
            })

            logger.info(
                "Stage 3b temporal-window case sampling applied: "
                "population_cases=%s, "
                "sampled_cases=%s, "
                "threshold=%s, "
                "confidence=%.1f%%, "
                "margin=%.1f%%, "
                "seed=%s",

                n_cases,
                sample_n,
                threshold,

                float(
                    FINAL_CALIBRATION_CONFIDENCE_LEVEL
                ),

                float(
                    FINAL_CALIBRATION_MARGIN_PERCENT
                ),

                int(
                    FINAL_CALIBRATION_SAMPLE_SEED
                ),
            )

            return (
                calibration_log,
                info,
            )

        except Exception as exc:

            # ======================================================
            # 出错时回退完整日志
            # ======================================================
            info[
                "reason"
            ] = (
                "sampling_failed_fallback_to_full_log"
            )

            info[
                "error"
            ] = str(exc)

            logger.warning(
                "Stage 3b calibration sampling failed; "
                "falling back to full "
                "train_validation_log: %s",
                exc,
            )

            return (
                train_validation_log,
                info,
            )

    def _calibrate_top_activity_scales_after_rediscovery(
            self,
            search_log,
            confirmation_log,
            selected_quantile: float,
    ) -> dict:
        """
        固定 Stage 2c 已选出的 External quantile，仅校准 Top-K activity scales。

        当前协议：
        - selected runtime-recency/recent window 用于候选搜索；
        - global scale 始终固定为 1.0；
        - search_log 很大时仍可使用固定 case-level sample 加速搜索；
        - 最终 baseline 与候选回到“完整 recent/runtime window”重复确认；
        - 不再使用完整历史 train_validation_log 做最终 veto；
        - scale 大小本身不构成拒绝条件，4x/5x 等候选只按 recent-window score 判断；
        - test_log 从不参与选择。

        confirmation_log 正常应与 search_log 指向同一个 selected recent/runtime window。
        之所以保留独立参数，是为了在 search 使用抽样时仍能在完整 recent window 上确认。
        """
        if confirmation_log is None:
            confirmation_log = search_log

        result = {
            "enabled": bool(TOP_ACTIVITY_SCALE_CALIBRATION_ENABLED),
            "selected_quantile": float(selected_quantile),
            "selected_activity_scales": {},
            "uses_test_log": False,
            "global_scale_fixed": 1.0,
            "calibration_sampling": {},
            "recent_validation_confirmation": {},
            # 兼容旧结果消费者，同时明确 full-history veto 已关闭。
            "full_log_confirmation": {
                "status": "disabled",
                "reason": "replaced_by_recent_validation_confirmation",
                "accepted": None,
            },
            "search_cases": int(self._count_log_cases(search_log)),
            "confirmation_cases": int(self._count_log_cases(confirmation_log)),
            "confirmation_scope": "selected_runtime_recency_log",
            "confirmation_runs": int(FINAL_CALIBRATION_CONFIRMATION_RUNS),
            "large_scale_magnitude_veto": False,
        }
        if not bool(TOP_ACTIVITY_SCALE_CALIBRATION_ENABLED):
            result["skipped_reason"] = "disabled_by_config"
            return result

        params = getattr(self.integration, "drq_params", {}) or {}
        ext = params.get("external_delay_params", {}) if isinstance(params, dict) else {}
        if not isinstance(ext, dict):
            result["enabled"] = False
            result["skipped_reason"] = "external_delay_disabled"
            return result

        # 固定 Stage 2c 选出的 q；Stage 3b 只搜索 activity-level scales。
        self.integration._set_external_delay_runtime_config(
            global_scale=1.0,
            max_delay_quantile=float(selected_quantile),
        )
        try:
            self.integration._set_external_delay_activity_scales({}, reset_others=True)
        except Exception:
            pass

        # ----------------------------------------------------------
        # Search: selected recent/runtime window（必要时再做 case-level sample）
        # ----------------------------------------------------------
        calibration_log, sampling_info = self._prepare_final_calibration_log(search_log)
        sampling_info = dict(sampling_info or {})
        sampling_info["search_source_cases"] = int(self._count_log_cases(search_log))
        sampling_info["confirmation_source_cases"] = int(self._count_log_cases(confirmation_log))
        sampling_info["confirmation_scope"] = "selected_runtime_recency_log"
        sampling_info["full_history_veto"] = False
        result["calibration_sampling"] = sampling_info

        metrics_labels = list(FINAL_CALIBRATION_METRICS)
        search_baseline_eval = self._evaluate_final_calibration_candidate(
            event_log=calibration_log,
            metrics_labels=metrics_labels,
        )
        result["search_baseline"] = search_baseline_eval

        activity_calibration = self._calibrate_top_external_activity_scales(
            calibration_log=calibration_log,
            metrics_labels=metrics_labels,
            base_eval=search_baseline_eval,
        )
        result["top_activity_scale_calibration"] = activity_calibration
        candidate_scales = dict(
            (activity_calibration or {}).get("selected_activity_scales", {}) or {}
        )
        result["candidate_selected_activity_scales"] = dict(candidate_scales)

        # 若 recent confirmation 显式关闭，直接采用 recent search 结果。
        if not bool(FINAL_CALIBRATION_RECENT_CONFIRMATION_ENABLED):
            self.integration._set_external_delay_activity_scales(
                candidate_scales, reset_others=True
            )
            result["selected_activity_scales"] = candidate_scales
            result["recent_validation_confirmation"] = {
                "status": "not_required",
                "reason": "disabled_by_config",
                "accepted": True,
                "candidate_scales": dict(candidate_scales),
            }
            return result

        # ----------------------------------------------------------
        # Stable recent confirmation:
        # baseline 与 search 选出的 candidate 都在完整 recent/runtime window
        # 上重复评价。这里没有 full-history veto，也没有针对“大 scale”的额外门槛。
        # ----------------------------------------------------------
        confirmation_runs = max(1, int(FINAL_CALIBRATION_CONFIRMATION_RUNS))
        try:
            self.integration._set_external_delay_activity_scales({}, reset_others=True)
            recent_baseline_eval = self._evaluate_final_calibration_candidate_repeated(
                event_log=confirmation_log,
                metrics_labels=metrics_labels,
                n_runs=confirmation_runs,
            )

            self.integration._set_external_delay_activity_scales(
                candidate_scales, reset_others=True
            )
            recent_selected_eval = self._evaluate_final_calibration_candidate_repeated(
                event_log=confirmation_log,
                metrics_labels=metrics_labels,
                n_runs=confirmation_runs,
            )

            baseline_score = float(recent_baseline_eval.get("score", float("inf")))
            selected_score = float(recent_selected_eval.get("score", float("inf")))
            required_score = baseline_score * (
                1.0 - float(TOP_ACTIVITY_SCALE_MIN_IMPROVEMENT)
            )
            accepted = bool(
                np.isfinite(selected_score)
                and (not np.isfinite(baseline_score) or selected_score <= required_score)
            )

            metric_deltas = {}
            baseline_metrics = recent_baseline_eval.get("metrics", {}) or {}
            selected_metrics = recent_selected_eval.get("metrics", {}) or {}
            for metric_name in metrics_labels:
                b = baseline_metrics.get(metric_name)
                c = selected_metrics.get(metric_name)
                if not (
                    isinstance(b, (int, float, np.integer, np.floating))
                    and isinstance(c, (int, float, np.integer, np.floating))
                    and np.isfinite(float(b))
                    and np.isfinite(float(c))
                ):
                    continue
                b = float(b)
                c = float(c)
                metric_deltas[metric_name] = {
                    "baseline": b,
                    "candidate": c,
                    "delta": c - b,
                    "relative_delta": ((c - b) / b) if abs(b) > 1e-12 else None,
                }

            result["recent_validation_confirmation"] = {
                "status": "ok",
                "accepted": accepted,
                "scope": "selected_runtime_recency_log",
                "n_runs": int(confirmation_runs),
                "cases": int(self._count_log_cases(confirmation_log)),
                "baseline_score": baseline_score,
                "selected_score": selected_score,
                "required_score": required_score,
                "selected_quantile": float(selected_quantile),
                "candidate_scales": dict(candidate_scales),
                "baseline_eval": recent_baseline_eval,
                "selected_eval": recent_selected_eval,
                "metric_deltas": metric_deltas,
                "large_scale_magnitude_veto": False,
            }

            logger.info(
                "[Stage3b Recent Confirm] runs=%s, recent_cases=%s, baseline_score=%s, "
                "candidate_score=%s, required_score=%s, accepted=%s, scales=%s",
                confirmation_runs,
                int(self._count_log_cases(confirmation_log)),
                baseline_score,
                selected_score,
                required_score,
                accepted,
                candidate_scales,
            )
            logger.info(
                "[Stage3b Recent Confirm] metric_deltas=%s",
                metric_deltas,
            )

            if accepted:
                result["selected_activity_scales"] = candidate_scales
            else:
                # 仅 recent confirmation 可以否决；完整历史不再参与。
                self.integration._set_external_delay_activity_scales({}, reset_others=True)
                result["selected_activity_scales"] = {}
                result["recent_validation_confirmation"]["rollback_to_unit_scales"] = True
        except Exception as exc:
            # 搜索结果已经在 recent window 上完成评价；确认阶段如果只是技术失败，
            # 不应因为失败而把大 scale 候选无条件回滚掉。
            logger.warning(
                "Top-K activity-scale recent confirmation failed; "
                "keeping recent-search candidate scales: %s",
                exc,
            )
            try:
                self.integration._set_external_delay_activity_scales(
                    candidate_scales, reset_others=True
                )
            except Exception:
                pass
            result["selected_activity_scales"] = candidate_scales
            result["recent_validation_confirmation"] = {
                "status": "failed",
                "accepted": True,
                "fallback": "keep_recent_search_candidate",
                "candidate_scales": dict(candidate_scales),
                "error": str(exc),
            }

        return result

    def _calibrate_final_model_after_rediscovery(
            self,
            train_validation_log,
            selected_hyperparameters: dict,
            quantile_candidates=None,
    ) -> dict:
        """Stage 3a/3b：最终模型上的 External Delay 自动校准。

        统一协议：
        1. discovery 始终保留完整 residual 分布；
        2. Stage 3a 仅搜索 runtime tail quantile；
        3. Stage 3b 固定 quantile 后搜索 Top-K activity scales；
        4. N_cases > 5000 时只对搜索使用固定 case-level sample；
        5. 大日志最终候选必须在完整 train_validation_log 上确认；
        6. test_log 从不参与任何参数选择。
        """
        quantile_candidates = self._normalize_quantile_candidates(
            quantile_candidates or EXTERNAL_DELAY_QUANTILE_CANDIDATES
        )

        result = {
            "enabled": bool(FINAL_REDISCOVERY_CALIBRATION_ENABLED),
            "partition": "train_validation_log",
            "uses_test_log": False,
            "calibration_sampling": {
                "enabled": bool(FINAL_CALIBRATION_SAMPLING_ENABLED),
                "applied": False,
                "reason": "not_prepared",
            },
            "global_scale_calibration": {
                "enabled": False,
                "fixed_global_scale": 1.0,
                "reason": "kept_fixed_for_identifiability",
            },
            "quantile_calibration": {},
            "top_activity_scale_calibration": {},
            "selected_quantile": 1.0,
            "selected_activity_scales": {},
            "candidate_selected_activity_scales": {},
            "baseline": {},
            "search_baseline": {},
            "full_log_confirmation": {},
            "final_global_scale": 1.0,
        }

        if not bool(FINAL_REDISCOVERY_CALIBRATION_ENABLED):
            result["skipped_reason"] = "disabled_by_config"
            return result

        params = getattr(self.integration, "drq_params", {}) or {}
        ext = params.get("external_delay_params", {}) if isinstance(params, dict) else {}
        if not isinstance(ext, dict):
            result["enabled"] = False
            result["skipped_reason"] = "external_delay_disabled"
            return result

        metrics_labels = list(FINAL_CALIBRATION_METRICS)
        calibration_log, calibration_sampling = self._prepare_final_calibration_log(
            train_validation_log
        )
        calibration_sampling = dict(calibration_sampling or {})
        result["calibration_sampling"] = calibration_sampling
        sampling_applied = bool(calibration_sampling.get("applied", False))
        result["partition"] = (
            "sampled_train_validation_log" if sampling_applied else "train_validation_log"
        )

        # Baseline 必须是完整 tail + unit activity scales。
        try:
            self.integration._set_external_delay_activity_scales({}, reset_others=True)
        except Exception:
            pass
        self.integration._set_external_delay_runtime_config(
            global_scale=1.0,
            max_delay_quantile=1.0,
        )

        full_baseline_eval = self._evaluate_final_calibration_candidate(
            event_log=train_validation_log,
            metrics_labels=metrics_labels,
        )
        result["baseline"] = full_baseline_eval

        if sampling_applied:
            search_baseline_eval = self._evaluate_final_calibration_candidate(
                event_log=calibration_log,
                metrics_labels=metrics_labels,
            )
        else:
            search_baseline_eval = full_baseline_eval
        result["search_baseline"] = search_baseline_eval

        # ----------------------------------------------------------
        # Stage 3a：External Delay runtime tail quantile
        # ----------------------------------------------------------
        quantile_calibration = self._calibrate_external_delay_quantile(
            calibration_log=calibration_log,
            metrics_labels=metrics_labels,
            base_eval=search_baseline_eval,
            quantile_candidates=quantile_candidates,
        )
        result["quantile_calibration"] = quantile_calibration
        selected_quantile = float(quantile_calibration.get("selected_quantile", 1.0))
        result["selected_quantile"] = selected_quantile
        self.integration._set_external_delay_runtime_config(
            global_scale=1.0,
            max_delay_quantile=selected_quantile,
        )
        post_quantile_eval = quantile_calibration.get("selected_eval") or search_baseline_eval
        result["post_quantile"] = post_quantile_eval

        # ----------------------------------------------------------
        # Stage 3b：固定 quantile 后做 Top-K activity scale
        # ----------------------------------------------------------
        if bool(TOP_ACTIVITY_SCALE_CALIBRATION_ENABLED):
            activity_calibration = self._calibrate_top_external_activity_scales(
                calibration_log=calibration_log,
                metrics_labels=metrics_labels,
                base_eval=post_quantile_eval,
            )
            candidate_selected_scales = dict(
                activity_calibration.get("selected_activity_scales", {}) or {}
            )
        else:
            activity_calibration = {
                "enabled": False,
                "skipped_reason": "disabled_by_config",
                "selected_activity_scales": {},
            }
            candidate_selected_scales = {}

        result["top_activity_scale_calibration"] = activity_calibration
        result["candidate_selected_activity_scales"] = dict(candidate_selected_scales)
        result["selected_activity_scales"] = dict(candidate_selected_scales)
        result["post_activity_scale"] = (
            activity_calibration.get("selected_eval") or post_quantile_eval
        )

        # 小/中型日志：所有 candidate 已经在完整 train_validation 上评价。
        if not sampling_applied:
            result["full_log_confirmation"] = {
                "status": "not_required",
                "reason": "calibration_sampling_not_applied",
                "accepted": True,
                "selected_quantile": selected_quantile,
                "selected_activity_scales": dict(candidate_selected_scales),
            }
            return result

        # 大日志若显式关闭 confirmation，则直接采用 sample 候选。
        if not bool(FINAL_CALIBRATION_FULL_CONFIRMATION_ENABLED):
            result["full_log_confirmation"] = {
                "status": "not_required",
                "reason": "disabled_by_config",
                "accepted": True,
                "selected_quantile": selected_quantile,
                "selected_activity_scales": dict(candidate_selected_scales),
            }
            return result

        # ----------------------------------------------------------
        # Full-log confirmation：同时确认 quantile + activity scales。
        # ----------------------------------------------------------
        try:
            self.integration._set_external_delay_runtime_config(
                global_scale=1.0,
                max_delay_quantile=selected_quantile,
            )
            self.integration._set_external_delay_activity_scales(
                candidate_selected_scales,
                reset_others=True,
            )
            full_selected_eval = self._evaluate_final_calibration_candidate(
                event_log=train_validation_log,
                metrics_labels=metrics_labels,
            )

            full_baseline_score = float(
                (full_baseline_eval or {}).get("score", float("inf"))
            )
            full_selected_score = float(
                (full_selected_eval or {}).get("score", float("inf"))
            )
            required_score = full_baseline_score * (
                1.0 - float(TOP_ACTIVITY_SCALE_MIN_IMPROVEMENT)
            )
            accepted = bool(
                np.isfinite(full_selected_score)
                and (
                    not np.isfinite(full_baseline_score)
                    or full_selected_score <= required_score
                )
            )

            result["full_log_confirmation"] = {
                "status": "ok",
                "accepted": accepted,
                "baseline_score": full_baseline_score,
                "selected_score": full_selected_score,
                "required_score": required_score,
                "selected_quantile": selected_quantile,
                "candidate_scales": dict(candidate_selected_scales),
                "selected_eval": full_selected_eval,
            }

            if accepted:
                logger.info(
                    "Stage 3a/3b full-log confirmation accepted: q=%s, scales=%s, "
                    "baseline_score=%s, selected_score=%s",
                    selected_quantile,
                    candidate_selected_scales,
                    full_baseline_score,
                    full_selected_score,
                )
            else:
                # 保守回退：完整 tail + unit scales。
                self.integration._set_external_delay_runtime_config(
                    global_scale=1.0,
                    max_delay_quantile=1.0,
                )
                self.integration._set_external_delay_activity_scales(
                    {},
                    reset_others=True,
                )
                result["selected_quantile"] = 1.0
                result["selected_activity_scales"] = {}
                result["full_log_confirmation"]["rollback_to_baseline"] = True
                logger.info(
                    "Stage 3a/3b full-log confirmation rejected candidate; "
                    "rolling back to q=1.0 and unit activity scales."
                )
        except Exception as exc:
            logger.warning(
                "Stage 3a/3b full-log confirmation failed; rolling back to baseline: %s",
                exc,
            )
            try:
                self.integration._set_external_delay_runtime_config(
                    global_scale=1.0,
                    max_delay_quantile=1.0,
                )
                self.integration._set_external_delay_activity_scales(
                    {},
                    reset_others=True,
                )
            except Exception:
                pass
            result["selected_quantile"] = 1.0
            result["selected_activity_scales"] = {}
            result["full_log_confirmation"] = {
                "status": "failed",
                "accepted": False,
                "rollback_to_baseline": True,
                "error": str(exc),
            }

        return result

    def _calibrate_external_delay_quantile(
            self,
            calibration_log,
            metrics_labels: list[str],
            base_eval: dict,
            quantile_candidates,
    ) -> dict:
        """Stage 3a：在固定 global_scale=1 下选择 runtime tail quantile。"""
        candidates = self._normalize_quantile_candidates(quantile_candidates)
        try:
            self.integration._set_external_delay_activity_scales({}, reset_others=True)
        except Exception:
            pass

        base_eval = dict(base_eval or {})
        base_eval.setdefault(
            "score",
            self._final_calibration_score(
                base_eval.get("metrics", {}) or {},
                base_eval.get("case_duration_guard", {}) or {},
            ),
        )

        results = []
        best_quantile = 1.0
        best_eval = base_eval
        best_score = float(base_eval.get("score", float("inf")))

        for quantile in candidates:
            quantile = float(quantile)
            try:
                self.integration._set_external_delay_runtime_config(
                    global_scale=1.0,
                    max_delay_quantile=quantile,
                )
                if abs(quantile - 1.0) <= 1e-12:
                    trial_eval = base_eval
                else:
                    trial_eval = self._evaluate_final_calibration_candidate(
                        event_log=calibration_log,
                        metrics_labels=list(metrics_labels),
                    )
            except Exception as exc:
                trial_eval = {
                    "metrics": {},
                    "case_duration_guard": {},
                    "score": float("inf"),
                    "error": str(exc),
                }

            score = float(trial_eval.get("score", float("inf")))
            results.append({
                "candidate_quantile": quantile,
                "metrics": trial_eval.get("metrics", {}),
                "case_duration_guard": trial_eval.get("case_duration_guard", {}),
                "score": score,
                "time_decomposition_overlap": trial_eval.get("time_decomposition_overlap", {}),
                "error": trial_eval.get("error"),
            })

            # 最小 score；近似并列时优先较大的 quantile，避免无必要截尾。
            if (
                score < best_score - 1e-12
                or (abs(score - best_score) <= 1e-12 and quantile > best_quantile)
            ):
                best_quantile = quantile
                best_eval = trial_eval
                best_score = score

        self.integration._set_external_delay_runtime_config(
            global_scale=1.0,
            max_delay_quantile=best_quantile,
        )
        return {
            "enabled": True,
            "candidate_quantiles": [float(x) for x in candidates],
            "selected_quantile": float(best_quantile),
            "baseline_quantile": 1.0,
            "baseline_score": float(base_eval.get("score", float("inf"))),
            "selected_score": float(best_score),
            "selected_eval": best_eval,
            "results": results,
            "tie_breaker": "prefer_larger_quantile_when_scores_equal",
            "note": "Only runtime tail truncation is optimized; discovery retains the full residual distribution.",
        }

    def _evaluate_final_calibration_candidate(
            self,
            event_log,
            metrics_labels: list[str],
    ) -> dict:
        metrics = self._evaluate(
            event_log=event_log,
            n_runs=1,
            metrics_labels=list(metrics_labels),
            enable_time_diagnostics=True,
        )
        flat_metrics = self.integration._flatten_formatted_metrics(metrics)
        time_diagnostics = getattr(self.integration, "_last_time_diagnostics", {}) or {}
        duration_guard = self.integration._extract_case_duration_guard(time_diagnostics)
        external_delay_ratio = self._extract_external_delay_ratio(time_diagnostics)
        if external_delay_ratio is not None:
            duration_guard["external_delay_ratio"] = float(external_delay_ratio)
        return {
            "metrics": flat_metrics,
            "case_duration_guard": duration_guard,
            "score": self._final_calibration_score(flat_metrics, duration_guard),
            "time_decomposition_overlap": self._extract_queue_external_overlap(time_diagnostics),
        }

    def _evaluate_final_calibration_candidate_repeated(
            self,
            event_log,
            metrics_labels: list[str],
            n_runs: int,
    ) -> dict:
        """Repeated Stage-3b confirmation evaluation using averaged metrics/diagnostics."""
        n_runs = max(1, int(n_runs))
        if n_runs <= 1:
            result = self._evaluate_final_calibration_candidate(
                event_log=event_log,
                metrics_labels=list(metrics_labels),
            )
            result["n_runs"] = 1
            result["repeated_metric_details"] = {}
            return result

        metrics = self._evaluate(
            event_log=event_log,
            n_runs=n_runs,
            metrics_labels=list(metrics_labels),
            enable_time_diagnostics=True,
        )
        flat_metrics = self.integration._flatten_formatted_metrics(metrics)
        time_diagnostics = getattr(self.integration, "_last_time_diagnostics", {}) or {}
        duration_guard = self.integration._extract_case_duration_guard(time_diagnostics)
        external_delay_ratio = self._extract_external_delay_ratio(time_diagnostics)
        if external_delay_ratio is not None:
            duration_guard["external_delay_ratio"] = float(external_delay_ratio)

        return {
            "metrics": flat_metrics,
            "case_duration_guard": duration_guard,
            "score": self._final_calibration_score(flat_metrics, duration_guard),
            "time_decomposition_overlap": self._extract_queue_external_overlap(time_diagnostics),
            "n_runs": int(n_runs),
            "repeated_metric_details": copy.deepcopy(
                getattr(self.integration, "repeated_metric_details", {}) or {}
            ),
        }

    def _final_calibration_score(self, metrics: dict, duration_guard: dict) -> float:
        # Top-K scale 校准与 Stage 2c 使用同一组六个 runtime metrics，
        # 避免 activity scale 只针对 RED/CTD 等局部指标进行后验拟合。
        weighted_values = []
        weight_sum = 0.0
        for name in FINAL_CALIBRATION_METRICS:
            weight = float(RUNTIME_METRIC_WEIGHTS.get(name, 1.0))
            value = metrics.get(name)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                weighted_values.append(weight * float(np.log1p(max(0.0, float(value)))))
                weight_sum += weight
        if weighted_values and weight_sum > 0.0:
            score = float(sum(weighted_values) / weight_sum)
        else:
            score = 100.0
        ratio = (duration_guard or {}).get("duration_ratio")
        if (
            isinstance(ratio, (int, float, np.integer, np.floating))
            and np.isfinite(float(ratio))
            and float(ratio) > 0.0
        ):
            # 关于 ratio=1 对称：2.0 与 0.5 获得同样的偏差惩罚。
            score += float(FINAL_CALIBRATION_DURATION_LOG_WEIGHT) * abs(
                float(np.log(max(float(ratio), 1e-9)))
            )
        else:
            score += float(FINAL_CALIBRATION_DURATION_LOG_WEIGHT) * 2.0
        return float(score)

    def _resolve_top_activity_scale_k(self, n_external_delay_activities: int) -> int:
        """
        解析活动级外部延迟缩放校准的自适应 Top-K。

        K = min(max_k, max(min_k, ceil(ratio * N)))
        其中 N 为发现的外部延迟活动数量。
        """
        try:
            n = int(n_external_delay_activities)
        except Exception:
            n = 0
        if n <= 0:
            return 0

        try:
            ratio = float(TOP_ACTIVITY_SCALE_RATIO)
            if not np.isfinite(ratio) or ratio <= 0:
                ratio = 0.25
        except Exception:
            ratio = 0.25

        try:
            min_k = max(1, int(TOP_ACTIVITY_SCALE_MIN_K))
        except Exception:
            min_k = 3

        try:
            max_k = max(min_k, int(TOP_ACTIVITY_SCALE_MAX_K))
        except Exception:
            max_k = 8

        k = int(np.ceil(ratio * n))
        k = max(min_k, k)
        k = min(max_k, k)
        k = min(k, n)
        return int(k)

    def _calibrate_top_external_activity_scales(
            self,
            calibration_log,
            metrics_labels: list[str],
            base_eval: dict,
    ) -> dict:
        try:
            all_activities = self.integration._get_top_external_delay_activities(top_k=100000)
        except Exception as exc:
            return {
                "enabled": False,
                "skipped_reason": f"cannot_get_external_delay_activities: {exc}",
            }

        if not all_activities:
            return {
                "enabled": False,
                "skipped_reason": "no_enabled_external_delay_activities",
            }

        n_external_delay_activities = int(len(all_activities))
        top_k = self._resolve_top_activity_scale_k(n_external_delay_activities)
        if top_k <= 0:
            return {
                "enabled": False,
                "skipped_reason": "resolved_top_k_is_zero",
                "n_external_delay_activities": int(n_external_delay_activities),
            }

        top_activities = list(all_activities[:top_k])

        selected_scales = {str(activity): 1.0 for activity in top_activities}
        self.integration._set_external_delay_activity_scales(selected_scales, reset_others=True)

        current_eval = dict(base_eval or {})
        current_eval.setdefault(
            "score",
            self._final_calibration_score(
                current_eval.get("metrics", {}) or {},
                current_eval.get("case_duration_guard", {}) or {},
            ),
        )
        current_score = float(current_eval.get("score", float("inf")))
        results = []

        for activity in top_activities:
            best_scale = float(selected_scales.get(str(activity), 1.0))
            best_eval = current_eval
            best_score = current_score
            activity_results = []

            scale_jobs = []
            baseline_scale = float(selected_scales.get(str(activity), 1.0))
            for scale in TOP_ACTIVITY_SCALE_CANDIDATES:
                scale = float(scale)
                trial_scales = dict(selected_scales)
                trial_scales[str(activity)] = scale
                reused_current_baseline = bool(
                    REUSE_IDENTITY_SCALE_BASELINE
                    and abs(scale - baseline_scale) <= 1e-12
                )
                scale_jobs.append((scale, trial_scales, reused_current_baseline))

            nonbaseline_jobs = [job for job in scale_jobs if not job[2]]
            logger.info(
                "[Serial Eval] Stage 3b activity=%s evaluating %s non-baseline scales sequentially",
                activity,
                len(nonbaseline_jobs),
            )

            for scale, trial_scales, reused_current_baseline in scale_jobs:
                if reused_current_baseline:
                    # current_eval 已经对应“其它已选 activity scales + 当前 activity 当前 scale”。
                    trial_eval = current_eval
                else:
                    logger.info(
                        "[Serial Eval] Stage 3b activity=%s evaluating scale=%s",
                        activity,
                        scale,
                    )
                    self.integration._set_external_delay_activity_scales(trial_scales, reset_others=True)
                    try:
                        trial_eval = self._evaluate_final_calibration_candidate(
                            event_log=calibration_log,
                            metrics_labels=list(metrics_labels),
                        )
                    except Exception as exc:
                        logger.warning(
                            "[Serial Eval] Stage 3b activity=%s scale=%s failed: %s",
                            activity,
                            scale,
                            exc,
                        )
                        trial_eval = {
                            "metrics": {},
                            "case_duration_guard": {},
                            "score": float("inf"),
                            "error": str(exc),
                        }

                row = {
                    "activity": str(activity),
                    "candidate_activity_scale": float(scale),
                    "metrics": trial_eval.get("metrics", {}),
                    "case_duration_guard": trial_eval.get("case_duration_guard", {}),
                    "score": float(trial_eval.get("score", float("inf"))),
                    "time_decomposition_overlap": trial_eval.get("time_decomposition_overlap", {}),
                    "reused_current_baseline": bool(reused_current_baseline),
                    "parallel_isolated_eval": False,
                    "candidate_eval_execution_mode": CANDIDATE_EVAL_EXECUTION_MODE,
                }
                activity_results.append(row)

                if row["score"] < best_score * (1.0 - float(TOP_ACTIVITY_SCALE_MIN_IMPROVEMENT)):
                    best_scale = float(scale)
                    best_eval = trial_eval
                    best_score = float(row["score"])

            selected_scales[str(activity)] = float(best_scale)
            self.integration._set_external_delay_activity_scales(selected_scales, reset_others=True)
            current_eval = best_eval
            current_score = best_score
            results.extend(activity_results)

        return {
            "enabled": True,
            "top_k_policy": "adaptive_ratio_with_min_max_caps",
            "top_k_ratio": float(TOP_ACTIVITY_SCALE_RATIO),
            "top_k_min": int(TOP_ACTIVITY_SCALE_MIN_K),
            "top_k_max": int(TOP_ACTIVITY_SCALE_MAX_K),
            "n_external_delay_activities": int(n_external_delay_activities),
            "resolved_top_k": int(top_k),
            "top_activities": list(top_activities),
            "candidate_scales": [float(x) for x in TOP_ACTIVITY_SCALE_CANDIDATES],
            "selected_activity_scales": selected_scales,
            "baseline_score": float((base_eval or {}).get("score", float("inf"))),
            "selected_score": float(current_score),
            "selected_eval": current_eval,
            "results": results,
            "note": "仅在 train_validation_log 上针对自适应 Top-K 外部延迟活动进行坐标搜索。",
        }

    @staticmethod
    def _extract_queue_external_overlap(time_diagnostics: dict):
        if not isinstance(time_diagnostics, dict):
            return {}
        for key in ("queue_external_overlap_diagnostics", "wait_component_consistency"):
            value = time_diagnostics.get(key)
            if isinstance(value, dict):
                return value
        rep = time_diagnostics.get("representative_report")
        if isinstance(rep, dict):
            for key in ("queue_external_overlap_diagnostics", "wait_component_consistency"):
                value = rep.get(key)
                if isinstance(value, dict):
                    return value
        return {}

    def _apply_selected_runtime_parameters(self, selected_hyperparameters: dict):
        if self.integration.drq_params is None:
            return

        case_duration_calibration = selected_hyperparameters.get("case_duration_calibration") or {}
        try:
            self.integration.drq_params["case_duration_calibration"] = dict(case_duration_calibration)
            modular = self.integration.drq_params.get("modular_params", {})
            if isinstance(modular, dict):
                runtime_controls = modular.setdefault("runtime_controls", {})
                runtime_controls["case_duration_calibration"] = dict(case_duration_calibration)
        except Exception:
            pass

        fixed_controls = selected_hyperparameters.get("fixed_runtime_controls") or {}
        arrival_scale = selected_hyperparameters.get(
            "arrival_time_scale",
            fixed_controls.get("arrival_time_scale"),
        )
        arrival_demand_multiplier = selected_hyperparameters.get(
            "arrival_demand_multiplier",
            fixed_controls.get("arrival_demand_multiplier", 1.0),
        )
        if arrival_scale is not None:
            try:
                arrival_scale = float(arrival_scale)
                arrival_demand_multiplier = float(arrival_demand_multiplier)
                if not np.isfinite(arrival_demand_multiplier) or arrival_demand_multiplier <= 0:
                    arrival_demand_multiplier = 1.0
                if np.isfinite(arrival_scale):
                    arrival_params = self.integration.drq_params.setdefault("arrival_params", {})
                    arrival_params["arrival_time_scale"] = arrival_scale
                    arrival_params["arrival_demand_multiplier"] = arrival_demand_multiplier

                    modular = self.integration.drq_params.get("modular_params", {})
                    if isinstance(modular, dict):
                        arrival_module = modular.get("modules", {}).get("arrival")
                        if isinstance(arrival_module, dict):
                            arrival_module["model_type"] = "case_arrival_decision_tree_interarrival"
                            arrival_module["arrival_time_scale"] = arrival_scale
                            arrival_module["arrival_demand_multiplier"] = arrival_demand_multiplier
            except Exception:
                pass

        global_scale = selected_hyperparameters.get(
            "external_delay_global_scale",
            fixed_controls.get("external_delay_global_scale"),
        )
        max_delay_quantile = selected_hyperparameters.get(
            "external_delay_max_delay_quantile",
            fixed_controls.get("external_delay_max_delay_quantile"),
        )

        if global_scale is not None or max_delay_quantile is not None:
            self.integration._set_external_delay_runtime_config(
                global_scale=global_scale,
                max_delay_quantile=max_delay_quantile,
            )

        activity_scales = selected_hyperparameters.get("external_delay_activity_scales")
        if isinstance(activity_scales, dict) and activity_scales:
            try:
                self.integration._set_external_delay_activity_scales(
                    activity_scales,
                    reset_others=True,
                )
            except Exception:
                pass

        queue_weights = selected_hyperparameters.get("queue_score_mixing_weights")
        queue_strategy_name = selected_hyperparameters.get(
            "queue_strategy_name",
            fixed_controls.get("queue_strategy_name"),
        )

        if queue_weights is not None or queue_strategy_name is not None:
            self.integration._set_queue_scheduling_runtime_config(
                score_mixing_weights=queue_weights,
                strategy_name=queue_strategy_name,
            )

    # ------------------------------------------------------------------
    # 评分、剪枝与排序工具
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_queue_strategy_candidates(candidates) -> list[dict]:
        normalized = []

        for item in candidates:
            if isinstance(item, str):
                matched = [
                    strategy
                    for strategy in QUEUE_STRATEGIES
                    if strategy["name"] == item
                ]
                if matched:
                    normalized.append(matched[0])
                continue

            if isinstance(item, dict):
                name = item.get("name", "custom_queue_strategy")
                weights = item.get("weights", item)
                normalized.append({
                    "name": str(name),
                    "weights": {
                        "tree": float(weights.get("tree", 0.0)),
                        "frequency": float(weights.get("frequency", 0.0)),
                        "availability": float(weights.get("availability", 0.0)),
                        "calendar_fit": float(weights.get("calendar_fit", 0.0)),
                    },
                })

        return normalized or list(QUEUE_STRATEGIES)

    @staticmethod
    def _get_numeric_metric(record: dict, metric_name: str):
        value = (record.get("metrics") or {}).get(metric_name)
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            if np.isfinite(value):
                return value
        return None

    @staticmethod
    def _extract_external_delay_ratio(time_diagnostics: dict):
        """从嵌套诊断数据中稳健地提取 external_delay_ratio。"""

        def _walk(obj):
            if isinstance(obj, dict):
                if "external_delay_ratio" in obj:
                    value = obj.get("external_delay_ratio")
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        value = float(value)
                        if np.isfinite(value):
                            return value
                for value in obj.values():
                    found = _walk(value)
                    if found is not None:
                        return found
            elif isinstance(obj, (list, tuple)):
                for value in obj:
                    found = _walk(value)
                    if found is not None:
                        return found
            return None

        return _walk(time_diagnostics or {})

    def _scalar_runtime_objective(
            self,
            metrics: dict,
            duration_guard: dict,
            metric_names: list[str],
            min_duration_ratio: float,
            max_duration_ratio: float,
            max_external_delay_ratio: float | None,
            ctd_metric_weight: float = 1.0,
            duration_guard_weight: float = 0.0,
            external_delay_ratio_weight: float = 0.0,
    ) -> float:
        """
        运行时阶段的标量目标函数。

        CTD/RED 具有更高的通用权重，并且 duration_ratio 在 Optuna 搜索期间就会受到惩罚，
        而不是仅在搜索后惩罚。这防止了选择那些 CAR/AED/CED 很好但 case duration 被严重压缩的候选。
        """
        weighted_values = []
        weight_sum = 0.0
        missing_penalty = 0.0

        metric_weights = globals().get(
            "RUNTIME_METRIC_WEIGHTS",
            {m: 1.0 for m in metric_names},
        )

        for metric_name in metric_names:
            value = metrics.get(metric_name)
            weight = float(metric_weights.get(metric_name, 1.0))

            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                weighted_values.append(weight * float(np.log1p(max(0.0, float(value)))))
                weight_sum += weight
            else:
                missing_penalty += 5.0 * weight

        if weighted_values and weight_sum > 0:
            score = float(sum(weighted_values) / weight_sum)
        else:
            score = 100.0
        score += missing_penalty

        duration_ratio = (duration_guard or {}).get("duration_ratio")
        if isinstance(duration_ratio, (int, float, np.integer, np.floating)):
            duration_ratio = float(duration_ratio)
            if np.isfinite(duration_ratio):
                if duration_ratio < float(min_duration_ratio):
                    under = (float(min_duration_ratio) - duration_ratio) / max(float(min_duration_ratio), 1e-6)
                    # 在硬剪枝前施加较强的柔和惩罚。这使 Optuna 优先选择那些已经能恢复 case duration 的候选。
                    score += 8.0 * under * under
                elif duration_ratio > float(max_duration_ratio):
                    over = (duration_ratio - float(max_duration_ratio)) / max(float(max_duration_ratio), 1e-6)
                    score += 4.0 * over * over
            else:
                score += 4.0
        else:
            score += 4.0

        external_ratio = (duration_guard or {}).get("external_delay_ratio")
        if (
                max_external_delay_ratio is not None
                and isinstance(external_ratio, (int, float, np.integer, np.floating))
        ):
            external_ratio = float(external_ratio)
            if np.isfinite(external_ratio) and external_ratio > float(max_external_delay_ratio):
                over = (
                    external_ratio - float(max_external_delay_ratio)
                ) / max(1.0 - float(max_external_delay_ratio), 1e-6)
                score += 1.5 * over * over

        return float(score)

    def _hard_prune_runtime_records(
            self,
            records: list[dict],
            metric_names: list[str],
            min_duration_ratio: float,
            max_duration_ratio: float,
            max_external_delay_ratio: float | None,
            ctd_tolerance: float | None = None,
            arrival_span_min_ratio: float | None = None,
            arrival_span_max_ratio: float | None = None,
            arrival_case_count_min_ratio: float | None = None,
    ) -> list[dict]:
        """返回指标有效的运行记录，并附加 guard 诊断信息。

        早期版本将 case-duration、arrival、external-delay 和 CTD guard 用作硬过滤器。
        这对异构日志来说过于脆弱：例如 ACR 日志可能所有候选都在 duration guard 之外，
        导致整个 Web 工作流失败。在统一的五日志协议中，guard 仅作为诊断/软约束。
        它们为每个候选添加注释，并在最终选择时贡献柔和惩罚，但不会移除其他有效候选。
        """
        kept = []

        # 仅为诊断预计算 CTD 参考值。没有候选会因为该诊断 CTD 容差而被丢弃。
        ctd_tolerance_value = None
        best_ctd = None
        if ctd_tolerance is not None and "ctd" in metric_names:
            try:
                ctd_tolerance_value = float(ctd_tolerance)
                if not np.isfinite(ctd_tolerance_value) or ctd_tolerance_value < 1.0:
                    ctd_tolerance_value = None
            except Exception:
                ctd_tolerance_value = None

            ctd_values = []
            for row in records or []:
                if row.get("status") != "ok":
                    continue
                ctd_value = self._get_numeric_metric(row, "ctd")
                if ctd_value is not None:
                    try:
                        ctd_value = float(ctd_value)
                        if np.isfinite(ctd_value):
                            ctd_values.append(ctd_value)
                    except Exception:
                        pass
            if ctd_values:
                best_ctd = min(ctd_values)

        for row in records:
            if row.get("status") != "ok":
                continue

            numeric_count = sum(
                self._get_numeric_metric(row, metric_name) is not None
                for metric_name in metric_names
            )
            if numeric_count == 0:
                continue

            ratio = (row.get("case_duration_guard") or {}).get("duration_ratio")
            duration_diagnostic = {
                "enabled": True,
                "hard_guard_enabled": False,
                "configured_min_ratio": float(min_duration_ratio),
                "configured_max_ratio": float(max_duration_ratio),
                "observed_duration_ratio": None,
                "passed_min_ratio": None,
                "passed_max_ratio": None,
                "passed_diagnostic": None,
            }
            if isinstance(ratio, (int, float, np.integer, np.floating)):
                ratio = float(ratio)
                if np.isfinite(ratio):
                    duration_diagnostic["observed_duration_ratio"] = float(ratio)
                    duration_diagnostic["passed_min_ratio"] = bool(ratio >= float(min_duration_ratio))
                    duration_diagnostic["passed_max_ratio"] = bool(ratio <= float(max_duration_ratio))
                    duration_diagnostic["passed_diagnostic"] = bool(
                        float(min_duration_ratio) <= ratio <= float(max_duration_ratio)
                    )
            row["duration_diagnostic_guard"] = duration_diagnostic

            external_ratio = (row.get("case_duration_guard") or {}).get("external_delay_ratio")
            external_diagnostic = {
                "enabled": max_external_delay_ratio is not None,
                "max_external_delay_ratio": float(max_external_delay_ratio) if max_external_delay_ratio is not None else None,
                "observed_external_delay_ratio": None,
                "passed": None,
            }
            if isinstance(external_ratio, (int, float, np.integer, np.floating)):
                external_ratio = float(external_ratio)
                if np.isfinite(external_ratio):
                    external_diagnostic["observed_external_delay_ratio"] = float(external_ratio)
                    if max_external_delay_ratio is not None:
                        external_diagnostic["passed"] = bool(external_ratio <= float(max_external_delay_ratio))
            row["external_delay_diagnostic_guard"] = external_diagnostic

            arrival_guard = row.get("arrival_guard") or {}
            span_ratio = arrival_guard.get("arrival_span_ratio")
            case_count_ratio = arrival_guard.get("case_count_ratio")
            arrival_diagnostic = {
                "enabled": bool(
                    arrival_span_min_ratio is not None
                    or arrival_span_max_ratio is not None
                    or arrival_case_count_min_ratio is not None
                ),
                "span_min_ratio": float(arrival_span_min_ratio) if arrival_span_min_ratio is not None else None,
                "span_max_ratio": float(arrival_span_max_ratio) if arrival_span_max_ratio is not None else None,
                "case_count_min_ratio": float(arrival_case_count_min_ratio) if arrival_case_count_min_ratio is not None else None,
                "observed_span_ratio": None,
                "observed_case_count_ratio": None,
                "passed_span": None,
                "passed_case_count": None,
                "passed": None,
            }
            arrival_passed = True
            if isinstance(span_ratio, (int, float, np.integer, np.floating)):
                span_ratio = float(span_ratio)
                if np.isfinite(span_ratio):
                    arrival_diagnostic["observed_span_ratio"] = float(span_ratio)
                    span_passed = True
                    if arrival_span_min_ratio is not None and span_ratio < float(arrival_span_min_ratio):
                        span_passed = False
                    if arrival_span_max_ratio is not None and span_ratio > float(arrival_span_max_ratio):
                        span_passed = False
                    arrival_diagnostic["passed_span"] = bool(span_passed)
                    arrival_passed = arrival_passed and span_passed
            if isinstance(case_count_ratio, (int, float, np.integer, np.floating)):
                case_count_ratio = float(case_count_ratio)
                if np.isfinite(case_count_ratio):
                    arrival_diagnostic["observed_case_count_ratio"] = float(case_count_ratio)
                    case_passed = True
                    if arrival_case_count_min_ratio is not None and case_count_ratio < float(arrival_case_count_min_ratio):
                        case_passed = False
                    arrival_diagnostic["passed_case_count"] = bool(case_passed)
                    arrival_passed = arrival_passed and case_passed
            arrival_diagnostic["passed"] = bool(arrival_passed)
            row["arrival_diagnostic_guard"] = arrival_diagnostic

            ctd_value = self._get_numeric_metric(row, "ctd")
            ctd_diagnostic = {
                "enabled": bool(ctd_tolerance_value is not None and best_ctd is not None),
                "best_ctd": float(best_ctd) if best_ctd is not None else None,
                "ctd_tolerance": float(ctd_tolerance_value) if ctd_tolerance_value is not None else None,
                "max_allowed_ctd": float(best_ctd * ctd_tolerance_value) if best_ctd is not None and ctd_tolerance_value is not None else None,
                "observed_ctd": float(ctd_value) if ctd_value is not None else None,
                "passed": None,
            }
            if ctd_value is not None and best_ctd is not None and ctd_tolerance_value is not None:
                ctd_diagnostic["passed"] = bool(float(ctd_value) <= float(best_ctd) * float(ctd_tolerance_value))
            row["ctd_guard"] = ctd_diagnostic

            # 柔和 guard 惩罚：稍后作为次级选择惩罚使用，不作为硬性拒绝标准。
            row["runtime_guard_soft_penalty"] = self._guard_violation_score(
                row,
                min_duration_ratio=float(min_duration_ratio),
                max_duration_ratio=float(max_duration_ratio),
                max_external_delay_ratio=max_external_delay_ratio,
            )
            kept.append(row)

        if not kept:
            logger.warning(
                "Runtime guard diagnostics found no metric-valid candidate. "
                "This indicates simulation/evaluation failure, not guard rejection."
            )
            return []

        logger.info(
            "Runtime guard diagnostics kept %s/%s metric-valid candidates; "
            "guards are diagnostic/soft penalties only.",
            len(kept),
            len(records or []),
        )
        return kept


    def _extract_arrival_guard(self) -> dict:
        """从最新的验证仿真中提取到达跨度/数量 guard。"""
        diagnostics = getattr(self.integration, "_last_arrival_diagnostics", {}) or {}
        guard = {
            "enabled": bool(diagnostics),
            "arrival_span_ratio": None,
            "case_count_ratio": None,
            "real_cases": diagnostics.get("real_cases"),
            "sim_cases": diagnostics.get("sim_cases"),
            "weekday_hour_jaccard": diagnostics.get("weekday_hour_jaccard"),
        }

        try:
            span_ratio = float(diagnostics.get("span_ratio_sim_over_real"))
            if np.isfinite(span_ratio):
                guard["arrival_span_ratio"] = span_ratio
        except Exception:
            pass

        try:
            real_cases = float(diagnostics.get("real_cases"))
            sim_cases = float(diagnostics.get("sim_cases"))
            if np.isfinite(real_cases) and np.isfinite(sim_cases) and real_cases > 0:
                guard["case_count_ratio"] = sim_cases / real_cases
        except Exception:
            pass

        span = guard.get("arrival_span_ratio")
        case_ratio = guard.get("case_count_ratio")
        passed = True
        if isinstance(span, (int, float, np.integer, np.floating)) and np.isfinite(float(span)):
            passed = passed and (
                float(ARRIVAL_SPAN_MIN_RATIO) <= float(span) <= float(ARRIVAL_SPAN_MAX_RATIO)
            )
        if isinstance(case_ratio, (int, float, np.integer, np.floating)) and np.isfinite(float(case_ratio)):
            passed = passed and (float(case_ratio) >= float(ARRIVAL_CASE_COUNT_MIN_RATIO))
        guard["passed"] = bool(passed)
        return guard

    @staticmethod
    def _runtime_arrival_guard_passed(row: dict) -> bool:
        """返回运行时候选是否通过到达 guard。"""
        diag = (row or {}).get("arrival_diagnostic_guard") or (row or {}).get("arrival_guard") or {}
        passed = diag.get("passed")
        if passed is not None:
            return bool(passed)

        span_ratio = diag.get("observed_span_ratio", diag.get("arrival_span_ratio"))
        case_count_ratio = diag.get("observed_case_count_ratio", diag.get("case_count_ratio"))
        ok = True
        if isinstance(span_ratio, (int, float, np.integer, np.floating)) and np.isfinite(float(span_ratio)):
            ok = ok and (
                float(ARRIVAL_SPAN_MIN_RATIO)
                <= float(span_ratio)
                <= float(ARRIVAL_SPAN_MAX_RATIO)
            )
        if isinstance(case_count_ratio, (int, float, np.integer, np.floating)) and np.isfinite(float(case_count_ratio)):
            ok = ok and (float(case_count_ratio) >= float(ARRIVAL_CASE_COUNT_MIN_RATIO))
        return bool(ok)

    @staticmethod
    def _runtime_duration_ratio(row: dict) -> float | None:
        """从运行时候选中提取模拟/真实 case duration 比率。"""
        diag = (row or {}).get("duration_diagnostic_guard") or {}
        ratio = diag.get("observed_duration_ratio")
        if ratio is None:
            ratio = ((row or {}).get("case_duration_guard") or {}).get("duration_ratio")
        try:
            ratio = float(ratio)
            if np.isfinite(ratio) and ratio > 0.0:
                return float(ratio)
        except Exception:
            pass
        return None

    @staticmethod
    def _runtime_duration_guard_passed(row: dict) -> bool:
        """返回运行时候选是否通过配置的 duration guard。"""
        diag = (row or {}).get("duration_diagnostic_guard") or {}
        passed = diag.get("passed_diagnostic")
        if passed is not None:
            return bool(passed)

        ratio = StagewiseTuner._runtime_duration_ratio(row)
        if ratio is None:
            return False
        try:
            min_ratio = float(diag.get("configured_min_ratio", CASE_DURATION_MIN_RATIO))
            max_ratio = float(diag.get("configured_max_ratio", CASE_DURATION_MAX_RATIO))
        except Exception:
            min_ratio = float(CASE_DURATION_MIN_RATIO)
            max_ratio = float(CASE_DURATION_MAX_RATIO)
        return bool(min_ratio <= ratio <= max_ratio)

    @staticmethod
    def _runtime_duration_closeness(row: dict) -> float:
        """duration_ratio 到 1.0 的对称距离；越小越好。"""
        ratio = StagewiseTuner._runtime_duration_ratio(row)
        if ratio is None:
            return float("inf")
        return float(abs(np.log(max(float(ratio), 1e-9))))

    @staticmethod
    def _runtime_arrival_closeness(row: dict) -> float:
        """到达跨度/数量比率到安全值的对称距离；越小越好。"""
        diag = (row or {}).get("arrival_diagnostic_guard") or (row or {}).get("arrival_guard") or {}
        score = 0.0
        used = 0
        span_ratio = diag.get("observed_span_ratio", diag.get("arrival_span_ratio"))
        if isinstance(span_ratio, (int, float, np.integer, np.floating)) and np.isfinite(float(span_ratio)) and float(span_ratio) > 0:
            score += abs(float(np.log(max(float(span_ratio), 1e-9))))
            used += 1
        case_count_ratio = diag.get("observed_case_count_ratio", diag.get("case_count_ratio"))
        if isinstance(case_count_ratio, (int, float, np.integer, np.floating)) and np.isfinite(float(case_count_ratio)) and float(case_count_ratio) > 0:
            if float(case_count_ratio) < float(ARRIVAL_CASE_COUNT_MIN_RATIO):
                score += abs(float(np.log(max(float(case_count_ratio), 1e-9))))
            used += 1
        if used <= 0:
            return float("inf")
        return float(score)

    def _guard_violation_score(
            self,
            row: dict,
            min_duration_ratio: float,
            max_duration_ratio: float,
            max_external_delay_ratio: float | None,
    ) -> float:
        """
        仅在所有硬 guard 都失败时使用的回退分数。

        Duration 违规主导该分数；CTD 和 RED 比 CAR 更重要，因为即使 case 内部时间
        被严重压缩，CAR 也可能看起来不错。
        """
        guard = row.get("case_duration_guard") or {}

        ratio = guard.get("duration_ratio")
        if isinstance(ratio, (int, float, np.integer, np.floating)) and np.isfinite(float(ratio)):
            ratio = float(ratio)
            if ratio < float(min_duration_ratio):
                duration_violation = (
                    float(min_duration_ratio) - ratio
                ) / max(float(min_duration_ratio), 1e-6)
            elif ratio > float(max_duration_ratio):
                duration_violation = (
                    ratio - float(max_duration_ratio)
                ) / max(float(max_duration_ratio), 1e-6)
            else:
                duration_violation = 0.0
        else:
            duration_violation = 1.0

        external_ratio = guard.get("external_delay_ratio")
        if (
                max_external_delay_ratio is not None
                and isinstance(external_ratio, (int, float, np.integer, np.floating))
                and np.isfinite(float(external_ratio))
        ):
            external_ratio = float(external_ratio)
            if external_ratio > float(max_external_delay_ratio):
                external_violation = (
                    external_ratio - float(max_external_delay_ratio)
                ) / max(1.0 - float(max_external_delay_ratio), 1e-6)
            else:
                external_violation = 0.0
        else:
            external_violation = 0.0

        arrival_guard = row.get("arrival_guard") or {}
        arrival_violation = 0.0

        span_ratio = arrival_guard.get("arrival_span_ratio")
        if isinstance(span_ratio, (int, float, np.integer, np.floating)) and np.isfinite(float(span_ratio)):
            span_ratio = float(span_ratio)
            if span_ratio < float(ARRIVAL_SPAN_MIN_RATIO):
                arrival_violation += (
                    float(ARRIVAL_SPAN_MIN_RATIO) - span_ratio
                ) / max(float(ARRIVAL_SPAN_MIN_RATIO), 1e-6)
            elif span_ratio > float(ARRIVAL_SPAN_MAX_RATIO):
                arrival_violation += (
                    span_ratio - float(ARRIVAL_SPAN_MAX_RATIO)
                ) / max(float(ARRIVAL_SPAN_MAX_RATIO), 1e-6)

        case_count_ratio = arrival_guard.get("case_count_ratio")
        if isinstance(case_count_ratio, (int, float, np.integer, np.floating)) and np.isfinite(float(case_count_ratio)):
            case_count_ratio = float(case_count_ratio)
            if case_count_ratio < float(ARRIVAL_CASE_COUNT_MIN_RATIO):
                arrival_violation += (
                    float(ARRIVAL_CASE_COUNT_MIN_RATIO) - case_count_ratio
                ) / max(float(ARRIVAL_CASE_COUNT_MIN_RATIO), 1e-6)

        red = self._get_numeric_metric(row, "red")
        ctd = self._get_numeric_metric(row, "ctd")
        car = self._get_numeric_metric(row, "car")
        aed = self._get_numeric_metric(row, "aed")
        red_term = np.log1p(max(0.0, float(red))) / 8.0 if red is not None else 1.0
        ctd_term = np.log1p(max(0.0, float(ctd))) / 8.0 if ctd is not None else 1.0
        car_term = np.log1p(max(0.0, float(car))) / 8.0 if car is not None else 0.5
        aed_term = np.log1p(max(0.0, float(aed))) / 8.0 if aed is not None else 0.5
        external_weight = 0.25 if duration_violation > 0 else 0.8
        return float(
            8.0 * duration_violation
            + external_weight * external_violation
            + 2.4 * ctd_term
            + 2.0 * red_term
            + 0.8 * arrival_violation
            + 0.4 * car_term
            + 0.3 * aed_term
        )

    def _pareto_prune_records(
            self,
            records: list[dict],
            metric_names: list[str],
    ) -> list[dict]:
        if not records:
            return []
        non_dominated = []

        for i, row in enumerate(records):
            dominated = False
            for j, other in enumerate(records):
                if i == j:
                    continue
                if self._dominates(other, row, metric_names):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(row)
        return non_dominated or list(records)
    def _dominates(self, a: dict, b: dict, metric_names: list[str]) -> bool:
        better_or_equal_all = True
        strictly_better_one = False
        compared_any = False

        for metric_name in metric_names:
            av = self._get_numeric_metric(a, metric_name)
            bv = self._get_numeric_metric(b, metric_name)

            if av is None or bv is None:
                continue
            compared_any = True
            if av > bv:
                better_or_equal_all = False
                break
            if av < bv:
                strictly_better_one = True
        return bool(compared_any and better_or_equal_all and strictly_better_one)
    @staticmethod
    def _select_top_records(
            primary_records: list[dict],
            fallback_records: list[dict],
            top_k: int,
            score_key: str,
    ) -> list[dict]:
        pool = list(primary_records) if primary_records else list(fallback_records)
        pool = [r for r in pool if r.get("status") == "ok"]

        if not pool:
            return []

        def _depth_penalty(row: dict) -> tuple[int, int]:
            try:
                depth = int(row.get("max_depth_tree", row.get("tree_depth", 2)))
            except Exception:
                depth = 999
            # 当得分相同/非常接近时优先选择适中的 depth=2，然后按确定性的较低深度排序。
            # 这避免自动偏向深度 1。
            return abs(depth - 2), depth
        if any("guard_fallback_score" in r for r in pool):
            pool.sort(
                key=lambda r: (
                    r.get("guard_fallback_score", float("inf")),
                    r.get(score_key, float("inf")),
                    r.get("optuna_objective_score", float("inf")),
                    r.get("structure_score", float("inf")),
                    *_depth_penalty(r),
                )
            )
        else:
            pool.sort(
                key=lambda r: (
                    r.get(score_key, float("inf")),
                    r.get("optuna_objective_score", float("inf")),
                    r.get("structure_score", float("inf")),
                    *_depth_penalty(r),
                )
            )
        return pool[:max(1, int(top_k))]

    @staticmethod
    def _attach_ideal_point_scores(
            records: list[dict],
            metric_names: list[str],
            metrics_key: str = "metrics",
            score_key: str = "score",
            metric_weights: dict[str, float] | None = None,
    ):
        if metric_weights is None:
            metric_weights = {m: 1.0 for m in metric_names}
        values_by_metric = {m: [] for m in metric_names}
        for row in records:
            metrics = row.get(metrics_key, {}) or {}
            for metric_name in metric_names:
                value = metrics.get(metric_name)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    value = float(value)
                    if np.isfinite(value):
                        values_by_metric[metric_name].append(value)
        for row in records:
            metrics = row.get(metrics_key, {}) or {}
            weighted_sum = 0.0
            weight_sum = 0.0
            for metric_name in metric_names:
                value = metrics.get(metric_name)
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    continue
                value = float(value)
                if not np.isfinite(value):
                    continue
                candidates = values_by_metric.get(metric_name, [])
                if not candidates:
                    continue
                min_v = min(candidates)
                max_v = max(candidates)

                if max_v > min_v:
                    normalized = (value - min_v) / (max_v - min_v)
                else:
                    normalized = 0.0

                weight = float(metric_weights.get(metric_name, 1.0))
                weighted_sum += weight * normalized ** 2
                weight_sum += weight

            if weight_sum > 0:
                row[score_key] = float(np.sqrt(weighted_sum / weight_sum))
            else:
                row[score_key] = float("inf")