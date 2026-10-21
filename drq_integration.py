# -*- coding: utf-8 -*-
"""
Integration layer between the Flask application and the modular drq
simulation engine.

This refactored version treats the five-module model as the primary parameter
format and keeps legacy fields only for UI compatibility.
"""

from __future__ import annotations

import copy
import logging
import os
import inspect
from datetime import datetime

import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.visualization.petri_net import visualizer as pn_visualizer

from evaluation import evaluate
from drqsim.simulator import SimulatorEngine, SimulatorParameters
from drqsim.utils.log_partitioner import split_drq_xes_log
from drqsim.diagnostics.time_decomposition import (
    aggregate_time_decomposition_diagnostics,
    build_time_decomposition_diagnostics,
    format_time_diagnostics_for_log,
)

logger = logging.getLogger(__name__)


def _discover_im_process_tree_then_petri_net(event_log, noise_threshold_im: float = 0.0) -> dict:
    """
    使用同一次 Inductive Miner 结果同时获得流程树与 Petri 网。

    目的：
    1. 无 PNML 时只调用一次 Inductive Miner；
    2. 将该流程树转换成 Petri 网，后续决策点统一从 Petri 网结构提取；
    3. 流程树仅保留为来源/可视化元数据，不再定义决策点。
    """
    process_tree = None

    # 兼容不同 PM4Py 版本的流程树发现接口。
    try:
        process_tree = pm4py.discover_process_tree_inductive(
            event_log,
            noise_threshold=noise_threshold_im,
        )
    except TypeError:
        process_tree = pm4py.discover_process_tree_inductive(event_log)
    except Exception:
        from pm4py.algo.discovery.inductive import algorithm as inductive_miner
        try:
            process_tree = inductive_miner.apply(
                event_log,
                parameters={"noise_threshold": noise_threshold_im},
            )
        except TypeError:
            process_tree = inductive_miner.apply(event_log)

    # 由同一个流程树转换得到 Petri 网，保证二者结构同源。
    try:
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)
    except Exception:
        from pm4py.objects.conversion.process_tree import converter as pt_converter
        net, initial_marking, final_marking = pt_converter.apply(process_tree)

    return {
        "process_tree": process_tree,
        "net": net,
        "initial_marking": initial_marking,
        "final_marking": final_marking,
    }


class drqIntegration:
    """
    Application-facing wrapper around process discovery, modular parameter
    discovery, simulation, and evaluation.
    """

    def __init__(self):
        self.process_model = None
        self.event_log = None
        self.drq_params = None
        self.drq_metrics = None

        self.full_event_log = None
        self.log_partitions = None
        self.train_log = None
        self.validation_log = None
        self.train_validation_log = None
        self.test_log = None
        self.repeated_metric_details = {}
        self.time_diagnostics = {}

        # 用于外部延迟校准阶段读取最近一次 validation 仿真的时间诊断
        self._last_time_diagnostics = {}

        # 用于诊断 case arrival 生成质量。
        self._last_arrival_diagnostics = {}
        self._last_control_flow_diagnostics = {}
        self._last_arrival_bucket_diagnostics = {}
        self._repeated_arrival_diagnostics = []
        self._repeated_control_flow_diagnostics = {}

        # Optional UI progress callback. It is set by the Flask route during
        # long discovery/tuning calls, so the frontend can show the actual
        # business stage instead of a timer-based fake stage.
        self._progress_callback = None

    def _set_progress_callback(self, progress_callback=None):
        self._progress_callback = progress_callback

    def _emit_progress(self, stage: str, detail: str | None = None, title: str | None = None, status: str = "running"):
        """Emit a coarse discovery progress event to the web layer.

        Supported stages used by the current UI:
        - petri / structure / structure_search: Petri net mining
        - params / runtime / parameter_search: parameter mining
        """
        callback = getattr(self, "_progress_callback", None)
        if callback is None:
            return

        try:
            callback(
                stage=stage,
                title=title,
                detail=detail,
                status=status,
            )
        except TypeError:
            try:
                callback(stage, detail)
            except Exception:
                logger.warning("Failed to emit discovery progress", exc_info=True)
        except Exception:
            logger.warning("Failed to emit discovery progress", exc_info=True)

    # ------------------------------------------------------------------
    # Log split
    # ------------------------------------------------------------------
    def split_event_log(
            self,
            event_log,
            train_validation_ratio: float = 0.8,
            train_ratio_within_train_validation: float = 0.8,
            sort_temporally: bool = True
    ):
        self.full_event_log = event_log

        self.log_partitions = split_drq_xes_log(
            event_log,
            train_validation_ratio=train_validation_ratio,
            train_ratio_within_train_validation=train_ratio_within_train_validation,
            case_id_col="case:concept:name",
            activity_col="concept:name",
            resource_col="org:resource",
            start_time_col="start:timestamp",
            end_time_col="time:timestamp",
            sort_temporally=sort_temporally,
        )

        self.train_log = self.log_partitions.train_log
        self.validation_log = self.log_partitions.validation_log
        self.train_validation_log = self.log_partitions.train_validation_log
        self.test_log = self.log_partitions.test_log

        logger.info(
            "Log split completed: "
            f"train={len(self.train_log)}, "
            f"validation={len(self.validation_log)}, "
            f"train_validation={len(self.train_validation_log)}, "
            f"test={len(self.test_log)}"
        )

        return self.log_partitions

    # ------------------------------------------------------------------
    # Process model
    # ------------------------------------------------------------------
    def discover_process_model(
            self,
            noise_threshold=0.2,
            compute_quality: bool = True,
            generate_visualization: bool = True,
    ):
        try:
            self._emit_progress(
                "petri",
                detail=f"Discovering Petri net structure with noise threshold {noise_threshold}.",
                title="Petri net mining...",
            )

            logger.info(
                "Discovering process model via Inductive Miner with "
                f"noise_threshold={noise_threshold}"
            )

            discovered = _discover_im_process_tree_then_petri_net(
                self.event_log,
                noise_threshold_im=float(noise_threshold),
            )

            net = discovered["net"]
            initial_marking = discovered["initial_marking"]
            final_marking = discovered["final_marking"]
            process_tree = discovered.get("process_tree")

            self.process_model = self._build_process_model_dict(
                net,
                initial_marking,
                final_marking,
                process_tree=process_tree,
                process_tree_source="inductive_miner",
                compute_quality=compute_quality,
                generate_visualization=generate_visualization,
            )
            logger.info(
                f"Discovered IM process model with "
                f"{len(self.process_model['activities'])} activities, "
                f"process_tree_available={self.process_model.get('process_tree_available', False)}"
            )

        except Exception as e:
            logger.error(f"Error discovering process model: {str(e)}")
            raise

    def tune_and_evaluate_with_strict_split(self, *args, progress_callback=None, **kwargs):
        """Run strict split tuning with optional progress callback.

        The web route passes a callback here. We keep it on the integration
        instance so all methods called by StagewiseTuner can emit progress.
        If the installed StagewiseTuner.run also supports progress_callback,
        the callback is forwarded; otherwise this wrapper remains compatible.
        """
        from drqsim.tuning.stagewise_tuner import StagewiseTuner

        previous_callback = getattr(self, "_progress_callback", None)
        self._set_progress_callback(progress_callback)

        try:
            self._emit_progress(
                "petri",
                detail="Starting strict split tuning and Petri net structure search.",
                title="Petri net mining...",
            )

            tuner = StagewiseTuner(self)

            try:
                run_signature = inspect.signature(tuner.run)
                if "progress_callback" in run_signature.parameters:
                    kwargs["progress_callback"] = progress_callback
            except Exception:
                # Signature inspection is best-effort only; do not block tuning.
                pass

            return tuner.run(*args, **kwargs)

        finally:
            self._set_progress_callback(previous_callback)

    def import_petri_net_from_pnml(
            self,
            pnml_file_path,
            compute_quality: bool = True,
            generate_visualization: bool = True,
    ):
        try:
            self._emit_progress(
                "petri",
                detail="Validating and importing the uploaded PNML Petri net.",
                title="Petri net mining...",
            )

            logger.info(f"Importing Petri net structure from {pnml_file_path}")

            net, initial_marking, final_marking = pm4py.read_pnml(pnml_file_path)

            self.process_model = self._build_process_model_dict(
                net,
                initial_marking,
                final_marking,
                process_tree=None,
                process_tree_source="pnml",
                compute_quality=compute_quality,
                generate_visualization=generate_visualization,
            )

            logger.info(
                "PNML Petri net imported successfully: "
                f"{len(self.process_model['activities'])} activities, "
                f"{len(self.process_model['places'])} places"
            )

        except Exception as e:
            logger.error(f"Failed to import Petri net from PNML: {str(e)}")
            raise

    def _build_process_model_dict(
            self,
            net,
            initial_marking,
            final_marking,
            process_tree=None,
            process_tree_source: str = "unknown",
            compute_quality: bool = True,
            generate_visualization: bool = True,
    ) -> dict:
        activities = []
        transitions = []
        places = []

        for transition in net.transitions:
            transitions.append(transition.name)
            if transition.label:
                activities.append(transition.label)

        for place in net.places:
            places.append(place.name)

        map_transition_name_to_id = {}
        for transition in net.transitions:
            key = transition.label if transition.label else transition.name
            map_transition_name_to_id[key] = str(id(transition))

        if generate_visualization:
            visualization_svg = self._generate_petri_net_visualization(
                net,
                initial_marking,
                final_marking,
            )
        else:
            visualization_svg = None

        if compute_quality:
            try:
                fitness = fitness_evaluator.apply(
                    self.event_log,
                    net,
                    initial_marking,
                    final_marking,
                )["averageFitness"]
            except Exception:
                fitness = 0.0

            try:
                precision = precision_evaluator.apply(
                    self.event_log,
                    net,
                    initial_marking,
                    final_marking,
                )
            except Exception:
                precision = 0.0

            if fitness + precision > 0:
                f_measure = 2 * fitness * precision / (fitness + precision)
            else:
                f_measure = 0.0
        else:
            fitness = None
            precision = None
            f_measure = None

        return {
            "activities": activities,
            "transitions": transitions,
            "places": places,
            "map_transitionName_to_id": map_transition_name_to_id,
            "fitness": fitness,
            "precision": precision,
            "f_measure": f_measure,
            "visualization": visualization_svg,
            "net": net,
            "initial_marking": initial_marking,
            "final_marking": final_marking,
            "process_tree": process_tree,
            "process_tree_source": str(process_tree_source),
            "process_tree_available": process_tree is not None,
        }

    def _get_process_tree_for_parameters(self):
        """Return the optional IM process tree as provenance metadata only.

        Control-flow decision points are no longer derived from this object.
        Uploaded PNML therefore returns ``None`` here and remains the sole
        structural source through its Petri net.
        """
        if not isinstance(self.process_model, dict):
            return None
        return self.process_model.get("process_tree")

    def _get_control_flow_structure_source(self) -> str:
        """Return a stable description of the Petri-net structure source."""
        if not isinstance(self.process_model, dict):
            return "unknown"
        source = str(self.process_model.get("process_tree_source", "unknown") or "unknown")
        if source == "pnml_unavailable":  # backward compatibility with old in-memory models
            return "pnml"
        return source

    # ------------------------------------------------------------------
    # Parameter discovery
    # ------------------------------------------------------------------
    def discover_parameters(
            self,
            max_depth_tree=3,

            decision_point_max_depth: int | None = None,
            arrival_decision_tree_max_depth: int | None = None,
            case_arrival_max_depth: int | None = None,
            queue_scheduling_max_depth: int | None = None,
            conditional_prioritization_max_depth: int | None = None,
            external_delay_max_depth: int | None = None,

            queue_scheduling_force_empirical: bool = False,
            conditional_prioritization_max_training_winners: int | None = 12000,
            conditional_prioritization_min_training_pairs: int = 80,
            conditional_prioritization_max_competitors_per_winner: int = 8,
            external_delay_discovery_max_quantile: float = 1.0,
            external_delay_runtime_max_quantile: float = 1.0,
            external_delay_min_samples_per_activity: int = 80,
            external_delay_max_training_events: int | None = 10000,
            external_delay_max_leaf_nodes: int | None = 64,

            # 新增：活动候选资源过滤参数
            enable_candidate_resource_filter: bool = False,
            candidate_resource_coverage: float = 1.00,
            candidate_resource_max_k: int | None = None,
            candidate_resource_min_k: int = 1,

            # 新增：对齐超时保护（秒）
            alignment_timeout_seconds: int | None = None,

            # 兼容 StagewiseTuner / adaptive runtime recency
            precomputed_df_features=None,
            runtime_event_log=None,
            runtime_recency_modules=None,
    ):

        """
        Discover the complete modular drq parameter set.

        All simulation components are discovered by default.  Module-level
        enable/disable switches have been removed; CPT discovery is always run
        and the discovered data decides whether dispatch uses a pairwise CPT or
        the FCFS fallback.
        """

        try:
            self._emit_progress(
                "params",
                detail="Mining resource, queue, arrival, execution-time, and waiting-time simulation parameters.",
                title="Parameter mining...",
            )

            logger.info("Discovering modular drq parameters from event log")

            if not self.process_model:
                self._emit_progress(
                    "petri",
                    detail="No Petri net is available yet; discovering the process structure first.",
                    title="Petri net mining...",
                )
                self.discover_process_model()
                self._emit_progress(
                    "params",
                    detail="Petri net is ready. Continuing parameter mining.",
                    title="Parameter mining...",
                )

            net = self.process_model["net"]
            initial_marking = self.process_model["initial_marking"]
            final_marking = self.process_model["final_marking"]

            params = SimulatorParameters(
                net,
                initial_marking,
                final_marking,
                process_tree=self._get_process_tree_for_parameters(),
                control_flow_structure_source=self._get_control_flow_structure_source(),
            )

            logger.info(
                "Control-flow structure source for parameter discovery: "
                "petri_source=%s, process_tree_available=%s, decision_source=petri_net",
                self._get_control_flow_structure_source(),
                self.process_model.get("process_tree_available", False),
            )

            self._emit_progress(
                "params",
                detail="Fitting transition, resource, queue, arrival, execution-time, and waiting-time models from the event log.",
                title="Parameter mining...",
            )

            params.discover_from_eventlog(
                self.event_log,
                max_depth_tree=max_depth_tree,

                decision_point_max_depth=decision_point_max_depth,
                arrival_decision_tree_max_depth=arrival_decision_tree_max_depth,
                case_arrival_max_depth=case_arrival_max_depth,
                queue_scheduling_max_depth=queue_scheduling_max_depth,
                conditional_prioritization_max_depth=conditional_prioritization_max_depth,
                external_delay_max_depth=external_delay_max_depth,

                verbose=True,
                queue_scheduling_force_empirical=bool(queue_scheduling_force_empirical),
                conditional_prioritization_max_training_winners=conditional_prioritization_max_training_winners,
                conditional_prioritization_min_training_pairs=int(conditional_prioritization_min_training_pairs),
                conditional_prioritization_max_competitors_per_winner=int(conditional_prioritization_max_competitors_per_winner),
                external_delay_discovery_max_quantile=external_delay_discovery_max_quantile,
                external_delay_runtime_max_quantile=external_delay_runtime_max_quantile,
                external_delay_min_samples_per_activity=external_delay_min_samples_per_activity,
                external_delay_max_training_events=external_delay_max_training_events,
                external_delay_max_leaf_nodes=external_delay_max_leaf_nodes,

                # 新增：传递到 SimulatorParameters
                enable_candidate_resource_filter=bool(enable_candidate_resource_filter),
                candidate_resource_coverage=float(candidate_resource_coverage),
                candidate_resource_max_k=candidate_resource_max_k,
                candidate_resource_min_k=int(candidate_resource_min_k),
                alignment_timeout_seconds=alignment_timeout_seconds,
                precomputed_df_features=precomputed_df_features,
                runtime_event_log=runtime_event_log,
                runtime_recency_modules=runtime_recency_modules,
            )

            self._emit_progress(
                "params",
                detail="Converting discovered simulation parameters to DRQSim application format.",
                title="Parameter mining...",
            )

            self.drq_params = params.to_dict()

            resource_params = self.drq_params.get("resource_params", {})
            resources = resource_params.get("resources", [])
            act_to_resources = resource_params.get("act_to_resources", {})

            candidate_sizes = [
                len(v)
                for v in act_to_resources.values()
                if isinstance(v, list)
            ]

            if candidate_sizes:
                logger.info(
                    "[Candidate Resource Filter Check] resources=%d, "
                    "activity_candidate_size min=%d, mean=%.2f, max=%d",
                    len(resources),
                    min(candidate_sizes),
                    float(np.mean(candidate_sizes)),
                    max(candidate_sizes),
                )
            else:
                logger.info(
                    "[Candidate Resource Filter Check] resources=%d, no act_to_resources found",
                    len(resources),
                )

            logger.info("Modular drq parameters discovered and converted to dict")

            return self._convert_drq_to_app_format()

        except Exception as e:
            logger.error(f"Error discovering parameters: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Neutral fallbacks for always-on runtime components
    # ------------------------------------------------------------------
    def _default_resource_state_params(self) -> dict:
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

    def _default_queue_params(self) -> dict:
        return {
            "model_type": "runtime_queue_state_for_scheduling",
            "feature_names": [
                "global_queue_length",
                "activity_queue_length",
                "resource_queue_length",
                "resource_activity_queue_length",
                "queue_pressure",
                "system_wip",
            ],
            "use_log_scaled_features": True,
            "max_queue_count": 8,
            "max_wip_count": 30,
        }

    def _build_simulation_params(self) -> dict:
        """Build runtime parameters for the complete model.

        There is no experiment/runtime switch layer anymore.  Every discovered
        component is passed to the simulator.  Component-specific fallbacks
        (for example CPT -> FCFS when the log has too little overtaking) remain
        data-driven inside the corresponding discovery/runtime module.
        """
        sim_params = copy.deepcopy(self.drq_params)

        # Dynamic resource state is always part of the full model.  If DBN
        # discovery itself had to fall back, that fallback is preserved here.
        sim_params.setdefault("resource_state_params", self._default_resource_state_params())

        # Queue state and queue-aware resource allocation are always available
        # in the full runtime.
        sim_params.setdefault("queue_params", self._default_queue_params())
        sim_params.setdefault("queue_scheduling_params", {})
        qsp = sim_params["queue_scheduling_params"]
        if isinstance(qsp, dict):
            qsp.setdefault("resource_policy_mode", qsp.get("selected_strategy", "balanced"))
            qsp.setdefault("selected_resource_policy", qsp.get("resource_policy_mode", "balanced"))
            qsp.setdefault("adaptive_resource_capacity", True)
            qsp.setdefault("max_queue_wait_min", 480.0)
            qsp.setdefault("fallback_top_k_when_all_blocked", 3)
            qsp.setdefault(
                "conditional_prioritization",
                {
                    "model_type": "conditional_prioritization_tree",
                    "model_kind": "fcfs_fallback",
                    "feature_columns": [],
                    "rules": None,
                    "fallback_policy": "fcfs",
                    "tie_breaker": "enabled_time_then_case_id",
                },
            )

        # External-delay discovery may decide that individual activities have no
        # significant timer.  The module itself still remains in the runtime;
        # such activities simply return zero delay.
        sim_params.setdefault("external_delay_params", {"activities": {}})

        sim_params.setdefault("arrival_params", {})
        sim_params["arrival_params"]["arrival_time_scale"] = float(
            sim_params["arrival_params"].get("arrival_time_scale", 1.0)
        )
        sim_params["arrival_params"]["arrival_demand_multiplier"] = float(
            sim_params["arrival_params"].get("arrival_demand_multiplier", 1.0)
        )
        if sim_params["arrival_params"]["arrival_demand_multiplier"] <= 0:
            sim_params["arrival_params"]["arrival_demand_multiplier"] = 1.0
        sim_params["arrival_params"]["arrival_time_shift_min"] = float(
            sim_params["arrival_params"].get("arrival_time_shift_min", 0.0)
        )

        # Keep serialized modular metadata synchronized with runtime parameters.
        modular = sim_params.get("modular_params")
        if isinstance(modular, dict):
            modules = modular.setdefault("modules", {})
            if "queue_scheduling" in modules and isinstance(modules["queue_scheduling"], dict):
                if isinstance(modules["queue_scheduling"].get("queue_scheduling_model"), dict):
                    nested_cpt = modules["queue_scheduling"]["queue_scheduling_model"].get(
                        "conditional_prioritization", {}
                    )
                    if isinstance(nested_cpt, dict):
                        modules["queue_scheduling"]["conditional_prioritization_model"] = copy.deepcopy(nested_cpt)
            if "arrival" in modules and isinstance(modules["arrival"], dict):
                modules["arrival"]["model_type"] = "case_arrival_decision_tree_interarrival"
                modules["arrival"]["arrival_time_scale"] = sim_params["arrival_params"]["arrival_time_scale"]
                modules["arrival"]["arrival_demand_multiplier"] = sim_params["arrival_params"]["arrival_demand_multiplier"]
                modules["arrival"]["arrival_time_shift_min"] = sim_params["arrival_params"]["arrival_time_shift_min"]

        return sim_params

    # ------------------------------------------------------------------
    # Arrival diagnostics
    # ------------------------------------------------------------------
    def _extract_case_first_start_times(self, event_log) -> pd.Series:
        """Return sorted first start timestamp of each case."""
        if isinstance(event_log, pd.DataFrame):
            df = event_log.copy()
        else:
            df = pm4py.convert_to_dataframe(event_log)

        if "case:concept:name" not in df.columns or "start:timestamp" not in df.columns:
            return pd.Series(dtype="datetime64[ns, UTC]")

        df["case:concept:name"] = df["case:concept:name"].astype(str)
        df["start:timestamp"] = pd.to_datetime(
            df["start:timestamp"],
            utc=True,
            errors="coerce",
        )
        df.dropna(subset=["case:concept:name", "start:timestamp"], inplace=True)

        if df.empty:
            return pd.Series(dtype="datetime64[ns, UTC]")

        return (
            df.groupby("case:concept:name")["start:timestamp"]
            .min()
            .sort_values()
        )

    @staticmethod
    def _summarize_numeric_series(values: pd.Series) -> dict:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            return {
                "n": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "p10": None,
                "p90": None,
            }

        return {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p10": float(values.quantile(0.10)),
            "p90": float(values.quantile(0.90)),
        }

    def log_case_arrival_diagnostics(
            self,
            real_log,
            simulated_log,
            prefix: str = "",
    ) -> dict:
        """
        打印 case 到达诊断，用来判断 CAR 偏大的原因：
        1) 仿真 case 到达跨度是否太长/太短；
        2) inter-arrival 均值/中位数是否偏离；
        3) case 是否落在了不同的 hour / weekday-hour 桶中。
        """
        diagnostics = {}

        try:
            real_first = self._extract_case_first_start_times(real_log)
            sim_first = self._extract_case_first_start_times(simulated_log)

            if real_first.empty or sim_first.empty:
                logger.warning(
                    "%s[Arrival Diagnostics] skipped: empty case-arrival series. "
                    "real_cases=%s, sim_cases=%s",
                    prefix,
                    len(real_first),
                    len(sim_first),
                )
                self._last_arrival_diagnostics = {}
                return {}

            real_inter = real_first.diff().dt.total_seconds().dropna() / 60.0
            sim_inter = sim_first.diff().dt.total_seconds().dropna() / 60.0

            real_span = (real_first.iloc[-1] - real_first.iloc[0]).total_seconds() / 60.0
            sim_span = (sim_first.iloc[-1] - sim_first.iloc[0]).total_seconds() / 60.0
            span_ratio = sim_span / max(real_span, 1e-9)

            real_inter_stats = self._summarize_numeric_series(real_inter)
            sim_inter_stats = self._summarize_numeric_series(sim_inter)

            real_hour_counts = real_first.dt.floor("h").value_counts().sort_index()
            sim_hour_counts = sim_first.dt.floor("h").value_counts().sort_index()

            real_weekday_hour = (
                real_first.dt.dayofweek.astype(str)
                + "-"
                + real_first.dt.hour.astype(str).str.zfill(2)
            ).value_counts()
            sim_weekday_hour = (
                sim_first.dt.dayofweek.astype(str)
                + "-"
                + sim_first.dt.hour.astype(str).str.zfill(2)
            ).value_counts()

            common_wh = set(real_weekday_hour.index).intersection(set(sim_weekday_hour.index))
            union_wh = set(real_weekday_hour.index).union(set(sim_weekday_hour.index))
            weekday_hour_jaccard = len(common_wh) / max(1, len(union_wh))

            diagnostics = {
                "real_cases": int(len(real_first)),
                "sim_cases": int(len(sim_first)),
                "real_first_start": str(real_first.iloc[0]),
                "real_last_start": str(real_first.iloc[-1]),
                "sim_first_start": str(sim_first.iloc[0]),
                "sim_last_start": str(sim_first.iloc[-1]),
                "real_span_min": float(real_span),
                "sim_span_min": float(sim_span),
                "span_ratio_sim_over_real": float(span_ratio),
                "real_interarrival_min": real_inter_stats,
                "sim_interarrival_min": sim_inter_stats,
                "real_active_hours": int(len(real_hour_counts)),
                "sim_active_hours": int(len(sim_hour_counts)),
                "real_hourly_count_mean": float(real_hour_counts.mean()) if len(real_hour_counts) else None,
                "sim_hourly_count_mean": float(sim_hour_counts.mean()) if len(sim_hour_counts) else None,
                "real_weekday_hour_buckets": int(len(real_weekday_hour)),
                "sim_weekday_hour_buckets": int(len(sim_weekday_hour)),
                "weekday_hour_jaccard": float(weekday_hour_jaccard),
            }

            logger.info(
                "%s[Arrival Diagnostics] cases: real=%s, sim=%s, "
                "real_start=%s, real_end=%s, sim_start=%s, sim_end=%s",
                prefix,
                diagnostics["real_cases"],
                diagnostics["sim_cases"],
                diagnostics["real_first_start"],
                diagnostics["real_last_start"],
                diagnostics["sim_first_start"],
                diagnostics["sim_last_start"],
            )

            logger.info(
                "%s[Arrival Diagnostics] span_min: real=%.3f, sim=%.3f, "
                "sim/real=%.3f. Interpretation: >1 means simulated arrivals are too sparse; "
                "<1 means simulated arrivals are too dense.",
                prefix,
                diagnostics["real_span_min"],
                diagnostics["sim_span_min"],
                diagnostics["span_ratio_sim_over_real"],
            )

            logger.info(
                "%s[Arrival Diagnostics] inter_arrival_min: "
                "real_mean=%.3f, real_median=%.3f, real_p90=%.3f; "
                "sim_mean=%.3f, sim_median=%.3f, sim_p90=%.3f",
                prefix,
                real_inter_stats.get("mean") or 0.0,
                real_inter_stats.get("median") or 0.0,
                real_inter_stats.get("p90") or 0.0,
                sim_inter_stats.get("mean") or 0.0,
                sim_inter_stats.get("median") or 0.0,
                sim_inter_stats.get("p90") or 0.0,
            )

            logger.info(
                "%s[Arrival Diagnostics] hourly_distribution: "
                "real_active_hours=%s, sim_active_hours=%s, "
                "real_hourly_mean=%.3f, sim_hourly_mean=%.3f, "
                "weekday_hour_jaccard=%.3f",
                prefix,
                diagnostics["real_active_hours"],
                diagnostics["sim_active_hours"],
                diagnostics["real_hourly_count_mean"] or 0.0,
                diagnostics["sim_hourly_count_mean"] or 0.0,
                diagnostics["weekday_hour_jaccard"],
            )

            self._last_arrival_diagnostics = copy.deepcopy(diagnostics)
            return diagnostics

        except Exception as exc:
            logger.warning("%s[Arrival Diagnostics] failed: %s", prefix, exc)
            self._last_arrival_diagnostics = {}
            return {}

    def log_arrival_bucket_fallback_diagnostics(
            self,
            bucket_diagnostics: dict | None = None,
            prefix: str = "",
    ) -> dict:
        """
        打印 arrival sampler 在仿真中使用 weekday_hour / weekday / hour / global / fallback
        的次数，用于判断 weekday-hour 模型是否真正发挥作用。
        """
        if bucket_diagnostics is None:
            bucket_diagnostics = getattr(self, "_last_arrival_bucket_diagnostics", {})

        if not isinstance(bucket_diagnostics, dict) or not bucket_diagnostics:
            logger.info(
                "%s[Arrival Bucket Diagnostics] unavailable. "
                "Patch drq/simulator.py and drq/discovery/time_discovery.py "
                "to collect bucket fallback statistics.",
                prefix,
            )
            return {}

        source_counts = (
            bucket_diagnostics.get("source_counts")
            or bucket_diagnostics.get("bucket_source_counts")
            or {}
        )
        if not isinstance(source_counts, dict):
            source_counts = {}

        total = sum(
            int(v)
            for v in source_counts.values()
            if isinstance(v, (int, float, np.integer, np.floating))
        )

        if total <= 0:
            logger.info(
                "%s[Arrival Bucket Diagnostics] no sampled inter-arrival deltas recorded.",
                prefix,
            )
            return bucket_diagnostics

        preferred_order = [
            "weekday_hour", "weekday", "hour", "global",
            "fallback_distribution", "none", "legacy_global",
            "histogram_sampling", "distribution", "tuple_distribution", "tuple_mean",
        ]
        ordered_keys = [k for k in preferred_order if k in source_counts]
        ordered_keys += [k for k in sorted(source_counts.keys()) if k not in ordered_keys]

        parts = []
        for key in ordered_keys:
            count = int(source_counts.get(key, 0) or 0)
            parts.append(f"{key}={count}({count / total:.3f})")

        logger.info(
            "%s[Arrival Bucket Diagnostics] source_usage total_samples=%s: %s",
            prefix,
            total,
            ", ".join(parts),
        )

        if "intensity_trend_enabled" in bucket_diagnostics:
            trend_values = bucket_diagnostics.get("trend_multipliers", [])
            if isinstance(trend_values, list) and trend_values:
                trend_stats = self._summarize_numeric_series(
                    pd.Series(trend_values, dtype="float64")
                )
                logger.info(
                    "%s[Arrival Trend Diagnostics] enabled=%s, "
                    "forecast_multiplier=%.3f, applied_multiplier_mean=%.3f, "
                    "min=%.3f, max=%.3f, n=%s",
                    prefix,
                    bool(bucket_diagnostics.get("intensity_trend_enabled", False)),
                    float(bucket_diagnostics.get("intensity_trend_forecast_multiplier", 1.0) or 1.0),
                    trend_stats.get("mean") or 1.0,
                    trend_stats.get("min") or 1.0,
                    trend_stats.get("max") or 1.0,
                    trend_stats.get("n") or 0,
                )
            else:
                logger.info(
                    "%s[Arrival Trend Diagnostics] enabled=%s, forecast_multiplier=%.3f",
                    prefix,
                    bool(bucket_diagnostics.get("intensity_trend_enabled", False)),
                    float(bucket_diagnostics.get("intensity_trend_forecast_multiplier", 1.0) or 1.0),
                )

        sampled_by_source = bucket_diagnostics.get("sampled_delta_min_by_source", {})
        if isinstance(sampled_by_source, dict):
            for key in ordered_keys:
                values = sampled_by_source.get(key, [])
                if not values:
                    continue
                stats = self._summarize_numeric_series(pd.Series(values, dtype="float64"))
                logger.info(
                    "%s[Arrival Bucket Diagnostics] sampled_delta_min[%s]: "
                    "n=%s, mean=%.3f, median=%.3f, p90=%.3f, min=%.3f, max=%.3f",
                    prefix,
                    key,
                    stats.get("n"),
                    stats.get("mean") or 0.0,
                    stats.get("median") or 0.0,
                    stats.get("p90") or 0.0,
                    stats.get("min") or 0.0,
                    stats.get("max") or 0.0,
                )

        return bucket_diagnostics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _format_metrics(self, metrics: dict) -> dict:
        formatted_metrics = {
            "control-flow": {},
            "temporal": {},
            "congestion": {},
            "resource": {},
            "extra": {},
        }

        for key in ["cfld", "ngd", "2gd", "3gd"]:
            if key in metrics:
                formatted_metrics["control-flow"][key] = metrics[key]

        for key in ["aed", "ced", "red"]:
            if key in metrics:
                formatted_metrics["temporal"][key] = metrics[key]

        for key in ["car", "ctd"]:
            if key in metrics:
                formatted_metrics["congestion"][key] = metrics[key]

        for key in ["cwd"]:
            if key in metrics:
                formatted_metrics["resource"][key] = metrics[key]

        for key in [
            "r2gd", "r3gd", "r4gd", "r5gd",
            "rfld", "hwd",
            "car_entropy", "ctd_entropy", "etd_entropy",
        ]:
            if key in metrics:
                formatted_metrics["extra"][key] = metrics[key]

        return formatted_metrics

    def _prepare_evaluation_dataframe(self, event_log) -> pd.DataFrame:
        if isinstance(event_log, pd.DataFrame):
            df_event_log = event_log.copy()
        else:
            df_event_log = pm4py.convert_to_dataframe(event_log)

        df_event_log["case:concept:name"] = df_event_log["case:concept:name"].astype(str)

        df_event_log["time:timestamp"] = pd.to_datetime(
            df_event_log["time:timestamp"],
            utc=True,
            errors="coerce",
        )

        df_event_log["start:timestamp"] = pd.to_datetime(
            df_event_log["start:timestamp"],
            utc=True,
            errors="coerce",
        )

        df_event_log.dropna(
            subset=["case:concept:name", "start:timestamp", "time:timestamp"],
            inplace=True,
        )

        df_event_log.sort_values(
            by=["start:timestamp", "time:timestamp"],
            inplace=True,
        )

        df_event_log.reset_index(drop=True, inplace=True)

        if df_event_log.empty:
            raise ValueError("Evaluation event log is empty after preprocessing.")

        return df_event_log

    def evaluate_on_event_log(
            self,
            event_log,
            set_as_current_metrics: bool = False,
            metrics_labels=None,
            enable_time_diagnostics: bool = True,
    ) -> dict:
        try:
            if self.drq_params is None:
                raise ValueError("No parameters available.")

            if metrics_labels is None:
                metrics_labels = [
                    "cfld", "3gd", "ngd",
                    "aed", "ced", "red",
                    "car", "ctd", "cwd",
                ]

            df_event_log = self._prepare_evaluation_dataframe(event_log)

            n_traces = df_event_log["case:concept:name"].nunique()
            start_t = df_event_log.iloc[0]["start:timestamp"]

            simulated_event_log = self.run_simulation(
                n_traces=n_traces,
                start_timestamp=start_t,
            )

            logger.info(
                "Evaluation simulation completed. "
                f"Real cases={n_traces}, simulated events={len(simulated_event_log)}"
            )

            self.log_case_arrival_diagnostics(
                df_event_log,
                simulated_event_log,
                prefix="[Single Eval] ",
            )
            self.log_arrival_bucket_fallback_diagnostics(
                getattr(self, "_last_arrival_bucket_diagnostics", {}),
                prefix="[Single Eval] ",
            )

            raw_metrics = evaluate(
                df_event_log,
                simulated_event_log,
                metrics_labels=metrics_labels,
            )

            if enable_time_diagnostics:
                time_diagnostics = build_time_decomposition_diagnostics(
                    real_log=df_event_log,
                    simulated_log=simulated_event_log,
                    top_k=10,
                )

                self._last_time_diagnostics = copy.deepcopy(time_diagnostics)

                wait_consistency = self._compute_wait_component_consistency(simulated_event_log)
                time_diagnostics["queue_external_overlap_diagnostics"] = wait_consistency
                logger.info(
                    "[Wait Component Consistency] %s",
                    wait_consistency,
                )

                for line in format_time_diagnostics_for_log(time_diagnostics):
                    logger.info(line)
            else:
                time_diagnostics = {}
                self._last_time_diagnostics = {}

            formatted_metrics = self._format_metrics(raw_metrics)

            if set_as_current_metrics:
                self.drq_metrics = formatted_metrics
                self.time_diagnostics = time_diagnostics

                # 单次评估时清空 repeated details，避免返回上一次 validation calibration 的缓存。
                self.repeated_metric_details = {}

            return formatted_metrics

        except Exception as e:
            logger.error(f"Error evaluating drq model on event log: {str(e)}")
            raise

    def evaluate_on_event_log_repeated(
            self,
            event_log,
            n_runs: int = 5,
            set_as_current_metrics: bool = False,
            metrics_labels=None,
            enable_time_diagnostics: bool = True,
            save_simulated_logs: bool = False,
            simulation_log_dir: str | None = None,
            simulation_log_prefix: str = "test_evaluation",
    ) -> dict:
        try:
            if self.drq_params is None:
                raise ValueError("No parameters available.")

            if metrics_labels is None:
                metrics_labels = [
                    "cfld", "3gd", "ngd",
                    "aed", "ced", "red",
                    "car", "ctd", "cwd",
                ]

            n_runs = max(1, int(n_runs))
            df_event_log = self._prepare_evaluation_dataframe(event_log)

            n_traces = df_event_log["case:concept:name"].nunique()
            start_t = df_event_log.iloc[0]["start:timestamp"]

            raw_results = []
            diagnostic_results = []
            arrival_diagnostic_results = []
            control_flow_diagnostic_results = []
            self.repeated_simulation_log_files = []

            if save_simulated_logs:
                if simulation_log_dir is None:
                    simulation_log_dir = "simulations"

                os.makedirs(simulation_log_dir, exist_ok=True)

            for run_idx in range(n_runs):
                logger.info(f"Repeated evaluation run {run_idx + 1}/{n_runs} started.")

                simulated_event_log = self.run_simulation(
                    n_traces=n_traces,
                    start_timestamp=start_t,
                    )
                control_flow_diagnostic_results.append(
                    copy.deepcopy(getattr(self, "_last_control_flow_diagnostics", {}) or {})
                )
                if save_simulated_logs:
                    sim_log_filename = (
                        f"{simulation_log_prefix}_run_{run_idx + 1}.csv"
                    )
                    sim_log_path = os.path.join(
                        simulation_log_dir,
                        sim_log_filename,
                    )

                    simulated_event_log.to_csv(
                        sim_log_path,
                        index=False,
                        encoding="utf-8-sig",
                    )

                    self.repeated_simulation_log_files.append({
                        "run": run_idx + 1,
                        "filename": sim_log_filename,
                        "path": sim_log_path,
                        "num_events": int(len(simulated_event_log)),
                    })

                    logger.info(
                        f"Repeated simulation log saved: {sim_log_path}, "
                        f"events={len(simulated_event_log)}"
                    )

                arrival_diagnostic_results.append(
                    self.log_case_arrival_diagnostics(
                        df_event_log,
                        simulated_event_log,
                        prefix=f"[Repeated Eval run {run_idx + 1}/{n_runs}] ",
                    )
                )
                self.log_arrival_bucket_fallback_diagnostics(
                    getattr(self, "_last_arrival_bucket_diagnostics", {}),
                    prefix=f"[Repeated Eval run {run_idx + 1}/{n_runs}] ",
                )

                raw_metrics = evaluate(
                    df_event_log,
                    simulated_event_log,
                    metrics_labels=metrics_labels,
                )

                if enable_time_diagnostics:
                    try:
                        run_diagnostics = build_time_decomposition_diagnostics(
                            real_log=df_event_log,
                            simulated_log=simulated_event_log,
                            top_k=10,
                        )
                        run_diagnostics["queue_external_overlap_diagnostics"] = self._compute_wait_component_consistency(
                            simulated_event_log
                        )
                        diagnostic_results.append(run_diagnostics)
                    except Exception as diag_error:
                        logger.warning(
                            f"Time decomposition diagnostics failed in run {run_idx + 1}: {diag_error}"
                        )

                raw_results.append(raw_metrics)

            averaged_raw_metrics = {}
            repeated_metric_details = {}

            for metric_name in metrics_labels:
                values = []

                for result in raw_results:
                    value = result.get(metric_name)

                    if isinstance(value, (int, float, np.integer, np.floating)):
                        if np.isfinite(value):
                            values.append(float(value))

                if values:
                    averaged_raw_metrics[metric_name] = float(np.mean(values))
                    repeated_metric_details[metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "n_runs": len(values),
                        "values": values,
                    }

                    logger.info(
                        f"Repeated metric {metric_name}: "
                        f"mean={averaged_raw_metrics[metric_name]}, "
                        f"std={repeated_metric_details[metric_name]['std']}"
                    )
                else:
                    logger.warning(
                        f"Metric {metric_name} has no valid numeric values."
                    )

            formatted_metrics = self._format_metrics(averaged_raw_metrics)
            self.repeated_metric_details = repeated_metric_details
            self._repeated_arrival_diagnostics = [
                row for row in arrival_diagnostic_results
                if isinstance(row, dict) and row
            ]

            if self._repeated_arrival_diagnostics:
                span_values = [
                    float(row.get("span_ratio_sim_over_real"))
                    for row in self._repeated_arrival_diagnostics
                    if isinstance(row.get("span_ratio_sim_over_real"), (int, float, np.integer, np.floating))
                ]
                if span_values:
                    logger.info(
                        "[Arrival Diagnostics] repeated span_ratio_sim_over_real: "
                        "mean=%.3f, std=%.3f, min=%.3f, max=%.3f, n=%s",
                        float(np.mean(span_values)),
                        float(np.std(span_values, ddof=1)) if len(span_values) > 1 else 0.0,
                        float(np.min(span_values)),
                        float(np.max(span_values)),
                        len(span_values),
                    )

            valid_cf_runs = [
                row for row in control_flow_diagnostic_results
                if isinstance(row, dict) and row.get("per_decision_point")
            ]
            repeated_cf = {"n_runs": int(len(valid_cf_runs)), "per_decision_point": []}
            if valid_cf_runs:
                grouped_cf = {}
                for run_idx, run_diag in enumerate(valid_cf_runs, start=1):
                    for row in run_diag.get("per_decision_point", []) or []:
                        sig = str(row.get("decision_point_signature", ""))
                        entry = grouped_cf.setdefault(sig, {
                            "decision_point_signature": sig,
                            "source_place": row.get("source_place"),
                            "real_visits_per_case": float(row.get("real_visits_per_case", 0.0) or 0.0),
                            "unresolved_structural_branches": list(row.get("unresolved_structural_branches", []) or []),
                            "sim_visits_per_case_values": [],
                            "visit_ratio_values": [],
                            "branch_tv_values": [],
                            "visit_tv_values": [],
                            "sim_max_visits_values": [],
                            "suspicion_values": [],
                        })
                        entry["sim_visits_per_case_values"].append(float(row.get("sim_visits_per_case", 0.0) or 0.0))
                        ratio = row.get("visit_ratio")
                        if isinstance(ratio, (int, float, np.integer, np.floating)) and np.isfinite(ratio):
                            entry["visit_ratio_values"].append(float(ratio))
                        entry["branch_tv_values"].append(float(row.get("branch_tv_distance", 0.0) or 0.0))
                        entry["visit_tv_values"].append(float(row.get("max_visit_branch_tv", 0.0) or 0.0))
                        entry["sim_max_visits_values"].append(int(row.get("sim_max_visits", 0) or 0))
                        entry["suspicion_values"].append(float(row.get("suspicion_score", 0.0) or 0.0))

                for entry in grouped_cf.values():
                    sim_values = entry.pop("sim_visits_per_case_values")
                    ratio_values = entry.pop("visit_ratio_values")
                    branch_tv_values = entry.pop("branch_tv_values")
                    visit_tv_values = entry.pop("visit_tv_values")
                    max_values = entry.pop("sim_max_visits_values")
                    suspicion_values = entry.pop("suspicion_values")
                    entry.update({
                        "sim_visits_per_case_mean": float(np.mean(sim_values)) if sim_values else 0.0,
                        "sim_visits_per_case_std": float(np.std(sim_values, ddof=1)) if len(sim_values) > 1 else 0.0,
                        "visit_ratio_mean": float(np.mean(ratio_values)) if ratio_values else None,
                        "branch_tv_mean": float(np.mean(branch_tv_values)) if branch_tv_values else 0.0,
                        "visit_tv_mean": float(np.mean(visit_tv_values)) if visit_tv_values else 0.0,
                        "sim_max_visits_max": int(max(max_values)) if max_values else 0,
                        "suspicion_mean": float(np.mean(suspicion_values)) if suspicion_values else 0.0,
                    })
                    repeated_cf["per_decision_point"].append(entry)

                repeated_cf["per_decision_point"].sort(
                    key=lambda row: float(row.get("suspicion_mean", 0.0)),
                    reverse=True,
                )
                logger.info("[CF-AUDIT repeated] Top decision points across %s runs:", len(valid_cf_runs))
                for row in repeated_cf["per_decision_point"][:5]:
                    logger.info(
                        "[CF-AUDIT repeated] place=%s real_visits/case=%.3f "
                        "sim_visits/case=%.3f±%.3f ratio_mean=%s branch_TV=%.3f "
                        "visit_TV=%.3f sim_max=%s unresolved=%s",
                        row.get("source_place"),
                        row.get("real_visits_per_case", 0.0),
                        row.get("sim_visits_per_case_mean", 0.0),
                        row.get("sim_visits_per_case_std", 0.0),
                        (f"{row.get('visit_ratio_mean'):.3f}" if row.get("visit_ratio_mean") is not None else "inf/na"),
                        row.get("branch_tv_mean", 0.0),
                        row.get("visit_tv_mean", 0.0),
                        row.get("sim_max_visits_max", 0),
                        row.get("unresolved_structural_branches", []),
                    )
            self._repeated_control_flow_diagnostics = repeated_cf

            if enable_time_diagnostics and diagnostic_results:
                aggregated_diagnostics = aggregate_time_decomposition_diagnostics(
                    diagnostic_results
                )
                overlap_rows = [
                    row.get("queue_external_overlap_diagnostics")
                    for row in diagnostic_results
                    if isinstance(row, dict) and isinstance(row.get("queue_external_overlap_diagnostics"), dict)
                ]
                if overlap_rows:
                    try:
                        numeric_keys = [
                            "mean_queue_wait_min",
                            "mean_external_delay_min",
                            "mean_total_wait_min",
                            "mean_component_sum_min",
                            "mean_residual_min",
                            "mean_abs_residual_min",
                            "negative_residual_share_gt_1min",
                            "component_sum_over_total_gt_1_05_share",
                        ]
                        overlap_summary = {"available": True, "n_runs": len(overlap_rows)}
                        for key in numeric_keys:
                            vals = [
                                float(row.get(key)) for row in overlap_rows
                                if isinstance(row.get(key), (int, float, np.integer, np.floating))
                                and np.isfinite(float(row.get(key)))
                            ]
                            if vals:
                                overlap_summary[f"{key}.mean"] = float(np.mean(vals))
                        overlap_summary["possible_double_count"] = bool(
                            any(bool(row.get("possible_double_count", False)) for row in overlap_rows)
                        )
                        aggregated_diagnostics["queue_external_overlap_diagnostics"] = overlap_summary
                        logger.info("[Wait Component Consistency] repeated summary: %s", overlap_summary)
                    except Exception as overlap_error:
                        logger.warning("Wait component consistency aggregation failed: %s", overlap_error)

                self._last_time_diagnostics = copy.deepcopy(aggregated_diagnostics)

                for line in format_time_diagnostics_for_log(aggregated_diagnostics):
                    logger.info(line)
            else:
                aggregated_diagnostics = {}
                self._last_time_diagnostics = {}

            if set_as_current_metrics:
                self.drq_metrics = formatted_metrics
                self.time_diagnostics = aggregated_diagnostics

            return formatted_metrics

        except Exception as e:
            logger.error(f"Error in repeated evaluation: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def run_simulation(
            self,
            n_traces=100,
            start_timestamp=None,
    ):
        try:
            if not self.drq_params:
                raise ValueError("No parameters available. Call discover_parameters first.")

            if start_timestamp is None:
                start_timestamp = datetime.now()

            net = self.process_model["net"]
            im = self.process_model["initial_marking"]
            fm = self.process_model["final_marking"]

            params = SimulatorParameters(
                net,
                im,
                fm,
                process_tree=self._get_process_tree_for_parameters(),
                control_flow_structure_source=self._get_control_flow_structure_source(),
            )
            sim_params = self._build_simulation_params()
            params.from_dict(sim_params)

            simulator = SimulatorEngine(params)

            logger.info(
                "Running complete-model simulation with %s traces starting at %s; "
                "control flow, arrival, dynamic resource, queue scheduling and "
                "external delay are all part of the runtime.",
                n_traces,
                start_timestamp,
            )

            result_df = simulator.apply(
                n_traces=n_traces,
                t_start=start_timestamp,
            )

            self._last_arrival_bucket_diagnostics = copy.deepcopy(
                getattr(simulator, "arrival_diagnostics", {}) or {}
            )
            self._last_control_flow_diagnostics = copy.deepcopy(
                getattr(simulator, "control_flow_diagnostics", {}) or {}
            )

            logger.info(f"Simulation completed. Generated {len(result_df)} events")
            return result_df

        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def _generate_petri_net_visualization(
            self,
            net,
            initial_marking,
            final_marking
    ):
        try:
            gviz = pn_visualizer.apply(
                net,
                initial_marking,
                final_marking,
                parameters={
                    pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "svg"
                },
            )

            svg_content = gviz.pipe(format="svg", encoding="utf-8")
            return self._enhance_svg_visualization(svg_content)

        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return None

    def _enhance_svg_visualization(self, svg_content):
        try:
            enhanced_svg = svg_content.replace(
                "<svg",
                """<svg style="max-width: 100%; height: auto; background: white; border-radius: 8px;" """,
            )

            enhanced_svg = enhanced_svg.replace(
                'fill="lightblue"',
                'fill="#6f42c1" stroke="#5a2d91" stroke-width="2"',
            )
            enhanced_svg = enhanced_svg.replace(
                'fill="orange"',
                'fill="#28a745" stroke="#1e7e34" stroke-width="2"',
            )
            enhanced_svg = enhanced_svg.replace(
                'fill="black"',
                'fill="#333" stroke="#444" stroke-width="1"',
            )

            style_css = """
            <defs>
            <style type="text/css">
            <![CDATA[
            text {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 12px;
            }
            ]]>
            </style>
            </defs>
            """

            enhanced_svg = enhanced_svg.replace("<svg", style_css + "<svg", 1)
            return enhanced_svg

        except Exception as e:
            logger.error(f"Error enhancing SVG: {str(e)}")
            return svg_content

    def save_current_petri_net_to_pnml(self, output_path: str):
        """
        Save the current best Petri net to a PNML file.
        """
        try:
            if not self.process_model:
                raise ValueError("No process model available.")

            net = self.process_model.get("net")
            initial_marking = self.process_model.get("initial_marking")
            final_marking = self.process_model.get("final_marking")

            if net is None or initial_marking is None or final_marking is None:
                raise ValueError("Current process model does not contain net/markings.")

            pm4py.write_pnml(
                net,
                initial_marking,
                final_marking,
                output_path,
            )

            logger.info(f"Current Petri net saved to PNML: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving current Petri net to PNML: {str(e)}")
            raise
    # ------------------------------------------------------------------
    # Save/load
    # ------------------------------------------------------------------
    def save_parameters_to_json(self, output_path):
        try:
            if not self.drq_params:
                raise ValueError("No parameters discovered yet.")

            net = self.process_model["net"]
            im = self.process_model["initial_marking"]
            fm = self.process_model["final_marking"]

            params = SimulatorParameters(
                net,
                im,
                fm,
                process_tree=self._get_process_tree_for_parameters(),
                control_flow_structure_source=self._get_control_flow_structure_source(),
            )
            params.from_dict(self.drq_params)
            params.to_json(output_path)

            logger.info(f"Parameters saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving parameters: {str(e)}")
            raise

    def load_parameters_from_json_file(self, json_path):
        try:
            net = self.process_model["net"]
            im = self.process_model["initial_marking"]
            fm = self.process_model["final_marking"]

            params = SimulatorParameters(
                net,
                im,
                fm,
                process_tree=self._get_process_tree_for_parameters(),
                control_flow_structure_source=self._get_control_flow_structure_source(),
            )
            params.from_json(json_path)

            logger.info(f"Loading parameters from JSON file: {json_path}")

            self.drq_params = params.to_dict()
            return self._convert_drq_to_app_format()

        except Exception as e:
            logger.error(f"Error loading parameters from JSON file: {str(e)}")
            raise


    def _format_decision_point_label(
            self,
            index: int,
            branches: list[str],
            branch_display_names: dict,
            model_kind: str,
    ) -> str:
        """
        Build a readable label for the frontend decision-point selector.

        The current control-flow model is decision-point based rather than
        transition based.  Therefore the UI should show one option per
        decision point, not one option per Petri-net transition.
        """
        branch_labels = [
            str(branch_display_names.get(branch, branch))
            for branch in branches
        ]
        short_labels = branch_labels[:4]
        suffix = "" if len(branch_labels) <= 4 else f" / ... (+{len(branch_labels) - 4})"
        kind_label = "Decision tree" if model_kind == "decision_tree" else "Empirical"
        return f"Decision Point {index}: " + " / ".join(short_labels) + suffix + f"  ·  {kind_label}"

    def _collect_decision_tree_safe_edit_nodes(self, rules: dict, path=None, condition_path=None) -> list[dict]:
        """
        Collect editable node/leaf positions for first-version safe what-if editing.

        Returned paths are compatible with routes.py safe-edit operations:
        - set_threshold: node_path of an internal node;
        - set_leaf_probs: leaf_path of a leaf node;
        - prune_subtree: node_path of either an internal node or a leaf node.
        """
        path = list(path or [])
        condition_path = list(condition_path or [])
        rows = []

        if not isinstance(rules, dict):
            return rows

        node_type = rules.get("type")
        path_string = ".".join(path) if path else "(root)"

        if node_type == "leaf":
            rows.append({
                "path": path,
                "path_string": path_string,
                "type": "leaf",
                "display_label": f"Leaf {path_string}",
                "condition_path": condition_path,
                "editable_operations": ["set_leaf_probs", "prune_subtree"],
                "value": rules.get("value"),
                "branch_probs": rules.get("branch_probs", {}),
            })
            return rows

        feature = rules.get("feature")
        threshold = rules.get("threshold")
        rows.append({
            "path": path,
            "path_string": path_string,
            "type": "node",
            "display_label": f"Rule node {path_string}: {feature} <= {threshold}",
            "condition_path": condition_path,
            "editable_operations": ["set_threshold", "prune_subtree"],
            "feature": feature,
            "threshold": threshold,
        })

        children = rules.get("children", {})
        if isinstance(children, dict):
            if "true" in children:
                rows.extend(
                    self._collect_decision_tree_safe_edit_nodes(
                        children.get("true"),
                        path + ["true"],
                        condition_path + [
                            {
                                "feature": feature,
                                "operator": "<=",
                                "threshold": threshold,
                            }
                        ],
                    )
                )
            if "false" in children:
                rows.extend(
                    self._collect_decision_tree_safe_edit_nodes(
                        children.get("false"),
                        path + ["false"],
                        condition_path + [
                            {
                                "feature": feature,
                                "operator": ">",
                                "threshold": threshold,
                            }
                        ],
                    )
                )

        return rows

    def _build_control_flow_what_if_summary(self, decision_point_models: dict) -> dict:
        """
        Build a frontend-friendly Scheme-B summary.

        Scheme B shows *all* editable decision points:
        - empirical decision points: show a branch-probability editor;
        - decision-tree decision points: show safe tree edits.

        This intentionally replaces the legacy "Select Transition" frontend
        logic.  The selector should use decision_point_options, not
        process_model.transitions.
        """
        summary = {
            "editor_mode": "decision_point",
            "selector_label": "Select Decision Point",
            "legacy_transition_selector_deprecated": True,
            "show_all_decision_points": True,
            "supported_operations": [
                {
                    "operation": "set_empirical_probs",
                    "scope": "empirical decision point",
                    "description": "Modify empirical branch probabilities for an empirical decision point.",
                },
                {
                    "operation": "set_leaf_probs",
                    "scope": "decision-tree leaf",
                    "description": "Modify branch probabilities of an existing decision-tree leaf.",
                },
                {
                    "operation": "set_threshold",
                    "scope": "decision-tree internal node",
                    "description": "Modify the threshold of an existing decision-tree internal node without changing its feature.",
                },
                {
                    "operation": "prune_subtree",
                    "scope": "decision-tree subtree",
                    "description": "Replace an existing subtree with a safe leaf containing branch probabilities.",
                },
            ],
            "counts": {
                "total": 0,
                "empirical": 0,
                "decision_tree": 0,
                "disabled": 0,
            },
            "decision_point_options": [],
            "decision_points": [],
        }

        if not isinstance(decision_point_models, dict):
            return summary

        decision_points = decision_point_models.get("decision_points", {})
        if not isinstance(decision_points, dict):
            return summary

        for index, (signature, model) in enumerate(decision_points.items(), start=1):
            if not isinstance(model, dict):
                continue

            model_kind = str(model.get("model_kind", "empirical"))
            if model_kind not in ("empirical", "decision_tree"):
                model_kind = "empirical"

            branches = [str(b) for b in model.get("branches", []) if str(b)]
            branch_display_names = model.get("branch_display_names", {})
            if not isinstance(branch_display_names, dict):
                branch_display_names = {}

            branch_labels = [
                str(branch_display_names.get(branch, branch))
                for branch in branches
            ]

            enabled = bool(model.get("enabled", True))
            label = self._format_decision_point_label(
                index=index,
                branches=branches,
                branch_display_names=branch_display_names,
                model_kind=model_kind,
            )

            safe_operations = []
            editable_nodes = []
            if model_kind == "empirical":
                safe_operations = ["set_empirical_probs"]
                editable_nodes = [
                    {
                        "path": [],
                        "path_string": "(empirical)",
                        "type": "empirical_probability_table",
                        "display_label": "Empirical branch probabilities",
                        "editable_operations": ["set_empirical_probs"],
                        "branch_probs": model.get("empirical_probs", {}),
                    }
                ]
                summary["counts"]["empirical"] += 1
            elif model_kind == "decision_tree":
                safe_operations = [
                    "set_leaf_probs",
                    "set_threshold",
                    "prune_subtree",
                ]
                editable_nodes = self._collect_decision_tree_safe_edit_nodes(
                    model.get("rules", {})
                )
                summary["counts"]["decision_tree"] += 1

            if not enabled:
                summary["counts"]["disabled"] += 1

            option = {
                "label": label,
                "value": str(signature),
                "signature": str(signature),
                "model_kind": model_kind,
                "model_kind_label": "Decision tree" if model_kind == "decision_tree" else "Empirical probabilities",
                "enabled": enabled,
                "n_samples": model.get("n_samples"),
                "dominant_prob": model.get("dominant_prob"),
                "branch_count": len(branches),
                "branch_labels": branch_labels,
                "source_place": model.get("source_place"),
                "is_free_choice": model.get("is_free_choice"),
                "has_invisible_branch": model.get("has_invisible_branch"),
                "unobserved_in_replay": model.get("unobserved_in_replay", False),
                "safe_operations": safe_operations,
            }

            row = {
                **option,
                "branches": branches,
                "branch_display_names": branch_display_names,
                "empirical_probs": model.get("empirical_probs", {}),
                "feature_columns": model.get("feature_columns", []) if model_kind == "decision_tree" else [],
                "editable_nodes": editable_nodes,
                "rules": model.get("rules") if model_kind == "decision_tree" else None,
            }

            summary["decision_point_options"].append(option)
            summary["decision_points"].append(row)

        summary["counts"]["total"] = len(summary["decision_points"])
        return summary

    # ------------------------------------------------------------------
    # App-format conversion
    # ------------------------------------------------------------------
    def _convert_drq_to_app_format(self):
        try:
            logger.info("Converting modular drq parameters to application format")

            process_model = {
                "activities": self.process_model["activities"],
                "places": self.process_model["places"],
                "transitions": self.process_model["transitions"],
                "map_transitionName_to_id": self.process_model["map_transitionName_to_id"],
                "visualization": self.process_model["visualization"],
                "fitness": self.process_model["fitness"],
                "precision": self.process_model["precision"],
                "f_measure": self.process_model["f_measure"],
                "process_tree_available": bool(self.process_model.get("process_tree_available", False)),
                "process_tree_source": self.process_model.get("process_tree_source", "unknown"),
                "control_flow_structure": "petri_structural_conflict_decision_points",
                "control_flow_structure_source": self._get_control_flow_structure_source(),
            }

            modular_params = self.drq_params.get("modular_params", {})

            control_flow_params = self.drq_params.get("control_flow_params", {})
            decision_point_models = control_flow_params.get(
                "decision_point_branch_models",
                modular_params.get("modules", {})
                .get("control_flow", {})
                .get("decision_point_branch_models", {}),
            )
            control_flow_what_if = self._build_control_flow_what_if_summary(
                decision_point_models
            )
            process_model["control_flow_editor_mode"] = "decision_point"
            process_model["decision_points"] = control_flow_what_if.get(
                "decision_point_options",
                [],
            )

            resource_params = self.drq_params.get("resource_params", {})
            arrival_params = self.drq_params.get("arrival_params", {})
            execution_params = self.drq_params.get("execution_time_params", {})
            waiting_params = self.drq_params.get("waiting_time_params", {})
            data_attribute_params = self.drq_params.get("data_attribute_params", {})

            arrival_dist = arrival_params.get("arrival_time_distributions", {})
            arrival_calendar = arrival_params.get("arrival_calendar", {})

            if (
                    isinstance(arrival_dist, dict)
                    and arrival_dist.get("model_type") == "case_arrival_decision_tree_interarrival"
            ):
                inter_arrival_time = {
                    "distribution": "context_observed_histogram",
                    "model_type": arrival_dist.get("model_type"),
                    "calendar": arrival_calendar,
                    "global_histogram": arrival_dist.get("global_inter_arrival", {}),
                    "weekday_hour_histograms": arrival_dist.get("weekday_hour_inter_arrivals", {}),
                    "weekday_histograms": arrival_dist.get("weekday_inter_arrivals", {}),
                    "hour_histograms": arrival_dist.get("hour_inter_arrivals", {}),
                    "fallback_order": arrival_dist.get("fallback_order", []),
                    "statistics": arrival_dist.get("statistics", {}),
                    "sampling": arrival_dist.get("sampling", {}),
                    "regularization": arrival_dist.get("regularization", {}),
                    "intensity_trend": arrival_dist.get("intensity_trend", {}),
                    "arrival_time_scale": arrival_params.get("arrival_time_scale", 1.0),
                    "arrival_demand_multiplier": arrival_params.get("arrival_demand_multiplier", 1.0),
                    "arrival_time_shift_min": arrival_params.get("arrival_time_shift_min", 0.0),
                    "raw_model": arrival_dist,
                }
            elif (
                    isinstance(arrival_dist, dict)
                    and arrival_dist.get("model_type") == "case_arrival_calendar_observed_histogram"
            ):
                # Legacy global histogram support for old saved parameter files.
                inter_arrival_time = {
                    "distribution": "observed_histogram",
                    "model_type": arrival_dist.get("model_type"),
                    "calendar": arrival_calendar,
                    "histogram": arrival_dist.get("inter_arrival_times", {}),
                    "statistics": arrival_dist.get("statistics", {}),
                    "arrival_time_scale": arrival_params.get("arrival_time_scale", 1.0),
                    "arrival_demand_multiplier": arrival_params.get("arrival_demand_multiplier", 1.0),
                    "arrival_time_shift_min": arrival_params.get("arrival_time_shift_min", 0.0),
                    "raw_model": arrival_dist,
                }
            elif isinstance(arrival_dist, dict) and "dist_name" in arrival_dist:
                arr_params = arrival_dist.get("params", [1.0])
                inter_arrival_time = {
                    "distribution": arrival_dist.get("dist_name", "fixed"),
                    "parameters": self._convert_drq_params_to_app(
                        arrival_dist.get("dist_name", "fixed"),
                        arr_params,
                        arrival_dist.get("min_value", 0.0),
                        arrival_dist.get("max_value", 0.0),
                        arrival_dist.get("mean_value", arr_params[0] if arr_params else 0.0),
                    ),
                    "calendar": arrival_calendar,
                    "arrival_time_scale": arrival_params.get("arrival_time_scale", 1.0),
                    "arrival_demand_multiplier": arrival_params.get("arrival_demand_multiplier", 1.0),
                }
            else:
                inter_arrival_time = {
                    "distribution": arrival_dist,
                    "calendar": arrival_calendar,
                    "arrival_time_scale": arrival_params.get("arrival_time_scale", 1.0),
                    "arrival_demand_multiplier": arrival_params.get("arrival_demand_multiplier", 1.0),
                }

            app_data_attributes = {
                "label_data_attributes": data_attribute_params.get("label_data_attributes", []),
                "label_data_attributes_categorical": data_attribute_params.get(
                    "label_data_attributes_categorical",
                    [],
                ),
                "attribute_values_label_categorical": data_attribute_params.get(
                    "attribute_values_label_categorical",
                    {},
                ),
                "distribution_data_attributes": data_attribute_params.get(
                    "distribution_data_attributes",
                    {},
                ),
            }

            base_service = execution_params.get(
                "base_service_time_distributions",
                {},
            )

            queue_scheduling_params = self.drq_params.get(
                "queue_scheduling_params",
                {},
            )

            external_delay_params = self.drq_params.get(
                "external_delay_params",
                {},
            )

            dynamic_resource_params = {
                "resources": resource_params.get("resources", []),
                "act_to_resources": resource_params.get("act_to_resources", {}),
                "multitasking_resource": resource_params.get("multitasking_resource", []),
                "calendars": resource_params.get("calendars", {}),
                "base_service_time_distributions": base_service,
                "resource_state_dbn": self.drq_params.get(
                    "resource_state_params",
                    self._default_resource_state_params(),
                ),
            }

            parameters = {
                "process_model": process_model,
                "drq_metrics": self.drq_metrics,
                "time_diagnostics": getattr(self, "time_diagnostics", {}),
                "runtime_recency": self.drq_params.get(
                    "runtime_recency",
                    modular_params.get("runtime_recency", {}) if isinstance(modular_params, dict) else {},
                ),

                # Primary five-module view
                "modular_params": modular_params,
                "control_flow_params": {
                    "editor_mode": "decision_point",
                    "selector_label": "Select Decision Point",
                    "decision_point_branch_models": decision_point_models,
                    "decision_point_options": control_flow_what_if.get(
                        "decision_point_options",
                        [],
                    ),
                    "safe_what_if": control_flow_what_if,
                },
                "arrival_params": {
                    "arrival_calendar": arrival_calendar,
                    "arrival_model": arrival_dist,
                    "arrival_time_scale": arrival_params.get("arrival_time_scale", 1.0),
                    "arrival_demand_multiplier": arrival_params.get("arrival_demand_multiplier", 1.0),
                    "arrival_time_shift_min": arrival_params.get("arrival_time_shift_min", 0.0),
                },
                "dynamic_resource_params": dynamic_resource_params,
                "queue_scheduling_params": queue_scheduling_params,
                "conditional_prioritization_params": (
                    queue_scheduling_params.get("conditional_prioritization", {})
                    if isinstance(queue_scheduling_params, dict) else {}
                ),
                "external_delay_params": external_delay_params,

                # Legacy-compatible view for the current front end
                "transition_params": {
                    "deprecated_for_control_flow_what_if": True,
                    "note": (
                        "The active control-flow model is decision-point based. "
                        "Use control_flow_params.decision_point_options instead of selecting all transitions."
                    ),
                    "transition_weights": self.drq_params
                    .get("transition_params", {})
                    .get("transition_weights", {}),
                },
                "execution_time_params": {
                    "activity_durations": execution_params.get(
                        "execution_time_distributions",
                        {},
                    ),
                    "base_service_time_distributions": base_service,
                },
                "resource_params": {
                    "resources": resource_params.get("resources", []),
                    "resource_weights": resource_params.get("resource_weights", {}),
                    "multitasking_resource": resource_params.get("multitasking_resource", []),
                    "act_to_resources": resource_params.get("act_to_resources", {}),
                    "calendars": resource_params.get("calendars", {}),
                    "queue_scheduling_model": queue_scheduling_params,
                },
                "waiting_time_params": {
                    "resource_waiting_times": waiting_params.get(
                        "waiting_time_distributions",
                        {},
                    ),
                    "external_delay_model": external_delay_params,
                    "note": (
                        "Waiting time is generated by queue scheduling plus "
                        "an external residual-delay model when enabled."
                    ),
                },
                "inter_arrival_params": {
                    "inter_arrival_time": inter_arrival_time,
                },
                "data_attribute_params": app_data_attributes,
                "resource_state_params": self.drq_params.get(
                    "resource_state_params",
                    self._default_resource_state_params(),
                ),
                "queue_params": self.drq_params.get(
                    "queue_params",
                    self._default_queue_params(),
                ),
            }

            logger.info(
                "Converted modular parameters: "
                f"{len(resource_params.get('resources', []))} resources, "
                f"{len(process_model.get('activities', []))} activities"
            )

            return parameters

        except Exception as e:
            logger.error(f"Error converting drq parameters: {str(e)}")
            raise

    def _convert_drq_params_to_app(
            self,
            dist_name,
            params,
            min_val,
            max_val,
            mean_val
    ):
        try:
            app_params = {
                "min_value": float(min_val),
                "max_value": float(max_val),
                "mean_value": float(mean_val),
            }

            if dist_name == "fixed":
                app_params["value"] = float(params[0]) if params else float(mean_val)

            elif dist_name == "norm":
                app_params["mean"] = float(params[0]) if len(params) > 0 else float(mean_val)
                app_params["std"] = float(params[1]) if len(params) > 1 else 1.0

            elif dist_name == "expon":
                scale_val = float(params[1]) if len(params) > 1 else float(mean_val)
                app_params["scale"] = scale_val
                app_params["mean"] = scale_val

            elif dist_name == "uniform":
                if len(params) >= 2:
                    app_params["min"] = float(params[0])
                    app_params["max"] = float(params[1])
                else:
                    app_params["min"] = float(min_val)
                    app_params["max"] = float(max_val)

            return app_params

        except Exception as e:
            logger.error(f"Error converting parameters for {dist_name}: {str(e)}")
            return {
                "mean": 15.0,
                "std": 2.0,
                "min_value": 0.0,
                "max_value": 60.0,
                "mean_value": 15.0,
            }

    # ------------------------------------------------------------------
    # 验证集运行参数校准：到达时间、队列策略、外部延迟
    # ------------------------------------------------------------------
    def _flatten_formatted_metrics(self, metrics: dict) -> dict:
        """
        Convert grouped metrics returned by _format_metrics() into a flat dict.
        Example: {"temporal": {"red": 1}} -> {"red": 1}
        """
        flat = {}
        if not isinstance(metrics, dict):
            return flat

        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[sub_key] = sub_value
            else:
                flat[key] = value
        return flat

    def _get_external_delay_global_scale(self) -> float:
        if not isinstance(self.drq_params, dict):
            return 1.0

        ext = self.drq_params.get("external_delay_params", {})
        if not isinstance(ext, dict):
            return 1.0

        runtime = ext.get("runtime", {}) if isinstance(ext.get("runtime", {}), dict) else {}
        value = runtime.get("global_scale", ext.get("global_scale", 1.0))
        try:
            value = float(value)
            if np.isfinite(value):
                return value
        except Exception:
            pass
        return 1.0

    def _get_external_delay_max_delay_quantile(self) -> float:
        if not isinstance(self.drq_params, dict):
            return 0.95

        ext = self.drq_params.get("external_delay_params", {})
        if not isinstance(ext, dict):
            return 0.95

        runtime = ext.get("runtime", {}) if isinstance(ext.get("runtime", {}), dict) else {}
        # ``None`` means "inherit", not "invalid -> default".  This mirrors
        # ExternalDelayRuntime so a runtime placeholder cannot hide a selected
        # top-level quantile.
        value = runtime.get("max_delay_quantile")
        if value is None:
            value = ext.get("max_delay_quantile", 0.95)

        try:
            value = float(value)
            if np.isfinite(value):
                return float(np.clip(value, 0.0, 1.0))
        except Exception:
            pass

        return 0.95

    def _set_queue_scheduling_runtime_config(
            self,
            score_mixing_weights: dict | None = None,
            strategy_name: str | None = None,
    ):
        """
        设置队列调度运行时策略。

        该函数只改变运行时队列调度打分方式，不重新发现队列调度树。
        QueueSchedulingRuntime 会优先读取：
            queue_scheduling_params["score_mixing_weights"]
        或：
            queue_scheduling_params["hybrid_score_weights"]
        """

        if not isinstance(self.drq_params, dict):
            return

        queue_model = self.drq_params.setdefault("queue_scheduling_params", {})
        if not isinstance(queue_model, dict):
            self.drq_params["queue_scheduling_params"] = {}
            queue_model = self.drq_params["queue_scheduling_params"]

        if score_mixing_weights is not None:
            weights = self._normalize_queue_score_weights(score_mixing_weights)
            queue_model["score_mixing_weights"] = weights
            queue_model["hybrid_score_weights"] = weights

        if strategy_name is not None:
            queue_model["selected_strategy"] = str(strategy_name)
            queue_model["resource_policy_mode"] = str(strategy_name)
            queue_model["selected_resource_policy"] = str(strategy_name)

        queue_model.setdefault("adaptive_resource_capacity", True)
        queue_model.setdefault("max_queue_wait_min", 480.0)
        queue_model.setdefault("fallback_top_k_when_all_blocked", 3)

        # 同步 modular schema
        modular = self.drq_params.get("modular_params", {})
        if isinstance(modular, dict):
            modules = modular.get("modules", {})
            if isinstance(modules, dict):
                queue_module = modules.get("queue_scheduling")
                if isinstance(queue_module, dict):
                    queue_module["selected_strategy"] = strategy_name
                    queue_module["resource_policy_mode"] = strategy_name

                    nested_model = queue_module.get("queue_scheduling_model")
                    if isinstance(nested_model, dict):
                        if score_mixing_weights is not None:
                            nested_model["score_mixing_weights"] = weights
                            nested_model["hybrid_score_weights"] = weights
                        if strategy_name is not None:
                            nested_model["selected_strategy"] = str(strategy_name)
                            nested_model["resource_policy_mode"] = str(strategy_name)
                            nested_model["selected_resource_policy"] = str(strategy_name)
                        nested_model.setdefault("adaptive_resource_capacity", True)
                        nested_model.setdefault("max_queue_wait_min", 480.0)
                        nested_model.setdefault("fallback_top_k_when_all_blocked", 3)
                        if strategy_name is not None:
                            nested_model["selected_strategy"] = str(strategy_name)

    def _normalize_queue_score_weights(self, weights: dict) -> dict:
        raw = {
            "tree": weights.get("tree", 0.0),
            "frequency": weights.get("frequency", 0.0),
            "availability": weights.get("availability", 0.0),
            "calendar_fit": weights.get("calendar_fit", 0.0),
        }

        clean = {}

        for key, value in raw.items():
            try:
                value = float(value)
                if not np.isfinite(value):
                    value = 0.0
            except Exception:
                value = 0.0

            clean[key] = max(0.0, value)

        total = sum(clean.values())

        if total <= 0:
            clean = {
                "tree": 0.0,
                "frequency": 1.0,
                "availability": 0.0,
                "calendar_fit": 0.0,
            }
            total = 1.0

        return {
            key: float(value) / total
            for key, value in clean.items()
        }
    def _set_external_delay_runtime_config(
            self,
            global_scale: float | None = None,
            max_delay_quantile: float | None = None,
    ):
        if not isinstance(self.drq_params, dict):
            return

        ext = self.drq_params.setdefault("external_delay_params", {})
        if not isinstance(ext, dict):
            self.drq_params["external_delay_params"] = {}
            ext = self.drq_params["external_delay_params"]

        ext.setdefault("runtime", {})
        if not isinstance(ext["runtime"], dict):
            ext["runtime"] = {}

        if global_scale is not None:
            try:
                global_scale = float(global_scale)
                if not np.isfinite(global_scale):
                    global_scale = 1.0
            except Exception:
                global_scale = 1.0

            global_scale = max(0.0, global_scale)

            ext["global_scale"] = global_scale
            ext["runtime"]["global_scale"] = global_scale

        if max_delay_quantile is not None:
            try:
                max_delay_quantile = float(max_delay_quantile)
                if not np.isfinite(max_delay_quantile):
                    max_delay_quantile = 0.95
            except Exception:
                max_delay_quantile = 0.95

            max_delay_quantile = float(np.clip(max_delay_quantile, 0.0, 1.0))

            ext["max_delay_quantile"] = max_delay_quantile
            ext["runtime"]["max_delay_quantile"] = max_delay_quantile

        # Keep modular schema consistent.
        modular = self.drq_params.get("modular_params", {})
        if isinstance(modular, dict):
            modules = modular.get("modules", {})
            if isinstance(modules, dict):
                external_module = modules.get("external_delay")
                if isinstance(external_module, dict):
                    if global_scale is not None:
                        external_module["global_scale"] = global_scale
                    if max_delay_quantile is not None:
                        external_module["max_delay_quantile"] = max_delay_quantile

                    external_model = external_module.get("external_delay_model")
                    if isinstance(external_model, dict):
                        external_model.setdefault("runtime", {})
                        if not isinstance(external_model["runtime"], dict):
                            external_model["runtime"] = {}

                        if global_scale is not None:
                            external_model["global_scale"] = global_scale
                            external_model["runtime"]["global_scale"] = global_scale

                        if max_delay_quantile is not None:
                            external_model["max_delay_quantile"] = max_delay_quantile
                            external_model["runtime"]["max_delay_quantile"] = max_delay_quantile


    def _get_top_external_delay_activities(self, top_k: int = 3) -> list[str]:
        """Return Top-K enabled external-delay activities by residual contribution."""
        if not isinstance(self.drq_params, dict):
            return []
        ext = self.drq_params.get("external_delay_params", {})
        if not isinstance(ext, dict):
            return []
        activities = ext.get("activities", {})
        if not isinstance(activities, dict):
            return []

        rows = []
        for activity, model in activities.items():
            if not isinstance(model, dict):
                continue
            enabled = bool(
                model.get("enabled", model.get("external_delay_enabled", model.get("timer_delay_enabled", False)))
            )
            if not enabled:
                continue
            stats = model.get("residual_statistics", {}) if isinstance(model.get("residual_statistics", {}), dict) else {}
            try:
                n = float(stats.get("n", model.get("n_samples", 0)) or 0.0)
                total = float(stats.get("total_residual_min", 0.0) or 0.0)
                p95 = float(stats.get("p95", model.get("activity_default_p95", 0.0)) or 0.0)
                p98 = float(stats.get("p98", model.get("activity_default_p98", 0.0)) or 0.0)
                mean = float(stats.get("mean", 0.0) or 0.0)
                significance = float(stats.get("significance_score", 0.0) or 0.0)
            except Exception:
                continue
            score = total
            if score <= 0.0:
                score = max(0.0, p95) * max(1.0, n)
            score += 0.25 * max(0.0, p98) + 0.10 * max(0.0, mean) * max(1.0, n) + 0.01 * max(0.0, significance)
            rows.append((str(activity), float(score), float(total), float(p95), float(n)))

        rows.sort(key=lambda item: (-item[1], item[0]))
        try:
            top_k = max(1, int(top_k))
        except Exception:
            top_k = 3
        return [activity for activity, *_ in rows[:top_k]]

    def _set_external_delay_activity_scales(
            self,
            activity_scales: dict | None = None,
            reset_others: bool = False,
    ):
        """Set per-activity external timer scale on top-level and modular params."""
        if not isinstance(self.drq_params, dict):
            return

        clean_scales = {}
        for activity, scale in (activity_scales or {}).items():
            try:
                scale = float(scale)
                if np.isfinite(scale):
                    clean_scales[str(activity)] = max(0.0, scale)
            except Exception:
                continue

        def _apply_to_external_model(ext_model: dict):
            if not isinstance(ext_model, dict):
                return
            activities = ext_model.get("activities", {})
            if not isinstance(activities, dict):
                return
            for activity, model in activities.items():
                if not isinstance(model, dict):
                    continue
                if str(activity) in clean_scales:
                    scale = float(clean_scales[str(activity)])
                elif reset_others:
                    scale = 1.0
                else:
                    continue
                model.setdefault("runtime", {})
                if not isinstance(model["runtime"], dict):
                    model["runtime"] = {}
                model["runtime"]["activity_scale"] = float(scale)
                model["activity_scale"] = float(scale)
            ext_model.setdefault("runtime", {})
            if isinstance(ext_model["runtime"], dict):
                ext_model["runtime"]["activity_scale_mode"] = "top_activity_scale_calibrated"
            ext_model["activity_scales"] = dict(clean_scales)

        ext = self.drq_params.setdefault("external_delay_params", {})
        if isinstance(ext, dict):
            _apply_to_external_model(ext)

        modular = self.drq_params.get("modular_params", {})
        if isinstance(modular, dict):
            modules = modular.get("modules", {})
            if isinstance(modules, dict):
                external_module = modules.get("external_delay")
                if isinstance(external_module, dict):
                    external_model = external_module.get("external_delay_model")
                    if isinstance(external_model, dict):
                        _apply_to_external_model(external_model)
                    external_module["activity_scales"] = dict(clean_scales)

    def _compute_wait_component_consistency(self, simulated_log: pd.DataFrame) -> dict:
        """Check whether Queue/Calendar/External waiting components are double counted."""
        required = {"queue_wait_min", "external_delay_min", "total_wait_min"}
        if simulated_log is None or not isinstance(simulated_log, pd.DataFrame) or not required.issubset(simulated_log.columns):
            return {"available": False, "reason": "missing_wait_component_columns"}

        cols = list(required) + (["calendar_wait_min"] if "calendar_wait_min" in simulated_log.columns else [])
        df = simulated_log[cols].copy()
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        if df.empty:
            return {"available": False, "reason": "empty_wait_component_columns"}

        queue = df["queue_wait_min"].clip(lower=0.0)
        calendar = df["calendar_wait_min"].clip(lower=0.0) if "calendar_wait_min" in df.columns else pd.Series(0.0, index=df.index)
        external = df["external_delay_min"].clip(lower=0.0)
        total = df["total_wait_min"].clip(lower=0.0)
        component_sum = queue + calendar + external
        residual = total - component_sum
        eps = 1e-9
        positive_total = total > eps
        ratio = component_sum[positive_total] / total[positive_total] if positive_total.any() else pd.Series(dtype="float64")

        try:
            corr = float(queue.corr(external)) if len(df) > 1 and queue.std() > 0 and external.std() > 0 else None
        except Exception:
            corr = None

        mean_total = float(total.mean()) if len(total) else 0.0
        mean_abs_residual = float(residual.abs().mean()) if len(residual) else 0.0
        negative_residual_share = float((residual < -1.0).mean()) if len(residual) else 0.0
        high_sum_ratio_share = float((ratio > 1.05).mean()) if len(ratio) else 0.0
        possible_double_count = bool(
            mean_total > 0.0
            and (
                mean_abs_residual / max(mean_total, eps) > 0.05
                or negative_residual_share > 0.10
                or high_sum_ratio_share > 0.10
            )
        )

        return {
            "available": True,
            "n_events": int(len(df)),
            "mean_queue_wait_min": float(queue.mean()),
            "mean_calendar_wait_min": float(calendar.mean()),
            "mean_external_delay_min": float(external.mean()),
            "mean_total_wait_min": float(total.mean()),
            "mean_component_sum_min": float(component_sum.mean()),
            "mean_residual_min": float(residual.mean()),
            "mean_abs_residual_min": mean_abs_residual,
            "max_abs_residual_min": float(residual.abs().max()),
            "negative_residual_share_gt_1min": negative_residual_share,
            "component_sum_over_total_mean": float(ratio.mean()) if len(ratio) else None,
            "component_sum_over_total_p95": float(ratio.quantile(0.95)) if len(ratio) else None,
            "component_sum_over_total_gt_1_05_share": high_sum_ratio_share,
            "queue_external_corr": corr,
            "possible_double_count": possible_double_count,
            "interpretation": (
                "possible double counting or inconsistent diagnostics"
                if possible_double_count else
                "queue_wait, calendar_wait and external_delay are additively consistent with total_wait"
            ),
        }


    def _nested_get(self, data: dict, path: tuple, default=None):
        cur = data
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def _extract_case_duration_guard(self, diagnostics: dict) -> dict:
        """
        从单次或多次聚合的 time diagnostics 中提取 case duration 对比信息。
        返回：
            real_case_duration_mean
            sim_case_duration_mean
            duration_ratio = sim / real
        """

        guard = {
            "real_case_duration_mean": None,
            "sim_case_duration_mean": None,
            "duration_ratio": None,
        }

        if not isinstance(diagnostics, dict) or not diagnostics:
            return guard

        # repeated diagnostics 的结构
        sim_mean = self._nested_get(
            diagnostics,
            ("mean_std", "summary.simulated.case_duration_min.mean", "mean"),
        )

        real_mean = self._nested_get(
            diagnostics,
            ("real_summary", "case_duration_min", "mean"),
        )

        if real_mean is None:
            real_mean = self._nested_get(
                diagnostics,
                ("representative_report", "summary", "real", "case_duration_min", "mean"),
            )

        # single diagnostics 的结构
        if sim_mean is None:
            sim_mean = self._nested_get(
                diagnostics,
                ("summary", "simulated", "case_duration_min", "mean"),
            )

        if real_mean is None:
            real_mean = self._nested_get(
                diagnostics,
                ("summary", "real", "case_duration_min", "mean"),
            )

        try:
            real_mean = float(real_mean)
            sim_mean = float(sim_mean)

            if np.isfinite(real_mean) and np.isfinite(sim_mean) and real_mean > 0:
                guard["real_case_duration_mean"] = real_mean
                guard["sim_case_duration_mean"] = sim_mean
                guard["duration_ratio"] = sim_mean / real_mean

        except Exception:
            pass

        return guard

    def _extract_arrival_guard(self, diagnostics: dict | None = None) -> dict:
        """Extract arrival span/count quality from the latest arrival diagnostics.

        StagewiseTuner uses this as a hard runtime guard so CAR is not only a
        soft metric in the final ideal-point score.
        """
        if diagnostics is None:
            diagnostics = getattr(self, "_last_arrival_diagnostics", {}) or {}

        guard = {
            "enabled": bool(diagnostics),
            "arrival_span_ratio": None,
            "case_count_ratio": None,
            "weekday_hour_jaccard": None,
            "real_cases": None,
            "sim_cases": None,
        }

        if not isinstance(diagnostics, dict) or not diagnostics:
            return guard

        try:
            guard["arrival_span_ratio"] = float(diagnostics.get("span_ratio_sim_over_real"))
        except Exception:
            pass
        try:
            real_cases = float(diagnostics.get("real_cases"))
            sim_cases = float(diagnostics.get("sim_cases"))
            guard["real_cases"] = int(real_cases)
            guard["sim_cases"] = int(sim_cases)
            if real_cases > 0:
                guard["case_count_ratio"] = float(sim_cases / real_cases)
        except Exception:
            pass
        try:
            guard["weekday_hour_jaccard"] = float(diagnostics.get("weekday_hour_jaccard"))
        except Exception:
            pass

        return guard

    def calibrate_external_delay_runtime_params(
            self,
            validation_log,
            candidate_scales=(1.0, 1.2, 1.4, 1.6),
            candidate_max_delay_quantiles=(0.95, 0.98, 1.00),
            metrics_labels=("aed", "ced", "red", "ctd", "cwd"),
            evaluation_runs: int = 2,
            conservative_tolerance: float = 0.03,

            # 新增：CTD 保护
            # 只允许候选 CTD 不超过 validation 最小 CTD 的 1.15 倍
            ctd_tolerance: float = 1.15,

            # 新增：case duration 保护
            # 仿真 case duration 均值不能明显短于或长于真实日志
            case_duration_min_ratio: float = 0.85,
            case_duration_max_ratio: float = 1.15,
    ) -> dict:
        """
        Jointly select external_delay.global_scale and
        external_delay.runtime.max_delay_quantile on the validation split.

        The external-delay tree is already discovered. This function only changes
        runtime scaling and runtime cap, then simulates the validation split.
        """

        if self.drq_params is None:
            return {
                "global_scale": 1.0,
                "max_delay_quantile": 0.95,
            }

        ext = self.drq_params.get("external_delay_params", {})
        if not isinstance(ext, dict):
            logger.info("External-delay runtime calibration skipped: model is not enabled.")
            return {
                "global_scale": self._get_external_delay_global_scale(),
                "max_delay_quantile": self._get_external_delay_max_delay_quantile(),
            }

        original_scale = self._get_external_delay_global_scale()
        original_quantile = self._get_external_delay_max_delay_quantile()

        candidate_scales = [
            float(s)
            for s in candidate_scales
            if np.isfinite(float(s)) and float(s) >= 0.0
        ]

        candidate_max_delay_quantiles = [
            float(np.clip(float(q), 0.0, 1.0))
            for q in candidate_max_delay_quantiles
            if np.isfinite(float(q))
        ]

        candidate_max_delay_quantiles = sorted(set(candidate_max_delay_quantiles))

        evaluation_runs = max(1, int(evaluation_runs))

        logger.info(
            "Calibrating external delay runtime parameters on validation log: "
            f"scales={candidate_scales}, "
            f"max_delay_quantiles={candidate_max_delay_quantiles}, "
            f"metrics={list(metrics_labels)}, "
            f"evaluation_runs={evaluation_runs}"
        )

        results = []

        for max_q in candidate_max_delay_quantiles:
            for scale in candidate_scales:
                self._set_external_delay_runtime_config(
                    global_scale=scale,
                    max_delay_quantile=max_q,
                )

                try:
                    if evaluation_runs > 1:
                        metrics = self.evaluate_on_event_log_repeated(
                            validation_log,
                            n_runs=evaluation_runs,
                            set_as_current_metrics=False,
                            metrics_labels=list(metrics_labels),
                                    )
                    else:
                        metrics = self.evaluate_on_event_log(
                            validation_log,
                            set_as_current_metrics=False,
                            metrics_labels=list(metrics_labels),
                                    )

                    flat_metrics = self._flatten_formatted_metrics(metrics)
                    duration_guard = self._extract_case_duration_guard(
                        getattr(self, "_last_time_diagnostics", {})
                    )

                except Exception as exc:
                    logger.warning(
                        "External-delay runtime calibration failed for "
                        "scale=%s, max_delay_quantile=%s: %s",
                        scale,
                        max_q,
                        exc,
                    )
                    flat_metrics = {}
                    duration_guard = self._extract_case_duration_guard({})
                row = {
                    "global_scale": float(scale),
                    "max_delay_quantile": float(max_q),
                    "metrics": flat_metrics,
                    "case_duration_guard": duration_guard,
                }

                results.append(row)

                logger.info(
                    "External-delay runtime calibration candidate: "
                    f"global_scale={scale}, "
                    f"max_delay_quantile={max_q}, "
                    f"metrics={flat_metrics}, "
                    f"case_duration_guard={duration_guard}"
                )

        # Normalized ideal-point distance. Smaller is better for all metrics.
        values_by_metric = {m: [] for m in metrics_labels}
        for row in results:
            for metric_name in metrics_labels:
                value = row["metrics"].get(metric_name)
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    values_by_metric[metric_name].append(float(value))

        for row in results:
            squared_sum = 0.0
            used = 0

            for metric_name in metrics_labels:
                value = row["metrics"].get(metric_name)

                if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
                    continue

                values = values_by_metric.get(metric_name, [])
                if not values:
                    continue

                min_v = min(values)
                max_v = max(values)

                if max_v > min_v:
                    normalized = (float(value) - min_v) / (max_v - min_v)
                else:
                    normalized = 0.0

                squared_sum += normalized ** 2
                used += 1

            row["score"] = float(np.sqrt(squared_sum / max(1, used)))

        valid_results = [
            row for row in results
            if "score" in row and np.isfinite(row["score"])
        ]

        if not valid_results:
            self._set_external_delay_runtime_config(
                global_scale=original_scale,
                max_delay_quantile=original_quantile,
            )
            logger.warning(
                "External-delay runtime calibration found no valid candidate. "
                f"Restored global_scale={original_scale}, "
                f"max_delay_quantile={original_quantile}."
            )
            return {
                "global_scale": original_scale,
                "max_delay_quantile": original_quantile,
            }

        # ==========================================================
        # 保护规则 1：CTD guard
        # ==========================================================
        ctd_values = [
            float(row["metrics"].get("ctd"))
            for row in valid_results
            if isinstance(row["metrics"].get("ctd"), (int, float, np.integer, np.floating))
               and np.isfinite(float(row["metrics"].get("ctd")))
        ]

        selection_pool = list(valid_results)

        ctd_guard_info = {
            "enabled": bool(ctd_values),
            "min_ctd": None,
            "ctd_tolerance": float(ctd_tolerance),
            "max_allowed_ctd": None,
            "n_before": len(selection_pool),
            "n_after": len(selection_pool),
        }

        if ctd_values:
            min_ctd = float(min(ctd_values))
            max_allowed_ctd = float(min_ctd * ctd_tolerance)

            ctd_safe_results = [
                row for row in selection_pool
                if isinstance(row["metrics"].get("ctd"), (int, float, np.integer, np.floating))
                   and np.isfinite(float(row["metrics"].get("ctd")))
                   and float(row["metrics"].get("ctd")) <= max_allowed_ctd
            ]

            ctd_guard_info.update({
                "min_ctd": min_ctd,
                "max_allowed_ctd": max_allowed_ctd,
                "n_after": len(ctd_safe_results) if ctd_safe_results else len(selection_pool),
            })

            if ctd_safe_results:
                selection_pool = ctd_safe_results

        # ==========================================================
        # 保护规则 2：case duration guard
        # ==========================================================
        def _duration_guard_passes(row):
            ratio = row.get("case_duration_guard", {}).get("duration_ratio")

            if not isinstance(ratio, (int, float, np.integer, np.floating)):
                return False

            if not np.isfinite(float(ratio)):
                return False

            ratio = float(ratio)

            return case_duration_min_ratio <= ratio <= case_duration_max_ratio

        duration_guard_info = {
            "enabled": True,
            "min_ratio": float(case_duration_min_ratio),
            "max_ratio": float(case_duration_max_ratio),
            "n_before": len(selection_pool),
            "n_after": len(selection_pool),
            "fallback_used": False,
        }

        duration_safe_results = [
            row for row in selection_pool
            if _duration_guard_passes(row)
        ]

        if duration_safe_results:
            selection_pool = duration_safe_results
            duration_guard_info["n_after"] = len(duration_safe_results)

        else:
            # 如果 CTD-safe 候选都不满足 duration ratio，
            # 就在全部 valid_results 中寻找 duration-safe 候选，避免约束过强。
            all_duration_safe = [
                row for row in valid_results
                if _duration_guard_passes(row)
            ]

            if all_duration_safe:
                selection_pool = all_duration_safe
                duration_guard_info["n_after"] = len(all_duration_safe)
                duration_guard_info["fallback_used"] = True
            else:
                duration_guard_info["n_after"] = 0
                duration_guard_info["fallback_used"] = True

        if not selection_pool:
            selection_pool = valid_results
            logger.warning(
                "External-delay calibration protection rules removed all candidates; "
                "falling back to all valid candidates."
            )

        # ==========================================================
        # 在保护后的候选集合中再做理想点距离 + 保守选择
        # ==========================================================
        best_score = min(row["score"] for row in selection_pool)

        near_best = [
            row for row in selection_pool
            if row["score"] <= best_score * (1.0 + float(conservative_tolerance))
        ]

        def delay_aggressiveness(row):
            return (
                float(row["global_scale"]),
                float(row["max_delay_quantile"]),
            )

        selected = min(near_best, key=delay_aggressiveness)

        selected_scale = float(selected["global_scale"])
        selected_quantile = float(selected["max_delay_quantile"])

        self._set_external_delay_runtime_config(
            global_scale=selected_scale,
            max_delay_quantile=selected_quantile,
        )

        ext = self.drq_params.setdefault("external_delay_params", {})
        if isinstance(ext, dict):
            ext["calibration"] = {
                "method": "validation_normalized_ideal_point_distance_joint_scale_quantile_with_ctd_and_duration_guards",
                "candidate_scales": candidate_scales,
                "candidate_max_delay_quantiles": candidate_max_delay_quantiles,
                "selected_global_scale": selected_scale,
                "selected_max_delay_quantile": selected_quantile,
                "selected_score": float(selected["score"]),
                "best_score": float(best_score),
                "conservative_tolerance": float(conservative_tolerance),

                # 新增
                "ctd_guard": ctd_guard_info,
                "case_duration_guard": duration_guard_info,
                "selected_case_duration_guard": selected.get("case_duration_guard", {}),

                "metrics_labels": list(metrics_labels),
                "evaluation_runs": evaluation_runs,
                "results": results,
            }

        logger.info(
            "External delay runtime parameters calibrated on validation log: "
            f"old_scale={original_scale}, old_max_delay_quantile={original_quantile}, "
            f"selected_scale={selected_scale}, "
            f"selected_max_delay_quantile={selected_quantile}, "
            f"selected_score={selected['score']}, "
            f"selected_metrics={selected['metrics']}, "
            f"selected_case_duration_guard={selected.get('case_duration_guard')}, "
            f"ctd_guard={ctd_guard_info}, "
            f"case_duration_guard={duration_guard_info}"
        )

        return {
            "global_scale": selected_scale,
            "max_delay_quantile": selected_quantile,
            "score": float(selected["score"]),
            "metrics": selected["metrics"],
            "case_duration_guard": selected.get("case_duration_guard", {}),
            "ctd_guard": ctd_guard_info,
            "duration_guard": duration_guard_info,
        }
