# drq/config/modular_schema.py

from __future__ import annotations

from typing import Any, Dict


def build_empty_modular_params() -> Dict[str, Any]:
    """
    构建五模块仿真模型的空参数骨架。
    """
    return {
        "modeling_version": "modular_drq_bps_v1",
        "modeling_description": (
            "Five-module white-box process simulation model with preprocessing, "
            "decision-point control flow, inter-arrival distribution, dynamic "
            "resource service model, and queue scheduling."
        ),
        "modules": {
            "preprocessing": {
                "description": (
                    "Log loading, log-model alignment, enabled-time computation, "
                    "resource-state reconstruction, and queue-state reconstruction."
                ),
                "log_format": None,
                "split_strategy": "temporal_case_split",
                "state_reconstruction": {}
            },
            "control_flow": {
                "model_type": "petri_net_with_decision_point_branch_trees",
                "petri_net": {},
                "decision_point_branch_models": {},
                "fallback": {
                    "single_enabled_transition": "fire_directly",
                    "missing_decision_point": "uniform_over_enabled_transitions"
                }
            },
            "arrival": {
                "model_type": "inter_arrival_distribution",
                "arrival_calendar": {},
                "inter_arrival_distribution": None,
                "arrival_time_scale": 1.0
            },
            "dynamic_resource": {
                "model_type": "base_service_distribution_plus_resource_state_dbn",
                "resources": [],
                "activity_to_resources": {},
                "resource_calendars": {},
                "multitasking_resources": [],
                "base_service_time_distributions": {},
                "resource_state_dbn": {}
            },
            "queue_scheduling": {
                "model_type": "resource_allocation_plus_conditional_prioritization",
                "queue_scheduling_model": {},
                "conditional_prioritization_model": {},
                "queue_feature_config": {
                    "max_queue_count": 8.0,
                    "max_wip_count": 30.0,
                    "disable_global_queue_features": True
                },
                "fallback": {
                    "missing_model": "fcfs",
                    "empty_ready_queue": "wait",
                    "no_available_resource": "wait_until_next_resource_available"
                }
            }
        },
        "legacy_fallback": {
            "transition_weights": None,
            "resource_weights": None,
            "execution_time_distributions": None,
            "waiting_time_distributions": None,
            "arrival_time_distribution": None
        }
    }


def attach_module_params(
        modular_params: Dict[str, Any],
        module_name: str,
        module_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    将某个模块发现得到的参数写入五模块参数结构。
    """
    if modular_params is None:
        modular_params = build_empty_modular_params()

    modular_params.setdefault("modules", {})

    if module_name not in modular_params["modules"]:
        modular_params["modules"][module_name] = {}

    modular_params["modules"][module_name].update(module_params or {})

    return modular_params
