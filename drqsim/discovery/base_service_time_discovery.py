# drq/discovery/base_service_time_discovery.py

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from drqsim.utils.common_utils import count_false_hours
from drqsim.utils.distribution_utils import (
    return_best_distribution,
    remove_outliers,
    sampling_from_dist,
)
from drqsim.utils.save_and_load_utils import fromstr_to_scipy


DIST_SEARCH = ["fixed", "norm", "expon", "lognorm", "gamma", "uniform"]
ACTIVITY_DEFAULT_KEY = "__activity_default__"


def discover_base_service_time_distributions(
        df_features: pd.DataFrame,
        activities: list[str],
        resources: list[str],
        calendars: dict,
        act_to_resources: Optional[dict] = None,
        min_samples_per_activity_resource: int = 5,
        use_outlier_removal: bool = True,
) -> Dict[str, Any]:
    """
    发现动态资源模块中的基础服务时间分布。

    与 DBPS 的执行时间决策树不同，本函数不训练上下文相关的执行时间树，
    而是为 activity-resource 组合拟合基础服务时间分布：

        ET_e = ET_base(activity, resource) * psi_ET(S_r(t))

    其中 ET_base 由本函数发现，psi_ET 由 DBN 动态资源状态模块提供。

    回退机制：
    1. 若某 activity-resource 组合样本数充足，则使用该组合分布；
    2. 若样本不足，则仿真时回退到 activity 级默认分布；
    3. 若 activity 级样本也为空，则使用 fixed(1) 分布。
    """

    activities = [str(a) for a in (activities or [])]
    resources = [str(r) for r in (resources or [])]
    act_to_resources = act_to_resources or {a: resources for a in activities}

    empty_result = {
        "enabled": False,
        "model_type": "activity_resource_base_service_time_distribution",
        "reason": "No valid execution samples.",
        "min_samples_per_activity_resource": int(min_samples_per_activity_resource),
        "activities": {},
    }

    required_cols = ["transition_label", "resource", "start_t", "end_t"]
    missing = [c for c in required_cols if c not in df_features.columns]
    if missing:
        result = dict(empty_result)
        result["reason"] = f"df_features misses required columns: {missing}"
        return result

    df = df_features[required_cols].copy()
    df = df[(~df["transition_label"].isna()) & (~df["resource"].isna())].copy()
    df["transition_label"] = df["transition_label"].astype(str)
    df["resource"] = df["resource"].astype(str)
    df["start_t"] = pd.to_datetime(df["start_t"], errors="coerce")
    df["end_t"] = pd.to_datetime(df["end_t"], errors="coerce")
    df = df[(~df["start_t"].isna()) & (~df["end_t"].isna())].copy()

    if df.empty:
        return empty_result

    df["service_time_min"] = df.apply(
        lambda row: _compute_working_minutes(
            start_time=row["start_t"],
            end_time=row["end_t"],
            resource=row["resource"],
            calendars=calendars,
        ),
        axis=1,
    )

    df = df[df["service_time_min"] > 0].copy()

    if df.empty:
        return empty_result

    activity_models = {}

    for activity in activities:
        df_act = df[df["transition_label"] == str(activity)].copy()
        act_values = df_act["service_time_min"].dropna().astype(float).tolist()

        activity_default = _fit_distribution_tuple(
            values=act_values,
            use_outlier_removal=use_outlier_removal,
            default_value=1.0,
        )

        candidate_resources = [
            str(r) for r in act_to_resources.get(activity, resources)
        ]

        resource_models = {}
        resource_sample_counts = {}

        for resource in candidate_resources:
            values = (
                df_act[df_act["resource"] == str(resource)]["service_time_min"]
                .dropna()
                .astype(float)
                .tolist()
            )

            resource_sample_counts[str(resource)] = int(len(values))

            if len(values) >= int(min_samples_per_activity_resource):
                resource_models[str(resource)] = _fit_distribution_tuple(
                    values=values,
                    use_outlier_removal=use_outlier_removal,
                    default_value=float(activity_default[4]),
                )

        activity_models[str(activity)] = {
            "activity": str(activity),
            "n_samples": int(len(act_values)),
            "activity_default": activity_default,
            "resources": resource_models,
            "resource_sample_counts": resource_sample_counts,
            "fallback": "activity_default",
        }

    return {
        "enabled": True,
        "model_type": "activity_resource_base_service_time_distribution",
        "description": (
            "Base service-time distributions for activity-resource pairs. "
            "Sparse pairs fall back to activity-level distributions. "
            "Dynamic resource DBN adjusts sampled base service times at runtime."
        ),
        "min_samples_per_activity_resource": int(min_samples_per_activity_resource),
        "use_outlier_removal": bool(use_outlier_removal),
        "activities": activity_models,
    }


def sample_base_service_time(
        base_service_time_params: Dict[str, Any],
        activity: str,
        resource: str,
        deterministic_time: bool = False,
) -> float:
    """
    从 activity-resource 基础服务时间分布中取值。

    若 deterministic_time=True，则返回分布均值；
    否则从分布中随机采样一个服务时间。
    """

    dist_tuple = get_base_service_distribution(
        base_service_time_params=base_service_time_params,
        activity=activity,
        resource=resource,
    )

    if deterministic_time:
        try:
            return max(1.0, float(dist_tuple[4]))
        except Exception:
            return 1.0

    try:
        value = sampling_from_dist(
            dist_tuple[0],
            dist_tuple[1],
            dist_tuple[2],
            dist_tuple[3],
            dist_tuple[4],
            n_sample=1,
        )[0]
        return max(1.0, float(value))
    except Exception:
        try:
            return max(1.0, float(dist_tuple[4]))
        except Exception:
            return 1.0


def get_base_service_distribution(
        base_service_time_params: Dict[str, Any],
        activity: str,
        resource: str,
) -> tuple:
    """
    返回某 activity-resource 的基础服务时间分布。
    """

    default_dist = ("fixed", (1.0,), 1.0, 1.0, 1.0)

    if not isinstance(base_service_time_params, dict):
        return default_dist

    activities = base_service_time_params.get("activities", {})
    act_params = activities.get(str(activity))

    if not isinstance(act_params, dict):
        return default_dist

    res_params = act_params.get("resources", {}).get(str(resource))

    if _is_distribution_tuple(res_params):
        return res_params

    act_default = act_params.get("activity_default")

    if _is_distribution_tuple(act_default):
        return act_default

    return default_dist


def serialize_distribution_container(obj: Any) -> Any:
    """
    将任意嵌套容器中的 scipy 分布元组转换为 JSON 友好字典。
    """

    if _is_distribution_tuple(obj):
        dist = obj[0]
        dist_name = dist if isinstance(dist, str) else getattr(dist, "name", str(dist))
        result = {
            "dist_name": dist_name,
            "params": list(obj[1]) if isinstance(obj[1], tuple) else obj[1],
            "min_value": float(obj[2]),
            "max_value": float(obj[3]),
        }
        if len(obj) >= 5:
            result["mean_value"] = float(obj[4])
        return result

    if isinstance(obj, dict):
        return {str(k): serialize_distribution_container(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [serialize_distribution_container(v) for v in obj]

    if isinstance(obj, tuple):
        return [serialize_distribution_container(v) for v in obj]

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def deserialize_distribution_container(obj: Any) -> Any:
    """
    将 JSON 中的分布字典恢复为运行时可采样的分布元组。
    """

    if isinstance(obj, dict) and "dist_name" in obj and "params" in obj:
        dist = fromstr_to_scipy(obj.get("dist_name"))
        params = tuple(obj.get("params", ()))
        min_value = float(obj.get("min_value", 0.0))
        max_value = float(obj.get("max_value", min_value))
        mean_value = float(obj.get("mean_value", min_value))
        return dist, params, min_value, max_value, mean_value

    if isinstance(obj, dict):
        return {str(k): deserialize_distribution_container(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [deserialize_distribution_container(v) for v in obj]

    return obj


def _compute_working_minutes(start_time, end_time, resource: str, calendars: dict) -> float:
    try:
        total_min = (end_time - start_time).total_seconds() / 60.0
    except Exception:
        return 0.0

    if total_min <= 0:
        return 0.0

    calendar = calendars.get(str(resource)) if isinstance(calendars, dict) else None

    if calendar is None:
        return max(1.0, float(total_min))

    try:
        non_working_min = count_false_hours(calendar, start_time, end_time) * 60.0
        return max(1.0, float(total_min - non_working_min))
    except Exception:
        return max(1.0, float(total_min))


def _fit_distribution_tuple(
        values: list[float],
        use_outlier_removal: bool,
        default_value: float = 1.0,
) -> tuple:
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]

    if use_outlier_removal and values:
        values = remove_outliers(values)

    if not values:
        default_value = max(1.0, float(default_value))
        return "fixed", (default_value,), default_value, default_value, default_value

    values = [max(1.0, float(v)) for v in values]

    dist, params = return_best_distribution(values, dist_search=DIST_SEARCH)

    return (
        dist,
        params,
        float(np.min(values)),
        float(np.max(values)),
        float(np.mean(values)),
    )


def _is_distribution_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) >= 4
