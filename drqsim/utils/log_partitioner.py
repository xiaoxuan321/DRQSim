"""
utils/drq_xes_partitioner.py

该文件用于将 drq 使用的 XES / pm4py EventLog 日志划分为：
1. train_log
2. validation_log
3. train_validation_log
4. test_log

保证同一个 case / trace 不会被拆散；
默认按照 case 的时间顺序划分
"""

from dataclasses import dataclass
from typing import Union

import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog


@dataclass
class drqLogPartitions:
    """
    drq 日志划分结果。

    同时保存 DataFrame 和 pm4py EventLog 两种格式。
    """

    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    train_validation_df: pd.DataFrame
    test_df: pd.DataFrame

    train_log: EventLog
    validation_log: EventLog
    train_validation_log: EventLog
    test_log: EventLog

    full_df: pd.DataFrame


def split_drq_xes_log(
    event_log: Union[EventLog, pd.DataFrame],
    train_validation_ratio: float = 0.8,
    train_ratio_within_train_validation: float = 0.8,
    case_id_col: str = "case:concept:name",
    activity_col: str = "concept:name",
    resource_col: str = "org:resource",
    start_time_col: str = "start:timestamp",
    end_time_col: str = "time:timestamp",
    sort_temporally: bool = True,
) -> drqLogPartitions:
    """
    将 drq 的 XES 日志划分为训练集、验证集和测试集。

    参数
    ----------
    event_log:
        pm4py EventLog 或 pandas.DataFrame。

    train_validation_ratio:
        第一阶段划分比例。
        默认 0.8 表示 80% case 用作 train+validation，20% case 用作 test。

    train_ratio_within_train_validation:
        第二阶段划分比例。
        默认 0.8 表示 train+validation 中 80% case 用作 train，
        20% case 用作 validation。

    sort_temporally:
        是否按照 case 开始时间排序后划分。
        建议保持 True，以模拟 SIMOD 的时间维度划分思想。

    返回
    ----------
    drqLogPartitions
        日志划分结果。
    """

    _check_ratio(train_validation_ratio, "train_validation_ratio")
    _check_ratio(train_ratio_within_train_validation, "train_ratio_within_train_validation")

    df = _to_dataframe(event_log)

    _check_required_columns(
        df,
        [
            case_id_col,
            activity_col,
            resource_col,
            start_time_col,
            end_time_col,
        ],
    )

    df[case_id_col] = df[case_id_col].astype(str)
    df[start_time_col] = pd.to_datetime(df[start_time_col], utc=True, errors="coerce")
    df[end_time_col] = pd.to_datetime(df[end_time_col], utc=True, errors="coerce")

    df = df.dropna(subset=[case_id_col, start_time_col, end_time_col]).copy()

    case_ids = _get_ordered_case_ids(
        df=df,
        case_id_col=case_id_col,
        start_time_col=start_time_col,
        end_time_col=end_time_col,
        sort_temporally=sort_temporally,
    )

    if len(case_ids) < 3:
        raise ValueError("case 数量过少，无法划分 train / validation / test。")

    # ================================
    # 第一步：完整日志 -> train_validation + test
    # ================================
    train_validation_case_ids, test_case_ids = _split_case_ids(
        case_ids,
        train_validation_ratio,
    )

    train_validation_df = _filter_by_case_ids(
        df,
        train_validation_case_ids,
        case_id_col,
        start_time_col,
        end_time_col,
    )

    test_df = _filter_by_case_ids(
        df,
        test_case_ids,
        case_id_col,
        start_time_col,
        end_time_col,
    )

    # ================================
    # 第二步：train_validation -> train + validation
    # ================================
    train_validation_case_ids_ordered = _get_ordered_case_ids(
        df=train_validation_df,
        case_id_col=case_id_col,
        start_time_col=start_time_col,
        end_time_col=end_time_col,
        sort_temporally=sort_temporally,
    )

    train_case_ids, validation_case_ids = _split_case_ids(
        train_validation_case_ids_ordered,
        train_ratio_within_train_validation,
    )

    train_df = _filter_by_case_ids(
        train_validation_df,
        train_case_ids,
        case_id_col,
        start_time_col,
        end_time_col,
    )

    validation_df = _filter_by_case_ids(
        train_validation_df,
        validation_case_ids,
        case_id_col,
        start_time_col,
        end_time_col,
    )

    return drqLogPartitions(
        train_df=train_df,
        validation_df=validation_df,
        train_validation_df=train_validation_df,
        test_df=test_df,
        train_log=_to_event_log(train_df, case_id_col),
        validation_log=_to_event_log(validation_df, case_id_col),
        train_validation_log=_to_event_log(train_validation_df, case_id_col),
        test_log=_to_event_log(test_df, case_id_col),
        full_df=df,
    )


def _to_dataframe(event_log: Union[EventLog, pd.DataFrame]) -> pd.DataFrame:
    """
    将 pm4py EventLog 或 DataFrame 统一转换为 DataFrame。
    """

    if isinstance(event_log, pd.DataFrame):
        return event_log.copy()

    return pm4py.convert_to_dataframe(event_log).copy()


def _to_event_log(df: pd.DataFrame, case_id_col: str) -> EventLog:
    """
    将 DataFrame 转回 pm4py EventLog。
    """

    df_copy = df.copy()

    if case_id_col != "case:concept:name":
        df_copy["case:concept:name"] = df_copy[case_id_col].astype(str)
    else:
        df_copy["case:concept:name"] = df_copy["case:concept:name"].astype(str)

    return pm4py.convert_to_event_log(df_copy)


def _check_ratio(value: float, name: str):
    """
    检查比例参数是否合法。
    """

    if not 0 < value < 1:
        raise ValueError(f"{name} 必须在 0 和 1 之间，当前值为 {value}。")


def _check_required_columns(df: pd.DataFrame, required_cols: list[str]):
    """
    检查 DataFrame 中是否包含必要字段。
    """

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"事件日志缺少必要字段：{missing_cols}")


def _get_ordered_case_ids(
    df: pd.DataFrame,
    case_id_col: str,
    start_time_col: str,
    end_time_col: str,
    sort_temporally: bool,
) -> list[str]:
    """
    获取排序后的 case ID。

    sort_temporally=True:
        按每个 case 的最早开始时间和最晚完成时间排序。

    sort_temporally=False:
        按日志中首次出现顺序排序。
    """

    if sort_temporally:
        case_order = (
            df.groupby(case_id_col)
            .agg(
                case_start=(start_time_col, "min"),
                case_end=(end_time_col, "max"),
            )
            .sort_values(["case_start", "case_end"])
            .index
            .astype(str)
            .tolist()
        )
    else:
        case_order = df[case_id_col].drop_duplicates().astype(str).tolist()

    return case_order


def _split_case_ids(
    case_ids: list[str],
    first_ratio: float,
) -> tuple[list[str], list[str]]:
    """
    按 case ID 列表进行二分。
    """

    n_cases = len(case_ids)

    split_index = int(n_cases * first_ratio)

    split_index = max(1, min(split_index, n_cases - 1))

    return case_ids[:split_index], case_ids[split_index:]


def _filter_by_case_ids(
    df: pd.DataFrame,
    case_ids: list[str],
    case_id_col: str,
    start_time_col: str,
    end_time_col: str,
) -> pd.DataFrame:
    """
    根据 case ID 过滤日志，并按 case 和时间排序。
    """

    case_id_set = set(case_ids)

    filtered_df = df[df[case_id_col].astype(str).isin(case_id_set)].copy()

    filtered_df.sort_values(
        by=[case_id_col, start_time_col, end_time_col],
        inplace=True,
    )

    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df