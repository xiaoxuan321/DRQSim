import pandas as pd
import pm4py
import numpy as np

from scipy.stats import entropy
from scipy.stats import wasserstein_distance

from log_distance_measures.config import EventLogIDs

from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance, discretize_to_hour
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.config import AbsoluteTimestampType
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance

import datetime



import time
import logging

logger = logging.getLogger(__name__)


def evaluate(original_log, simulated_log, metrics_labels=None, print_timing=True):
    """
    计算仿真日志与真实日志之间的评估指标，并打印每个指标的耗时。

    参数：
        original_log:
            真实日志 DataFrame。

        simulated_log:
            仿真日志 DataFrame。

        metrics_labels:
            需要计算的指标列表。

        print_timing:
            是否在控制台打印每个指标耗时。
    """

    if metrics_labels is None:
        metrics_labels = [
            "cfld",
            "3gd",
            "ngd",
            "aed",
            "ced",
            "red",
            "car",
            "ctd",
            "cwd"
        ]

    # 避免直接修改外部传入的 DataFrame
    original_log = original_log.copy()
    simulated_log = simulated_log.copy()

    event_log_ids = EventLogIDs(
        case="case:concept:name",
        activity="concept:name",
        resource="org:resource",
        start_time="start:timestamp",
        end_time="time:timestamp"
    )

    metrics = {}
    metric_times = {}

    total_start = time.perf_counter()

    def _print_or_log(message: str):
        logger.info(message)
        if print_timing:
            print(message, flush=True)

    def _run_metric(metric_name: str, metric_func):
        """
        执行单个指标，并统计耗时。
        """

        start = time.perf_counter()

        try:
            value = metric_func()
            elapsed = time.perf_counter() - start

            metrics[metric_name] = value
            metric_times[metric_name] = elapsed

            _print_or_log(
                f"[Evaluation Timing] {metric_name.upper()} "
                f"耗时: {elapsed:.4f} 秒, 结果: {value}"
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            metric_times[metric_name] = elapsed

            _print_or_log(
                f"[Evaluation Timing] {metric_name.upper()} "
                f"计算失败，已耗时: {elapsed:.4f} 秒, 错误: {str(e)}"
            )

            raise

    # =========================
    # 0. 时间字段预处理计时
    # =========================
    preprocess_start = time.perf_counter()

    original_log[event_log_ids.start_time] = pd.to_datetime(
        original_log[event_log_ids.start_time],
        format="ISO8601",
        utc=True,
        errors="coerce"
    )
    original_log[event_log_ids.end_time] = pd.to_datetime(
        original_log[event_log_ids.end_time],
        format="ISO8601",
        utc=True,
        errors="coerce"
    )

    simulated_log[event_log_ids.start_time] = pd.to_datetime(
        simulated_log[event_log_ids.start_time],
        format="ISO8601",
        utc=True,
        errors="coerce"
    )
    simulated_log[event_log_ids.end_time] = pd.to_datetime(
        simulated_log[event_log_ids.end_time],
        format="ISO8601",
        utc=True,
        errors="coerce"
    )

    preprocess_time = time.perf_counter() - preprocess_start
    metric_times["preprocess"] = preprocess_time

    _print_or_log(
        f"[Evaluation Timing] PREPROCESS 耗时: {preprocess_time:.4f} 秒"
    )

    # =========================
    # 1. CFLD
    # =========================
    if "cfld" in metrics_labels:
        _run_metric(
            "cfld",
            lambda: control_flow_log_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids,
                parallel=False
            )
        )

    # =========================
    # 2. NGD，默认 n=2
    # =========================
    if "ngd" in metrics_labels:
        _run_metric(
            "ngd",
            lambda: n_gram_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids,
                n=2
            )
        )

    # 可选：2GD
    if "2gd" in metrics_labels:
        _run_metric(
            "2gd",
            lambda: n_gram_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids,
                n=2
            )
        )

    # 可选：3GD
    if "3gd" in metrics_labels:
        _run_metric(
            "3gd",
            lambda: n_gram_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids,
                n=3
            )
        )

    # =========================
    # 3. AED
    # =========================
    if "aed" in metrics_labels:
        _run_metric(
            "aed",
            lambda: absolute_event_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids,
                AbsoluteTimestampType.BOTH,
                discretize_to_hour
            )
        )

    # =========================
    # 4. RED
    # =========================
    if "red" in metrics_labels:
        _run_metric(
            "red",
            lambda: relative_event_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids,
                AbsoluteTimestampType.BOTH
            )
        )

    # =========================
    # 5. CED
    # =========================
    if "ced" in metrics_labels:
        _run_metric(
            "ced",
            lambda: circadian_event_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids,
                AbsoluteTimestampType.BOTH
            )
        )

    # =========================
    # 6. CWD
    # =========================
    if "cwd" in metrics_labels:
        _run_metric(
            "cwd",
            lambda: circadian_workforce_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids
            )
        )

    # =========================
    # 7. CAR
    # =========================
    if "car" in metrics_labels:
        _run_metric(
            "car",
            lambda: case_arrival_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids
            )
        )

    # =========================
    # 8. CTD
    # =========================
    if "ctd" in metrics_labels:
        _run_metric(
            "ctd",
            lambda: cycle_time_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids,
                datetime.timedelta(hours=1)
            )
        )

    # =========================
    # 9. 扩展指标
    # =========================
    if "hwd" in metrics_labels:
        _run_metric(
            "hwd",
            lambda: compute_handover_error(
                simulated_log,
                original_log
            )
        )

    if "rfld" in metrics_labels:
        _run_metric(
            "rfld",
            lambda: compute_resource_flow_distance(
                simulated_log,
                original_log
            )
        )

    if "r2gd" in metrics_labels:
        _run_metric(
            "r2gd",
            lambda: compute_resource_ngd(
                simulated_log,
                original_log,
                n=2
            )
        )

    if "r3gd" in metrics_labels:
        _run_metric(
            "r3gd",
            lambda: compute_resource_ngd(
                simulated_log,
                original_log,
                n=3
            )
        )

    if "r4gd" in metrics_labels:
        _run_metric(
            "r4gd",
            lambda: compute_resource_ngd(
                simulated_log,
                original_log,
                n=4
            )
        )

    if "r5gd" in metrics_labels:
        _run_metric(
            "r5gd",
            lambda: compute_resource_ngd(
                simulated_log,
                original_log,
                n=5
            )
        )

    if "car_entropy" in metrics_labels:
        _run_metric(
            "car_entropy",
            lambda: compute_atd_entropy(simulated_log)
        )

    if "ctd_entropy" in metrics_labels:
        _run_metric(
            "ctd_entropy",
            lambda: compute_ctd_entropy(simulated_log)
        )

    if "etd_entropy" in metrics_labels:
        _run_metric(
            "etd_entropy",
            lambda: compute_etd_entropy(simulated_log)
        )

    total_elapsed = time.perf_counter() - total_start
    metric_times["total"] = total_elapsed

    _print_or_log(
        f"[Evaluation Timing] TOTAL 总耗时: {total_elapsed:.4f} 秒"
    )

    # 可选：把耗时也放进 metrics，便于后续调试或前端展示
    metrics["_timing_seconds"] = metric_times

    return metrics

def compute_atd_entropy(df_log: pd.DataFrame) -> float:

    first_ts = df_log.groupby('case:concept:name')["start:timestamp"].min()
    ordered_first_ts_list = first_ts.sort_values().tolist()

    arrival_times = []
    for i in range(1, len(ordered_first_ts_list)):
        arrival_times.append((ordered_first_ts_list[i] - ordered_first_ts_list[i-1]).total_seconds()/60)

    hist, _ = np.histogram(arrival_times, bins='auto', density=True)
    hist = hist[hist > 0]
    probs = hist / np.sum(hist)
    atd_entr = entropy(probs)

    return atd_entr


def compute_ctd_entropy(df_log: pd.DataFrame) -> float:

    log = pm4py.convert_to_event_log(df_log)
    cycle_times = []
    for trace in log:
        start = trace[0]['start:timestamp']
        end = trace[-1]['time:timestamp']
        cycle_times.append((end-start).total_seconds()//60)
    
    hist, _ = np.histogram(cycle_times, bins='auto', density=True)
    probs = hist / np.sum(hist)
    ctd_entr = entropy(probs)

    return ctd_entr


def compute_etd_entropy(df_log: pd.DataFrame) -> float:

    activities = list(df_log["concept:name"].unique())
    etd_entropies = []
    for act in activities:
        df_log_act = df_log[df_log["concept:name"] == act]
        ex_times = (df_log_act["time:timestamp"] - df_log_act["start:timestamp"]).apply(lambda x: x.total_seconds() // 60)
        hist, _ = np.histogram(list(ex_times), bins='auto', density=True)
        probs = hist / np.sum(hist)
        etd_entropies.append(entropy(probs))

    return np.mean(etd_entropies)


def build_handover_matrix(df: pd.DataFrame, resources: list, case_id_col: str='case:concept:name', resource_col: str='org:resource') -> pd.DataFrame:

    res_idx = {res: i for i, res in enumerate(resources)}
    matrix = np.zeros((len(resources), len(resources)), dtype=int)

    for case_id, group in df.groupby(case_id_col):
        resource_sequence = group[resource_col].tolist()
        for i in range(len(resource_sequence) - 1):
            from_res = resource_sequence[i]
            to_res = resource_sequence[i + 1]
            if from_res in res_idx and to_res in res_idx:
                matrix[res_idx[from_res], res_idx[to_res]] += 1

    handover_matrix = pd.DataFrame(matrix, index=resources, columns=resources)

    return handover_matrix


def compute_handover_error(df_sim: pd.DataFrame, df_test: pd.DataFrame, case_id_col: str='case:concept:name', resource_col: str='org:resource') -> int:
    
    resources = list(set(df_sim[resource_col]) | set(df_test[resource_col]))

    handover_matrix_sim = build_handover_matrix(df_sim, resources, case_id_col=case_id_col, resource_col=resource_col)
    handover_matrix_test = build_handover_matrix(df_test, resources, case_id_col=case_id_col, resource_col=resource_col)

    return float((handover_matrix_sim - handover_matrix_test).abs().sum().sum())


def compute_resource_flow_distance(df_sim: pd.DataFrame, df_test: pd.DataFrame):
    
    event_log_ids = EventLogIDs(  
        case="case:concept:name",
        activity="org:resource",
        start_time="start:timestamp",
        end_time="time:timestamp"
    )


    df_test[event_log_ids.start_time] = pd.to_datetime(df_test[event_log_ids.start_time], format='ISO8601', utc=True)
    df_test[event_log_ids.end_time] = pd.to_datetime(df_test[event_log_ids.end_time], format='ISO8601', utc=True)

    df_sim[event_log_ids.start_time] = pd.to_datetime(df_sim[event_log_ids.start_time], format='ISO8601', utc=True)
    df_sim[event_log_ids.end_time] = pd.to_datetime(df_sim[event_log_ids.end_time], format='ISO8601', utc=True)

    rfld = control_flow_log_distance(
        df_test, 
        event_log_ids, 
        df_sim, 
        event_log_ids
    )

    return rfld


def compute_resource_ngd(df_sim: pd.DataFrame, df_test: pd.DataFrame, n=3):
    
    event_log_ids = EventLogIDs(  
        case="case:concept:name",
        activity="org:resource",
        start_time="start:timestamp",
        end_time="time:timestamp"
    )

    df_test[event_log_ids.start_time] = pd.to_datetime(df_test[event_log_ids.start_time], format='ISO8601', utc=True)
    df_test[event_log_ids.end_time] = pd.to_datetime(df_test[event_log_ids.end_time], format='ISO8601', utc=True)

    df_sim[event_log_ids.start_time] = pd.to_datetime(df_sim[event_log_ids.start_time], format='ISO8601', utc=True)
    df_sim[event_log_ids.end_time] = pd.to_datetime(df_sim[event_log_ids.end_time], format='ISO8601', utc=True)

    ngd = n_gram_distribution_distance(
            df_test, 
            event_log_ids, 
            df_sim, 
            event_log_ids, 
            n=n,
        )

    return ngd