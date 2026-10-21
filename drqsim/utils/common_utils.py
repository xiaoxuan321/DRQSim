# 导入 random，用于基于权重随机选择变迁或资源
import random
import os as _os
import signal as _signal
import sys as _sys
import time as _time

# 导入 pandas，用于构建和处理特征数据表
import pandas as pd

# 导入 datetime 和 timedelta，用于计算时间戳和时间增量
from datetime import datetime, timedelta

# 导入 copy，用于复制 Petri 网当前标识，避免直接修改原对象
from copy import copy

# 导入 pm4py，用于事件日志转换、对齐和流程挖掘相关操作
import pm4py

# 导入 tqdm，用于显示遍历事件日志时的进度条
from tqdm import tqdm

# 导入 Petri 网对齐算法，用于将事件日志 trace 与 Petri 网模型进行对齐
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

# 导入 PetriNet 和 Marking 类型，用于 Petri 网结构和标识表示
from pm4py.objects.petri_net.obj import PetriNet, Marking

# 导入 EventLog 类型，用于事件日志类型标注
from pm4py.objects.log.obj import EventLog

# 导入若干 sklearn 分类模型，用于识别模型类型并计算预测概率
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 导入自定义决策规则类，用于根据输入特征计算规则输出
from drqsim.utils.rule_utils import DecisionRules

# 导入 river 在线自适应 Hoeffding 决策树分类器
from river.tree.hoeffding_adaptive_tree_classifier import HoeffdingAdaptiveTreeClassifier


CONTROL_FLOW_DEBUG_BUILD = "2026-09-16-v3-alignment-path"


def return_transitions_frequency(
        log: EventLog,
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking
    ) -> dict:
    """
    统计事件日志中各 Petri 网变迁的触发频率。

    该函数首先将日志与 Petri 网模型进行对齐，
    然后根据对齐结果统计每个变迁被实际触发的次数。

    参数：
        log:
            输入事件日志。
        net:
            Petri 网模型。
        initial_marking:
            初始标识。
        final_marking:
            终止标识。

    返回：
        frequency_t:
            每个变迁对应的触发次数，格式为 {transition: count}。
    """

    # 对事件日志和 Petri 网模型进行对齐
    alignments_ = alignments.apply_log(
        log,
        net,
        initial_marking,
        final_marking,
        parameters={"ret_tuple_as_trans_desc": True}
    )

    # 提取对齐轨迹中的模型变迁信息
    # 过滤掉日志移动，即 y[0][1] == '>>' 的情况
    aligned_traces = [
        [y[0] for y in x['alignment'] if y[0][1] != '>>']
        for x in alignments_
    ]

    # 初始化每个变迁的频率为 0
    frequency_t = {
        t: 0
        for t in net.transitions
    }

    # 将 Petri 网变迁集合转换为列表，便于遍历查找
    list_transitions = list(net.transitions)

    # 遍历每条对齐后的轨迹
    for trace in aligned_traces:
        # 遍历轨迹中的每个对齐步骤
        for align in trace:
            # align[1] 表示触发变迁的名称
            name_t = align[1]

            # 在 Petri 网变迁中查找对应名称的变迁对象
            for t in list_transitions:
                if t.name == name_t:
                    # 找到后将该变迁触发次数加 1
                    frequency_t[t] += 1
                    break

    # 返回变迁触发频率
    return frequency_t


def return_enabled_and_fired_transitions(
        net: PetriNet,
        initial_marking: Marking,
        final_marking: Marking,
        trace_aligned: list
    ) -> tuple:
    """
    根据对齐轨迹返回所有访问过的可使能变迁，以及它们是否被触发。

    对每一步实际触发的变迁：
    - 当前可使能但未触发的变迁标记为 0；
    - 实际触发的变迁标记为 1。

    该函数常用于构造变迁选择模型的二分类训练数据。

    返回：
        visited_transitions:
            所有被访问的候选变迁。
        is_fired:
            每个候选变迁是否被实际触发，1 表示触发，0 表示未触发。
    """

    # 保存每一步访问到的候选变迁
    visited_transitions = []

    # 保存对应候选变迁是否被触发
    is_fired = []

    # 使用初始标识作为当前 token 状态
    tkns = list(initial_marking)

    # 计算初始状态下的可使能变迁
    enabled_transitions = return_enabled_transitions(net, tkns)

    # 遍历对齐轨迹中实际触发的变迁
    for t_fired_name in trace_aligned:
        # 根据变迁名称找到对应的 Petri 网变迁对象
        for t in net.transitions:
            if t.name == t_fired_name[1]:
                t_fired = t
                break

        # 当前可使能但没有被触发的变迁
        not_fired_transitions = list(enabled_transitions - {t_fired})

        # 将未触发的候选变迁加入数据，并标记为 0
        for t_not_fired in not_fired_transitions:
            visited_transitions.append(t_not_fired)
            is_fired.append(0)

        # 将实际触发的变迁加入数据，并标记为 1
        visited_transitions.append(t_fired)
        is_fired.append(1)

        # 根据触发变迁更新当前标识
        tkns = update_current_marking(tkns, t_fired)

        # 如果已经到达终止标识，则直接返回结果
        if set(tkns) == set(final_marking):
            return visited_transitions, is_fired

        # 继续计算新标识下的可使能变迁
        enabled_transitions = return_enabled_transitions(net, tkns)

    return visited_transitions, is_fired


def update_current_marking(
        m: Marking,
        t_fired: PetriNet.Transition
    ) -> Marking:
    """
    根据触发的变迁更新 Petri 网当前标识。

    触发变迁后：
    1. 从输入库所中消耗 token；
    2. 向输出库所中添加 token。

    参数：
        m:
            当前标识。
        t_fired:
            被触发的变迁。

    返回：
        m_out:
            更新后的标识。
    """

    # 复制当前标识，避免直接修改原始对象
    m_out = copy(m)

    # 遍历触发变迁的输入弧，从输入库所中消耗 token
    for a in t_fired.in_arcs:
        m_out[a.source] -= a.weight

        # 如果某个库所中的 token 数量变为 0，则从标识中删除该库所
        if m_out[a.source] == 0:
            del m_out[a.source]

    # 遍历触发变迁的输出弧，向输出库所添加 token
    for a in t_fired.out_arcs:
        m_out[a.target] += a.weight

    return m_out


def return_enabled_transitions(net: PetriNet, tkns: Marking) -> set:
    """
    返回当前标识下所有可使能变迁。

    判断逻辑：
        如果某个变迁的所有输入库所都包含在当前标识中，
        则该变迁可使能。

    参数：
        net:
            Petri 网模型。
        tkns:
            当前标识。

    返回：
        enabled_t:
            当前可使能变迁集合。
    """

    # 初始化可使能变迁集合
    enabled_t = set()

    # 遍历 Petri 网中的所有变迁
    list_transitions = list(net.transitions)
    for t in list_transitions:
        # 如果变迁所有输入库所都存在于当前 token 标识中，则该变迁可触发
        if {a.source for a in t.in_arcs}.issubset(tkns):
            enabled_t.add(t)

    return enabled_t


def return_fired_transition(
        transition_weights: dict,
        enabled_transitions: list
    ) -> PetriNet.Transition:
    """
    根据变迁选择权重，从当前可使能变迁中随机选择一个触发。

    参数：
        transition_weights:
            每个变迁对应的选择权重。
        enabled_transitions:
            当前可使能变迁列表。

    返回：
        被选择触发的变迁。
    """

    # 计算当前可使能变迁的总权重
    total_weight = sum(
        transition_weights[s]
        for s in enabled_transitions
    )

    # 在 0 到总权重之间随机生成一个数
    random_value = random.uniform(0, total_weight)

    # 累加权重，并根据随机数落入的位置选择变迁
    cumulative_weight = 0
    for s in enabled_transitions:
        cumulative_weight += transition_weights[s]
        if random_value <= cumulative_weight:
            return s


def compute_transition_weights_from_model(
        models_t: dict,
        dict_x: dict
    ) -> dict:
    """
    根据训练好的变迁选择模型，计算每个变迁在当前上下文下的选择权重。

    参数：
        models_t:
            每个变迁对应的模型或规则。
        dict_x:
            当前上下文特征字典。

    返回：
        transition_weights:
            当前上下文下每个变迁的选择权重。
    """

    # 初始化变迁权重字典
    transition_weights = dict()

    # 获取所有建模过的变迁
    list_transitions = list(models_t.keys())

    # 遍历每个变迁及其对应模型
    for t in list_transitions:

        # 如果模型是 sklearn 分类器，则使用 predict_proba 计算类别 1 的概率
        if type(models_t[t]) in [
            LogisticRegression,
            DecisionTreeClassifier,
            RandomForestClassifier
        ]:
            # 将特征字典转换为单行 DataFrame
            X = pd.DataFrame({
                k: [dict_x[k]]
                for k in dict_x.keys()
            })

            # 计算该变迁被选择的概率
            transition_weights[t] = compute_proba(models_t, t, X)

        # 如果模型是 river 在线 Hoeffding 自适应树分类器
        elif type(models_t[t]) == HoeffdingAdaptiveTreeClassifier:
            try:
                # 取类别 1 的预测概率作为选择权重
                transition_weights[t] = models_t[t].predict_proba_one(dict_x)[1]
            except:
                # 如果预测失败，则权重置为 0
                transition_weights[t] = 0

        # 如果模型是自定义 DecisionRules，则调用 apply 方法
        elif type(models_t[t]) == DecisionRules:
            transition_weights[t] = models_t[t].apply(dict_x)

        # 如果模型已经是浮点数，则直接作为权重
        elif type(models_t[t]) == float:
            transition_weights[t] = models_t[t]

        # 其他情况使用默认权重 1
        else:
            transition_weights[t] = 1

    return transition_weights


def return_resource(resource_weights: dict, enabled_resources: list) -> str:
    """
    根据资源选择权重，从当前可用资源中随机选择一个资源。

    参数：
        resource_weights:
            每个资源对应的选择权重。
        enabled_resources:
            当前可用资源列表。

    返回：
        被选择的资源名称。
    """

    # 计算当前可用资源的总权重
    total_weight = sum(
        resource_weights[s]
        for s in enabled_resources
    )

    # 在 0 到总权重之间随机生成一个值
    random_value = random.uniform(0, total_weight)

    # 按累计权重选择资源
    cumulative_weight = 0
    for s in enabled_resources:
        cumulative_weight += resource_weights[s]
        if random_value <= cumulative_weight:
            return s


def compute_resource_weights_from_model(
        models_r: dict,
        enabled_resources: list,
        dict_x: dict
    ) -> dict:
    """
    根据资源选择模型，计算当前上下文下可用资源的选择权重。

    参数：
        models_r:
            每个资源对应的模型或规则。
        enabled_resources:
            当前可用于执行活动的资源集合。
        dict_x:
            当前上下文特征。

    返回：
        resource_weights:
            当前可用资源对应的选择权重。
    """

    # 初始化资源权重字典
    resource_weights = dict()

    # 只计算当前可用资源的权重
    for r in enabled_resources:
        # 如果资源模型是 DecisionRules，则调用规则计算权重
        if type(models_r[r]) == DecisionRules:
            resource_weights[r] = models_r[r].apply(dict_x)

        # 如果资源模型是浮点数，则直接作为权重
        elif type(models_r[r]) == float:
            resource_weights[r] = models_r[r]

        # 其他情况设为 0
        else:
            resource_weights[r] = 0

    return resource_weights


def compute_proba(
        models_t: dict,
        t: PetriNet.Transition,
        X: pd.DataFrame
    ) -> float:
    """
    使用 sklearn 分类器计算某个变迁被选择的概率。

    参数：
        models_t:
            变迁到分类模型的映射。
        t:
            当前变迁。
        X:
            当前上下文特征，单行 DataFrame。

    返回：
        float:
            类别 1 的预测概率，即该变迁被选择的概率。
    """

    # 取出当前变迁对应的分类模型
    clf_t = models_t[t]

    # 返回类别 1 的预测概率
    return clf_t.predict_proba(X)[0, 1]


def count_concurrent_events(schedule, t_enabled) -> int:
    """
    统计某个资源在指定时间点正在并发执行的事件数量。

    参数：
        schedule:
            某个资源的任务时间表，元素形式为 (start_time, end_time)。
        t_enabled:
            当前活动使能时间。

    返回：
        count:
            在 t_enabled 时刻仍在执行的任务数量。
    """

    # 初始化并发任务计数
    count = 0

    # 从后往前遍历资源任务时间表
    # 后面的任务通常时间更接近 t_enabled
    for start, end in reversed(schedule):
        # 如果任务结束时间早于或等于 t_enabled，则之前任务也不会与当前时间重叠
        if end <= t_enabled:
            break

        # 如果 t_enabled 落在某个任务执行区间内，则说明该任务正在并发执行
        if start <= t_enabled < end:
            count += 1

    return count


def count_false_hours(
        calendar: dict,
        start_ts: datetime,
        end_ts: datetime
    ) -> int:
    """
    统计两个时间戳之间有多少个小时不在工作日历内。

    参数：
        calendar:
            工作日历，格式为 {weekday: {hour: True/False}}。
        start_ts:
            起始时间。
        end_ts:
            结束时间。

    返回：
        false_hours_count:
            非工作小时数量。
    """

    # 初始化非工作小时计数
    false_hours_count = 0

    # 从起始时间开始逐小时向后遍历
    current_time = start_ts

    while current_time < end_ts:
        # 获取当前时间对应的星期和小时
        weekday = current_time.weekday()
        hour = current_time.hour

        # 如果该星期该小时在日历中被标记为 False，则计为非工作小时
        if calendar.get(weekday, {}).get(hour) == False:
            false_hours_count += 1

        # 时间推进 1 小时
        current_time += timedelta(hours=1)

    return false_hours_count


def add_minutes_with_calendar(
        start_ts: datetime,
        minutes_to_add: int,
        calendar: dict
    ) -> datetime:
    """
    在给定工作日历约束下，为时间戳增加指定分钟数。

    只有 calendar 中标记为 True 的小时才计入有效时间；
    如果当前小时不可工作，则跳到下一个小时整点。

    参数：
        start_ts:
            起始时间。
        minutes_to_add:
            需要增加的有效分钟数。
        calendar:
            工作日历。

    返回：
        current_time:
            增加有效工作分钟后的时间。
    """

    # 剩余需要增加的有效分钟数
    remaining_minutes = minutes_to_add

    # 当前时间指针
    current_time = start_ts

    # 直到所有有效分钟都被消耗
    while remaining_minutes > 0:
        # 获取当前星期和小时
        weekday = current_time.weekday()
        hour = current_time.hour

        # 如果当前小时在工作日历中可用
        if calendar.get(weekday, {}).get(hour, False):
            # 当前小时剩余可用分钟数
            minutes_in_current_hour = min(
                remaining_minutes,
                60 - current_time.minute
            )

            # 时间推进相应分钟
            current_time += timedelta(minutes=minutes_in_current_hour)

            # 扣除已经增加的有效分钟数
            remaining_minutes -= minutes_in_current_hour

        # 如果当前小时不可用，则直接跳到下一个小时整点
        else:
            current_time = (
                current_time + timedelta(hours=1)
            ).replace(minute=0)

    return current_time


def get_transition_from_name(
        t_fired_name: str,
        net: PetriNet
    ) -> PetriNet.Transition:
    """
    根据变迁名称从 Petri 网中找到对应的变迁对象。

    参数：
        t_fired_name:
            变迁名称。
        net:
            Petri 网模型。

    返回：
        匹配名称的 Petri 网变迁对象。
    """

    # 遍历 Petri 网中的所有变迁
    for t in net.transitions:
        # 根据 transition.name 进行匹配
        if t.name == t_fired_name:
            return t


class AlignmentTimeoutError(Exception):
    """当 Petri 网对齐操作超时时抛出。"""
    pass


def _mp_alignment_worker(conn, log, net, im, fm):
    """
    模块级多进程对齐工作函数。

    必须在模块级别定义（不能是嵌套函数），否则 Windows spawn 模式
    无法 pickle 序列化。
    """
    try:
        try:
            conn.send(("started", {
                "pid": int(_os.getpid()),
                "traces": int(len(log)) if log is not None else 0,
            }))
        except Exception:
            pass

        result = alignments.apply_log(
            log, net, im, fm,
            parameters={
                "ret_tuple_as_trans_desc": True,
                "show_progress_bar": True,
            }
        )
        conn.send(("result", result))
    except BaseException as e:
        try:
            conn.send(("error", e))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _apply_alignment_with_timeout(log, net, im, fm, timeout_seconds=None):
    """
    调用 pm4py alignments.apply_log，带超时保护与 v3 路径诊断。

    v3 的关键点：
    1. 在任何 early-return 之前打印真实代码文件与传入 timeout；
    2. 环境变量 drq_ALIGNMENT_TIMEOUT_SECONDS 会在 early-return 之前生效；
    3. Windows 上在 proc.start() 之前就打印，因此可以区分
       "没有进入 wrapper" 与 "卡在 proc.start()"；
    4. 所有打印 flush=True，避免 Flask/Windows 缓冲造成假卡死。
    """
    raw_timeout = timeout_seconds
    try:
        env_timeout = float(_os.environ.get("drq_ALIGNMENT_TIMEOUT_SECONDS", "0") or 0)
    except Exception:
        env_timeout = 0.0

    if env_timeout > 0:
        if timeout_seconds is None or float(timeout_seconds) <= 0:
            timeout_seconds = float(env_timeout)
        else:
            timeout_seconds = min(float(timeout_seconds), float(env_timeout))

    print(
        f"[Alignment Watchdog v3] enter file={__file__}, build={CONTROL_FLOW_DEBUG_BUILD}, "
        f"platform={_sys.platform}, raw_timeout={raw_timeout!r}, "
        f"effective_timeout={timeout_seconds!r}, traces={len(log) if log is not None else 0}",
        flush=True,
    )

    # No timeout requested: keep the original direct PM4Py behavior, but make
    # that path visible so a stall can no longer masquerade as a missing patch.
    if timeout_seconds is None or float(timeout_seconds) <= 0:
        print(
            "[Alignment Watchdog v3] direct PM4Py path selected (no timeout). "
            "Set drq_ALIGNMENT_TIMEOUT_SECONDS=180 while diagnosing.",
            flush=True,
        )
        started = _time.monotonic()
        result = alignments.apply_log(
            log, net, im, fm,
            parameters={
                "ret_tuple_as_trans_desc": True,
                "show_progress_bar": True,
            },
        )
        print(
            f"[Alignment Watchdog v3] direct alignment finished in "
            f"{_time.monotonic() - started:.1f}s",
            flush=True,
        )
        return result

    timeout_seconds = float(timeout_seconds)

    # Diagnostic escape hatch for Windows: bypass multiprocessing entirely.
    # This is useful to distinguish a proc.start()/spawn stall from a PM4Py
    # alignment stall.  It intentionally does not promise hard timeout.
    force_direct = str(_os.environ.get("drq_ALIGNMENT_FORCE_DIRECT", "0")).strip().lower() in {
        "1", "true", "yes", "on"
    }
    if force_direct:
        print(
            f"[Alignment Watchdog v3] FORCE_DIRECT enabled; bypassing multiprocessing; "
            f"nominal_timeout={timeout_seconds:.1f}s",
            flush=True,
        )
        started = _time.monotonic()
        result = alignments.apply_log(
            log, net, im, fm,
            parameters={
                "ret_tuple_as_trans_desc": True,
                "show_progress_bar": True,
            },
        )
        print(
            f"[Alignment Watchdog v3] forced-direct alignment finished in "
            f"{_time.monotonic() - started:.1f}s",
            flush=True,
        )
        return result

    # Unix: signal can interrupt the alignment in the current process.
    if hasattr(_signal, "SIGALRM"):
        def _alarm_handler(signum, frame):
            raise AlignmentTimeoutError(
                f"Alignment timed out after {timeout_seconds:.1f} seconds (SIGALRM)"
            )

        print(
            f"[Alignment Watchdog v3] unix SIGALRM path timeout={timeout_seconds:.1f}s",
            flush=True,
        )
        old_handler = _signal.signal(_signal.SIGALRM, _alarm_handler)
        _signal.alarm(max(1, int(timeout_seconds)))
        started = _time.monotonic()
        try:
            result = alignments.apply_log(
                log, net, im, fm,
                parameters={
                    "ret_tuple_as_trans_desc": True,
                    "show_progress_bar": True,
                },
            )
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, old_handler)
        print(
            f"[Alignment Watchdog v3] unix alignment finished in "
            f"{_time.monotonic() - started:.1f}s",
            flush=True,
        )
        return result

    # Windows / spawn platforms.
    import multiprocessing as _mp

    print(
        f"[Alignment Watchdog v3] preparing multiprocessing context; "
        f"start_method={_mp.get_start_method(allow_none=True)!r}",
        flush=True,
    )
    parent_conn, child_conn = _mp.Pipe(duplex=False)
    proc = _mp.Process(
        target=_mp_alignment_worker,
        args=(child_conn, log, net, im, fm),
    )
    proc.daemon = True

    # This print is intentionally BEFORE proc.start().  If the next line never
    # appears, Windows spawn/pickling is the stall point rather than PM4Py.
    print(
        "[Alignment Watchdog v3] calling proc.start() now...",
        flush=True,
    )
    start_call = _time.monotonic()
    try:
        proc.start()
    except Exception as e:
        parent_conn.close()
        child_conn.close()
        raise AlignmentTimeoutError(
            f"Failed to start alignment subprocess: {e}"
        )

    start_return_elapsed = _time.monotonic() - start_call
    child_conn.close()
    child_pid = getattr(proc, "pid", None)
    print(
        f"[Alignment Watchdog v3] proc.start() returned in {start_return_elapsed:.3f}s; "
        f"pid={child_pid}, timeout={timeout_seconds:.1f}s",
        flush=True,
    )

    start_monotonic = _time.monotonic()
    deadline = start_monotonic + timeout_seconds
    last_heartbeat = start_monotonic
    child_started = False

    try:
        while True:
            now = _time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                raise AlignmentTimeoutError(
                    f"Alignment timed out after {timeout_seconds:.1f} seconds "
                    f"(multiprocessing, pid={child_pid}, child_started={child_started})"
                )

            if parent_conn.poll(min(5.0, remaining)):
                try:
                    kind, payload = parent_conn.recv()
                except EOFError:
                    raise AlignmentTimeoutError(
                        f"Alignment subprocess pipe closed unexpectedly "
                        f"(exitcode={proc.exitcode}, pid={child_pid})"
                    )

                if kind == "started":
                    child_started = True
                    if isinstance(payload, dict):
                        child_pid = payload.get("pid", child_pid)
                    print(
                        f"[Alignment Watchdog v3] child started pid={child_pid}; "
                        "PM4Py alignment is running",
                        flush=True,
                    )
                    continue
                if kind == "result":
                    elapsed = _time.monotonic() - start_monotonic
                    print(
                        f"[Alignment Watchdog v3] alignment finished in {elapsed:.1f}s "
                        f"(pid={child_pid})",
                        flush=True,
                    )
                    return payload
                if kind == "error":
                    raise payload
                print(
                    f"[Alignment Watchdog v3] unexpected child message kind={kind!r}",
                    flush=True,
                )

            now = _time.monotonic()
            if not proc.is_alive():
                if parent_conn.poll(0.2):
                    continue
                raise AlignmentTimeoutError(
                    f"Alignment subprocess exited unexpectedly "
                    f"(exitcode={proc.exitcode}, pid={child_pid})"
                )

            if now - last_heartbeat >= 30.0:
                elapsed = now - start_monotonic
                print(
                    f"[Alignment Watchdog v3] still running: elapsed={elapsed:.1f}s, "
                    f"pid={child_pid}, alive={proc.is_alive()}, child_started={child_started}",
                    flush=True,
                )
                last_heartbeat = now
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=10)
        parent_conn.close()

def build_df_features(
        log,
        net,
        im,
        fm,
        act_to_resources,
        net_transition_labels,
        resources,
        label_data_attributes=[],
        alignment_timeout_seconds=None,
    ):
    """
    从事件日志和 Petri 网模型中构建仿真参数学习所需的特征数据集。

    每一行对应一次变迁执行或模型移动，包含：
    1. 案例编号；
    2. 实际触发的变迁及其标签；
    3. 执行资源；
    4. 变迁使能时间、开始时间、结束时间；
    5. 执行前可使能变迁集合；
    6. 执行前可用资源集合；
    7. 当前资源负载；
    8. 案例数据属性；
    9. 资源历史使用次数；
    10. 活动历史执行次数。

    参数：
        log:
            输入事件日志。
        net:
            Petri 网模型。
        im:
            初始标识。
        fm:
            终止标识。
        act_to_resources:
            活动到可执行资源集合的映射。
        net_transition_labels:
            Petri 网中所有活动标签。
        resources:
            资源列表。
        label_data_attributes:
            需要保留的数据属性列表。

    返回：
        df:
            构造好的特征数据表。
    """

    print(
        f"[Feature Build v3] ENTER file={__file__}, build={CONTROL_FLOW_DEBUG_BUILD}, "
        f"traces={len(log) if log is not None else 0}, alignment_timeout={alignment_timeout_seconds!r}",
        flush=True,
    )
    _feature_build_t0 = _time.monotonic()

    # 将 EventLog 转换为 DataFrame，便于计算资源负载
    df_log = pm4py.convert_to_dataframe(log)
    print(
        f"[Feature Build v3] dataframe ready rows={len(df_log)}, "
        f"elapsed={_time.monotonic() - _feature_build_t0:.3f}s",
        flush=True,
    )

    # 将 start:timestamp 转换为 Unix 时间戳
    # 注意：这里通过 str(x)[:-6] 去掉末尾时区信息后再解析
    df_log["start:timestamp"] = df_log["start:timestamp"].apply(
        lambda x: datetime.fromisoformat(str(x)[:-6]).timestamp()
    )

    # 将 time:timestamp 转换为 Unix 时间戳
    df_log["time:timestamp"] = df_log["time:timestamp"].apply(
        lambda x: datetime.fromisoformat(str(x)[:-6]).timestamp()
    )

    print(
        f"[Feature Build v3] timestamps converted; calling alignment; "
        f"elapsed={_time.monotonic() - _feature_build_t0:.3f}s",
        flush=True,
    )

    # 对事件日志和 Petri 网进行对齐（带超时保护）
    aligned_traces = _apply_alignment_with_timeout(
        log, net, im, fm,
        timeout_seconds=alignment_timeout_seconds,
    )

    print(
        f"[Feature Build v3] alignment returned traces={len(aligned_traces)}, "
        f"elapsed={_time.monotonic() - _feature_build_t0:.3f}s",
        flush=True,
    )

    # 用于保存所有训练样本
    dataset = []

    # 遍历日志中的每条 trace
    for i, trace in enumerate(tqdm(log)):

        # 获取当前 trace 的对齐结果
        trace_aligned = aligned_traces[i]["alignment"]

        # 当前案例编号
        case_id = trace[0]["case:concept:name"]

        # 初始化当前案例的活动历史计数
        history = {
            t_l: 0
            for t_l in net_transition_labels
        }

        # 初始化当前案例的资源历史使用计数
        history_res = {
            r: 0
            for r in resources
        }

        # 获取当前 trace 的数据属性
        if label_data_attributes:
            try:
                # 优先尝试从 trace 级别读取属性
                trace_attributes = [
                    trace[a]
                    for a in label_data_attributes
                ]
            except:
                # 如果 trace 级别读取失败，则从第一个事件读取属性
                trace_attributes = [
                    trace[0][a]
                    for a in label_data_attributes
                ]
        else:
            # 如果没有额外数据属性，则置为空列表
            trace_attributes = []

        # 当前 Petri 网标识初始化为初始标识
        marking = im

        # j 表示当前对应到原始 trace 中第几个事件
        j = 0

        # 记录每个可使能变迁第一次变为可使能的时间
        transition_enabled_times = dict()

        # 当前时间初始化为 trace 第一个事件的开始时间
        current_t = trace[0]["start:timestamp"]

        # 遍历当前 trace 的对齐步骤
        for step in trace_aligned:

            # 如果是 log move，说明日志中有事件但模型不同步执行
            # 此时只推进日志事件索引
            if step[0][1] == ">>":
                j += 1
                continue

            # 根据对齐步骤中的变迁名称获取 Petri 网变迁对象
            transition = get_transition_from_name(step[0][1], net)

            # 当前变迁对应的活动标签，静默变迁可能为 None
            transition_label = transition.label

            # 当前标识下执行前可使能的变迁集合
            prev_enabled_transitions = return_enabled_transitions(
                net,
                marking
            )

            # 对当前可使能变迁，若尚未记录使能时间，则记录为 current_t
            for enabled in prev_enabled_transitions:
                if enabled not in transition_enabled_times:
                    transition_enabled_times[enabled] = current_t

            # 如果是同步移动，说明日志事件和模型变迁匹配
            if step[1][0] == step[1][1]:
                # 当前事件的执行资源
                resource = trace[j]["org:resource"]

                # 当前事件开始时间
                start_t = trace[j]["start:timestamp"]

                # 当前事件完成时间
                end_t = trace[j]["time:timestamp"]

                # 当前变迁的使能时间
                enabled_t = transition_enabled_times[transition]

                # 当前时间推进到事件完成时间
                current_t = end_t

                # 从原始日志中筛选在 enabled_t 时刻仍在执行的事件
                # 条件：完成时间晚于 enabled_t，且开始时间早于 enabled_t
                df_log_filtered = df_log[
                    (df_log["time:timestamp"] > enabled_t.timestamp())
                    & (df_log["start:timestamp"] < enabled_t.timestamp())
                ]

                # 统计同一资源在 enabled_t 时刻的工作负载
                res_workload = (
                    df_log_filtered['org:resource'] == resource
                ).sum()

                # 当前活动在执行前可用的资源集合
                prev_enabled_resources = act_to_resources[transition_label]

                # 日志事件索引加 1
                j += 1

            # 如果是 model move，说明模型中发生了变迁，但日志中没有对应事件
            else:
                # model move 没有实际资源和时间信息
                resource = None
                enabled_t = None
                start_t = None
                end_t = None
                res_workload = None
                prev_enabled_resources = None

            # 当前变迁已经被处理，从使能时间字典中删除
            del transition_enabled_times[transition]

            # 根据当前变迁更新 Petri 网标识
            marking = update_current_marking(marking, transition)

            # 将当前步骤的所有特征保存为一行样本
            dataset.append(
                (
                    case_id,
                    transition,
                    transition_label,
                    resource,
                    enabled_t,
                    start_t,
                    end_t,
                    prev_enabled_transitions,
                    prev_enabled_resources,
                    res_workload
                )
                + tuple(trace_attributes)
                + tuple(history_res.values())
                + tuple(history.values())
            )

            # 如果当前变迁有活动标签，则更新活动历史和资源历史
            if transition_label:
                # 当前活动执行次数加 1
                history[transition_label] += 1

                # 如果资源在候选资源列表中，则该资源使用次数加 1
                if resource in resources:
                    history_res[resource] += 1

    # 将样本列表转换为 DataFrame
    df = pd.DataFrame(
        dataset,
        columns=[
            "case_id",
            "transition",
            "transition_label",
            "resource",
            "enabled_t",
            "start_t",
            "end_t",
            "prev_enabled_transitions",
            "prev_enabled_resources",
            "res_workload"
        ]
        + label_data_attributes
        + resources
        + net_transition_labels
    )

    # 返回特征数据集
    return df