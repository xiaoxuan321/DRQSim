# 导入自定义决策规则类，用于保存和恢复决策树规则
from drqsim.utils.rule_utils import DecisionRules

# 导入 PetriNet 类型，用于处理 Petri 网中的变迁对象
from pm4py.objects.petri_net.obj import PetriNet

# 导入 copy，用于深拷贝规则结构，避免修改原始对象
import copy

# 导入自定义采样函数，用于根据分布参数重新生成 sampled 采样列表
from drqsim.utils.distribution_utils import sampling_from_dist

# 导入 scipy.stats，用于根据字符串名称恢复 scipy 概率分布对象
import scipy.stats as stats


def transition_to_name(t: PetriNet.Transition) -> str:
    """
    将 Petri 网中的变迁对象转换为字符串名称。

    如果变迁存在 label，则优先使用 label；
    如果 label 为空，则使用变迁自身的 name。

    参数：
        t:
            Petri 网变迁对象。

    返回：
        name:
            变迁名称字符串。
    """

    # 如果变迁有业务标签，则使用业务标签作为名称
    if t.label is not None:
        name = t.label

    # 如果变迁没有业务标签，通常是静默变迁，则使用内部名称
    else:
        name = t.name

    return name


def name_to_transition(s: str, net: PetriNet) -> PetriNet.Transition:
    """
    根据变迁名称字符串，从 Petri 网中找到对应的变迁对象。

    该函数既支持通过 transition.name 查找，
    也支持通过 transition.label 查找。

    参数：
        s:
            变迁名称字符串。
        net:
            Petri 网对象。

    返回：
        t:
            与名称匹配的 Petri 网变迁对象。
    """

    # 遍历 Petri 网中的所有变迁
    for t in net.transitions:
        # 如果字符串与变迁内部名称一致，则返回该变迁
        if s == t.name:
            return t

        # 如果字符串与变迁业务标签一致，则返回该变迁
        if s == t.label:
            return t

def _dist_tuple_to_dict(t):
    """
    将不同长度的分布 tuple 统一转换为 JSON 可保存格式。

    支持：
    1. (dist, params, min_value, max_value, mean_value)
    2. (dist, params, min_value, max_value)
    3. (dist, params)
    4. ('fixed', value)
    """

    if not isinstance(t, tuple):
        return t

    dist = t[0] if len(t) > 0 else "fixed"

    params = t[1] if len(t) > 1 else (0.0,)

    if not isinstance(params, (list, tuple)):
        params = (params,)

    params = list(params)

    if len(t) > 2:
        min_value = t[2]
    else:
        min_value = params[0] if params else 0.0

    if len(t) > 3:
        max_value = t[3]
    else:
        max_value = min_value

    if len(t) > 4:
        mean_value = t[4]
    else:
        if params:
            mean_value = params[0]
        else:
            mean_value = min_value

    # scipy 分布对象转字符串
    if hasattr(dist, "__module__") and str(dist.__module__).startswith("scipy.stats"):
        dist_name = dist.name
    else:
        dist_name = str(dist)

    return {
        "dist_name": dist_name,
        "params": params,
        "min_value": float(min_value),
        "max_value": float(max_value),
        "mean_value": float(mean_value)
    }
def decision_rules_to_dict(d: DecisionRules) -> dict:
    """
    将 DecisionRules 对象或概率分布元组转换为可 JSON 序列化的字典。

    主要作用：
    1. 将 DecisionRules.rules 中的规则结构转换为普通字典；
    2. 删除 sampled 字段，避免 JSON 文件过大；
    3. 将 scipy.stats 分布对象转换为字符串；
    4. 将分布元组转换为包含 dist_name、params、min_value、max_value 的字典。

    参数：
        d:
            DecisionRules 对象、分布元组、普通数值或 None。

    返回：
        dict 或其他可序列化对象。
    """

    def convert(obj):
        """
        递归转换规则对象中的元素，使其可以被 JSON 序列化。
        """

        # 如果当前对象是字典，则递归处理每个键值对
        if isinstance(obj, dict):
            new_obj = {}

            for k, v in obj.items():
                # sampled 通常包含大量采样值，保存 JSON 时跳过该字段
                if k == "sampled":
                    continue

                # 递归转换当前值
                new_obj[k] = convert(v)

            return new_obj

        # 如果当前对象是元组，通常表示一个概率分布：
        # (dist, params, min_value, max_value)
        elif isinstance(obj, tuple):
            return _dist_tuple_to_dict(obj)

        # 如果当前对象是 scipy.stats 中的分布对象，则转换为分布名称字符串
        elif hasattr(obj, '__module__') and obj.__module__.startswith("scipy.stats"):
            return obj.name

        # 其他普通对象直接返回
        else:
            return obj

    # 如果 d 本身是一个分布元组，通常格式为：
    # (dist, params, min_value, max_value, mean_value)
    if isinstance(d, tuple):
        return _dist_tuple_to_dict(d)

    # 如果 d 不是 DecisionRules 对象，说明它可能是普通权重、None 或其他值
    if not isinstance(d, DecisionRules):
        # 如果为 None，则默认转换为 1
        if d is None:
            d = 1

        return d

    # 如果 d 是 DecisionRules 对象，则深拷贝其 rules 后递归转换
    # 深拷贝可以避免转换过程影响原始规则对象
    return convert(copy.deepcopy(d.rules))


def convert_calendar_names(calendar: dict, to_number=False):
    """
    转换日历中的星期表示方式。

    当 to_number=False 时：
        将数字星期转换为英文星期名称，例如 0 -> Monday。

    当 to_number=True 时：
        将英文星期名称转换为数字星期，例如 Monday -> 0。

    参数：
        calendar:
            日历字典，格式一般为 {weekday: {hour: True/False}}。
        to_number:
            是否将星期名称转换为数字。

    返回：
        new_calendar:
            转换后的日历字典。
    """

    # 如果需要转换为数字形式
    if to_number:
        weekday_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }

    # 如果需要转换为英文星期名称
    else:
        weekday_map = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday"
        }

    # 初始化新的日历字典
    new_calendar = {}

    # 遍历原始日历中的星期和小时配置
    for weekday, hours in calendar.items():
        # 转换星期键，并将小时键统一转换为 int 类型
        new_calendar[weekday_map[weekday]] = {
            int(k): v
            for k, v in hours.items()
        }

    return new_calendar


def dict_to_decrules(d: dict) -> DecisionRules:
    """
    将字典格式的规则恢复为 DecisionRules 对象。

    该函数通常用于从 JSON 文件中读取规则后，
    将其恢复为仿真引擎可以直接调用的 DecisionRules 结构。

    主要处理内容：
    1. 将字符串 "true" / "false" 恢复为布尔值 True / False；
    2. 将数字字符串键恢复为 int；
    3. 将 dist 字段中的分布名称恢复为 scipy 分布对象；
    4. 根据分布参数重新生成 sampled 采样列表。

    参数：
        d:
            从 JSON 中读取出来的规则字典。

    返回：
        dr:
            恢复后的 DecisionRules 对象。
    """

    def convert(obj):
        """
        递归恢复字典中的键和值。
        """

        # 如果当前对象是字典，则递归恢复
        if isinstance(obj, dict):
            new_obj = {}

            for k, v in obj.items():
                # JSON 中布尔值键可能被保存为字符串 "true"
                if k == "true":
                    new_k = True

                # JSON 中布尔值键可能被保存为字符串 "false"
                elif k == "false":
                    new_k = False

                else:
                    # 默认保留原始键
                    new_k = k

                    # 尝试将字符串数字键转换为 int
                    try:
                        new_k = int(k)
                    except:
                        pass

                # 如果当前键是 dist，说明该字段保存的是概率分布信息
                if k == "dist":
                    # 将分布名称恢复为 scipy 分布对象或 fixed 字符串
                    # 同时恢复分布参数、最小值和最大值
                    new_obj[k] = (
                        fromstr_to_scipy(obj[k]["dist_name"]),
                        tuple(obj[k]["params"]),
                        obj[k]["min_value"],
                        obj[k]["max_value"]
                    )

                    # 根据恢复后的分布参数重新生成 sampled 采样列表
                    # sampled 用于仿真时随机抽样
                    new_obj["sampled"] = sampling_from_dist(
                        new_obj[k][0],
                        new_obj[k][1],
                        new_obj[k][2],
                        new_obj[k][3],
                        obj["value"]
                    )

                else:
                    # 其他字段递归恢复
                    new_obj[new_k] = convert(v)

            return new_obj

        # 如果不是字典，则直接返回
        else:
            return obj

    # 创建 DecisionRules 对象
    dr = DecisionRules()

    # 将字典递归恢复为规则结构
    dr.rules = convert(d)

    return dr


def fromstr_to_scipy(s):
    """
    将字符串形式的分布名称恢复为 scipy.stats 中的分布对象。

    参数：
        s:
            分布名称字符串，例如 "norm"、"expon"、"uniform" 或 "fixed"。

    返回：
        scipy 分布对象或 "fixed" 字符串。
    """

    # fixed 是自定义固定值分布，不属于 scipy.stats，因此直接返回字符串
    if s == "fixed":
        return s

    # 其他分布名称从 scipy.stats 中获取对应分布对象
    else:
        return getattr(stats, s)