# 导入 Counter，用于统计类别型属性中各取值出现的次数
from collections import Counter

# 导入 pm4py，用于处理流程挖掘中的事件日志
import pm4py

# 导入 EventLog 类型，用于指定函数输入日志的类型
from pm4py.objects.log.obj import EventLog

# 导入自定义函数，用于为数值型属性拟合最优概率分布
from drqsim.utils.distribution_utils import return_best_distribution

# 导入 numpy，用于计算最小值、最大值和均值等统计量
import numpy as np


def discover_attributes_distribution(
        log: EventLog,
        label_data_attributes: list,
        label_data_attributes_categorical: list
    ) -> dict:
    """
    发现案例级数据属性的分布。

    该函数用于统计事件日志中案例级属性的分布情况。
    对于类别型属性，使用频率统计得到每个取值的概率；
    对于数值型属性，拟合最优概率分布，并记录最小值、最大值和均值。

    参数：
        log:
            输入事件日志。
        label_data_attributes:
            需要分析的数据属性列表。
        label_data_attributes_categorical:
            其中属于类别型变量的数据属性列表。

    返回：
        data_attributes_distribution:
            每个数据属性对应的分布信息。
            类别型属性返回 {取值: 概率}；
            数值型属性返回 (分布类型, 分布参数, 最小值, 最大值, 均值)。
    """

    # 初始化每个数据属性的观测值列表
    # key 为属性名，value 为该属性在所有 trace 中的取值列表
    observed_values = {
        l: []
        for l in label_data_attributes
    }

    # 遍历事件日志中的每条轨迹
    for trace in log:
        # 遍历需要统计的数据属性
        for l in label_data_attributes:
            # 这里取 trace[0][l]，把每条案例第一个事件的各个属性值收集起来
            # 因此该函数默认这些属性是案例级属性，或者至少在同一个 trace 内保持一致
            observed_values[l].append(trace[0][l])

    # 用于保存每个数据属性的分布结果
    data_attributes_distribution = dict()

    # 遍历每个数据属性，分别计算其分布
    for l in label_data_attributes:

        # 如果当前属性属于类别型属性
        if l in label_data_attributes_categorical:
            # 统计该属性不同取值出现的次数
            frequency = Counter(observed_values[l])

            # 统计该属性的观测值总数
            total = len(observed_values[l])

            # 将频数转换为概率分布
            # 例如：{'A': 0.6, 'B': 0.4}
            data_attributes_distribution[l] = {
                lst: count / total
                for lst, count in frequency.items()
            }

        # 如果当前属性不是类别型属性，则将其视为数值型属性
        else:
            # 在候选分布中寻找最适合当前属性观测值的概率分布
            # dist_search 指定候选分布包括：
            # fixed：固定值分布
            # norm：正态分布
            # expon：指数分布
            # uniform：均匀分布
            dist, params = return_best_distribution(
                observed_values[l],
                dist_search=['fixed', 'norm', 'expon', 'uniform']
            )

            # 计算当前数值型属性的最小值
            min_value = np.min(observed_values[l])

            # 计算当前数值型属性的最大值
            max_value = np.max(observed_values[l])

            # 计算当前数值型属性的均值
            mean_value = np.mean(observed_values[l])

            # 保存当前数值型属性的分布信息和基础统计量
            data_attributes_distribution[l] = (
                dist,
                params,
                min_value,
                max_value,
                mean_value
            )

    # 返回所有数据属性的分布结果
    return data_attributes_distribution


def return_label_data_attributes(log: EventLog) -> tuple:
    """
    从事件日志中识别数据属性及其中的类别型属性。

    该函数首先将 EventLog 转换为 DataFrame，
    然后去除标准 XES 字段，剩余字段被视为额外的数据属性。
    接着根据字段首个值的类型判断该属性是否为类别型属性。

    参数：
        log:
            输入事件日志。

    返回：
        label_data_attributes:
            日志中的额外数据属性列表。
        label_data_attributes_categorical:
            额外数据属性中属于类别型变量的属性列表。
    """

    # 定义标准 XES 日志字段
    # 这些字段通常是流程挖掘事件日志中的基础字段，不作为额外数据属性处理
    standard_xes_columns = {
        "case:concept:name",  # 案例编号
        "concept:name",      # 活动名称
        "time:timestamp",    # 事件完成时间
        "start:timestamp",   # 事件开始时间
        "org:resource",      # 执行资源
        "org:role"           # 资源角色
    }

    # 将 pm4py 的 EventLog 对象转换为 pandas DataFrame，便于读取字段名和字段类型
    df_log = pm4py.convert_to_dataframe(log)

    # 识别额外数据属性：
    # 即日志中除标准 XES 字段之外的所有字段
    label_data_attributes = list(
        set(df_log.columns) - standard_xes_columns
    )

    # 初始化类别型数据属性列表
    label_data_attributes_categorical = []

    # 遍历所有额外数据属性
    for l in label_data_attributes:
        # 如果该字段第一个值的类型是字符串，则将其视为类别型属性，取l这一列的第一行数据
        if type(df_log[l].iloc[0]) == str:
            label_data_attributes_categorical.append(l)

    # 返回所有额外数据属性，以及其中的类别型属性
    return label_data_attributes, label_data_attributes_categorical