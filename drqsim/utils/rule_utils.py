# 导入正则表达式模块，用于处理和重写 Graphviz DOT 字符串
import re

# 导入自定义分布采样函数，用于根据概率分布生成采样值
from drqsim.utils.distribution_utils import sampling_from_dist

# 导入 sklearn 的决策树导出工具，用于将决策树转换为 Graphviz DOT 格式
from sklearn.tree import export_graphviz

# 导入 graphviz，用于生成和渲染决策树图
import graphviz

# 导入 random，用于从采样列表中随机选择一个值
import random

# 导入 numpy，用于数值计算，例如开方、正态采样等
import numpy as np

# 导入 pandas，用于处理 river 决策树转换得到的 DataFrame
import pandas as pd

# 导入 scipy.stats，用于保存概率分布对象，例如正态分布 stats.norm
import scipy.stats as stats


def build_graph_vis(model_t, model_distributions=False):
    """
    将 sklearn 决策树模型转换为 Graphviz 可视化对象。

    参数：
        model_t:
            sklearn 决策树模型。
        model_distributions:
            是否以分布形式重新格式化叶子节点显示。

    返回：
        graph:
            graphviz.Source 对象，可用于可视化或导出决策树。
    """

    try:
        # 尝试导出带填充颜色的决策树 DOT 字符串
        dot_data = export_graphviz(
            model_t,
            feature_names=model_t.feature_names_in_,  # 使用模型中的输入特征名
            label='none',                             # 不显示默认标签名称
            filled=True,                              # 节点填充颜色
            rounded=True,                             # 节点圆角显示
            impurity=False,                           # 不显示 impurity 信息
            proportion=True                           # 显示样本比例
        )
    except:
        # 如果 filled=True 导出失败，则使用不带填充颜色的简化导出方式
        dot_data = export_graphviz(
            model_t,
            feature_names=model_t.feature_names_in_,
            label='none',
            rounded=True,
            impurity=False,
            proportion=True
        )

    # 对导出的 DOT 字符串进行重新格式化，使其更适合后续解析和展示
    new_dot_data = reformat_dot_str(dot_data, model_distributions)

    # 将 DOT 字符串转换为 graphviz.Source 对象
    graph = graphviz.Source(new_dot_data)

    return graph


def reformat_dot_str(input_str, dot_distributions=False):
    """
    重新格式化 Graphviz DOT 字符串。

    主要作用：
    1. 删除节点中多余的百分比信息；
    2. 简化叶子节点中的数值表达；
    3. 如果 dot_distributions=True，则将双数值叶子节点格式化为区间形式。

    参数：
        input_str:
            原始 DOT 字符串。
        dot_distributions:
            是否按照分布或区间形式格式化叶子节点。

    返回：
        result:
            重新格式化后的 DOT 字符串。
    """

    # 删除类似 “xx%\\n” 的样本比例显示
    result = re.sub(r'\d+\.?\d*\s?%\\n', '', input_str)

    # 将类似 “[0.0, 1.0]” 的数组标签简化为第二个数值
    result = re.sub(r'\[\d+\.?\d*,\s*(\d+\.?\d*)\]', r'\1', result)

    # 如果需要处理分布形式的节点显示
    if dot_distributions:
        # 匹配类似 [[num1]\n[num2]] 的模式
        pattern = r'\[\[(\d+\.?\d*)\]\\n\[(\d+\.?\d*)\]\]'

        def round_and_replace(match):
            """
            将匹配到的两个数字四舍五入，并格式化为 (num1, num2)。
            """
            num1, num2 = match.groups()
            rounded1 = round(float(num1))
            rounded2 = round(float(num2))
            return f'({rounded1}, {rounded2})'

        # 对匹配到的分布节点进行替换
        result = re.sub(pattern, round_and_replace, result)

    return result


def parse_tree(dot_string):
    """
    从 Graphviz DOT 字符串中解析决策树结构。

    该函数提取：
    1. 节点信息，包括判断特征、阈值或叶子节点值；
    2. 边信息，即父节点到子节点的连接关系。

    参数：
        dot_string:
            决策树的 DOT 字符串。

    返回：
        nodes:
            节点字典。
            决策节点格式为 {'feature': 特征名, 'threshold': 阈值}；
            叶子节点格式为 {'value': 叶子值}。
        edges:
            边列表，每个元素为 (parent, child)。
    """

    # 匹配节点，例如：0 [label="feature <= threshold"]
    node_pattern = r'(\d+) \[label="([^"]+)"'

    # 匹配边，例如：0 -> 1 或带有额外标签的边
    edge_pattern = r'(\d+) -> (\d+)(?: \[labeldistance=[^,]+, labelangle=[^,]+, headlabel="([^"]+)"\])?'

    # 保存解析得到的节点
    nodes = {}

    # 保存解析得到的边
    edges = []

    # 解析所有节点
    for match in re.findall(node_pattern, dot_string):
        # 节点 ID
        node_id = int(match[0])

        # 节点标签内容，按换行符拆分
        label_info = match[1].split("\\n")

        # 如果标签中包含多行信息，一般表示这是一个决策节点
        if len(label_info) > 1:
            # 第一行通常是 “feature <= threshold”
            feature, threshold = label_info[0].split(" <= ")

            # 将阈值转换为浮点数
            threshold = float(threshold)

            # 保存决策节点信息
            nodes[node_id] = {
                'feature': feature,
                'threshold': threshold
            }

        # 否则认为是叶子节点
        else:
            try:
                # 尝试将叶子节点值解析为浮点数
                nodes[node_id] = {
                    'value': float(label_info[0])
                }
            except:
                # 如果不能解析为单个浮点数，则尝试解析为类似 "(a, b)" 的二元组
                nodes[node_id] = {
                    'value': (
                        int(label_info[0][1:-1].split(', ')[0]),
                        int(label_info[0][1:-1].split(', ')[1])
                    )
                }

    # 解析所有边
    for match in re.findall(edge_pattern, dot_string):
        # 父节点 ID 和子节点 ID
        parent, child = int(match[0]), int(match[1])

        # 保存边
        edges.append((parent, child))

    return nodes, edges


def build_tree_structure(nodes, edges):
    """
    根据节点和边构造可遍历的树结构。

    该函数会在每个决策节点中添加 children 字段：
        children[True] 表示左分支，即满足 feature <= threshold；
        children[False] 表示右分支，即不满足 feature <= threshold。

    参数：
        nodes:
            节点字典。
        edges:
            边列表。

    返回：
        nodes:
            添加 children 信息后的树结构字典。
    """

    # 该变量原代码中创建但没有使用，保留原逻辑不做修改
    tree = {}

    def add_edge(parent, child, edge_index):
        """
        给父节点添加一个子节点。

        edge_index=0 表示第一条边，默认对应 True 分支；
        edge_index=1 表示第二条边，默认对应 False 分支。
        """

        # 如果父节点还没有 children 字段，则先初始化
        if 'children' not in nodes[parent]:
            nodes[parent]['children'] = {}

        # 第一条边认为是 True 分支，第二条边认为是 False 分支
        condition = (edge_index == 0)

        # 保存当前条件对应的子节点
        nodes[parent]['children'][condition] = child

    # 记录每个父节点已经添加了多少条边
    parent_edge_count = {}

    # 遍历所有边，构造父子关系
    for parent, child in edges:
        # 如果当前父节点第一次出现，则初始化边计数为 0
        if parent not in parent_edge_count:
            parent_edge_count[parent] = 0

        # 添加边
        add_edge(parent, child, parent_edge_count[parent])

        # 更新该父节点的边计数
        parent_edge_count[parent] += 1

    return nodes


def traverse_tree(tree, features):
    """
    根据输入特征遍历决策树，并返回叶子节点的预测值。

    参数：
        tree:
            决策树规则结构。
        features:
            输入特征字典，例如 {'x1': 3.2, 'x2': 1.0}。

    返回：
        value:
            决策树叶子节点的预测值。
    """

    # 如果 tree 不是字典，说明它本身就是一个固定值，直接返回
    if type(tree) != dict:
        return tree

    # 从根节点开始遍历，默认根节点 ID 为 0
    current_node = 0

    # 只要当前节点不是叶子节点，就继续向下判断
    while 'value' not in tree[current_node]:
        # 当前节点使用的判断特征
        feature = tree[current_node]['feature']

        # 当前节点的判断阈值
        threshold = tree[current_node]['threshold']

        # 如果输入特征值小于等于阈值，则进入 True 分支
        if features[feature] <= threshold:
            current_node = tree[current_node]['children'][True]

        # 否则进入 False 分支
        else:
            current_node = tree[current_node]['children'][False]

    # 返回叶子节点的值
    return tree[current_node]['value']


def traverse_tree_distribution(tree, features):
    """
    根据输入特征遍历决策树，并从叶子节点的 sampled 列表中随机采样一个值。

    该函数常用于时间建模场景：
    决策树叶子节点不仅保存均值 value，还保存一个采样列表 sampled，
    仿真时可以从 sampled 中随机取值，体现随机性。

    参数：
        tree:
            决策树规则结构。
        features:
            输入特征字典。

    返回：
        sampled_value:
            从叶子节点采样列表中随机选出的值。
    """

    # 如果 tree 不是字典，则认为它是一个采样列表，从中随机选一个值
    if type(tree) != dict:
        return random.choice(tree)

    # 从根节点开始遍历
    current_node = 0

    # 遍历直到叶子节点
    while 'value' not in tree[current_node]:
        feature = tree[current_node]['feature']
        threshold = tree[current_node]['threshold']

        # 满足阈值条件走 True 分支
        if features[feature] <= threshold:
            current_node = tree[current_node]['children'][True]

        # 不满足阈值条件走 False 分支
        else:
            current_node = tree[current_node]['children'][False]

    # 从当前叶子节点保存的 sampled 采样列表中随机取值
    return random.choice(tree[current_node]['sampled'])


def transform_river_decision_tree_data(
        decision_tree,
        distribution=True,
        min_value=0,
        max_value=60 * 24
    ) -> dict:
    """
    将 river 在线决策树模型转换为本代码统一使用的规则字典格式。

    该函数兼容两类任务：
    1. distribution=True：
       用于回归或时间分布建模，叶子节点保存 value、sampled 和 dist；
    2. distribution=False：
       用于分类或概率建模，叶子节点只保存 value。

    参数：
        decision_tree:
            river 决策树模型。
        distribution:
            是否按照分布建模方式处理叶子节点。
        min_value:
            采样值允许的最小值。
        max_value:
            采样值允许的最大值。

    返回：
        transformed_data_sorted:
            转换后的规则字典。
    """

    # 如果决策树高度为 0，说明没有有效分裂，只有一个根节点
    if decision_tree.height == 0:
        if distribution:
            # 分布模式下，返回固定为 0 的默认分布节点
            return {
                0: {
                    'value': 0,
                    'sampled': [0],
                    'dist': ("fixed", (0,), 0, 0)
                }
            }
        else:
            # 非分布模式下，返回默认值 1
            return {
                0: {
                    "value": 1
                }
            }

    # 将 river 决策树转换为 DataFrame
    df = decision_tree.to_dataframe()

    # 如果 to_dataframe() 返回 None，则说明树结构不可用，需要从模型调试信息中提取统计量
    if df is None:
        if distribution:
            # debug_one({}) 返回模型在空输入上的调试信息
            mean_var = decision_tree.debug_one({})

            # 从调试字符串中解析均值和方差信息
            pred_split = mean_var.split("\n")[-2].split(" | ")

            # 解析均值
            value = float(pred_split[0][6:].replace(",", ""))

            # 解析方差
            variance = float(pred_split[1][5:].replace(",", ""))

            # 根据方差计算标准差，并确保方差非负
            std_dev = np.sqrt(max(0, variance))

            # 如果标准差为 0，则所有采样值都等于均值
            if std_dev == 0:
                sampled_values = [value]
            else:
                # 从正态分布中采样 1000 个值
                sampled_values = np.random.normal(
                    loc=value,
                    scale=std_dev,
                    size=1000
                )

                # 将小于最小值的采样值替换为均值
                sampled_values[sampled_values < min_value] = value

                # 将大于最大值的采样值替换为均值
                sampled_values[sampled_values > max_value] = value

                # 转换为 Python 列表
                sampled_values = sampled_values.tolist()

            # 返回单节点的分布规则
            return {
                0: {
                    'value': value,
                    'sampled': sampled_values,
                    'dist': (
                        getattr(stats, "norm"),
                        (value, std_dev),
                        min_value,
                        max_value
                    )
                }
            }
        else:
            # 非分布模式下，使用 predict_proba_one 获取类别 1 的概率
            value = decision_tree.predict_proba_one({})[1]

            # 返回单节点概率规则
            return {
                0: {
                    "value": value
                }
            }

    # 保存转换后的树结构
    transformed_data = {}

    # 遍历 DataFrame 中的每一行，每一行表示一个节点
    for node_id, row in df.iterrows():
        # 判断当前节点是否为叶子节点
        is_leaf = row['is_leaf']

        # 如果不是叶子节点，并且存在 feature 与 threshold，则认为是决策节点
        if not is_leaf and pd.notna(row['feature']) and pd.notna(row['threshold']):
            # 当前节点的判断特征
            feature = row['feature']

            # 当前节点的判断阈值
            threshold = row['threshold']

            # 查找当前节点的直接子节点
            # 子节点的 parent 字段等于当前 node_id
            children_nodes_df = df[df['parent'] == node_id].sort_values(
                df.index.name if df.index.name else df.index.values[0]
            )

            # 获取子节点 ID
            children_node_ids = children_nodes_df.index.tolist()

            # 保存子节点分支映射
            children_dict = {}

            # 二叉树情况下，默认两个子节点分别对应 False 和 True 分支
            if len(children_node_ids) == 2:
                # 原逻辑中较小节点 ID 对应 False 分支
                children_dict[False] = int(children_node_ids[0])

                # 较大节点 ID 对应 True 分支
                children_dict[True] = int(children_node_ids[1])

            # 如果只有一个子节点，则默认放到 True 分支
            elif len(children_node_ids) == 1:
                children_dict[True] = int(children_node_ids[0])

            # 保存当前决策节点
            transformed_data[node_id] = {
                'feature': feature,
                'threshold': threshold,
                'children': children_dict
            }

        else:
            # 当前节点是叶子节点，或者由于缺少信息无法作为决策节点处理

            if distribution:
                # 分布模式下，从叶子节点统计量中获取均值
                value = row['stats'].mean.get()

                # 获取方差
                variance = row['stats'].get()

                # 计算标准差，确保方差非负
                std_dev = np.sqrt(max(0, variance))

                # 如果标准差为 0，则采样值只有均值本身
                if std_dev == 0:
                    sampled_values = [value]
                else:
                    # 根据叶子节点统计量，从正态分布中采样
                    # 采样数量至少为 1000，或等于该叶子节点样本数
                    sampled_values = np.random.normal(
                        loc=value,
                        scale=std_dev,
                        size=max(1000, int(row['stats'].n))
                    )

                    # 对超出下界的采样值进行修正
                    sampled_values[sampled_values < min_value] = value

                    # 对超出上界的采样值进行修正
                    sampled_values[sampled_values > max_value] = value

                    # 转换为列表
                    sampled_values = sampled_values.tolist()

                # 保存叶子节点的均值、采样列表和分布信息
                transformed_data[node_id] = {
                    'value': value,
                    'sampled': sampled_values,
                    'dist': (
                        getattr(stats, "norm"),
                        (value, std_dev),
                        min_value,
                        max_value
                    )
                }

            else:
                # 非分布模式下，通常用于二分类概率建模

                # 如果类别 1 不存在，则补 0
                if 1 not in row['stats']:
                    row['stats'][1] = 0

                # 如果类别 0 不存在，则补 0
                if 0 not in row['stats']:
                    row['stats'][0] = 0

                # 计算类别 1 的经验概率
                value = row['stats'][1] / (
                    row['stats'][0] + row['stats'][1]
                )

                # 保存叶子节点预测概率
                transformed_data[node_id] = {
                    'value': value
                }

    # 按节点 ID 排序，保证输出结构稳定
    transformed_data_sorted = dict(sorted(transformed_data.items()))

    return transformed_data_sorted


class DecisionRules:
    """
    决策规则封装类。

    该类用于统一封装不同来源的决策树模型：
    1. sklearn 决策树；
    2. river 在线决策树；
    3. 字典或分布形式的规则。

    它提供统一的 apply 和 apply_distribution 接口，
    供仿真阶段根据输入特征预测确定值或随机采样值。
    """

    def __init__(self):
        # 保存规则字典
        self.rules = None

        # 保存 graphviz 可视化对象
        self.graph = None

    def from_decision_tree(self, decision_tree):
        """
        从 sklearn 决策树构建规则。

        参数：
            decision_tree:
                sklearn 训练好的决策树模型。
        """

        # 保存原始决策树模型
        self.decision_tree = decision_tree

        # 构建决策树可视化图
        self.graph = build_graph_vis(decision_tree, True)

        # 从 DOT 字符串中解析节点和边
        nodes, edges = parse_tree(self.graph.source)

        # 根据节点和边构造可遍历的规则结构
        self.rules = build_tree_structure(nodes, edges)

    def from_river_decision_tree(
            self,
            decision_tree,
            distribution=False,
            min_value=0,
            max_value=60 * 24
        ):
        """
        从 river 在线决策树构建规则。

        参数：
            decision_tree:
                river 决策树模型。
            distribution:
                是否按照分布模式转换叶子节点。
            min_value:
                采样值下界。
            max_value:
                采样值上界。
        """

        # 保存原始 river 决策树模型
        self.decision_tree = decision_tree

        # 将 river 决策树转换为统一规则结构
        self.rules = transform_river_decision_tree_data(
            decision_tree,
            distribution,
            min_value,
            max_value
        )

    def from_dict(self, dict_value):
        """
        从字典或分布元组中恢复规则。

        参数：
            dict_value:
                如果是 float，则直接作为固定规则；
                否则尝试按照分布参数进行采样恢复。
        """

        # 如果传入的是浮点数，则直接作为规则值
        if type(dict_value) == float:
            self.rules = dict_value

        # 否则按分布参数进行采样，构造采样列表
        else:
            self.rules = sampling_from_dist(
                dict_value[0],
                dict_value[1],
                dict_value[2],
                dict_value[3],
                dict_value[1]
            )

    def apply(self, features):
        """
        根据输入特征应用规则，返回确定性预测值。

        参数：
            features:
                输入特征字典。

        返回：
            决策树叶子节点的 value。
        """

        return traverse_tree(self.rules, features)

    def apply_distribution(self, features):
        """
        根据输入特征应用规则，并从叶子节点分布中随机采样。

        参数：
            features:
                输入特征字典。

        返回：
            叶子节点 sampled 列表中的随机采样值。
        """

        return traverse_tree_distribution(self.rules, features)

    def write_dot(self, file_name='decision_tree.dot'):
        """
        将决策树图导出为 DOT/Graphviz 文件。

        参数：
            file_name:
                输出文件名，默认 decision_tree.dot。
        """

        # 只有存在 graph 对象时才进行渲染
        if self.graph:
            self.graph.render(file_name)