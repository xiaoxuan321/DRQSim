# 导入 numpy，用于数组计算、均值计算、生成数组等
import numpy as np

# 导入 scipy.stats，用于概率分布拟合、采样和 Wasserstein 距离计算
import scipy.stats as stats

# 导入 matplotlib，用于绘制数据分布和拟合分布曲线
import matplotlib.pyplot as plt


def fit_distribution(data: any, dist_name: str) -> tuple:
    """
    拟合指定类型的概率分布，并计算拟合效果。

    参数：
        data:
            待拟合的数据。
        dist_name:
            分布名称，例如 fixed、norm、expon、uniform 等。

    返回：
        params:
            拟合得到的分布参数。
        wass_distance:
            原始数据与拟合分布采样数据之间的 Wasserstein 距离。
            该距离越小，说明拟合效果越好。
        dist:
            拟合得到的分布对象。
            如果是 fixed，则返回字符串 'fixed'。
    """

    # 如果指定的是 fixed 分布，即固定值分布
    if dist_name == 'fixed':
        # 使用数据均值作为固定分布的参数
        params = np.mean(data)

        # 生成与原始数据长度相同的固定值序列
        generated_values = np.array([params] * len(data))

        # 计算原始数据与固定值序列之间的 Wasserstein 距离
        wass_distance = stats.wasserstein_distance(data, generated_values)

        # 返回固定分布参数、拟合误差和分布名称
        return (params,), wass_distance, 'fixed'

    # 根据分布名称，从 scipy.stats 中获取对应的分布对象
    dist = getattr(stats, dist_name)

    # 使用 scipy 的 fit 方法拟合当前分布，得到分布参数
    params = dist.fit(data)

    # 根据拟合得到的分布参数进行采样
    # 采样数量与原始数据长度一致
    generated_values = sampling_from_dist(
        dist,
        params,
        min(data),
        max(data),
        np.mean(data),
        n_sample=len(data)
    )

    # 计算原始数据与采样数据之间的 Wasserstein 距离
    wass_distance = stats.wasserstein_distance(data, generated_values)

    # 返回分布参数、拟合误差和分布对象
    return params, wass_distance, dist


def return_best_distribution(
        data: any,
        dist_search: list = ['fixed', 'norm', 'expon', 'lognorm', 'uniform']
    ) -> tuple:
    """
    在候选分布集合中选择最优拟合分布。

    该函数会遍历 dist_search 中的候选分布，
    分别拟合数据并计算 Wasserstein 距离，
    最后选择距离最小的分布作为最优分布。

    参数：
        data:
            待拟合的数据。
        dist_search:
            候选分布名称列表。

    返回：
        best_fit_dist:
            最优拟合分布对象，或字符串 'fixed'。
        best_fit_dist_params:
            最优拟合分布对应的参数。
    """

    # 保存每个候选分布对应的分布对象
    dict_fitting_distributions = dict()

    # 保存每个候选分布对应的拟合参数
    dict_fitting_dist_params = dict()

    # 保存每个候选分布对应的 Wasserstein 距离
    dict_wass = dict()

    # 如果所有数据取值都相同，则直接返回 fixed 分布
    if len(set(data)) == 1:
        # 如果 data 是 list，则取第一个元素作为固定值
        if type(data) == list:
            return 'fixed', (data[0],)

        # 如果 data 是 pandas Series，则使用 iloc[0] 取第一个元素
        else:
            return 'fixed', (data.iloc[0],)

    # 如果数据为空，则返回固定值为 0 的 fixed 分布
    if len(data) == 0:
        return 'fixed', (0,)

    # 遍历所有候选分布
    for dist_name in dist_search:
        # 拟合当前分布，并计算拟合效果
        params, goodness_of_fit, dist = fit_distribution(
            data,
            dist_name
        )

        # 保存当前分布对象
        dict_fitting_distributions[dist_name] = dist

        # 保存当前分布的 Wasserstein 距离
        dict_wass[dist_name] = goodness_of_fit

        # 保存当前分布的拟合参数
        dict_fitting_dist_params[dist_name] = params

    # 选择 Wasserstein 距离最小的分布名称
    best_fit_dist_name = min(dict_wass, key=dict_wass.get)

    # 获取最优分布对象
    best_fit_dist = dict_fitting_distributions[best_fit_dist_name]

    # 获取最优分布参数
    best_fit_dist_params = dict_fitting_dist_params[best_fit_dist_name]

    # 返回最优分布及其参数
    return best_fit_dist, best_fit_dist_params


def sampling_from_dist(
        dist: any,
        params: tuple,
        min_value: float,
        max_value: float,
        mean_value: float,
        n_sample: int = 1000
    ) -> np.array:
    """
    根据给定分布和参数进行采样。

    采样后会进行边界修正：
    如果采样值小于历史最小值，则替换为均值；
    如果采样值大于历史最大值，则替换为均值。

    参数：
        dist:
            概率分布对象，或字符串 'fixed'。
        params:
            分布参数。
        min_value:
            允许采样值的最小边界。
        max_value:
            允许采样值的最大边界。
        mean_value:
            超出边界时使用的替代均值。
        n_sample:
            采样数量，默认 1000。

    返回：
        l:
            采样得到的 numpy 数组。
    """

    # 如果是 fixed 分布，则直接返回 n_sample 个均值
    if dist == 'fixed':
        return np.array([mean_value] * n_sample)

    # 根据 scipy 分布对象和参数进行随机采样，根据设定的概率分布，生成一批随机数
    l = dist.rvs(*params, n_sample)

    # 将小于最小值的采样结果替换为均值
    l[l < min_value] = mean_value

    # 将大于最大值的采样结果替换为均值
    l[l > max_value] = mean_value

    # 返回修正后的采样结果
    return l


def plot_distribution(data: any, params: tuple, dist: any):
    """
    绘制原始数据直方图和拟合分布曲线。

    参数：
        data:
            原始数据。
        params:
            拟合分布参数。
        dist:
            scipy 分布对象。
    """

    # 绘制原始数据的概率密度直方图
    plt.hist(
        data,
        bins=30,
        density=True,
        alpha=0.6,
        color='skyblue',
        label="Data Histogram"
    )

    # 在数据最小值和最大值之间生成连续横坐标
    x = np.linspace(min(data), max(data), 1000)

    # 根据拟合分布计算每个 x 对应的概率密度值
    y = dist.pdf(x, *params)

    # 绘制拟合分布的概率密度曲线
    plt.plot(
        x,
        y,
        'r-',
        lw=2,
        label=f'Fitted {dist.name} Distribution'
    )

    # 设置 x 轴标签
    plt.xlabel("Value")

    # 设置 y 轴标签
    plt.ylabel("Density")

    # 显示图例
    plt.legend()

    # 设置图标题
    plt.title(f"Fitted {dist.name.capitalize()} Distribution")

    # 显示图像
    plt.show()


def remove_outliers(data: list, m: float = 20.0) -> list:
    """
    基于中位数绝对偏差移除异常值。

    该方法通过计算每个数据点与中位数的距离，
    并使用中位数绝对偏差进行归一化，
    最后保留归一化距离小于阈值 m 的数据。

    参数：
        data:
            原始数据列表。
        m:
            异常值过滤阈值，默认值为 20.0。
            数值越小，异常值过滤越严格。

    返回：
        list:
            移除异常值后的数据列表。
    """

    # 将输入数据转换为 numpy 数组，便于向量化计算
    data = np.asarray(data)

    # 计算每个数据点与数据中位数之间的绝对距离
    d = np.abs(data - np.median(data))

    # 计算上述距离的中位数，即 Median Absolute Deviation，MAD
    mdev = np.median(d)

    # 对距离进行归一化
    # 如果 mdev 为 0，则用 1.0 避免除零错误
    s = d / (mdev if mdev else 1.0)

    # 仅保留归一化距离小于阈值 m 的数据，并转换为列表返回
    return data[s < m].tolist()