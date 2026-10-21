# DRQSim

**Dynamic Resource Modeling and Queue Scheduling for Simulation Model Discovery**  
**面向仿真模型发现的动态资源建模与队列调度工具**

DRQSim 是一个**数据驱动、可解释的业务流程仿真模型发现与执行框架**，能够从事件日志中自动发现仿真模型。它将基于 Petri 网的控制流发现、上下文感知的案例到达建模、资源特定的服务时间建模、动态资源状态建模以及显式的队列感知调度集成在统一的离散事件仿真引擎中。

与将所有等待时间视为独立采样的活动延迟的做法不同，DRQSim 显式表示**就绪任务、竞争资源、资源可用性及任务调度决策**，从而有助于区分资源竞争和调度产生的等待，以及工作日历或其他因素引起的延迟。

## 核心亮点

- **可解释的仿真模型发现：** 从 XES 事件日志中学习可解释的控制流决策、资源分配规则、案例到达行为和任务优先级规则。
- **基于 Petri 网的控制流建模：** 使用 PM4Py 发现流程模型，或导入 PNML 模型；学习流程结构中决策点的分支选择规律。
- **上下文感知的案例到达：** 基于时间上下文建模案例到达间隔，并通过到达日历与可配置的工作负载变化支持不同场景。
- **资源行为建模：** 发现活动—资源关系、工作日历、多任务处理/容量信息、基础服务时间分布，以及资源状态动态贝叶斯网络（DBN）。
- **两阶段队列调度：** 将“**任务分配给哪个资源**”与“**该资源接下来执行哪个符合条件的等待任务**”区分建模。
- **显式离散事件仿真：** 维护就绪队列、资源调度状态、任务完成事件及等待时间诊断信息。
- **剩余延迟建模：** 将难以由资源竞争和工作日历解释的外部延迟作为独立的辅助机制估计。
- **分阶段校准与评估：** 使用按时间顺序划分的训练集和验证集、基于 Optuna 的模型选择，以及重复仿真比较。
- **场景分析：** 调整发现的参数、到达行为和流程模型，生成仿真日志以便进一步分析。

## DRQSim 的工作原理

DRQSim 将仿真模型发现划分为**五个主要模块**，并使用外部延迟建模作为辅助组件。

| 组成模块 | 发现或建模的内容 | 主要实现文件 |
| --- | --- | --- |
| **1. 数据预处理** | 事件日志转换、对齐与特征提取、任务使能时间及队列/资源状态重建 | `discovery/resource_state_features.py`、`utils/log_partitioner.py` 及发现流程中的预处理工具 |
| **2. 控制流建模** | Petri 网与可解释的决策点分支模型 | `discovery/decision_point_discovery.py`、`discovery/observable_petri.py`、`discovery/tau_abstraction.py` |
| **3. 案例到达建模** | 随上下文变化的到达间隔行为与到达日历 | `discovery/time_discovery.py`、`discovery/calendar_discovery.py` |
| **4. 动态资源建模** | 资源日历、活动—资源关系、基础服务时间以及资源状态 DBN 参数 | `discovery/resource_discovery.py`、`discovery/base_service_time_discovery.py`、`discovery/resource_state_discovery.py` |
| **5. 队列调度建模** | 资源分配模型，以及用于就绪任务调度的条件优先级树（Conditional Prioritization Tree, CPT） | `discovery/queue_scheduling_discovery.py`、`discovery/conditional_prioritization_discovery.py` |
| **辅助：外部延迟建模** | 无法完全由资源竞争和日历约束解释的、依赖上下文的剩余等待行为 | `discovery/external_delay_discovery.py`、`runtime/external_delay_runtime.py` |

### 1. 数据预处理与控制流建模

DRQSim 导入 XES 日志，并使用案例 ID、活动名称、资源、活动开始和结束时间戳等事件属性。它既可以使用 Inductive Miner 发现流程树并将其转换为 Petri 网，也可以采用导入的 PNML Petri 网。系统结合 Petri 网结构和事件日志观测结果学习决策点模型，并在需要时使用基于规则的回退策略。

### 2. 案例到达建模

案例到达模块学习连续案例之间的到达时间间隔。已实现的决策树模型利用**星期、小时、月份和年内周次**等时间上下文，选择相应的到达间隔行为。到达日历和可配置的到达量缩放机制支持不同需求负载下的仿真场景。

### 3. 动态资源建模

资源模块主要估计以下内容：

- 各活动的候选资源及其历史分配频率。
- 资源可用性日历与并发/容量特征。
- 各活动—资源组合的基础服务时间分布。
- 使用**二时间片动态贝叶斯网络（Two-Time-Slice Dynamic Bayesian Network, DBN）**表示的离散资源状态转移。

DBN 建模的状态变量包括**疲劳（Fatigue）、工作负载（Workload）、技能（Skill）、效率（Efficiency）和可用性（Availability）**。系统根据重建的资源历史学习状态转移参数，并将其保存为条件概率表（Conditional Probability Tables, CPTs）。在完整设计中，资源状态可以对从基础分布采样的服务时间产生影响；但在本次提供的消融版源码中，该运行时作用已被禁用。

### 4. 资源分配与条件优先级调度

DRQSim 将调度过程拆分为两个问题：

1. **资源分配（Resource Allocation）——由谁执行任务？** 资源感知模型结合学习得到的决策树得分、历史活动—资源频率、资源可用性、日历兼容性和队列相关上下文，进行资源选择。
2. **条件优先级（Conditional Prioritization）——先执行哪个等待任务？** 成对比较的**条件优先级树（CPT）**学习同时竞争的任务之间的优先关系。运行时将成对比较得分汇总为队列排序结果。

当事件日志缺乏足够可靠的任务竞争样本，或者没有明显的任务超越（overtaking）行为时，优先级模型会明确回退为**先来先服务（First-Come, First-Served, FCFS）**策略。

### 5. 离散事件仿真与等待时间分解

仿真引擎以事件驱动方式维护案例进展、资源调度状态和各资源的就绪队列。当任务满足执行条件时，资源分配与任务调度共同决定实际开始时间；任务完成事件释放资源容量，并推动流程继续执行。

生成的仿真事件中包含可解释的时间字段，例如：

- `enabled:timestamp`、`queue_start:timestamp`、`start:timestamp`、`time:timestamp`
- `queue_wait_min`、`calendar_wait_min`、`external_delay_min`、`total_wait_min`
- `resource_contention_wait_min`、`priority_wait_min`、`wait_decomposition_residual_min`
- `base_service_time_min`、`adjusted_service_time_min`、`service_time_min`
- `dispatch_policy`、`conditional_priority_score`

通过这些诊断信息，可以检查不同等待机制对事件时间的贡献。其中，剩余项（residual）用于表示等待时间分解中仍然存在的差异，**不应将其误认为另一个独立学习的延迟模型**。

## 输入数据

### 事件日志属性

DRQSim 在执行资源和时间感知的模型发现流程时，期望输入包含以下属性的 **XES 事件日志**：

| 属性 | 含义 |
| --- | --- |
| `case:concept:name` | 唯一案例标识符 |
| `concept:name` | 活动名称 |
| `start:timestamp` | 活动执行开始时间 |
| `time:timestamp` | 活动完成时间 |
| `org:resource` | 活动执行资源 |

一些中间时间戳（例如任务使能时间与进入队列时间）以及队列状态特征，会在模型发现流程中重建或推导。输入日志中可选的案例属性，在相应模块支持时也可以参与决策规则学习。

**可选流程模型：** 如果希望使用给定的 Petri 网结构而非自动发现的流程模型，可提供 `.pnml` 文件。

## 快速开始

> **仓库完整性要求：** 上传的 `code.zip` 包含核心源码模块和部分 Flask 集成文件，**但不包含完整、可直接运行的 Web 应用**。其中缺少应用入口（`main.py`）、Flask 初始化代码（`app.py`）、数据库模型（`models.py`）、`evaluation` 模块、依赖清单、UI 模板以及 Docker/安装脚本等文件。以下安装说明适用于原 README 所描述的、包含这些文件的**完整项目仓库**；不能仅凭当前局部源码包验证这些安装命令。

### 方式 A：Docker 部署（完整 Web 应用）

首先安装 [Docker](https://docs.docker.com/get-docker/) 和 Docker Compose，然后在完整项目的根目录运行：

```bash
docker compose up -d --build
```

原始 Web 应用文档将 **http://localhost:5050** 作为默认访问地址。如果实际 `docker-compose.yml` 中的端口映射不同，请以该配置为准。

原项目文档还列出了以下平台专用安装/启动脚本（如果完整仓库中包含这些脚本）：

| 平台 | 安装脚本 | 启动脚本 |
| --- | --- | --- |
| Windows | `installer\windows\setup-windows.bat` | `installer\windows\run-docker.bat` |
| Ubuntu/Linux | `installer/ubuntu/setup-ubuntu.sh` | `installer/ubuntu/run-docker.sh` |
| macOS | `installer/macos/setup-macos.sh` | `installer/macos/run-docker.sh` |

**仅在完整代码仓库确实包含这些脚本时运行。** Linux/macOS 脚本可能需要先执行 `chmod +x` 赋予可执行权限。

### 方式 B：本地 Python 环境（完整 Web 应用）

原始应用的安装说明要求 **Python 3.10+** 以及 Conda/Miniconda。在完整仓库中执行：

```bash
# 请在包含 environment.yml 和 main.py 的项目根目录运行。
conda env create -f environment.yml

# 激活 environment.yml 中定义的环境。
conda activate DRQSim

pip install -e .
python main.py
```

`conda activate` 后的环境名称必须与实际环境配置文件一致。当前提取出的源码包不包含 `environment.yml` 或 `main.py`，因此**不能直接使用这些命令启动**。

## 使用流程

完整 Web 应用的使用流程如下：

1. **上传数据：** 提供训练用 XES 日志，并可选择提供 PNML 模型。对于当前发现接口，需将对应的 `_test.xes` 文件放在 `example_data/` 中。
2. **模型发现与调优：** 发现 Petri 网（如果没有导入现成模型），拟合五个模型模块及外部延迟模型，并使用按时间顺序划分的验证集选择参数。
3. **查看参数：** 检查发现的控制流规则、资源与日历设置、案例到达行为及调度模型。可通过 Web 界面或 API 修改受支持的参数。
4. **运行仿真：** 指定案例数量，并按需设置仿真起始时间与需求控制参数，由事件驱动引擎生成合成事件日志。
5. **评估与导出：** 比较真实行为和仿真行为，检查时间诊断与可视化结果，下载参数 JSON、PNML 以及仿真 CSV 日志。

### API 接口参考

| 方法 | 接口 | 功能 |
| --- | --- | --- |
| `POST` | `/api/upload` | 上传 XES 日志和可选 PNML 模型 |
| `POST` | `/api/discover/<session_id>` | 发现、调优并评估仿真参数 |
| `GET` | `/api/discover_progress/<session_id>` | 查询模型发现进度 |
| `GET`、`PUT` | `/api/parameters/<session_id>` | 读取或更新仿真参数 |
| `GET` | `/api/parameters/<session_id>/control_flow_decision_points` | 查看可编辑的控制流决策点 |
| `POST`、`PUT` | `/api/parameters/<session_id>/control_flow_edits` | 应用受支持的控制流规则编辑 |
| `POST` | `/api/simulate/<session_id>` | 生成仿真事件日志 |
| `GET` | `/api/export_parameters/<session_id>` | 导出发现或编辑后的参数 JSON |
| `GET` | `/api/petri_net/<session_id>/download` | 下载当前 Petri 网的 PNML 文件 |
| `POST` | `/api/petri_net/<session_id>/replace` | 上传修改后的 PNML 模型；之后需要重新执行模型发现 |
| `GET` | `/api/download/<filename>` | 下载仿真 CSV 日志 |
| `GET` | `/api/get_visualization/<session_id>/<visualization_type>` | 请求支持的结果可视化 |
| `GET` | `/api/get_activities_and_resources/<session_id>` | 列出仿真结果中的活动与资源 |

## 模型选择与评估

DRQSim 的 `tuning/stagewise_tuner.py` 实现了多阶段调优流程，旨在将模型发现过程与验证过程分开：

- **流程结构选择：** 依据控制流相似度指标，调整 Inductive Miner 的噪声阈值和控制流决策树复杂度。
- **运行时模型选择：** 依据到达、时间和资源相关指标，调优到达行为、资源分配、条件优先级和外部延迟机制。
- **重新校准与最终拟合：** 在验证数据上评估可选的近期数据感知时间/到达模型及选定的校准参数，然后使用指定的训练集与验证集联合数据重新发现最终模型参数。
- **独立测试集评估：** 使用提供的测试日志评估最终模型，并在配置允许时执行重复仿真。

实现中使用 **Optuna** 进行流程结构搜索，并包含树复杂度搜索、按时间顺序划分数据集，以及针对大规模日志的搜索和确认策略。

调优代码引用的指标键包括用于控制流比较的 `cfld`、`ngd`、`3gd`，以及用于运行时评估的 `aed`、`ced`、`red`、`car`、`ctd`、`cwd`。这些指标的实际可用性取决于完整应用中的外部 `evaluation` 模块。

## 项目结构

以下目录结构反映了**本次提供的源码压缩包**，其顶层文件夹为 `code/`：

```text
code/
├── config/
│   └── modular_schema.py              # 五模块参数结构定义
├── discovery/
│   ├── decision_point_discovery.py     # Petri 网分支决策规则
│   ├── time_discovery.py               # 案例到达及时间模型
│   ├── calendar_discovery.py           # 到达日历与资源日历
│   ├── resource_discovery.py           # 资源特征发现
│   ├── base_service_time_discovery.py  # 活动—资源服务时间
│   ├── resource_state_features.py      # DBN 训练特征
│   ├── resource_state_discovery.py     # DBN 发现和条件概率表
│   ├── queue_scheduling_discovery.py   # 资源分配模型
│   ├── conditional_prioritization_discovery.py # 成对任务优先级
│   └── external_delay_discovery.py     # 外部延迟建模
├── runtime/
│   ├── dynamic_resource_state.py       # DBN 运行时（当前关闭）
│   ├── queue_state_runtime.py          # 队列及在制任务状态
│   ├── queue_scheduling_runtime.py     # 资源分配
│   ├── conditional_prioritization_runtime.py # 就绪任务调度
│   └── external_delay_runtime.py       # 外部延迟采样
├── tuning/
│   └── stagewise_tuner.py              # 分阶段模型选择
├── diagnostics/
│   └── time_decomposition.py           # 时间分解诊断
├── utils/                              # 分布拟合、日志划分、输入输出等工具
├── simulator.py                        # 模型参数及仿真引擎
├── prosit_integration.py              # 集成及评估封装层
└── routes.py                           # Flask API 接口
```

当前代码依然使用 **`prosit` 作为内部 Python 包名/导入命名空间**，并使用 `ProSiTIntegration` 作为集成类名称。这些属于保留的旧实现标识；工具的正式名称为 **DRQSim**。如果要在整个代码库中统一重命名，必须同步修改导入语句、打包配置和集成代码，而不只是更改 README。

## 常见问题排查

| 问题 | 建议检查 |
| --- | --- |
| 无法直接从 `code.zip` 启动应用 | 压缩包仅包含部分源码，并非可独立部署的完整仓库；需要补齐应用与依赖文件。 |
| 模型发现拒绝上传的文件名 | 当前 Web 流程要求文件名包含 `_train`，并在 `example_data/` 中存在匹配的 `_test.xes`。 |
| XES 解析或资源发现失败 | 检查必需的事件属性、时间戳格式、案例 ID 和资源值。 |
| 导入 PNML 失败 | 检查 PNML Petri 网的有效性；路由还会尝试进行合理性（soundness）检查并提供反馈。 |
| 仿真生成的等待时间不符合预期 | 分别检查队列等待、日历等待、外部延迟、资源竞争等待及优先级等待诊断信息。 |
| 看不到动态资源状态产生的影响 | 检查当前运行时代码中明确设置的 `self.enabled = False` 消融开关。 |
| 无法通过 5050 端口访问 Docker 应用 | 检查 Docker 状态、容器日志以及实际的 Compose 端口映射。 |

对于完整的 Docker 部署，可使用 `docker compose ps` 和 `docker compose logs -f` 排查问题。

## 研究与开发说明

DRQSim 强调通过**可解释的运行机制**建模流程，而不是将资源分配、任务等待和执行时间视为互不相关的随机采样结果。这一设计特别适合研究资源竞争、调度策略、随时间变化的工作负载以及 What-if 场景变化对流程整体性能的影响。

进行严格的实验比较时，建议记录训练/验证/测试集划分、参数搜索配置、仿真随机种子与重复次数，以及资源状态作用是否启用。本次文档修订并未对所提供的局部源码包进行端到端运行验证。

## 许可证

原项目 README 声明使用 **MIT 许可证**。具体许可条款请以完整仓库中的 `LICENSE` 文件为准；当前提供的源码压缩包不包含该文件。
