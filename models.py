# 从 app 模块中导入数据库对象 db
# db 通常是 Flask-SQLAlchemy 初始化得到的数据库实例
from app import db

# 导入 datetime，用于记录会话创建时间
from datetime import datetime

# 导入 json，用于在 Python 字典和 JSON 字符串之间进行转换
import json


class SimulationSession(db.Model):
    """
    仿真会话数据库模型。

    该类用于在数据库中保存一次仿真会话的相关信息，包括：
    1. 上传的日志文件名；
    2. 流程发现时使用的噪声阈值；
    3. 发现得到的仿真参数；
    4. 会话创建时间；
    5. 当前处理状态；
    6. 仿真结果文件名。
    """

    # 会话 ID，作为数据库表的主键
    id = db.Column(db.Integer, primary_key=True)

    # 上传的事件日志文件名
    # nullable=False 表示该字段不能为空
    filename = db.Column(db.String(255), nullable=False)

    # 流程模型发现时使用的噪声阈值
    # 默认值为 0.2
    noise_threshold = db.Column(db.Float, default=0.2)

    # 保存发现得到的仿真参数
    # 这里使用 Text 类型存储 JSON 字符串
    parameters = db.Column(db.Text)

    # 当前仿真会话的创建时间
    # 默认使用 UTC 时间
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # 当前会话状态
    # 可能的状态包括：
    # uploaded：文件已上传
    # discovered：流程模型或参数已发现
    # ready：已准备好仿真
    # simulating：正在仿真
    # completed：仿真完成
    status = db.Column(db.String(50), default='uploaded')

    # 保存仿真生成的事件日志 DataFrame 文件名
    # 默认值为空字符串
    simulation_df_filename = db.Column(db.String(255), default='')

    def __init__(self, filename, noise_threshold=0.2, **kwargs):
        """
        初始化仿真会话对象。

        参数：
            filename:
                上传的事件日志文件名。

            noise_threshold:
                流程模型发现时使用的噪声阈值，默认值为 0.2。

            **kwargs:
                传递给父类 db.Model 的其他初始化参数。
        """

        # 调用父类初始化方法，保证 SQLAlchemy 模型能够正确初始化
        super().__init__(**kwargs)

        # 设置上传文件名
        self.filename = filename

        # 设置噪声阈值
        self.noise_threshold = noise_threshold

        # 新建会话的初始状态设置为 uploaded
        self.status = 'uploaded'

    def get_parameters(self):
        """
        获取仿真参数，并将 JSON 字符串转换为 Python 字典。

        返回：
            dict:
                如果 parameters 字段不为空，则返回解析后的参数字典。

            None:
                如果 parameters 字段为空，则返回 None。
        """

        # 如果数据库中已经保存了参数 JSON 字符串
        if self.parameters:
            # 将 JSON 字符串反序列化为 Python 字典
            return json.loads(self.parameters)

        # 如果没有保存参数，则返回 None
        return None

    def set_parameters(self, params_dict):
        """
        设置仿真参数，并将 Python 字典转换为 JSON 字符串保存。

        参数：
            params_dict:
                需要保存的仿真参数字典。
        """

        # 将 Python 字典序列化为格式化后的 JSON 字符串
        # indent=2 表示使用 2 个空格缩进，便于阅读和调试
        self.parameters = json.dumps(params_dict, indent=2)