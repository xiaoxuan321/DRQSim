"""
Flask 路由模块：用于完成 drq 仿真系统的文件上传、参数发现、参数修改、仿真执行、结果下载和可视化展示。

主要功能包括：
1. 上传 XES 事件日志，以及可选的 PNML Petri 网模型；
2. 调用 drq 从 XES 日志中发现流程模型和仿真参数；
3. 将发现得到的参数保存为 JSON，并支持前端修改后写回 JSON；
4. 基于保存或修改后的参数运行仿真；
5. 下载仿真生成的 CSV 日志；
6. 生成流程图、案例周期时间分布、资源热力图、活动执行时间箱线图、等待时间分布图等可视化结果。

注意：
当前代码中，/api/discover/<session_id> 会直接读取上传的完整 XES 文件，
并将其赋值给 drq.event_log。如果没有额外加入日志划分逻辑，
后续 discover_process_model() 和 discover_parameters() 默认使用完整日志。
"""

# os 用于处理路径、判断文件是否存在、创建目录等操作
import os

# json 用于读取和保存 drq 参数 JSON 文件
import json

# copy 用于对参数 JSON 进行安全复制，避免直接修改原对象
import copy

# logging 用于记录系统运行日志，便于调试和定位错误
import logging

# io 用于在内存中创建字符串缓冲区，这里主要用于保存 matplotlib 生成的 SVG 图像
import io

# tempfile 用于创建临时文件，例如导出参数 JSON 时先写入临时文件再返回给前端下载
import tempfile

# datetime 用于生成时间戳文件名，以及处理仿真开始时间
from datetime import datetime, timezone

# Flask 常用函数：
# render_template：渲染 HTML 页面
# request：读取前端请求参数和上传文件
# jsonify：返回 JSON 响应
# send_file：返回文件下载响应
from flask import render_template, request, jsonify, send_file

# secure_filename 用于清洗上传文件名，避免路径穿越等安全问题
from werkzeug.utils import secure_filename

# 导入 Flask 应用对象和数据库对象
from app import app, db

# 导入数据库模型 SimulationSession，用于保存每次上传、发现和仿真的会话信息
from models import SimulationSession

# 导入 drq 集成类，封装流程发现、参数发现、仿真运行和参数读写等功能
from drq_integration import drqIntegration

# PM4Py 的 XES 导入器，用于读取上传的 .xes 事件日志文件
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net.utils import check_soundness

# numpy 用于数值计算，这里主要在可视化中生成完整的星期/小时索引
import numpy as np

# pandas 用于读取仿真 CSV 日志、处理时间字段、分组统计等
import pandas as pd

# matplotlib 用于绘图
import matplotlib

# 设置 matplotlib 使用非交互式后端 Agg。
# Web 服务器通常没有图形界面，因此不能使用默认交互式后端。
matplotlib.use('Agg')

# pyplot 是 matplotlib 的常用绘图接口
import matplotlib.pyplot as plt

# seaborn 用于绘制更美观的统计图，如直方图、热力图和箱线图
import seaborn as sns

# pm4py 用于流程挖掘、事件日志转换和 DFG 发现
import pm4py

# PM4Py 的 performance DFG 可视化工具，用于生成带性能信息的流程图
from pm4py.visualization.dfg.variants import performance as dfg_perf_visualizer

# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)

# 创建全局 drq 集成对象。
# 注意：这是一个全局对象，如果系统存在多用户并发请求，可能会出现不同 session 之间状态互相覆盖的问题。
# 更稳妥的生产级设计是：按 session 创建独立的 drqIntegration 对象，或者把状态持久化并在每个请求中恢复。
drq = drqIntegration()
DISCOVERY_PROGRESS = {}


TIMESTAMP_COLUMNS = (
    "time:timestamp",
    "start:timestamp",
    "end:timestamp",
    "enabled:timestamp",
    "case:start:timestamp",
    "case:end:timestamp",
)


def parse_timestamp_series_mixed(series, column_name="timestamp"):
    """
    Robustly parse timestamp columns that may mix formats such as:
    - 2026-06-29 21:00:00.000000+00:00
    - 2026-06-29 21:00:00+00:00
    - 2026-06-29T21:00:00+00:00

    pandas may infer one fixed format from the first non-null row and then fail
    on later rows with a slightly different representation. format="mixed"
    fixes this in pandas >= 2.0. The element-wise fallback keeps compatibility
    with older pandas versions.
    """
    if series is None:
        return series

    s = series.copy()

    # Normalize common empty placeholders before parsing.
    s = s.replace(["", "None", "none", "NULL", "null", "NaN", "nan", "NaT", "nat"], pd.NA)

    try:
        parsed = pd.to_datetime(
            s,
            utc=True,
            format="mixed",
            errors="coerce",
        )
    except Exception:
        # Compatible fallback for pandas versions without format="mixed".
        def _safe_parse_scalar(x):
            if pd.isna(x):
                return pd.NaT
            try:
                return pd.Timestamp(x)
            except Exception:
                return pd.NaT

        parsed = s.apply(_safe_parse_scalar)
        parsed = pd.to_datetime(parsed, utc=True, errors="coerce")

    original_not_empty = s.notna()
    bad_count = int((parsed.isna() & original_not_empty).sum())
    if bad_count > 0:
        examples = s[parsed.isna() & original_not_empty].astype(str).head(5).tolist()
        logger.warning(
            "Timestamp parse warning: column=%s, bad_count=%s, examples=%s",
            column_name,
            bad_count,
            examples,
        )

    return parsed


def normalize_timestamp_columns(df, required_columns=None, drop_bad_required=True):
    """
    Normalize all timestamp-like columns in a DataFrame to timezone-aware UTC.

    required_columns:
        Columns that must be valid for visualization. Rows with NaT in these
        columns can be dropped to avoid PM4Py / plotting errors.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    required_columns = list(required_columns or [])

    for col in df.columns:
        col_lower = str(col).lower()
        is_timestamp_column = (
            col in TIMESTAMP_COLUMNS
            or col_lower.endswith(":timestamp")
            or "timestamp" in col_lower
        )

        if is_timestamp_column:
            df[col] = parse_timestamp_series_mixed(df[col], column_name=col)

    existing_required = [col for col in required_columns if col in df.columns]
    if drop_bad_required and existing_required:
        before = len(df)
        df = df.dropna(subset=existing_required)
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(
                "Dropped %s rows with invalid required timestamps: %s",
                dropped,
                existing_required,
            )

    return df


def format_timestamp_columns_for_csv(df):
    """
    Save timestamp columns with one stable textual format.
    This prevents pandas from writing mixed strings with and without microseconds.
    """
    if df is None or df.empty:
        return df

    df = normalize_timestamp_columns(df, drop_bad_required=False)

    for col in df.columns:
        col_lower = str(col).lower()
        is_timestamp_column = (
            col in TIMESTAMP_COLUMNS
            or col_lower.endswith(":timestamp")
            or "timestamp" in col_lower
        )

        if is_timestamp_column:
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S.%f%z")

    return df


def to_utc_aware_timestamp(value):
    """
    Convert a scalar timestamp-like value to timezone-aware UTC datetime.
    """
    if value is None:
        return None

    parsed = parse_timestamp_series_mixed(
        pd.Series([value]),
        column_name="simulation_start_timestamp",
    )

    if parsed.empty or pd.isna(parsed.iloc[0]):
        return None

    return parsed.iloc[0].to_pydatetime()


def find_start_timestamp_from_params(params):
    """
    Try to find original simulation start timestamp from saved drq parameters.
    Adjust keys if your JSON uses different field names.
    """
    if not isinstance(params, dict):
        return None

    candidate_keys = [
        "start_timestamp",
        "start_ts_simulation",
        "simulation_start_timestamp",
        "log_start_timestamp",
        "first_event_timestamp",
    ]

    for key in candidate_keys:
        value = params.get(key)
        if value is not None:
            return value

    arrival_params = params.get("arrival_params", {})
    if isinstance(arrival_params, dict):
        for key in candidate_keys:
            value = arrival_params.get(key)
            if value is not None:
                return value

    modular_params = params.get("modular_params", {})
    if isinstance(modular_params, dict):
        modules = modular_params.get("modules", {})
        if isinstance(modules, dict):
            arrival_module = modules.get("arrival", {})
            if isinstance(arrival_module, dict):
                for key in candidate_keys:
                    value = arrival_module.get(key)
                    if value is not None:
                        return value

    return None
def set_discovery_progress(session_id, stage, title, detail, status="running"):
    """
    保存当前发现进度，供前端轮询。
    stage 只使用两个业务阶段：
    - petri: Petri 网挖掘 / 验证 / 结构搜索
    - params: 参数挖掘 / 运行参数搜索
    """
    DISCOVERY_PROGRESS[int(session_id)] = {
        "session_id": int(session_id),
        "stage": stage,
        "title": title,
        "detail": detail,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }


@app.route("/api/discover_progress/<int:session_id>", methods=["GET"])
def get_discover_progress(session_id):
    """
    前端轮询该接口，获取当前发现阶段。
    """
    progress = DISCOVERY_PROGRESS.get(int(session_id))

    if not progress:
        return jsonify({
            "success": True,
            "stage": "petri",
            "title": "Petri net mining...",
            "detail": "Waiting for discovery to start.",
            "status": "idle",
        })

    return jsonify({
        "success": True,
        **progress
    })
def render_petri_net_svg_from_pnml(pnml_filepath):
    """
    使用 PM4Py 读取 PNML，并生成 SVG 可视化。

    返回：
    - process_model_summary
    - net
    - initial_marking
    - final_marking
    """
    net, initial_marking, final_marking = pnml_importer.apply(pnml_filepath)

    if net is None:
        raise ValueError("Failed to read PNML as Petri net.")

    if len(net.places) == 0:
        raise ValueError("Invalid PNML: no places found.")

    if len(net.transitions) == 0:
        raise ValueError("Invalid PNML: no transitions found.")

    # 可选 soundness 检查。不同 PM4Py 版本对 soundness 支持略有差异，所以这里做成 warning。
    soundness_warning = None
    try:
        is_sound = check_soundness.check_easy_soundness_net_in_fin_marking(
            net,
            initial_marking,
            final_marking
        )
        if not is_sound:
            soundness_warning = "The PNML can be imported, but PM4Py soundness check did not pass."
    except Exception as e:
        soundness_warning = f"PNML imported successfully, but soundness check failed to run: {str(e)}"

    # 生成 SVG
    gviz = pn_visualizer.apply(
        net,
        initial_marking,
        final_marking
    )

    tmp_svg_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp:
            tmp_svg_path = tmp.name

        pn_visualizer.save(gviz, tmp_svg_path)

        with open(tmp_svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()

    finally:
        if tmp_svg_path and os.path.exists(tmp_svg_path):
            try:
                os.remove(tmp_svg_path)
            except Exception:
                pass

    activities = []
    transitions = []
    map_transition_name_to_id = {}

    for t in net.transitions:
        transition_id = str(t.name)
        transition_label = t.label if t.label else transition_id

        transitions.append(transition_label)

        if t.label:
            activities.append(t.label)
            map_transition_name_to_id[t.label] = transition_id
        else:
            map_transition_name_to_id[transition_id] = transition_id

    places = [str(p.name) for p in net.places]

    process_model_summary = {
        "visualization": svg_content,
        "activities": sorted(list(set(activities))),
        "transitions": transitions,
        "map_transitionName_to_id": map_transition_name_to_id,
        "places": places,
        "fitness": 0,
        "precision": 0,
        "f_measure": 0,
        "soundness_warning": soundness_warning,
    }

    return process_model_summary, net, initial_marking, final_marking


def get_current_pnml_path_from_session(session):
    """
    获取当前 session 对应的 Petri 网 PNML 路径。
    优先级：
    1. 用户刚上传的手动修改 PNML；
    2. 当前优化后的 PNML；
    3. 初始上传的 PNML。
    """
    session_metadata = session.get_parameters() or {}

    candidates = []

    manual_modified_pnml_path = session_metadata.get("manual_modified_pnml_path")
    if manual_modified_pnml_path:
        candidates.append(manual_modified_pnml_path)

    optimized_info = session_metadata.get("optimized_petri_net", {})
    if isinstance(optimized_info, dict):
        optimized_pnml_path = optimized_info.get("current_best_pnml_path")
        if optimized_pnml_path:
            candidates.append(optimized_pnml_path)

    pnml_filename = session_metadata.get("pnml_filename")
    if pnml_filename:
        candidates.append(os.path.join(app.config["UPLOAD_FOLDER"], pnml_filename))

    for path in candidates:
        if path and os.path.exists(path):
            return path

    return None
@app.route("/api/petri_net/<int:session_id>/download", methods=["GET"])
def download_current_petri_net(session_id):
    """
    下载当前 session 的 Petri 网 PNML。
    """
    try:
        session = SimulationSession.query.get_or_404(session_id)

        pnml_path = get_current_pnml_path_from_session(session)

        if not pnml_path:
            return jsonify({
                "success": False,
                "error": "No PNML file is available for this session."
            }), 404

        download_name = f"session_{session_id}_petri_net.pnml"

        return send_file(
            pnml_path,
            as_attachment=True,
            download_name=download_name,
            mimetype="application/xml"
        )

    except Exception as e:
        logger.error(f"Download Petri net PNML error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
@app.route("/api/petri_net/<int:session_id>/replace", methods=["POST"])
def replace_current_petri_net(session_id):
    """
    上传用户外部工具修改后的 PNML，并将其设置为当前 session 的 Petri 网输入。

    注意：
    这个接口只负责：
    1. 保存修改后的 PNML；
    2. 用 PM4Py 读取和校验；
    3. 生成临时 SVG 预览；
    4. 更新 session 中的 pnml_filename。

    真正的参数同步和重新发现由前端随后调用 /api/discover/<session_id> 完成。
    """
    try:
        session = SimulationSession.query.get_or_404(session_id)

        if "pnml_file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No PNML file provided."
            }), 400

        pnml_file = request.files["pnml_file"]

        if not pnml_file or not pnml_file.filename:
            return jsonify({
                "success": False,
                "error": "No PNML file selected."
            }), 400

        if not pnml_file.filename.lower().endswith(".pnml"):
            return jsonify({
                "success": False,
                "error": "Uploaded file must have .pnml extension."
            }), 400

        timestamp = int(datetime.now().timestamp())
        safe_filename = secure_filename(pnml_file.filename)
        saved_filename = f"{timestamp}_modified_{safe_filename}"
        saved_filepath = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)

        pnml_file.save(saved_filepath)

        # 用 PM4Py 读取和生成 SVG，作为硬校验。
        process_model_summary, _, _, _ = render_petri_net_svg_from_pnml(saved_filepath)

        session_metadata = session.get_parameters() or {}

        # 关键：把当前 session 的 pnml_filename 指向用户修改后的 PNML。
        # 后续 /api/discover/<session_id> 会读取这个 pnml_filename，并将该 PNML 作为结构输入。
        session_metadata["pnml_filename"] = saved_filename
        session_metadata["manual_modified_pnml_filename"] = saved_filename
        session_metadata["manual_modified_pnml_path"] = saved_filepath
        session_metadata["manual_modified_pnml_uploaded_at"] = datetime.now().isoformat()
        session_metadata["process_model"] = process_model_summary

        optimized_info = session_metadata.get("optimized_petri_net", {})
        if not isinstance(optimized_info, dict):
            optimized_info = {}

        optimized_info["manual_modified_pnml_filename"] = saved_filename
        optimized_info["manual_modified_pnml_path"] = saved_filepath
        optimized_info["reuse_policy"] = "manual_modified_pnml_requires_rediscovery"

        session_metadata["optimized_petri_net"] = optimized_info

        session.set_parameters(session_metadata)
        session.status = "uploaded"
        db.session.commit()

        logger.info(
            f"Modified PNML uploaded and validated for session {session_id}: {saved_filepath}"
        )

        return jsonify({
            "success": True,
            "message": "Modified PNML uploaded and validated successfully. Please rediscover parameters.",
            "pnml_filename": saved_filename,
            "process_model": process_model_summary,
            "need_rediscovery": True
        })

    except Exception as e:
        db.session.rollback()
        logger.error(f"Replace Petri net error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Failed to replace Petri net: {str(e)}"
        }), 500
def parse_bool(value, default=False):
    """
    Parse frontend bool-like values.
    Supports: 1/0, true/false, yes/no, on/off.
    """
    if value is None:
        return bool(default)

    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in ("1", "true", "yes", "on", "checked"):
        return True

    if value in ("0", "false", "no", "off", "unchecked"):
        return False

    return bool(default)


def parse_noise_threshold(value, default=0.0):
    """
    Parse and validate Inductive Miner noise threshold.
    """
    try:
        threshold = float(value)
    except Exception:
        threshold = float(default)

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            f"Noise threshold must be in [0.0, 1.0], got {threshold}."
        )

    return threshold
def allowed_file(filename):
    """检查上传文件是否为允许的类型。

    当前只允许上传：
    1. .xes 事件日志；
    2. .pnml Petri 网模型。
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['xes', 'pnml']


def get_log_name_from_uploaded_filename(filename: str) -> str:
    """
    从上传后的文件名中提取原始日志名。

    例如：
    1781699550_acr.xes -> acr
    1781699550_purchasing.xes -> purchasing
    """
    base = os.path.basename(filename or "")
    stem = os.path.splitext(base)[0]

    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        stem = parts[1]

    safe_name = secure_filename(stem).strip("._-")
    return safe_name or "unknown_log"


def build_experiment_output_dirs(session_id: int, session_filename: str, timestamp: int):
    """
    构建当前 session 的实验输出目录。

    Petri 网保存到：
        data/petrinet/<log_name>/session_<id>_<timestamp>/

    5 次测试仿真日志保存到：
        data/simulated_event_log/<log_name>/session_<id>_<timestamp>/
    """
    log_name = get_log_name_from_uploaded_filename(session_filename)
    session_dir_name = f"session_{session_id}_{timestamp}"

    petrinet_dir = os.path.join(
        "data",
        "petrinet",
        log_name,
        session_dir_name,
    )

    simulated_log_dir = os.path.join(
        "data",
        "simulated_event_log",
        log_name,
        session_dir_name,
    )

    os.makedirs(petrinet_dir, exist_ok=True)
    os.makedirs(simulated_log_dir, exist_ok=True)

    return log_name, session_dir_name, petrinet_dir, simulated_log_dir
@app.route('/')
def index():
    """主页面路由。

    功能：
    查询最近 10 个仿真会话，并传递给 index.html 页面显示。
    """
    sessions = SimulationSession.query.order_by(SimulationSession.created_at.desc()).limit(10).all()
    return render_template('index.html', sessions=sessions)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """处理 XES 文件上传，并支持可选 PNML 文件上传。

    请求内容：
    - xes_file：必选，事件日志文件；
    - pnml_file：可选，外部 Petri 网模型；
    - noise_threshold：可选，流程发现噪声阈值。

    返回：
    - session_id：本次上传对应的会话 ID；
    - filename：保存后的 XES 文件名；
    - pnml_filename：保存后的 PNML 文件名，若未上传则为 None。
    """
    try:
        # 检查请求中是否包含 XES 文件字段
        if 'xes_file' not in request.files:
            return jsonify({'error': 'No XES file provided'}), 400

        # 读取上传的 XES 文件对象
        xes_file = request.files['xes_file']

        # 读取上传的 PNML 文件对象。
        # 原代码使用 request.files['pnml_file']，如果前端没有传该字段可能报 KeyError。
        # 若前端总是传空 pnml_file 字段则可以正常运行；更稳妥写法是 request.files.get('pnml_file')。
        pnml_file = request.files.get('pnml_file')

        # 读取噪声阈值模式：
        # automatic_threshold_optimization = 1 表示自动优化；
        # automatic_threshold_optimization = 0 表示使用用户手动输入的 noise_threshold。
        automatic_threshold_optimization = parse_bool(
            request.form.get("automatic_threshold_optimization", "1"),
            default=True,
        )

        try:
            noise_threshold = parse_noise_threshold(
                request.form.get("noise_threshold", 0.0),
                default=0.0,
            )
        except ValueError as threshold_error:
            return jsonify({"error": str(threshold_error)}), 400

        # 检查 XES 文件名是否为空
        if xes_file.filename == '':
            return jsonify({'error': 'No XES file selected'}), 400

        # 检查上传的 XES 文件扩展名是否正确
        if not xes_file.filename.lower().endswith('.xes'):
            return jsonify({'error': 'Event log file must have .xes extension'}), 400

        # 如果上传了 PNML 文件，则检查 PNML 文件扩展名是否正确
        if pnml_file and pnml_file.filename and not pnml_file.filename.lower().endswith('.pnml'):
            return jsonify({'error': 'PNML file must have .pnml extension'}), 400

        # 保存 XES 文件前再次检查文件名合法性
        if xes_file.filename is None:
            return jsonify({'error': 'Invalid XES filename'}), 400

        # 使用 secure_filename 清理文件名，并加上当前时间戳，避免重名覆盖
        xes_filename = secure_filename(xes_file.filename)
        timestamp = int(datetime.now().timestamp())
        xes_filename = f"{timestamp}_{xes_filename}"

        # 拼接 XES 文件保存路径，并保存文件
        xes_filepath = os.path.join(app.config['UPLOAD_FOLDER'], xes_filename)
        xes_file.save(xes_filepath)

        # 如果提供了 PNML 文件，则保存 PNML 文件
        pnml_filename = None
        if pnml_file and pnml_file.filename:
            pnml_filename = secure_filename(pnml_file.filename)
            pnml_filename = f"{timestamp}_{pnml_filename}"
            pnml_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pnml_filename)
            pnml_file.save(pnml_filepath)

        # 创建新的仿真会话记录，保存 XES 文件名和噪声阈值
        session = SimulationSession(
            filename=xes_filename,
            noise_threshold=noise_threshold
        )

        # 如果上传了 PNML 文件，则将 PNML 文件名保存在 session 参数中
        # 注意：这里不是单独的数据库字段，而是通过 set_parameters 写入参数 JSON 字段
        session_metadata = {
            "automatic_threshold_optimization": automatic_threshold_optimization,
            "original_filename": xes_file.filename,
            "manual_noise_threshold": noise_threshold,
            "noise_threshold_mode": (
                "automatic"
                if automatic_threshold_optimization
                else "manual"
            ),
        }

        if pnml_filename:
            session_metadata["pnml_filename"] = pnml_filename

        session.set_parameters(session_metadata)

        # 写入数据库
        db.session.add(session)
        db.session.commit()

        # 记录上传成功日志
        logger.info(f"Files uploaded successfully: XES={xes_filename}, PNML={pnml_filename}")

        # 返回上传成功信息
        return jsonify({
            'success': True,
            'session_id': session.id,
            'filename': xes_filename,
            'pnml_filename': pnml_filename,
            'message': 'Files uploaded successfully'
        })

    except Exception as e:
        # 捕获上传过程中的异常并返回错误信息
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/discover/<int:session_id>', methods=['POST'])
def discover_parameters(session_id):
    """
    当前 /api/discover/<session_id> 将上传的训练日志交给严格分阶段调参流程。
    StagewiseTuner 在训练日志内部按 case 时间顺序划分 train / validation，
    完整上传日志作为 train_validation_log 用于最终重新发现。
    独立的 *_test.xes 日志仅用于 Stage 4 最终评估。
    """
    session = None

    try:
        # =========================
        # 一、读取 session 和上传文件路径
        # =========================
        session = SimulationSession.query.get_or_404(session_id)

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            session.filename
        )

        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        session.status = 'discovering'
        db.session.commit()
        set_discovery_progress(
            session_id,
            stage="petri",
            title="Petri net mining...",
            detail="Preparing event log and Petri net structure.",
            status="running"
        )
        logger.info(
            f"Discovering drq model from XES file for session {session_id}"
        )
        set_discovery_progress(
            session_id,
            stage="petri",
            title="Petri net mining...",
            detail="Checking uploaded PNML model or preparing Petri net discovery.",
            status="running"
        )
        # =========================
        # 三、读取 PNML 文件路径
        # =========================
        session_params = session.get_parameters() or {}
        # =========================
        # 读取噪声阈值模式
        # 优先级：discover query 参数 > upload 阶段保存的 session 参数 > 默认自动优化
        # =========================

        automatic_threshold_optimization = parse_bool(
            request.args.get(
                "automatic_threshold_optimization",
                session_params.get("automatic_threshold_optimization", True),
            ),
            default=True,
        )

        try:
            manual_noise_threshold = parse_noise_threshold(
                request.args.get(
                    "noise_threshold",
                    session_params.get(
                        "manual_noise_threshold",
                        getattr(session, "noise_threshold", 0.0),
                    ),
                ),
                default=0.0,
            )
        except ValueError as threshold_error:
            return jsonify({"error": str(threshold_error)}), 400

        pnml_filename = session_params.get('pnml_filename')
        pnml_filepath = None

        if pnml_filename:
            candidate_pnml_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                pnml_filename
            )

            if os.path.exists(candidate_pnml_path):
                pnml_filepath = candidate_pnml_path
                logger.info(f"Using uploaded PNML file: {pnml_filepath}")
            else:
                logger.warning(
                    f"PNML filename exists in session metadata, "
                    f"but file not found: {candidate_pnml_path}"
                )
        set_discovery_progress(
            session_id,
            stage="petri",
            title="Petri net mining...",
            detail="Loading event log and preparing process structure discovery.",
            status="running"
        )
        # =========================
        # 四、读取完整 XES 日志
        # =========================
        full_event_log = xes_importer.apply(filepath)

        logger.info(
            f"Loaded full event log: {len(full_event_log)} traces from {session.filename}"
        )

        # =========================
        # 五、训练集建模 + 测试集评估
        # =========================
        try:
            evaluation_runs = int(request.args.get("evaluation_runs", 5))
        except Exception:
            evaluation_runs = 5

        evaluation_runs = max(1, min(evaluation_runs, 10))


        # =========================
        # 严格分阶段搜索空间
        # =========================
        # 新版 StagewiseTuner 使用连续 noise_threshold_range。
        # noise_threshold=0.0 会在 StagewiseTuner 中通过 Optuna enqueue_trial 强制评估。
        # =========================
        # Petri 网噪声阈值策略
        # =========================
        # 优化参数范围统一在 stagewise_tuner.py 中管理。
        # routes.py 只保留 UI 开关：
        #   1) automatic_threshold_optimization=True: 传 None，使用 tuner 默认范围
        #   2) automatic_threshold_optimization=False: 传退化区间，固定 manual_noise_threshold
        # =========================

        if automatic_threshold_optimization:
            noise_threshold_range = None
            noise_threshold_mode = "automatic"
        else:
            noise_threshold_range = (
                float(manual_noise_threshold),
                float(manual_noise_threshold),
            )
            noise_threshold_mode = "manual"

        logger.info(
            "Noise threshold mode for session %s: mode=%s, manual_noise_threshold=%s, "
            "range=%s (None means using StagewiseTuner default)",
            session_id,
            noise_threshold_mode,
            manual_noise_threshold,
            noise_threshold_range,
        )

        # 控制流/运行时树深度、外部延迟缩放等优化参数范围
        # 统一由 drq/tuning/stagewise_tuner.py 中的常量控制，routes.py 不再覆盖。

        # 其他运行时设置在 StagewiseTuner 内统一固定：
        # arrival_time_scale=1.0, external_delay_global_scale=1.0,
        # external_delay_max_delay_quantile=1.0, queue_strategy=balanced。
        test_evaluation_runs = evaluation_runs

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
        # Stage 1 控制流指标
        structure_metrics = ("cfld", "ngd", "3gd")

        # Stage 2 运行阶段指标，统一比重
        runtime_metrics = ("aed", "ced", "red", "car", "ctd", "cwd")

        valid_pnml_filepath = (
            pnml_filepath
            if pnml_filepath and os.path.exists(pnml_filepath)
            else None
        )

        logger.info(
            f"Strict staged full tuning for session {session_id}: "
            f"evaluation_runs={evaluation_runs}"
        )

        # 本次发现/评估统一使用同一个时间戳，便于 Petri 网和仿真日志对应
        artifact_timestamp = int(datetime.now().timestamp())

        log_name, session_dir_name, petrinet_output_dir, test_simulation_log_dir = (
            build_experiment_output_dirs(
                session_id=session_id,
                session_filename=session.filename,
                timestamp=artifact_timestamp,
            )
        )

        test_simulation_log_prefix = f"{log_name}_test"
        set_discovery_progress(
            session_id,
            stage="params",
            title="Parameter mining...",
            detail="Mining simulation parameters and evaluating resource, queue, arrival, execution-time, and waiting-time models.",
            status="running"
        )

        def discovery_progress_callback(stage, detail=None, title=None, status="running", **kwargs):
            """
            Receive real progress events from drqIntegration / StagewiseTuner.
            Keep only two UI stages: Petri net mining and Parameter mining.
            """
            stage_key = str(stage or "petri").lower()

            if stage_key in (
                "petri", "structure", "structure_search", "control_flow",
                "control-flow", "process_model", "pnml"
            ):
                set_discovery_progress(
                    session_id,
                    stage="petri",
                    title=title or "Petri net mining...",
                    detail=detail or "Optimizing Petri net structure and control-flow behavior.",
                    status=status or "running"
                )
            elif stage_key in (
                "params", "parameter", "parameters", "runtime",
                "runtime_search", "parameter_search", "simulation_parameters"
            ):
                set_discovery_progress(
                    session_id,
                    stage="params",
                    title=title or "Parameter mining...",
                    detail=detail or "Mining resource, queue, arrival, execution-time, waiting-time, and queue parameters.",
                    status=status or "running"
                )
            else:
                # Unknown progress events are treated as parameter mining after structure preparation.
                set_discovery_progress(
                    session_id,
                    stage="params",
                    title=title or "Parameter mining...",
                    detail=detail or f"Processing discovery stage: {stage_key}.",
                    status=status or "running"
                )

        # =========================
        # 查找测试集文件（在 example_data/ 目录中查找 _test 文件）
        # =========================
        # 上传的文件名包含 _train（如 BPI_Challenge_2012_W_Two_TS_train.xes），
        # 对应的测试集文件将 _train 替换为 _test（如 BPI_Challenge_2012_W_Two_TS_test.xes），
        # 存放在 example_data/ 目录中。
        original_filename = session_params.get("original_filename", "")
        test_event_log = None
        test_log_filename = ""

        if original_filename and "_train" in original_filename:
            test_filename = original_filename.replace("_train", "_test", 1)
            example_data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "example_data"
            )
            test_filepath = os.path.join(example_data_dir, test_filename)

            if os.path.exists(test_filepath):
                try:
                    test_event_log = xes_importer.apply(test_filepath)
                    test_log_filename = test_filename
                    logger.info(
                        f"Loaded test event log from example_data/: {test_filename}, "
                        f"{len(test_event_log)} traces"
                    )
                except Exception as test_load_error:
                    error_msg = (
                        f"Failed to load test event log from example_data/{test_filename}: "
                        f"{test_load_error}"
                    )
                    logger.error(error_msg)
                    return jsonify({'error': error_msg}), 400
            else:
                error_msg = (
                    f"Test file not found: example_data/{test_filename}. "
                    f"Please place the test file (with '_test' replacing '_train' in the filename) "
                    f"in the example_data/ directory."
                )
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
        else:
            error_msg = (
                f"Cannot derive test filename from '{original_filename}': "
                f"filename must contain '_train' (e.g., XXX_train.xes). "
                f"The corresponding test file (XXX_test.xes) should be placed in example_data/."
            )
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400

        result = drq.tune_and_evaluate_with_strict_split(
            full_event_log=full_event_log,
            test_event_log=test_event_log,
            pnml_filepath=valid_pnml_filepath,
            progress_callback=discovery_progress_callback,

            # noise_threshold_range: 自动模式传 None，使用 stagewise_tuner.py 默认范围；
            # 手动模式传退化区间，固定 manual_noise_threshold。
            noise_threshold_range=noise_threshold_range,

            # train/validation 划分由 StagewiseTuner 内部统一处理；
            # routes.py 不再传入旧版 split-ratio 参数。
            metrics_labels=metrics_labels,

            # Stage 1：只优化控制流指标
            structure_metrics=structure_metrics,

            # Stage 2：只优化运行阶段指标，且统一比重
            runtime_metrics=runtime_metrics,

            test_evaluation_runs=test_evaluation_runs,
            save_test_simulation_logs=True,
            test_simulation_log_dir=test_simulation_log_dir,
            test_simulation_log_prefix=test_simulation_log_prefix,

        )

        parameters = result["parameters"]
        process_model = result["process_model"]
        test_drq_metrics = result["test_metrics"]
        test_metric_details = result.get("test_metric_details", {})
        log_partitions = result["log_partitions"]
        time_diagnostics = result.get("time_diagnostics", {})
        # 当前 discover_and_evaluate_with_split 只正式返回测试集指标
        # 如果后续你想保存训练集指标，可以在 drqIntegration 中额外实现 train_metrics
        train_drq_metrics = {}

        # =========================
        # 六、整理日志统计信息
        # =========================
        # 新流程：上传日志内部做 train/validation 划分；完整上传日志用于最终重新发现；
        # 独立 *_test.xes 日志只用于 Stage 4 最终评估。
        log_split_summary = {
            "full_traces": len(full_event_log),
            "train_traces": len(log_partitions.train_log),
            "validation_traces": len(log_partitions.validation_log),
            "train_validation_traces": len(log_partitions.train_validation_log),
            "test_traces": len(log_partitions.test_log),
            "train_validation_ratio": 1.0,
            "train_ratio_within_train_validation": 0.8,
            "train_partition_used": "internal_train_validation_split",
            "log_split_enabled": True,
            "test_log_source": test_log_filename if test_log_filename else "full_event_log",
        }

        logger.info(
            "Internal train/validation split (80/20): "
            f"full={log_split_summary['full_traces']}, "
            f"train={log_split_summary['train_traces']}, "
            f"validation={log_split_summary['validation_traces']}, "
            f"test={log_split_summary['test_traces']} traces "
            f"(source: {log_split_summary['test_log_source']})"
        )

        # =========================
        # 七、记录测试集评估指标
        # =========================
        for metric_cat, metric_values in test_drq_metrics.items():
            if not isinstance(metric_values, dict):
                continue

            for metric_name, metric_value in metric_values.items():
                logger.info(
                    f"Test accuracy result [{metric_cat}] {metric_name}: {metric_value}"
                )

        # =========================
        # 八、保存 drq 参数 JSON
        # =========================
        timestamp = artifact_timestamp
        json_filename = f"{timestamp}_parameters.json"
        json_filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            json_filename
        )

        drq.save_parameters_to_json(json_filepath)

        logger.info(
            f"drq parameters saved to JSON: {json_filepath}"
        )

        # =========================
        # 九、保存最终 Petri 网 PNML 和元信息
        # =========================
        try:
            current_best_pnml_filename = "optimized_petri_net.pnml"
            current_best_pnml_path = os.path.join(
                petrinet_output_dir,
                current_best_pnml_filename,
            )

            drq.save_current_petri_net_to_pnml(current_best_pnml_path)

            optimized_meta_filename = "optimized_petri_net_meta.json"
            optimized_meta_path = os.path.join(
                petrinet_output_dir,
                optimized_meta_filename,
            )

            optimized_petri_metadata = {
                "session_id": session_id,
                "source_xes_filename": session.filename,
                "uploaded_pnml_filename": pnml_filename,
                "used_pnml_filepath": valid_pnml_filepath,
                "noise_threshold_mode": noise_threshold_mode,
                "automatic_threshold_optimization": automatic_threshold_optimization,
                "manual_noise_threshold": manual_noise_threshold,
                "noise_threshold_range": list(noise_threshold_range) if noise_threshold_range is not None else None,
                "current_best_pnml_filename": current_best_pnml_filename,
                "current_best_pnml_path": current_best_pnml_path,
                "optimized_meta_filename": optimized_meta_filename,
                "optimized_meta_path": optimized_meta_path,
                "selected_hyperparameters": result.get("hyperparameter_search", {}).get(
                    "selected_hyperparameters", {}
                ),
                "top_structure_candidates": result.get("hyperparameter_search", {}).get(
                    "top_structure_candidates", []
                ),
                "process_model_quality": {
                    "fitness": process_model.get("fitness"),
                    "precision": process_model.get("precision"),
                    "f_measure": process_model.get("f_measure"),
                } if isinstance(process_model, dict) else {},
                "test_metrics": test_drq_metrics,
                "created_at": datetime.now().isoformat(),
            }

            with open(optimized_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    optimized_petri_metadata,
                    f,
                    indent=4,
                    ensure_ascii=False,
                    default=str,
                )

            logger.info(
                "Optimized Petri net and metadata saved: "
                f"pnml={current_best_pnml_path}, meta={optimized_meta_path}"
            )

        except Exception as pnml_save_error:
            logger.warning(
                f"Failed to save optimized Petri net PNML: {pnml_save_error}"
            )
            current_best_pnml_filename = None
            current_best_pnml_path = None
            optimized_meta_filename = None
            optimized_meta_path = None

        # =========================
        # 十、整理流程模型摘要
        # =========================
        process_model_summary = {
            'visualization': process_model.get('visualization'),
            'activities': process_model.get('activities', []),
            'transitions': process_model.get('transitions', []),
            'map_transitionName_to_id': process_model.get(
                'map_transitionName_to_id',
                {}
            ),
            'places': process_model.get('places', []),
            'fitness': process_model.get('fitness', 0),
            'precision': process_model.get('precision', 0),
            'f_measure': process_model.get('f_measure', 0)
        }

        # 确保返回给前端的 parameters 中包含测试集指标
        if isinstance(parameters, dict):
            parameters["drq_metrics"] = test_drq_metrics
            parameters["process_model"] = process_model_summary
            parameters["time_diagnostics"] = time_diagnostics

        # =========================
        # 十、写入数据库
        # =========================
        session_metadata = {
            'json_filename': json_filename,
            'pnml_filename': pnml_filename,
            'process_model': process_model_summary,
            'drq_metrics': test_drq_metrics,
            'train_drq_metrics': train_drq_metrics,
            'test_drq_metrics': test_drq_metrics,
            'log_split': log_split_summary,
            'metrics_labels': metrics_labels,
            # The simulator now has one complete pipeline. Individual module
            # enable/disable flags and experiment profiles no longer exist.
            'pipeline_mode': 'full_pipeline',
            'dispatch_policy_selection': 'data_driven_fcfs_or_cpt',
            'noise_threshold_mode': noise_threshold_mode,
            'automatic_threshold_optimization': automatic_threshold_optimization,
            'manual_noise_threshold': manual_noise_threshold,
            "time_diagnostics": time_diagnostics,
            'hyperparameter_search': result.get("hyperparameter_search", {}),
            'selected_hyperparameters': result.get("hyperparameter_search", {}).get(
                "selected_hyperparameters",
                {},
            ),
            'tuning_config': {
                'noise_threshold_mode': noise_threshold_mode,
                'manual_noise_threshold': manual_noise_threshold,
                'noise_threshold_range': list(noise_threshold_range) if noise_threshold_range is not None else None,
                'note': 'Hyperparameter search ranges are controlled by drq/tuning/stagewise_tuner.py',
                'grid_search_cv_effective': True,
                'pipeline_mode': 'full_pipeline',
                'dispatch_policy_selection': 'data_driven_fcfs_or_cpt',
                'external_delay_protocol': 'simod_lite_activity_specific_timer_no_delay_tree',
                'external_delay_tree_enabled': False,
                'fixed_runtime_controls': {
                    'arrival_time_scale': 1.0,
                    'arrival_demand_multiplier': 1.0,
                    'external_delay_global_scale': 1.0,
                    'external_delay_max_delay_quantile': 1.0,
                    'queue_strategy': 'balanced',
                },
                'external_delay_activity_scale_top_k_policy': 'ratio_with_min_max_caps',
                'external_delay_activity_scale_ratio': 0.25,
                'external_delay_activity_scale_min_k': 3,
                'external_delay_activity_scale_max_k': 8,
                'external_delay_activity_scale_candidates': [0.8, 1.0, 1.2],
                'test_evaluation_runs': test_evaluation_runs,
                'structure_metrics': list(structure_metrics),
                'runtime_metrics': list(runtime_metrics),
            },
            "optimized_petri_net": {
                "current_best_pnml_filename": current_best_pnml_filename,
                "current_best_pnml_path": current_best_pnml_path,
                "optimized_meta_filename": optimized_meta_filename,
                "optimized_meta_path": optimized_meta_path,
                "uploaded_pnml_filename": pnml_filename,
                "used_pnml_filepath": valid_pnml_filepath,
                "reuse_policy": "manual_upload_only",
            },
        }

        session.set_parameters(session_metadata)

        session.status = 'ready'
        db.session.commit()

        set_discovery_progress(
            session_id,
            stage="params",
            title="Discovery completed",
            detail="Petri net and simulation parameters have been discovered successfully.",
            status="ready"
        )

        # =========================
        # 十一、返回结果
        # =========================
        return jsonify({
            'success': True,
            'parameters': parameters,
            'json_filename': json_filename,
            'process_model': process_model_summary,
            'drq_metrics': test_drq_metrics,
            'train_drq_metrics': train_drq_metrics,
            'test_drq_metrics': test_drq_metrics,

            'evaluation_runs': evaluation_runs,
            'test_metric_details': test_metric_details,

            'message': 'Parameters processed successfully',
            "time_diagnostics": time_diagnostics,
        })

    except Exception as e:
        logger.error(f"Discovery error: {str(e)}")

        db.session.rollback()

        try:
            if session is None:
                session = SimulationSession.query.get(session_id)

            if session:
                session.status = 'error'
                db.session.commit()

            set_discovery_progress(
                session_id,
                stage="params",
                title="Discovery failed",
                detail=str(e),
                status="error"
            )

        except Exception:
            db.session.rollback()

        return jsonify({
            'error': f'Parameter discovery failed: {str(e)}'
        }), 500

@app.route('/api/parameters/<int:session_id>')
def get_parameters(session_id):
    """获取指定 session 的参数。

    优先从保存的 drq JSON 文件中加载最新参数；
    如果没有 JSON 文件，则返回数据库中保存的旧版参数。
    """
    try:
        # 查询指定会话
        session = SimulationSession.query.get_or_404(session_id)
        session_metadata = session.get_parameters()

        # 如果 session 中没有参数元数据，则返回 404
        if not session_metadata:
            return jsonify({'error': 'No session metadata found'}), 404

        # 从 session 元数据中读取 drq 参数 JSON 文件名
        json_filename = session_metadata.get('json_filename')
        if json_filename:
            json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
            if os.path.exists(json_filepath):
                # 从 JSON 文件加载参数
                parameters = drq.load_parameters_from_json_file(json_filepath)

                # 将数据库中保存的流程模型摘要合并到参数中，便于前端显示
                if 'process_model' in session_metadata:
                    parameters['process_model'].update(session_metadata['process_model'])

                # 将数据库中保存的 drq 指标合并到参数中
                if 'drq_metrics' in session_metadata:
                    parameters['drq_metrics'].update(session_metadata['drq_metrics'])

                return jsonify({
                    'success': True,
                    'parameters': parameters,
                    'status': session.status,
                    'json_filename': json_filename
                })

        # 如果没有 JSON 文件，返回数据库中的元数据，兼容旧版本数据
        return jsonify({
            'success': True,
            'parameters': session_metadata,
            'status': session.status
        })

    except Exception as e:
        logger.error(f"Get parameters error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ------------------------------------------------------------------
# Safe control-flow what-if editing helpers
# ------------------------------------------------------------------
def _normalize_control_flow_path(path_value):
    """
    将前端传入的规则路径统一转换为 ['true', 'false', ...]。

    支持形式：
    - [] 或 None：表示根节点；
    - "true.false" / "true/false" / "true,false"；
    - ["true", "false"]；
    - [True, False]。
    """
    if path_value is None or path_value == "":
        return []

    if isinstance(path_value, str):
        raw = path_value.strip()
        if not raw:
            return []
        for sep in ["/", ",", "|"]:
            raw = raw.replace(sep, ".")
        parts = [p.strip() for p in raw.split(".") if p.strip()]
    elif isinstance(path_value, (list, tuple)):
        parts = list(path_value)
    else:
        raise ValueError("node_path / leaf_path must be a list or a string.")

    normalized = []
    for item in parts:
        if isinstance(item, bool):
            step = "true" if item else "false"
        else:
            step = str(item).strip().lower()
            if step in ("t", "yes", "1", "left"):
                step = "true"
            elif step in ("f", "no", "0", "right"):
                step = "false"
        if step not in ("true", "false"):
            raise ValueError(
                f"Invalid decision-tree path step '{item}'. "
                "Only true/false branches are allowed."
            )
        normalized.append(step)

    return normalized


def _normalize_decision_point_prob_dict(probs, branches, base_probs=None):
    """
    清洗并归一化分支概率。

    只允许 branches 中存在的分支；负数、非数字会被置 0。
    如果 preserve_unspecified=True 的场景需要保留原概率，可先在外部把
    base_probs 与新 probs 合并后再调用本函数。
    """
    if not isinstance(probs, dict):
        probs = {}

    branches = [str(b) for b in branches if str(b)]
    if not branches:
        return {}

    merged = {}
    if isinstance(base_probs, dict):
        for branch in branches:
            try:
                merged[branch] = float(base_probs.get(branch, 0.0))
            except Exception:
                merged[branch] = 0.0

    for branch in branches:
        if branch in probs:
            try:
                value = float(probs.get(branch, 0.0))
            except Exception:
                value = 0.0
            merged[branch] = max(0.0, value)
        elif branch not in merged:
            merged[branch] = 0.0

    total = sum(float(v) for v in merged.values())
    if total <= 0:
        uniform = 1.0 / len(branches)
        return {branch: float(uniform) for branch in branches}

    return {branch: float(merged[branch]) / total for branch in branches}


def _make_decision_point_leaf(branches, branch_probs=None, value=None, base_probs=None):
    """构造一个合法的决策点叶子节点。"""
    probs = _normalize_decision_point_prob_dict(
        branch_probs or {},
        branches,
        base_probs=base_probs,
    )

    branches = [str(b) for b in branches if str(b)]
    if value is not None:
        value = str(value)
    if value not in branches:
        value = max(probs, key=probs.get) if probs else (branches[0] if branches else None)

    return {
        "type": "leaf",
        "value": value,
        "branch_probs": probs,
    }


def _normalize_decision_point_rule_node(node, branches, feature_columns):
    """
    递归校验并归一化已有决策树规则。

    注意：该函数只用于保存前的结构修复，不开放任意 feature 修改。
    feature 必须属于 feature_columns。
    """
    if not isinstance(node, dict):
        return _make_decision_point_leaf(branches)

    node_type = str(node.get("type", "leaf")).lower()

    if node_type == "leaf":
        return _make_decision_point_leaf(
            branches=branches,
            branch_probs=node.get("branch_probs", {}),
            value=node.get("value"),
        )

    feature = str(node.get("feature", ""))
    feature_columns = [str(c) for c in (feature_columns or [])]
    if feature_columns and feature not in feature_columns:
        raise ValueError(
            f"Invalid decision-tree feature '{feature}'. "
            f"Available features: {feature_columns}"
        )

    try:
        threshold = float(node.get("threshold", 0.0))
        if not np.isfinite(threshold):
            threshold = 0.0
    except Exception:
        threshold = 0.0

    children = node.get("children", {})
    if not isinstance(children, dict):
        children = {}

    return {
        "type": "node",
        "feature": feature,
        "threshold": float(threshold),
        "children": {
            "true": _normalize_decision_point_rule_node(
                children.get("true"),
                branches,
                feature_columns,
            ),
            "false": _normalize_decision_point_rule_node(
                children.get("false"),
                branches,
                feature_columns,
            ),
        },
    }


def _get_decision_point_rule_node(root, path):
    """按照 path 返回 rules 中的目标节点。"""
    node = root
    for step in _normalize_control_flow_path(path):
        if not isinstance(node, dict) or node.get("type") == "leaf":
            raise ValueError(f"Path {path} reaches a leaf before the target node.")
        children = node.get("children", {})
        if not isinstance(children, dict) or step not in children:
            raise ValueError(f"Path {path} does not exist in the decision tree.")
        node = children[step]
    if not isinstance(node, dict):
        raise ValueError(f"Path {path} does not point to a valid rule node.")
    return node


def _set_decision_point_rule_node(root, path, replacement_node):
    """按照 path 替换 rules 中的目标节点。path=[] 表示替换整棵树。"""
    path = _normalize_control_flow_path(path)
    if not path:
        return replacement_node

    parent = _get_decision_point_rule_node(root, path[:-1])
    if parent.get("type") == "leaf":
        raise ValueError(f"Parent path {path[:-1]} points to a leaf.")

    children = parent.setdefault("children", {})
    last = path[-1]
    if last not in children:
        raise ValueError(f"Path {path} does not exist in the decision tree.")
    children[last] = replacement_node
    return root


def _normalize_decision_point_params_for_json(dp_params):
    """
    归一化 decision_point_branch_models。

    该函数保证：
    - empirical_probs 合法且归一化；
    - decision_tree 的每个 leaf.branch_probs 合法且归一化；
    - threshold 是有限浮点数；
    - feature 不会脱离 feature_columns。
    """
    if not isinstance(dp_params, dict):
        raise ValueError("decision_point_branch_models must be a dictionary.")

    normalized = copy.deepcopy(dp_params)
    normalized["enabled"] = bool(normalized.get("enabled", True))
    normalized["model_type"] = normalized.get("model_type", "decision_point_branch_tree")

    decision_points = normalized.get("decision_points", {})
    if not isinstance(decision_points, dict):
        decision_points = {}

    normalized_points = {}
    for signature, model in decision_points.items():
        if not isinstance(model, dict):
            continue

        model = copy.deepcopy(model)
        branches = [str(b) for b in model.get("branches", []) if str(b)]
        if not branches:
            continue

        model["enabled"] = bool(model.get("enabled", True))
        model["signature"] = str(model.get("signature", signature))
        model["branches"] = branches

        model_kind = str(model.get("model_kind", "empirical"))
        if model_kind not in ("empirical", "decision_tree"):
            model_kind = "empirical"
        model["model_kind"] = model_kind

        model["empirical_probs"] = _normalize_decision_point_prob_dict(
            model.get("empirical_probs", {}),
            branches,
        )

        if model_kind == "decision_tree":
            feature_columns = [str(c) for c in model.get("feature_columns", [])]
            model["feature_columns"] = feature_columns
            model["rules"] = _normalize_decision_point_rule_node(
                model.get("rules"),
                branches,
                feature_columns,
            )
        else:
            model["rules"] = None
            model["feature_columns"] = []

        normalized_points[str(signature)] = model

    normalized["decision_points"] = normalized_points
    normalized["num_decision_points"] = int(len(normalized_points))
    normalized.setdefault("metadata", {})
    normalized.setdefault("fallback", {
        "single_enabled_transition": "fire_directly",
        "missing_decision_point": "uniform_over_enabled_transitions",
        "missing_feature": 0.0,
    })

    return normalized


def _get_current_decision_point_params(drq_json):
    """从 control_flow_params 或 modular_params 中读取当前决策点模型。"""
    control_flow_params = drq_json.get("control_flow_params", {})
    if isinstance(control_flow_params, dict):
        dp_params = control_flow_params.get("decision_point_branch_models")
        if isinstance(dp_params, dict):
            return copy.deepcopy(dp_params)

    modular = drq_json.get("modular_params", {})
    if isinstance(modular, dict):
        dp_params = (
            modular.get("modules", {})
            .get("control_flow", {})
            .get("decision_point_branch_models")
        )
        if isinstance(dp_params, dict):
            return copy.deepcopy(dp_params)

    return {
        "enabled": False,
        "model_type": "decision_point_branch_tree",
        "num_decision_points": 0,
        "decision_points": {},
        "metadata": {},
        "fallback": {
            "single_enabled_transition": "fire_directly",
            "missing_decision_point": "uniform_over_enabled_transitions",
            "missing_feature": 0.0,
        },
    }

def _format_decision_point_option_label(index, branches, branch_display_names, model_kind):
    """生成 Select Decision Point 下拉框中的业务化显示名。"""
    labels = [
        str(branch_display_names.get(branch, branch))
        for branch in branches
    ]
    visible = labels[:4]
    suffix = "" if len(labels) <= 4 else f" / ... (+{len(labels) - 4})"
    kind_label = "Decision tree" if model_kind == "decision_tree" else "Empirical"
    return f"Decision Point {index}: " + " / ".join(visible) + suffix + f"  ·  {kind_label}"


def _collect_decision_tree_safe_edit_nodes_for_response(rules, path=None, condition_path=None):
    """收集前端可展示的安全编辑节点。"""
    path = list(path or [])
    condition_path = list(condition_path or [])
    rows = []

    if not isinstance(rules, dict):
        return rows

    node_type = rules.get("type")
    path_string = ".".join(path) if path else "(root)"

    if node_type == "leaf":
        rows.append({
            "path": path,
            "path_string": path_string,
            "type": "leaf",
            "display_label": f"Leaf {path_string}",
            "condition_path": condition_path,
            "editable_operations": ["set_leaf_probs", "prune_subtree"],
            "value": rules.get("value"),
            "branch_probs": rules.get("branch_probs", {}),
        })
        return rows

    feature = rules.get("feature")
    threshold = rules.get("threshold")
    rows.append({
        "path": path,
        "path_string": path_string,
        "type": "node",
        "display_label": f"Rule node {path_string}: {feature} <= {threshold}",
        "condition_path": condition_path,
        "editable_operations": ["set_threshold", "prune_subtree"],
        "feature": feature,
        "threshold": threshold,
    })

    children = rules.get("children", {})
    if isinstance(children, dict):
        if "true" in children:
            rows.extend(
                _collect_decision_tree_safe_edit_nodes_for_response(
                    children.get("true"),
                    path + ["true"],
                    condition_path + [{
                        "feature": feature,
                        "operator": "<=",
                        "threshold": threshold,
                    }],
                )
            )
        if "false" in children:
            rows.extend(
                _collect_decision_tree_safe_edit_nodes_for_response(
                    children.get("false"),
                    path + ["false"],
                    condition_path + [{
                        "feature": feature,
                        "operator": ">",
                        "threshold": threshold,
                    }],
                )
            )

    return rows


def build_control_flow_scheme_b_summary(dp_params):
    """
    生成方案 B 的控制流前端摘要。

    前端应使用 decision_point_options 构建下拉框：
    - empirical 决策点展示分支概率编辑器；
    - decision_tree 决策点展示树规则和安全编辑项。
    """
    summary = {
        "success": True,
        "editor_mode": "decision_point",
        "selector_label": "Select Decision Point",
        "legacy_transition_selector_deprecated": True,
        "show_all_decision_points": True,
        "supported_operations": [
            {
                "operation": "set_empirical_probs",
                "scope": "empirical decision point",
                "description": "Modify empirical branch probabilities for an empirical decision point.",
            },
            {
                "operation": "set_leaf_probs",
                "scope": "decision-tree leaf",
                "description": "Modify branch probabilities of an existing decision-tree leaf.",
            },
            {
                "operation": "set_threshold",
                "scope": "decision-tree internal node",
                "description": "Modify the threshold of an existing decision-tree internal node without changing its feature.",
            },
            {
                "operation": "prune_subtree",
                "scope": "decision-tree subtree",
                "description": "Replace an existing subtree with a safe leaf containing branch probabilities.",
            },
        ],
        "counts": {
            "total": 0,
            "empirical": 0,
            "decision_tree": 0,
            "disabled": 0,
        },
        "decision_point_options": [],
        "decision_points": [],
    }

    if not isinstance(dp_params, dict):
        return summary

    decision_points = dp_params.get("decision_points", {})
    if not isinstance(decision_points, dict):
        return summary

    for index, (signature, model) in enumerate(decision_points.items(), start=1):
        if not isinstance(model, dict):
            continue

        model_kind = str(model.get("model_kind", "empirical"))
        if model_kind not in ("empirical", "decision_tree"):
            model_kind = "empirical"

        branches = [str(b) for b in model.get("branches", []) if str(b)]
        branch_display_names = model.get("branch_display_names", {})
        if not isinstance(branch_display_names, dict):
            branch_display_names = {}
        branch_labels = [
            str(branch_display_names.get(branch, branch))
            for branch in branches
        ]
        enabled = bool(model.get("enabled", True))

        if model_kind == "empirical":
            safe_operations = ["set_empirical_probs"]
            editable_nodes = [{
                "path": [],
                "path_string": "(empirical)",
                "type": "empirical_probability_table",
                "display_label": "Empirical branch probabilities",
                "editable_operations": ["set_empirical_probs"],
                "branch_probs": model.get("empirical_probs", {}),
            }]
            summary["counts"]["empirical"] += 1
        else:
            safe_operations = ["set_leaf_probs", "set_threshold", "prune_subtree"]
            editable_nodes = _collect_decision_tree_safe_edit_nodes_for_response(
                model.get("rules", {})
            )
            summary["counts"]["decision_tree"] += 1

        if not enabled:
            summary["counts"]["disabled"] += 1

        label = _format_decision_point_option_label(
            index=index,
            branches=branches,
            branch_display_names=branch_display_names,
            model_kind=model_kind,
        )

        option = {
            "label": label,
            "value": str(signature),
            "signature": str(signature),
            "model_kind": model_kind,
            "model_kind_label": "Decision tree" if model_kind == "decision_tree" else "Empirical probabilities",
            "enabled": enabled,
            "n_samples": model.get("n_samples"),
            "dominant_prob": model.get("dominant_prob"),
            "branch_count": len(branches),
            "branch_labels": branch_labels,
            "safe_operations": safe_operations,
        }

        row = {
            **option,
            "branches": branches,
            "branch_display_names": branch_display_names,
            "empirical_probs": model.get("empirical_probs", {}),
            "feature_columns": model.get("feature_columns", []) if model_kind == "decision_tree" else [],
            "editable_nodes": editable_nodes,
            "rules": model.get("rules") if model_kind == "decision_tree" else None,
        }

        summary["decision_point_options"].append(option)
        summary["decision_points"].append(row)

    summary["counts"]["total"] = len(summary["decision_points"])
    return summary



def _sync_decision_point_params_to_drq_json(drq_json, dp_params):
    """将决策点模型同时写回 control_flow_params 和 modular_params。"""
    dp_params = _normalize_decision_point_params_for_json(dp_params)

    drq_json.setdefault("control_flow_params", {})
    drq_json["control_flow_params"]["decision_point_branch_models"] = dp_params

    modular = drq_json.setdefault("modular_params", {})
    modules = modular.setdefault("modules", {})
    control_flow_module = modules.setdefault("control_flow", {})
    control_flow_module["enabled"] = bool(dp_params.get("enabled", False))
    control_flow_module["model_type"] = dp_params.get(
        "model_type",
        "petri_net_with_decision_point_branch_trees",
    )
    control_flow_module["decision_point_branch_models"] = dp_params
    control_flow_module["fallback"] = (
        "petri_enabled_set_fallback_then_uniform"
        if isinstance(dp_params.get("petri_fallback_decision_points"), dict)
        else "uniform_enabled_transition"
    )

    return drq_json


def apply_decision_point_safe_edits(dp_params, edits):
    """
    应用第一版控制流安全修改。

    支持 4 类 operation：
    1. set_empirical_probs：修改 empirical 决策点的经验分支概率；
    2. set_leaf_probs：修改 decision_tree 某个叶子节点的 branch_probs；
    3. set_threshold：修改 decision_tree 某个内部节点的 threshold；
    4. prune_subtree：把 decision_tree 某棵子树替换成一个 leaf。

    不支持新增节点、删除任意分支、修改 feature、修改 branches、直接替换整棵 rules。
    """
    if not isinstance(edits, list):
        raise ValueError("decision_point_safe_edits must be a list.")

    updated = _normalize_decision_point_params_for_json(dp_params)
    decision_points = updated.setdefault("decision_points", {})
    audit_records = []

    for edit_index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise ValueError(f"Safe edit #{edit_index + 1} must be a dictionary.")

        operation = str(edit.get("operation", edit.get("op", ""))).strip().lower()
        operation_alias = {
            "empirical_probs": "set_empirical_probs",
            "update_empirical_probs": "set_empirical_probs",
            "set_empirical_branch_probs": "set_empirical_probs",
            "leaf_probs": "set_leaf_probs",
            "update_leaf_probs": "set_leaf_probs",
            "update_leaf_branch_probs": "set_leaf_probs",
            "threshold": "set_threshold",
            "update_threshold": "set_threshold",
            "replace_subtree_with_leaf": "prune_subtree",
            "prune_to_leaf": "prune_subtree",
        }
        operation = operation_alias.get(operation, operation)

        if operation not in {
            "set_empirical_probs",
            "set_leaf_probs",
            "set_threshold",
            "prune_subtree",
        }:
            raise ValueError(
                f"Unsupported control-flow safe edit operation: {operation}."
            )

        signature = edit.get("signature", edit.get("decision_point_signature"))
        if signature is None:
            raise ValueError(f"Safe edit #{edit_index + 1} misses signature.")
        signature = str(signature)

        if signature not in decision_points:
            raise ValueError(
                f"Decision point signature not found: {signature}"
            )

        model = decision_points[signature]
        if not isinstance(model, dict):
            raise ValueError(f"Decision point model is invalid: {signature}")

        branches = [str(b) for b in model.get("branches", []) if str(b)]
        if not branches:
            raise ValueError(f"Decision point has no valid branches: {signature}")

        model_kind = str(model.get("model_kind", "empirical"))

        if operation == "set_empirical_probs":
            if model_kind != "empirical":
                raise ValueError(
                    "set_empirical_probs can only be applied to empirical "
                    f"decision points. Current model_kind={model_kind}."
                )

            probs = (
                edit.get("branch_probs")
                or edit.get("empirical_probs")
                or edit.get("probs")
                or {}
            )
            preserve = bool(edit.get("preserve_unspecified", False))
            base_probs = model.get("empirical_probs", {}) if preserve else None
            model["empirical_probs"] = _normalize_decision_point_prob_dict(
                probs,
                branches,
                base_probs=base_probs,
            )
            model["dominant_prob"] = max(model["empirical_probs"].values()) if model["empirical_probs"] else 1.0

            audit_records.append({
                "operation": operation,
                "signature": signature,
                "branch_probs": model["empirical_probs"],
            })
            continue

        if model_kind != "decision_tree":
            raise ValueError(
                f"{operation} can only be applied to decision_tree decision points. "
                f"Current model_kind={model_kind}."
            )

        rules = model.get("rules")
        if not isinstance(rules, dict):
            raise ValueError(f"Decision point has no editable rules: {signature}")

        path = edit.get("node_path", edit.get("leaf_path", edit.get("path", [])))
        path = _normalize_control_flow_path(path)
        feature_columns = [str(c) for c in model.get("feature_columns", [])]

        if operation == "set_leaf_probs":
            node = _get_decision_point_rule_node(rules, path)
            if node.get("type") != "leaf":
                raise ValueError(
                    f"set_leaf_probs requires a leaf node, but path {path} "
                    f"points to type={node.get('type')}."
                )
            probs = edit.get("branch_probs") or edit.get("probs") or {}
            preserve = bool(edit.get("preserve_unspecified", False))
            base_probs = node.get("branch_probs", {}) if preserve else None
            new_leaf = _make_decision_point_leaf(
                branches=branches,
                branch_probs=probs,
                value=edit.get("value", node.get("value")),
                base_probs=base_probs,
            )
            rules = _set_decision_point_rule_node(rules, path, new_leaf)
            model["rules"] = _normalize_decision_point_rule_node(
                rules,
                branches,
                feature_columns,
            )
            audit_records.append({
                "operation": operation,
                "signature": signature,
                "path": path,
                "branch_probs": new_leaf["branch_probs"],
            })
            continue

        if operation == "set_threshold":
            node = _get_decision_point_rule_node(rules, path)
            if node.get("type") == "leaf":
                raise ValueError("set_threshold cannot be applied to a leaf node.")
            try:
                threshold = float(edit.get("threshold"))
                if not np.isfinite(threshold):
                    raise ValueError
            except Exception:
                raise ValueError("threshold must be a finite number.")

            # 只允许改阈值，不允许通过 patch 改 feature。
            old_threshold = node.get("threshold")
            node["threshold"] = float(threshold)
            model["rules"] = _normalize_decision_point_rule_node(
                rules,
                branches,
                feature_columns,
            )
            audit_records.append({
                "operation": operation,
                "signature": signature,
                "path": path,
                "feature": node.get("feature"),
                "old_threshold": old_threshold,
                "new_threshold": float(threshold),
            })
            continue

        if operation == "prune_subtree":
            probs = edit.get("branch_probs") or edit.get("probs") or {}
            try:
                old_node = _get_decision_point_rule_node(rules, path)
            except ValueError:
                raise
            base_probs = None
            if isinstance(old_node, dict) and old_node.get("type") == "leaf":
                base_probs = old_node.get("branch_probs", {})
            if bool(edit.get("preserve_unspecified", False)):
                base_probs = base_probs or model.get("empirical_probs", {})
            new_leaf = _make_decision_point_leaf(
                branches=branches,
                branch_probs=probs,
                value=edit.get("value"),
                base_probs=base_probs,
            )
            rules = _set_decision_point_rule_node(rules, path, new_leaf)
            model["rules"] = _normalize_decision_point_rule_node(
                rules,
                branches,
                feature_columns,
            )
            audit_records.append({
                "operation": operation,
                "signature": signature,
                "path": path,
                "replacement": new_leaf,
            })

    updated["decision_points"] = decision_points
    updated = _normalize_decision_point_params_for_json(updated)

    metadata = updated.setdefault("metadata", {})
    history = metadata.setdefault("what_if_safe_edit_history", [])
    history_entry = {
        "applied_at": datetime.now().isoformat(),
        "n_edits": int(len(audit_records)),
        "edits": audit_records,
    }
    history.append(history_entry)
    if len(history) > 100:
        metadata["what_if_safe_edit_history"] = history[-100:]

    return updated, audit_records


@app.route('/api/parameters/<int:session_id>/control_flow_decision_points', methods=['GET'])
def get_control_flow_decision_points(session_id):
    """
    返回方案 B 所需的控制流决策点列表。

    前端 Parameter Editor 应使用本接口返回的 decision_point_options
    替代旧的 Select Transition 列表。
    """
    try:
        session = SimulationSession.query.get_or_404(session_id)
        session_metadata = session.get_parameters() or {}

        json_filename = session_metadata.get('json_filename')
        if not json_filename:
            return jsonify({
                'success': False,
                'error': 'No JSON file associated with this session.'
            }), 400

        json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        if not os.path.exists(json_filepath):
            return jsonify({
                'success': False,
                'error': 'Parameter JSON file not found.'
            }), 404

        with open(json_filepath, 'r', encoding='utf-8') as f:
            drq_json = json.load(f)

        dp_params = _get_current_decision_point_params(drq_json)
        summary = build_control_flow_scheme_b_summary(dp_params)
        summary.update({
            'success': True,
            'session_id': int(session_id),
            'json_filename': json_filename,
            'message': 'Use decision_point_options for Select Decision Point. Do not use process_model.transitions for control-flow what-if editing.',
        })
        return jsonify(summary)

    except Exception as e:
        logger.error(f"Get control-flow decision points error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/parameters/<int:session_id>', methods=['PUT'])
def update_parameters(session_id):
    """更新指定 session 的参数，并保存回 drq JSON 文件。

    前端编辑参数后，会把修改后的参数发送到该接口。
    该接口负责将前端格式转换回 drq JSON 格式。
    """
    try:
        # 查询当前会话
        session = SimulationSession.query.get_or_404(session_id)
        session_metadata = session.get_parameters()

        # 获取前端提交的新参数
        new_parameters = request.get_json()
        if not new_parameters:
            return jsonify({'error': 'No parameters provided'}), 400

        # 获取当前 session 对应的 drq JSON 文件名
        json_filename = session_metadata.get('json_filename')
        if not json_filename:
            return jsonify({'error': 'No JSON file associated with this session'}), 400

        json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)

        # 读取当前 drq JSON 文件
        if os.path.exists(json_filepath):
            with open(json_filepath, 'r', encoding='utf-8') as f:
                drq_json = json.load(f)
        else:
            return jsonify({'error': 'Parameter JSON file not found'}), 404

        # 将前端参数格式转换为 drq 内部 JSON 格式
        drq_json = convert_app_to_drq_format(new_parameters, drq_json)

        # 保存更新后的 drq JSON
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(drq_json, f, indent=4, ensure_ascii=False)

        logger.info(f"Parameters updated for session {session_id} and saved to {json_filename}")
        return jsonify({
            'success': True,
            'message': 'Parameters updated successfully'
        })

    except Exception as e:
        logger.error(f"Update parameters error: {str(e)}")
        return jsonify({'error': str(e)}), 500


def convert_app_to_drq_format(app_params, drq_json):
    """将 Web 应用中的参数格式转换回 drq 仿真器 JSON 格式。

    参数
    ----
    app_params:
        前端提交的参数格式，通常更适合页面展示和编辑。

    drq_json:
        当前已有的 drq JSON 结构。
        本函数在该结构基础上局部更新，避免丢失未在前端展示的字段。

    返回
    ----
    drq_json:
        更新后的 drq JSON 参数。
    """
    try:
        logger.info("Converting app parameters to drq format")

        def _normalize_dist(dist_obj, leaf_value=None):
            """将前端传来的分布对象规范化为 drq 所需格式。

            支持两种前端参数形式：
            1. params 为 list，例如 [mean, std]；
            2. params 为 dict，例如 {'mean': 10, 'std': 2}。

            leaf_value 用于概率决策树叶节点，当叶节点没有完整分布时作为默认值。
            """
            # 如果不是字典，直接返回原对象
            if not isinstance(dist_obj, dict):
                return dist_obj

            # 获取分布名称，默认使用 expon
            dist_name = dist_obj.get('dist_name', 'expon')
            params_raw = dist_obj.get('params', {})
            result = {'dist_name': dist_name}

            # 情况一：params 已经是列表形式
            if isinstance(params_raw, list):
                if dist_name == 'fixed':
                    # fixed 分布只有一个固定值
                    value = params_raw[0] if params_raw else (leaf_value if leaf_value is not None else 1.0)
                    result['params'] = [value]
                    result['min_value'] = dist_obj.get('min_value', value)
                    result['max_value'] = dist_obj.get('max_value', value)
                    result['mean_value'] = dist_obj.get('mean_value', value)
                elif dist_name == 'norm':
                    # norm 分布参数为 [mean, std]
                    mean = params_raw[0] if len(params_raw) > 0 else dist_obj.get('mean_value', leaf_value or 15.0)
                    std = params_raw[1] if len(params_raw) > 1 else 5.0
                    result['params'] = [mean, std]
                    result['min_value'] = dist_obj.get('min_value', 0.0)
                    result['max_value'] = dist_obj.get('max_value', 60.0)
                    result['mean_value'] = mean
                elif dist_name == 'expon':
                    # expon 分布参数为 [loc, scale]
                    loc = params_raw[0] if len(params_raw) > 0 else 0.0
                    scale = params_raw[1] if len(params_raw) > 1 else dist_obj.get('mean_value', leaf_value or 15.0)
                    result['params'] = [loc, scale]
                    result['min_value'] = dist_obj.get('min_value', 0.0)
                    result['max_value'] = dist_obj.get('max_value', 60.0)
                    result['mean_value'] = scale
                elif dist_name == 'uniform':
                    # uniform 分布参数为 [low, high]
                    low = params_raw[0] if len(params_raw) > 0 else dist_obj.get('min_value', 0.0)
                    high = params_raw[1] if len(params_raw) > 1 else dist_obj.get('max_value', 60.0)
                    result['params'] = [low, high]
                    result['min_value'] = low
                    result['max_value'] = high
                    result['mean_value'] = (low + high) / 2.0
                return result

            # 情况二：params 是字典形式，例如 {'mean': 10, 'std': 2}
            if isinstance(params_raw, dict):
                mean = params_raw.get('mean', dist_obj.get('mean_value', leaf_value or 15.0))
                std = params_raw.get('std', 5.0)
                min_v = params_raw.get('min', dist_obj.get('min_value', 0.0))
                max_v = params_raw.get('max', dist_obj.get('max_value', 60.0))
                fixed_v = params_raw.get('value', mean)
            else:
                # 如果 params 既不是 list 也不是 dict，则使用默认值兜底
                mean, std, min_v, max_v, fixed_v = (leaf_value or 15.0), 5.0, 0.0, 60.0, (leaf_value or 15.0)

            # 按分布类型生成 drq 需要的字段
            if dist_name == 'fixed':
                result['params'] = [fixed_v]
                result['min_value'] = fixed_v
                result['max_value'] = fixed_v
                result['mean_value'] = fixed_v
            elif dist_name == 'norm':
                result['params'] = [mean, std]
                result['min_value'] = min_v
                result['max_value'] = max_v
                result['mean_value'] = mean
            elif dist_name == 'expon':
                result['params'] = [0.0, mean]
                result['min_value'] = min_v
                result['max_value'] = max_v
                result['mean_value'] = mean
            elif dist_name == 'uniform':
                result['params'] = [min_v, max_v]
                result['min_value'] = min_v
                result['max_value'] = max_v
                result['mean_value'] = (min_v + max_v) / 2.0
            return result

        def _transform_dist_tree(tree):
            """规范化概率决策树中的所有叶节点分布。

            drq 的概率决策树通常以字典形式存储，例如：
            {
                '0': {'feature': ..., 'threshold': ..., 'children': ...},
                '1': {'value': ..., 'dist': ...}
            }

            本函数主要处理叶节点：
            - 如果节点包含 value，则认为是叶节点；
            - 将叶节点中的 dist 规范化为 drq 分布格式。
            """
            if not isinstance(tree, dict):
                return tree

            normalized = {}
            for key, node in tree.items():
                if isinstance(node, dict) and 'value' in node:
                    # 叶节点：包含预测值和对应的分布
                    leaf_value = node.get('value')
                    dist_obj = node.get(
                        'dist',
                        {'dist_name': 'expon', 'params': {'mean': leaf_value if leaf_value is not None else 15.0}}
                    )
                    normalized[key] = {
                        'value': leaf_value,
                        'dist': _normalize_dist(dist_obj, leaf_value)
                    }
                elif isinstance(node, dict) and 'feature' in node:
                    # 分裂节点：保留 feature、threshold 和 children
                    normalized[key] = {
                        'feature': node.get('feature'),
                        'threshold': node.get('threshold'),
                        'children': node.get('children', {})
                    }
                else:
                    # 其他结构保持不变
                    normalized[key] = node
            return normalized

        # =========================
        # 一、更新控制流变迁选择参数
        # =========================
        if 'transition_params' in app_params and 'transition_weights' in app_params['transition_params']:
            drq_json['transition_params']['transition_weights'] = app_params['transition_params']['transition_weights']

        # =========================
        # 一点五、更新控制流决策点安全假设修改
        # =========================
        # 第一版不允许前端直接替换整棵 decision tree，只接受 4 类 safe edit：
        # 1) set_empirical_probs；2) set_leaf_probs；3) set_threshold；4) prune_subtree。
        control_flow_params = app_params.get('control_flow_params', {}) if isinstance(app_params, dict) else {}
        if isinstance(control_flow_params, dict):
            safe_edits = (
                control_flow_params.get('decision_point_safe_edits')
                or control_flow_params.get('safe_edits')
                or app_params.get('decision_point_safe_edits')
                or []
            )

            if safe_edits:
                current_dp_params = _get_current_decision_point_params(drq_json)
                edited_dp_params, audit_records = apply_decision_point_safe_edits(
                    current_dp_params,
                    safe_edits,
                )
                drq_json = _sync_decision_point_params_to_drq_json(
                    drq_json,
                    edited_dp_params,
                )
                logger.info(
                    "Applied %s safe decision-point what-if edits.",
                    len(audit_records),
                )
            elif 'decision_point_branch_models' in control_flow_params:
                logger.info(
                    "Ignoring direct replacement of decision_point_branch_models. "
                    "Use control_flow_params.decision_point_safe_edits for safe what-if edits."
                )

        # =========================
        # 二、更新活动执行时间参数
        # =========================
        if 'execution_time_params' in app_params and 'activity_durations' in app_params['execution_time_params']:
            for activity, params in app_params['execution_time_params']['activity_durations'].items():
                # 如果 params 是以 '0' 开头的字典，通常说明它是决策树结构
                if isinstance(params, dict) and '0' in params:
                    drq_entry = _transform_dist_tree(params)
                else:
                    # 否则认为是普通概率分布结构
                    dist = params.get('distribution', 'fixed')
                    param_values = params.get('parameters', {})

                    # 初始化 drq 分布条目
                    drq_entry = {
                        'dist_name': dist,
                        'min_value': param_values.get('min', param_values.get('min_value', 1.0)),
                        'max_value': param_values.get('max', param_values.get('max_value', 1.0)),
                        'mean_value': param_values.get('mean', param_values.get('mean_value', 1.0))
                    }

                    # 根据不同分布类型转换参数
                    if dist == 'fixed':
                        value = param_values.get('value', param_values.get('mean', param_values.get('mean_value', 1.0)))
                        drq_entry['params'] = [value]
                        drq_entry['min_value'] = value
                        drq_entry['max_value'] = value
                        drq_entry['mean_value'] = value
                    elif dist == 'norm':
                        drq_entry['params'] = [
                            param_values.get('mean', param_values.get('mean_value', 15.0)),
                            param_values.get('std', 5.0)
                        ]
                        drq_entry['min_value'] = param_values.get('min', param_values.get('min_value', 1.0))
                        drq_entry['max_value'] = param_values.get('max', param_values.get('max_value', 60.0))
                        drq_entry['mean_value'] = param_values.get('mean', param_values.get('mean_value', 15.0))
                    elif dist == 'expon':
                        drq_entry['params'] = [
                            0.0,
                            param_values.get('mean', param_values.get('mean_value', 15.0))
                        ]
                        drq_entry['min_value'] = param_values.get('min', param_values.get('min_value', 1.0))
                        drq_entry['max_value'] = param_values.get('max', param_values.get('max_value', 60.0))
                        drq_entry['mean_value'] = param_values.get('mean', param_values.get('mean_value', 15.0))
                    elif dist == 'uniform':
                        min_value = param_values.get('min', param_values.get('min_value', 0.0))
                        max_value = param_values.get('max', param_values.get('max_value', 60.0))
                        drq_entry['params'] = [min_value, max_value]
                        drq_entry['min_value'] = min_value
                        drq_entry['max_value'] = max_value
                        drq_entry['mean_value'] = (min_value + max_value) / 2

                # 写回 drq JSON 中的执行时间分布
                drq_json['execution_time_params']['execution_time_distributions'][activity] = drq_entry

        # =========================
        # 三、更新资源等待时间参数
        # =========================
        if 'waiting_time_params' in app_params:
            # 兼容不同前端字段名：优先使用 resource_waiting_times，否则使用 waiting_time
            src = app_params['waiting_time_params'].get('resource_waiting_times') or app_params['waiting_time_params'].get('waiting_time') or {}
            for resource, params in src.items():
                # 决策树结构
                if isinstance(params, dict) and '0' in params:
                    drq_entry = _transform_dist_tree(params)
                else:
                    # 普通分布结构
                    dist = params.get('distribution', 'fixed')
                    param_values = params.get('parameters', {})

                    drq_entry = {
                        'dist_name': dist,
                        'min_value': param_values.get('min', param_values.get('min_value', 1.0)),
                        'max_value': param_values.get('max', param_values.get('max_value', 1.0)),
                        'mean_value': param_values.get('mean', param_values.get('mean_value', 1.0))
                    }

                    if dist == 'fixed':
                        value = param_values.get('value', param_values.get('mean', param_values.get('mean_value', 1.0)))
                        drq_entry['params'] = [value]
                        drq_entry['min_value'] = value
                        drq_entry['max_value'] = value
                        drq_entry['mean_value'] = value
                    elif dist == 'norm':
                        drq_entry['params'] = [
                            param_values.get('mean', param_values.get('mean_value', 15.0)),
                            param_values.get('std', 5.0)
                        ]
                        drq_entry['min_value'] = param_values.get('min', param_values.get('min_value', 1.0))
                        drq_entry['max_value'] = param_values.get('max', param_values.get('max_value', 60.0))
                        drq_entry['mean_value'] = param_values.get('mean', param_values.get('mean_value', 15.0))
                    elif dist == 'expon':
                        drq_entry['params'] = [
                            0.0,
                            param_values.get('mean', param_values.get('mean_value', 15.0))
                        ]
                        drq_entry['min_value'] = param_values.get('min', param_values.get('min_value', 1.0))
                        drq_entry['max_value'] = param_values.get('max', param_values.get('max_value', 60.0))
                        drq_entry['mean_value'] = param_values.get('mean', param_values.get('mean_value', 15.0))
                    elif dist == 'uniform':
                        min_value = param_values.get('min', param_values.get('min_value', 0.0))
                        max_value = param_values.get('max', param_values.get('max_value', 60.0))
                        drq_entry['params'] = [min_value, max_value]
                        drq_entry['min_value'] = min_value
                        drq_entry['max_value'] = max_value
                        drq_entry['mean_value'] = (min_value + max_value) / 2

                # 写回 drq JSON 中的等待时间分布
                drq_json['waiting_time_params']['waiting_time_distributions'][resource] = drq_entry

        # =========================
        # 四、更新案例到达间隔参数
        # =========================
        if 'inter_arrival_params' in app_params and 'inter_arrival_time' in app_params['inter_arrival_params']:
            inter_arrival = app_params['inter_arrival_params']['inter_arrival_time']

            def _transform_arrival_tree(tree):
                """递归规范化到达时间决策树中的叶节点分布。"""
                if not isinstance(tree, dict):
                    return tree
                normalized = {}
                for key, node in tree.items():
                    if isinstance(node, dict) and 'value' in node:
                        leaf_value = node.get('value')
                        dist_obj = node.get(
                            'dist',
                            {'dist_name': 'expon', 'params': {'mean': leaf_value if leaf_value is not None else 15.0}}
                        )
                        normalized[key] = {
                            'value': leaf_value,
                            'dist': _normalize_dist(dist_obj, leaf_value)
                        }
                    elif isinstance(node, dict) and 'feature' in node:
                        normalized[key] = {
                            'feature': node.get('feature'),
                            'threshold': node.get('threshold'),
                            'children': node.get('children', {})
                        }
                    else:
                        normalized[key] = node
                return normalized

            # Full arrival model preservation:
            # 前端展示时 distribution 可能是 "decision_tree_interarrival" 或
            # "context_observed_histogram"，真正可仿真的完整模型保存在 raw_model 中。
            # 必须保留 raw_model，否则 PUT 更新参数会把完整到达模型覆盖成普通分布。
            if (
                    isinstance(inter_arrival, dict)
                    and inter_arrival.get('distribution') in {
                        'decision_tree_interarrival',
                        'context_observed_histogram',
                    }
                    and isinstance(inter_arrival.get('raw_model'), dict)
            ):
                drq_json.setdefault('arrival_params', {})
                drq_json['arrival_params']['arrival_time_distributions'] = inter_arrival['raw_model']

            # 如果 distribution 是旧版决策树结构，则规范化整棵树
            elif isinstance(inter_arrival.get('distribution'), dict) and '0' in inter_arrival['distribution']:
                normalized_tree = _transform_arrival_tree(inter_arrival['distribution'])
                drq_json['arrival_params']['arrival_time_distributions'] = normalized_tree
            else:
                # 普通分布结构
                dist = inter_arrival.get('distribution', 'expon') if isinstance(inter_arrival, dict) else 'expon'
                param_values = inter_arrival.get('parameters', {}) if isinstance(inter_arrival, dict) else {}

                drq_entry = {
                    'dist_name': dist,
                    'min_value': param_values.get('min', param_values.get('min_value', 1.0)),
                    'max_value': param_values.get('max', param_values.get('max_value', 1.0)),
                    'mean_value': param_values.get('mean', param_values.get('mean_value', 1.0))
                }

                if dist == 'fixed':
                    value = param_values.get('value', param_values.get('mean', param_values.get('mean_value', 1.0)))
                    drq_entry['params'] = [value]
                    drq_entry['min_value'] = value
                    drq_entry['max_value'] = value
                    drq_entry['mean_value'] = value
                elif dist == 'norm':
                    drq_entry['params'] = [
                        param_values.get('mean', param_values.get('mean_value', 15.0)),
                        param_values.get('std', 5.0)
                    ]
                    drq_entry['min_value'] = param_values.get('min', param_values.get('min_value', 1.0))
                    drq_entry['max_value'] = param_values.get('max', param_values.get('max_value', 60.0))
                    drq_entry['mean_value'] = param_values.get('mean', param_values.get('mean_value', 15.0))
                elif dist == 'expon':
                    drq_entry['params'] = [
                        0.0,
                        param_values.get('mean', param_values.get('mean_value', 15.0))
                    ]
                    drq_entry['min_value'] = param_values.get('min', param_values.get('min_value', 1.0))
                    drq_entry['max_value'] = param_values.get('max', param_values.get('max_value', 60.0))
                    drq_entry['mean_value'] = param_values.get('mean', param_values.get('mean_value', 15.0))
                elif dist == 'uniform':
                    min_value = param_values.get('min', param_values.get('min_value', 0.0))
                    max_value = param_values.get('max', param_values.get('max_value', 60.0))
                    drq_entry['params'] = [min_value, max_value]
                    drq_entry['min_value'] = min_value
                    drq_entry['max_value'] = max_value
                    drq_entry['mean_value'] = (min_value + max_value) / 2

                drq_json['arrival_params']['arrival_time_distributions'] = drq_entry

            # 如果前端传入到达日历，则同步更新 arrival_calendar
            if 'calendar' in inter_arrival:
                drq_json['arrival_params']['arrival_calendar'] = inter_arrival['calendar']

            # 支持 decision-tree inter-arrival 的场景负载参数更新。
            if 'arrival_time_scale' in inter_arrival:
                drq_json['arrival_params']['arrival_time_scale'] = float(
                    inter_arrival.get('arrival_time_scale', 1.0)
                )
            if 'arrival_demand_multiplier' in inter_arrival:
                multiplier = float(inter_arrival.get('arrival_demand_multiplier', 1.0))
                drq_json['arrival_params']['arrival_demand_multiplier'] = (
                    multiplier if multiplier > 0 else 1.0
                )

            modular = drq_json.get('modular_params', {})
            if isinstance(modular, dict):
                arrival_module = modular.get('modules', {}).get('arrival')
                if isinstance(arrival_module, dict):
                    arrival_module['arrival_time_scale'] = drq_json['arrival_params'].get('arrival_time_scale', 1.0)
                    arrival_module['arrival_demand_multiplier'] = drq_json['arrival_params'].get('arrival_demand_multiplier', 1.0)
                    arrival_distribution = drq_json['arrival_params'].get(
                        'arrival_time_distributions',
                        arrival_module.get('inter_arrival_distribution')
                    )
                    arrival_module['inter_arrival_distribution'] = arrival_distribution
                    if isinstance(arrival_distribution, dict):
                        arrival_module['model_type'] = arrival_distribution.get(
                            'model_type',
                            'case_arrival_decision_tree_interarrival'
                        )

        # =========================
        # 五、更新资源参数
        # =========================
        if 'resource_params' in app_params:
            resource_params = app_params['resource_params']

            # 更新主资源列表
            if 'resources' in resource_params:
                drq_json['resource_params']['resources'] = resource_params['resources']

            # current_resources 用于汇总所有资源，避免某些新增资源只出现在权重或日历中却不在资源列表里
            current_resources = set(drq_json['resource_params'].get('resources', []))

            # 更新资源选择权重
            if 'resource_weights' in resource_params:
                drq_json['resource_params']['resource_weights'] = resource_params['resource_weights']
                for resource in resource_params['resource_weights'].keys():
                    current_resources.add(resource)
            # 更新队列感知资源选择模型
            if 'queue_resource_selection' in resource_params:
                drq_json['resource_params']['queue_resource_selection'] = (
                    resource_params['queue_resource_selection']
                )
            # 更新多任务资源列表
            if 'multitasking_resource' in resource_params:
                drq_json['resource_params']['multitasking_resource'] = resource_params['multitasking_resource']

            # 更新活动到可执行资源的映射
            if 'act_to_resources' in resource_params:
                drq_json['resource_params']['act_to_resources'] = resource_params['act_to_resources']

            # 更新资源日历
            if 'calendars' in resource_params:
                drq_json['resource_params']['calendars'] = resource_params['calendars']
                for resource in resource_params['calendars'].keys():
                    current_resources.add(resource)

            # 将所有发现或新增资源写回主资源列表
            drq_json['resource_params']['resources'] = sorted(list(current_resources))

            # 为新增资源补齐默认参数，避免仿真时因为缺少权重、日历或等待时间分布而报错
            for resource in current_resources:
                # 如果资源没有选择权重，则赋予默认权重 0.1
                if resource not in drq_json['resource_params'].get('resource_weights', {}):
                    if 'resource_weights' not in drq_json['resource_params']:
                        drq_json['resource_params']['resource_weights'] = {}
                    drq_json['resource_params']['resource_weights'][resource] = 0.1

                # 如果资源没有日历，则赋予默认工作日日历
                if resource not in drq_json['resource_params'].get('calendars', {}):
                    if 'calendars' not in drq_json['resource_params']:
                        drq_json['resource_params']['calendars'] = {}
                    drq_json['resource_params']['calendars'][resource] = {
                        day: {str(hour): (9 <= hour <= 17) for hour in range(24)}
                        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    }

                # 如果资源没有等待时间分布，则补充一个默认 fixed 分布
                if resource not in drq_json.get('waiting_time_params', {}).get('waiting_time_distributions', {}):
                    if 'waiting_time_params' not in drq_json:
                        drq_json['waiting_time_params'] = {}
                    if 'waiting_time_distributions' not in drq_json['waiting_time_params']:
                        drq_json['waiting_time_params']['waiting_time_distributions'] = {}
                    drq_json['waiting_time_params']['waiting_time_distributions'][resource] = {
                        'dist_name': 'fixed',
                        'params': [1.0],
                        'min_value': 1.0,
                        'max_value': 1.0,
                        'mean_value': 1.0
                    }

        # =========================
        # 六、更新数据属性参数
        # =========================
        if 'data_attribute_params' in app_params:
            app_data_attrs = app_params['data_attribute_params']
            drq_dist_attrs = {}

            # 获取类别型属性列表
            categorical_attrs = app_data_attrs.get('label_data_attributes_categorical', [])

            # 遍历每个数据属性的分布信息
            for attr, params in app_data_attrs.get('distribution_data_attributes', {}).items():
                if attr in categorical_attrs:
                    # 类别型属性通常是经验概率分布，直接保存
                    drq_dist_attrs[attr] = params
                else:
                    # 数值型属性需要转换为 drq 分布格式
                    dist = params.get('distribution', 'fixed')
                    param_values = params.get('parameters', {})

                    drq_entry = {
                        'dist_name': dist,
                        'min_value': param_values.get('min', param_values.get('min_value', 1.0)),
                        'max_value': param_values.get('max', param_values.get('max_value', 1.0)),
                        'mean_value': param_values.get('mean', param_values.get('mean_value', 1.0))
                    }

                    if dist == 'fixed':
                        value = param_values.get('value', param_values.get('mean', param_values.get('mean_value', 1.0)))
                        drq_entry['params'] = [value]
                        drq_entry['min_value'] = value
                        drq_entry['max_value'] = value
                        drq_entry['mean_value'] = value
                    elif dist == 'norm':
                        drq_entry['params'] = [
                            param_values.get('mean', param_values.get('mean_value', 15.0)),
                            param_values.get('std', 5.0)
                        ]
                        drq_entry['min_value'] = param_values.get('min', param_values.get('min_value', 1.0))
                        drq_entry['max_value'] = param_values.get('max', param_values.get('max_value', 60.0))
                        drq_entry['mean_value'] = param_values.get('mean', param_values.get('mean_value', 15.0))
                    elif dist == 'expon':
                        drq_entry['params'] = [
                            0.0,
                            param_values.get('mean', param_values.get('mean_value', 15.0))
                        ]
                        drq_entry['min_value'] = param_values.get('min', param_values.get('min_value', 1.0))
                        drq_entry['max_value'] = param_values.get('max', param_values.get('max_value', 60.0))
                        drq_entry['mean_value'] = param_values.get('mean', param_values.get('mean_value', 15.0))
                    elif dist == 'uniform':
                        min_value = param_values.get('min', param_values.get('min_value', 0.0))
                        max_value = param_values.get('max', param_values.get('max_value', 60.0))
                        drq_entry['params'] = [min_value, max_value]
                        drq_entry['min_value'] = min_value
                        drq_entry['max_value'] = max_value
                        drq_entry['mean_value'] = (min_value + max_value) / 2

                    drq_dist_attrs[attr] = drq_entry

            # 写回数据属性分布和属性元信息
            drq_json['data_attribute_params']['distribution_data_attributes'] = drq_dist_attrs
            drq_json['data_attribute_params'].update({
                'label_data_attributes': app_data_attrs.get('label_data_attributes'),
                'label_data_attributes_categorical': app_data_attrs.get('label_data_attributes_categorical'),
                'attribute_values_label_categorical': app_data_attrs.get('attribute_values_label_categorical')
            })
        # =========================
        # 七、更新动态资源状态模型
        # =========================
        dynamic_resource_params = app_params.get('dynamic_resource_params', {}) if isinstance(app_params, dict) else {}
        if isinstance(dynamic_resource_params, dict):
            resource_state_dbn = dynamic_resource_params.get('resource_state_dbn')
            if isinstance(resource_state_dbn, dict):
                drq_json['resource_state_params'] = resource_state_dbn

            base_service_dists = dynamic_resource_params.get('base_service_time_distributions')
            if isinstance(base_service_dists, dict):
                drq_json.setdefault('execution_time_params', {})
                drq_json['execution_time_params']['base_service_time_distributions'] = base_service_dists

        if 'resource_state_params' in app_params and isinstance(app_params.get('resource_state_params'), dict):
            drq_json['resource_state_params'] = app_params['resource_state_params']

        modular = drq_json.get('modular_params', {})
        if isinstance(modular, dict):
            modules = modular.setdefault('modules', {})
            dynamic_resource_module = modules.get('dynamic_resource')
            if isinstance(dynamic_resource_module, dict):
                if 'resource_state_params' in drq_json:
                    dynamic_resource_module['resource_state_model'] = drq_json['resource_state_params']
                    dynamic_resource_module['resource_state_dbn'] = drq_json['resource_state_params']
                    dynamic_resource_module['enabled'] = bool(
                        drq_json['resource_state_params'].get('enabled', False)
                    )
                if isinstance(
                    drq_json.get('execution_time_params', {}).get('base_service_time_distributions'),
                    dict,
                ):
                    dynamic_resource_module['base_service_time_distributions'] = (
                        drq_json['execution_time_params']['base_service_time_distributions']
                    )

        # =========================
        # 八、更新队列调度模块
        # =========================
        if 'queue_params' in app_params and isinstance(app_params.get('queue_params'), dict):
            drq_json['queue_params'] = app_params['queue_params']

        if 'queue_scheduling_params' in app_params and isinstance(app_params.get('queue_scheduling_params'), dict):
            drq_json['queue_scheduling_params'] = app_params['queue_scheduling_params']

            qsp = drq_json['queue_scheduling_params']
            selected_strategy = qsp.get(
                'selected_strategy',
                qsp.get('resource_policy_mode', qsp.get('selected_resource_policy', 'balanced')),
            )
            qsp['selected_strategy'] = selected_strategy
            qsp['resource_policy_mode'] = selected_strategy
            qsp['selected_resource_policy'] = selected_strategy

            modular = drq_json.get('modular_params', {})
            if isinstance(modular, dict):
                modules = modular.setdefault('modules', {})
                queue_module = modules.get('queue_scheduling')
                if isinstance(queue_module, dict):
                    queue_module['enabled'] = bool(qsp.get('enabled', queue_module.get('enabled', True)))
                    queue_module['selected_strategy'] = selected_strategy
                    queue_module['resource_policy_mode'] = selected_strategy
                    queue_module['selected_resource_policy'] = selected_strategy
                    queue_model = queue_module.get('queue_scheduling_model')
                    if isinstance(queue_model, dict):
                        queue_model.update(qsp)
                    else:
                        queue_module['queue_scheduling_model'] = copy.deepcopy(qsp)

        # =========================
        # 九、更新外部残余等待模块
        # =========================
        if 'external_delay_params' in app_params and isinstance(app_params.get('external_delay_params'), dict):
            drq_json['external_delay_params'] = app_params['external_delay_params']

            ext = drq_json['external_delay_params']
            ext.setdefault('runtime', {})
            if not isinstance(ext['runtime'], dict):
                ext['runtime'] = {}

            if 'global_scale' in ext:
                ext['runtime']['global_scale'] = ext.get('global_scale')
            elif 'global_scale' in ext.get('runtime', {}):
                ext['global_scale'] = ext['runtime'].get('global_scale')

            if 'max_delay_quantile' in ext:
                ext['runtime']['max_delay_quantile'] = ext.get('max_delay_quantile')
            elif 'max_delay_quantile' in ext.get('runtime', {}):
                ext['max_delay_quantile'] = ext['runtime'].get('max_delay_quantile')

            modular = drq_json.get('modular_params', {})
            if isinstance(modular, dict):
                modules = modular.setdefault('modules', {})
                external_module = modules.get('external_delay')
                if isinstance(external_module, dict):
                    external_module['enabled'] = bool(ext.get('enabled', external_module.get('enabled', True)))
                    external_module['external_delay_model'] = copy.deepcopy(ext)
                    if 'global_scale' in ext:
                        external_module['global_scale'] = ext.get('global_scale')
                    if 'max_delay_quantile' in ext:
                        external_module['max_delay_quantile'] = ext.get('max_delay_quantile')

        # 返回更新后的 drq JSON
        return drq_json

    except Exception as e:
        logger.error(f"Error converting app to drq format: {str(e)}")
        raise


@app.route('/api/parameters/<int:session_id>/control_flow_edits', methods=['POST', 'PUT'])
def update_control_flow_safe_edits(session_id):
    """
    仅应用控制流决策点的安全假设修改，并保存回当前 drq JSON。

    请求格式示例：
    {
        "edits": [
            {
                "operation": "set_leaf_probs",
                "signature": "[\"t1\", \"t2\"]",
                "leaf_path": ["true", "false"],
                "branch_probs": {"t1": 0.1, "t2": 0.9}
            }
        ]
    }
    """
    try:
        session = SimulationSession.query.get_or_404(session_id)
        session_metadata = session.get_parameters() or {}
        payload = request.get_json() or {}

        edits = (
            payload.get('edits')
            or payload.get('decision_point_safe_edits')
            or payload.get('safe_edits')
            or []
        )

        if not isinstance(edits, list) or len(edits) == 0:
            return jsonify({
                'success': False,
                'error': 'No control-flow safe edits provided.'
            }), 400

        json_filename = session_metadata.get('json_filename')
        if not json_filename:
            return jsonify({
                'success': False,
                'error': 'No JSON file associated with this session.'
            }), 400

        json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        if not os.path.exists(json_filepath):
            return jsonify({
                'success': False,
                'error': 'Parameter JSON file not found.'
            }), 404

        with open(json_filepath, 'r', encoding='utf-8') as f:
            drq_json = json.load(f)

        current_dp_params = _get_current_decision_point_params(drq_json)
        edited_dp_params, audit_records = apply_decision_point_safe_edits(
            current_dp_params,
            edits,
        )
        drq_json = _sync_decision_point_params_to_drq_json(
            drq_json,
            edited_dp_params,
        )

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(drq_json, f, indent=4, ensure_ascii=False)

        logger.info(
            "Control-flow safe edits updated for session %s: %s edits",
            session_id,
            len(audit_records),
        )

        safe_summary = build_control_flow_scheme_b_summary(edited_dp_params)

        return jsonify({
            'success': True,
            'message': 'Control-flow safe edits applied successfully.',
            'applied_edits': audit_records,
            'decision_point_branch_models': edited_dp_params,
            'control_flow_safe_what_if': safe_summary,
            'decision_point_options': safe_summary.get('decision_point_options', []),
        })

    except Exception as e:
        logger.error(f"Update control-flow safe edits error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@app.route('/api/load_test_json')
def load_test_json():
    """加载测试 JSON 文件，主要用于调试参数展示和转换逻辑。"""
    try:
        # 测试 JSON 默认位于上传文件夹下，文件名为 test_params.json
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test_params.json')
        if not os.path.exists(json_path):
            return jsonify({'error': 'Test JSON file not found'}), 404

        # 调用 drq 集成对象加载 JSON 参数
        parameters = drq.load_parameters_from_json_file(json_path)

        return jsonify({
            'success': True,
            'parameters': parameters,
            'message': 'Test JSON loaded successfully'
        })

    except Exception as e:
        logger.error(f"Error loading test JSON: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulate/<int:session_id>', methods=['POST'])
def simulate(session_id):
    """基于已发现或已修改的 drq 参数生成仿真事件日志。"""
    try:
        # 查询当前会话
        session = SimulationSession.query.get_or_404(session_id)
        parameters = session.get_parameters()

        # 如果没有参数，则无法仿真
        if not parameters:
            return jsonify({'error': 'No parameters found'}), 404

        # 读取前端请求中的仿真案例数量，默认 100
        request_data = request.get_json() or {}
        num_instances = int(request_data.get('num_instances', 100))

        # 限制仿真案例数量，避免请求过大导致服务器压力过高
        if num_instances < 1 or num_instances > 10000:
            return jsonify({'error': 'Number of instances must be between 1 and 10000'}), 400

        # 更新会话状态为 simulating
        session.status = 'simulating'
        db.session.commit()

        logger.info(f"Starting simulation for session {session_id} with {num_instances} instances")

        # 获取 session 中保存的参数 JSON 文件名
        session_metadata = session.get_parameters()
        json_filename = session_metadata.get('json_filename')

        if not json_filename:
            return jsonify({'error': 'No stored parameters found for this session'}), 400

        # 拼接参数 JSON 路径
        json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)

        if not os.path.exists(json_filepath):
            return jsonify({'error': 'Stored parameter file not found'}), 404

        # 加载保存的参数 JSON。
        # 这里不会重新发现参数，而是使用用户已经修改并保存的参数。
        logger.info(f"Using stored modified parameters from {json_filename}")
        drq.load_parameters_from_json_file(json_filepath)
        # Resolve simulation start timestamp.
        # Priority:
        # 1) frontend request;
        # 2) stored drq parameters;
        # 3) current UTC time as fallback.
        start_timestamp_raw = request_data.get('start_timestamp')

        if not start_timestamp_raw:
            start_timestamp_raw = find_start_timestamp_from_params(drq.drq_params)

        if not start_timestamp_raw:
            logger.warning(
                "No start_timestamp found in request or parameters; "
                "falling back to current UTC time."
            )
            start_timestamp_raw = datetime.now(timezone.utc)

        start_timestamp = to_utc_aware_timestamp(start_timestamp_raw)

        logger.info(
            f"Resolved simulation start timestamp: {start_timestamp} "
            f"(source value: {start_timestamp_raw})"
        )
        # Optional what-if arrival workload control.
        # Priority:
        # 1) values explicitly sent by the simulation request;
        # 2) values already saved in the parameter JSON by the five-module editor;
        # 3) neutral defaults.
        arrival_params = drq.drq_params.setdefault('arrival_params', {})

        saved_arrival_demand_multiplier = arrival_params.get(
            'arrival_demand_multiplier',
            1.0,
        )
        saved_arrival_time_scale = arrival_params.get(
            'arrival_time_scale',
            1.0,
        )

        try:
            arrival_demand_multiplier = float(
                request_data.get(
                    'arrival_demand_multiplier',
                    saved_arrival_demand_multiplier,
                )
            )
            if not np.isfinite(arrival_demand_multiplier) or arrival_demand_multiplier <= 0:
                arrival_demand_multiplier = 1.0
        except Exception:
            arrival_demand_multiplier = 1.0

        try:
            arrival_time_scale = float(
                request_data.get(
                    'arrival_time_scale',
                    saved_arrival_time_scale,
                )
            )
            if not np.isfinite(arrival_time_scale) or arrival_time_scale < 0:
                arrival_time_scale = 1.0
        except Exception:
            arrival_time_scale = 1.0

        arrival_params['arrival_demand_multiplier'] = arrival_demand_multiplier
        arrival_params['arrival_time_scale'] = arrival_time_scale

        modular = drq.drq_params.get('modular_params', {})
        if isinstance(modular, dict):
            arrival_module = modular.get('modules', {}).get('arrival')
            if isinstance(arrival_module, dict):
                arrival_module['model_type'] = 'case_arrival_decision_tree_interarrival'
                arrival_module['arrival_demand_multiplier'] = arrival_demand_multiplier
                arrival_module['arrival_time_scale'] = arrival_time_scale

        result_df = drq.run_simulation(
            n_traces=num_instances,
            start_timestamp=start_timestamp,
        )

        # 生成仿真输出文件名
        timestamp = int(datetime.now().timestamp())
        output_filename = f'simulation_{timestamp}_{session_id}.csv'

        # 确定仿真结果保存路径
        output_path = os.path.join(app.config.get('SIMULATION_FOLDER', 'simulations'), output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存仿真日志为 CSV。
        # 先统一时间戳字符串格式，避免同一列中有的带微秒、有的不带微秒，
        # 进而导致后续 pd.to_datetime 默认推断固定格式时报错。
        result_df = format_timestamp_columns_for_csv(result_df)
        result_df.to_csv(output_path, index=False)

        # 更新会话状态和仿真输出文件名
        session.status = 'completed'
        session.simulation_df_filename = output_filename
        db.session.commit()

        return jsonify({
            'success': True,
            'sim_output_filename': output_filename,
            'num_events': len(result_df),
            'arrival_demand_multiplier': arrival_demand_multiplier,
            'arrival_time_scale': arrival_time_scale,
            'message': f'Simulation completed with {num_instances} instances, generated {len(result_df)} events'
        })

    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        session = SimulationSession.query.get(session_id)
        if session:
            session.status = 'error'
            db.session.commit()
        return jsonify({'error': f'Simulation failed: {str(e)}'}), 500


@app.route('/api/download/<path:filename>')
def download_file(filename):
    """下载仿真生成的 CSV 文件。"""
    try:
        # 拼接仿真结果文件路径
        filepath = os.path.join(app.config.get('SIMULATION_FOLDER', 'simulations'), filename)

        # 如果文件不存在则返回 404
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        # 返回文件下载响应
        return send_file(filepath, as_attachment=True)

    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export_parameters/<int:session_id>')
def export_parameters(session_id):
    """导出当前 session 的 drq 参数 JSON 文件。"""
    try:
        # 查询会话元数据
        session = SimulationSession.query.get_or_404(session_id)
        session_metadata = session.get_parameters()

        if not session_metadata:
            return jsonify({'error': 'No parameters found'}), 404

        # 获取参数 JSON 文件名
        json_filename = session_metadata.get('json_filename')
        if not json_filename:
            return jsonify({'error': 'No drq JSON file associated with this session'}), 400

        # 拼接参数 JSON 路径
        json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)

        if not os.path.exists(json_filepath):
            return jsonify({'error': 'drq parameter file not found'}), 404

        # 读取当前 drq JSON，包含前端已修改的参数
        with open(json_filepath, 'r') as f:
            drq_json = json.load(f)

        # 创建临时文件，用于返回下载
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(drq_json, f, indent=4)
            temp_path = f.name

        # 返回 JSON 文件下载响应
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f'drq_parameters_{session_id}.json',
            mimetype='application/json'
        )

    except Exception as e:
        logger.error(f"Export parameters error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_activities_and_resources/<int:session_id>')
def get_activities_and_resources(session_id):
    """从仿真日志中提取所有活动和资源名称，用于前端下拉筛选。"""
    try:
        # 查询会话，获取仿真日志文件名
        session = SimulationSession.query.get_or_404(session_id)
        sim_log_filename = session.simulation_df_filename

        if not sim_log_filename:
            return jsonify({'error': 'Simulated log file not found.'}), 404

        # 拼接仿真日志文件路径
        sim_log_filepath = os.path.join(app.config.get('SIMULATION_FOLDER', 'simulations'), sim_log_filename)
        if not os.path.exists(sim_log_filepath):
            return jsonify({'error': 'Simulated log file not found on server.'}), 404

        # 读取仿真日志 CSV
        df = pd.read_csv(sim_log_filepath)

        # 提取活动列表和资源列表
        activities = sorted(df['concept:name'].unique().tolist())
        resources = sorted(df['org:resource'].dropna().unique().tolist())

        return jsonify({'success': True, 'activities': activities, 'resources': resources})

    except Exception as e:
        logger.error(f"Get activities and resources error: {str(e)}")
        return jsonify({'error': f'Failed to get data: {str(e)}'}), 500


@app.route('/api/get_visualization/<int:session_id>/<visualization_type>')
def get_visualization(session_id, visualization_type):
    """根据仿真日志生成指定类型的可视化结果。

    支持的 visualization_type：
    1. process_map：性能 DFG 流程图；
    2. case_duration_dist：案例周期时间分布；
    3. resource_heatmap：资源在星期/小时维度上的事件数量热力图；
    4. activity_duration_boxplot：活动执行时间箱线图；
    5. waiting_time_dist：等待时间分布图。
    """
    try:
        # 查询会话和仿真日志文件名
        session = SimulationSession.query.get_or_404(session_id)
        sim_log_filename = session.simulation_df_filename

        if not sim_log_filename:
            return jsonify({'error': 'Simulated log file not found.'}), 404

        # 拼接仿真日志路径
        sim_log_filepath = os.path.join(app.config.get('SIMULATION_FOLDER', 'simulations'), sim_log_filename)
        if not os.path.exists(sim_log_filepath):
            return jsonify({'error': 'Simulated log file not found on server.'}), 404

        # 读取仿真日志
        df = pd.read_csv(sim_log_filepath)

        # 统一转换时间字段为 UTC datetime，便于计算时长。
        # 注意：仿真 CSV 中可能同时存在
        # 2026-06-29 21:00:00.000000+00:00 和
        # 2026-06-29 21:00:00+00:00。
        # 不能使用 pandas 默认推断或固定 format，否则 process_map / case_duration_dist
        # 会在混合格式上报错。
        df = normalize_timestamp_columns(
            df,
            required_columns=[
                "case:concept:name",
                "concept:name",
                "start:timestamp",
                "time:timestamp",
            ],
            drop_bad_required=True,
        )

        if "enabled:timestamp" not in df.columns:
            df["enabled:timestamp"] = df["start:timestamp"]
        else:
            df["enabled:timestamp"] = parse_timestamp_series_mixed(
                df["enabled:timestamp"],
                column_name="enabled:timestamp",
            ).fillna(df["start:timestamp"])

        if df.empty:
            return jsonify({
                "error": "No valid events remain after timestamp normalization."
            }), 400

        # 设置绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')

        if visualization_type == 'process_map':
            # 将 DataFrame 转为 pm4py EventLog
            log = pm4py.convert_to_event_log(df)

            # 发现 performance DFG，并获取开始活动和结束活动
            performance_dfg, start_activities, end_activities = pm4py.discover_performance_dfg(
                log,
                case_id_key='case:concept:name',
                activity_key='concept:name',
                timestamp_key='time:timestamp'
            )

            # 设置 DFG 可视化参数
            dfg_parameters = dfg_perf_visualizer.Parameters
            parameters = {}
            parameters[dfg_parameters.START_ACTIVITIES] = start_activities
            parameters[dfg_parameters.END_ACTIVITIES] = end_activities
            parameters[dfg_parameters.AGGREGATION_MEASURE] = "median"
            parameters["bgcolor"] = "white"

            # 生成 DFG 图对象
            gviz = dfg_perf_visualizer.apply(performance_dfg, parameters=parameters)

            # 导出 SVG 字符串
            svg_content = gviz.pipe(format='svg', encoding='utf-8')

            logger.info("Process Map Plot Generated.")
            return jsonify({'success': True, 'visualization': svg_content})

        elif visualization_type == 'case_duration_dist':
            # 计算每个 case 的开始时间和结束时间
            case_starts = df.groupby('case:concept:name')['start:timestamp'].min()
            case_ends = df.groupby('case:concept:name')['time:timestamp'].max()

            # 计算 case 周期时间，单位为分钟
            case_durations = (case_ends - case_starts).dt.total_seconds() // 60

            # 计算统计指标
            metrics = {
                'average_case_duration': case_durations.mean(),
                'median_case_duration': case_durations.median(),
                'min_case_duration': case_durations.min(),
                'max_case_duration': case_durations.max()
            }

            # 绘制案例周期时间直方图和 KDE 曲线
            plt.figure(figsize=(10, 6))
            sns.histplot(case_durations, kde=True)
            plt.title('Case Duration Density Plot')
            plt.xlabel('Duration (minutes)')
            plt.ylabel('Frequency')

            # 保存为 SVG 字符串
            svg_buffer = io.StringIO()
            plt.savefig(svg_buffer, format='svg')
            svg_buffer.seek(0)
            svg_content = svg_buffer.getvalue()

            # 释放绘图资源
            plt.close()
            svg_buffer.close()

            logger.info("Case Duration Distribution Plot Generated.")
            return jsonify({'success': True, 'visualization': svg_content, 'metrics': metrics})

        elif visualization_type == 'resource_heatmap':
            # 获取前端指定的资源名称，如果为空则统计所有资源
            resource_name = request.args.get('resource')

            # 如果指定资源，则筛选该资源的事件
            if resource_name:
                df = df[df['org:resource'] == resource_name]

            # 提取事件开始时间对应的小时和星期
            df['hour'] = df['start:timestamp'].dt.hour
            df['weekday_num'] = df['start:timestamp'].dt.dayofweek
            weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            df['weekday'] = df['weekday_num'].map(weekday_map)

            # 按星期和小时统计事件数量
            utilization = df.groupby(['weekday_num', 'hour']).size().unstack(fill_value=0)

            # 补齐 7 天 × 24 小时矩阵，避免没有事件的时段缺失
            utilization = utilization.reindex(index=np.arange(7), columns=np.arange(24), fill_value=0)

            # 绘制热力图
            plt.figure(figsize=(14, 6))
            sns.heatmap(utilization, cmap="YlGnBu", annot=True, fmt="d", cbar_kws={'label': 'Number of events'})

            # 设置标题和坐标轴
            title = f"Resource Event Count Heatmap: {resource_name}" if resource_name else "Resource Event Count Heatmap: All Resources"
            plt.title(title, fontsize=16)
            plt.xlabel("Hour of Day", fontsize=12)
            plt.ylabel("Weekday", fontsize=12)
            plt.yticks(ticks=np.arange(7) + 0.5, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)

            plt.tight_layout()

            # 保存为 SVG 字符串
            svg_buffer = io.StringIO()
            plt.savefig(svg_buffer, format='svg')
            svg_buffer.seek(0)
            svg_content = svg_buffer.getvalue()

            # 释放绘图资源
            plt.close()
            svg_buffer.close()

            logger.info("Resource Heatmap Plot Generated.")
            return jsonify({'success': True, 'visualization': svg_content})

        elif visualization_type == 'activity_duration_boxplot':
            # 获取前端指定的活动名称，如果为空则统计所有活动
            activity_name = request.args.get('activity')

            # 按活动筛选数据
            if activity_name:
                activity_df = df[df['concept:name'] == activity_name]
            else:
                activity_df = df

            # 计算活动执行时间，单位为分钟
            activity_df['duration'] = (activity_df['time:timestamp'] - activity_df['start:timestamp']).dt.total_seconds() // 60

            # 计算统计指标
            metrics = {
                'average_duration': activity_df['duration'].mean(),
                'median_duration': activity_df['duration'].median(),
                'min_duration': activity_df['duration'].min(),
                'max_duration': activity_df['duration'].max()
            }

            # 绘制箱线图，showfliers=False 表示不显示离群点
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=activity_df['duration'], showfliers=False)
            title = f'Activity Duration for "{activity_name}"' if activity_name else 'Activity Duration for All Activities'
            plt.title(title)
            plt.ylabel('Duration (minutes)')

            # 保存为 SVG 字符串
            svg_buffer = io.StringIO()
            plt.savefig(svg_buffer, format='svg')
            svg_buffer.seek(0)
            svg_content = svg_buffer.getvalue()

            # 释放绘图资源
            plt.close()
            svg_buffer.close()

            logger.info("Activity Duration Plot Generated.")
            return jsonify({'success': True, 'visualization': svg_content, 'metrics': metrics})

        elif visualization_type == 'waiting_time_dist':
            # 获取前端指定的活动和资源筛选条件
            activity_name = request.args.get('activity')
            resource_name = request.args.get('resource')

            # 获取日志中实际存在的活动和资源列表
            activities = list(df['concept:name'].unique())
            resources = list(df['org:resource'].unique())

            # 根据活动和资源条件筛选数据
            if activity_name and activity_name in activities:
                if resource_name and resource_name in resources:
                    df = df[(df["org:resource"] == resource_name) & (df["concept:name"] == activity_name)]
                else:
                    df = df[df["concept:name"] == activity_name]
            else:
                if resource_name and resource_name in resources:
                    df = df[df["org:resource"] == resource_name]

            # 计算等待时间：活动实际开始时间 - 活动使能时间
            df['waiting_time'] = (df['start:timestamp'] - df['enabled:timestamp']).dt.total_seconds() // 60

            # 计算统计指标
            metrics = {
                'average_duration': df['waiting_time'].mean(),
                'median_duration': df['waiting_time'].median(),
                'min_duration': df['waiting_time'].min(),
                'max_duration': df['waiting_time'].max()
            }

            # 绘制等待时间直方图和 KDE 曲线
            plt.figure(figsize=(10, 6))
            sns.histplot(df['waiting_time'], kde=True)

            # 根据筛选条件生成标题
            title_parts = []
            if resource_name and resource_name in resources:
                title_parts.append(f'Resource: {resource_name}')
            elif not resource_name:
                title_parts.append('Resource: Any')

            if activity_name and activity_name in activities:
                title_parts.append(f'Activity: {activity_name}')
            elif not activity_name:
                title_parts.append('Activity: Any')

            plt.title(' -- '.join(title_parts) if title_parts else 'Waiting Time Distribution')
            plt.xlabel('Duration (minutes)')
            plt.ylabel('Frequency')

            # 保存为 SVG 字符串
            svg_buffer = io.StringIO()
            plt.savefig(svg_buffer, format='svg')
            svg_buffer.seek(0)
            svg_content = svg_buffer.getvalue()

            # 释放绘图资源
            plt.close()
            svg_buffer.close()

            logger.info("Waiting Time Distribution Plot Generated.")
            return jsonify({'success': True, 'visualization': svg_content, 'metrics': metrics})

        else:
            # 如果 visualization_type 不在支持范围内，则返回错误
            return jsonify({'error': 'Invalid visualization type.'}), 400

    except Exception as e:
        logger.error(f"Visualization error for {visualization_type}: {str(e)}")
        return jsonify({'error': f'Failed to generate visualization: {str(e)}'}), 500


@app.errorhandler(404)
def not_found_error(error):
    """统一处理 404 错误。"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """统一处理 500 错误。

    如果请求过程中数据库事务未正常提交，则先回滚，避免数据库处于异常事务状态。
    """
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500
