# app.py
from flask import Flask, request, jsonify
from celery import Celery
from ultralytics import YOLO
import os
import zipfile
from pathlib import Path
import logging
import json
import shutil
import requests
from urllib.parse import urlparse
from PIL import Image
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 Flask 应用
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# 初始化 Celery 实例
celery = Celery(
    'app',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# ==================== 任务控制文件 ====================
CONTROL_DIR = Path("task_control")
CONTROL_DIR.mkdir(exist_ok=True)

def set_task_action(task_id, action):
    """设置任务控制状态: running / paused / stopped"""
    control_file = CONTROL_DIR / f"{task_id}.json"
    with open(control_file, "w", encoding="utf-8") as f:
        json.dump({"action": action}, f)

def get_task_action(task_id):
    """获取任务控制状态"""
    control_file = CONTROL_DIR / f"{task_id}.json"
    if not control_file.exists():
        return "running"
    try:
        with open(control_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("action", "running")
    except:
        return "running"
    
# ==================== 工具函数：下载远程 ZIP ====================
def download_dataset(dataset_url: str, download_folder: str) -> str:
    """
    从 HTTP/HTTPS URL 下载 ZIP 数据集
    :param dataset_url: 数据集 ZIP 的 URL
    :param download_folder: 下载和解压的目标目录
    :return: 解压后的数据源目录路径
    """
    download_path = Path(download_folder)
    download_path.mkdir(parents=True, exist_ok=True)
    # 构建本地文件名
    filename = os.path.basename(urlparse(dataset_url).path)
    if not filename.endswith(".zip"):
        filename = "dataset.zip"
    zip_path = download_path / filename
    extract_path = download_path / "extracted"

    logger.info(f"📥 开始下载数据集: {dataset_url}")
    try:
        with requests.get(dataset_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"✅ 下载完成: {zip_path}")
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        raise RuntimeError(f"无法下载数据集: {e}")

    # 解压
    extract_path.mkdir(exist_ok=True)
    logger.info(f"📦 正在解压到: {extract_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info("✅ 解压完成")
        # 可选：删除 ZIP 文件
        zip_path.unlink(missing_ok=True)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"ZIP 文件损坏: {e}")
    return str(extract_path)


# ==================== 从 Label_Format_Convert.txt 复用的函数 ====================
def convert_json_to_yolo_and_labelme(json_file, image_file, classes_file,
                                     output_detect_dir, output_segment_dir):
    """
    根据 JSON 中的 type 类型，分别转换为：
    - RECT: YOLO 检测格式 (.txt)
    - POLYGON: LabelMe 分割格式 (.json)
    """
    # 读取 classes.txt（使用 utf-8-sig 防止 BOM）
    with open(classes_file, 'r', encoding='utf-8-sig') as f:
        class_names = [line.strip() for line in f.readlines()]
    # 读取自定义 JSON
    with open(json_file, 'r', encoding='utf-8-sig') as f:
        annotations = json.load(f)
    # 读取图像尺寸
    img = Image.open(image_file)
    img_w, img_h = img.size
    
    # 区分处理：检测 vs 分割
    yolo_lines = []
    labelme_shapes = []
    has_rect = False
    has_polygon = False
    for ann in annotations:
        ann_type = ann.get("type")
        label = ann.get("label")
        if label not in class_names:
            logger.warning(f"警告：类别 '{label}' 未在 classes.txt 中找到，跳过。")
            continue
        class_id = class_names.index(label)
        if ann_type == "RECT":
            has_rect = True
            point = ann["points"][0]
            x, y, w, h = point['x'], point['y'], point['w'], point['h']
            # 计算归一化中心坐标
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        elif ann_type == "POLYGON":
            has_polygon = True
            points = ann["points"]
            polygon = [[round(p["x"], 3), round(p["y"], 3)] for p in points]
            shape = {
                "label": label,
                "points": polygon,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            labelme_shapes.append(shape)
        else:
            logger.warning(f"未知类型 '{ann_type}'，跳过。")
    # 输出检测结果（YOLO .txt）
    if has_rect:
        txt_path = Path(output_detect_dir) / (Path(json_file).stem + '.txt')
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_lines))
        logger.info(f"✅ 检测标签已生成: {txt_path}")
    # 输出分割结果（LabelMe .json）
    if has_polygon:
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": labelme_shapes,
            "imagePath": Path(image_file).name,
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w
        }
        json_path = Path(output_segment_dir) / Path(json_file).name
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 分割标签已生成: {json_path}")


def batch_convert(input_dir, output_detect_dir, output_segment_dir):
    input_path = Path(input_dir)
    classes_file = input_path / 'classes.txt'
    if not classes_file.exists():
        raise FileNotFoundError(f"未找到 classes.txt 文件: {classes_file}")
    # 支持的图像后缀
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    for json_file in input_path.glob("*.json"):
        if json_file.name == 'classes.txt':
            continue
        # 查找对应图像
        image_file = None
        for ext in img_extensions:
            img_path = json_file.with_suffix(ext)
            if img_path.exists():
                image_file = img_path
                break
        if image_file is None:
            logger.warning(f"⚠️ 警告：未找到与 {json_file} 对应的图像文件。")
            continue
        # 执行转换
        convert_json_to_yolo_and_labelme(
            json_file=json_file,
            image_file=image_file,
            classes_file=classes_file,
            output_detect_dir=output_detect_dir,
            output_segment_dir=output_segment_dir
        )
    # ✅ 拷贝 classes.txt 到 YOLO 输出目录
    output_detect_path = Path(output_detect_dir)
    output_detect_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(classes_file, output_detect_path / 'classes.txt')
    logger.info(f"📄 classes.txt 已复制到: {output_detect_path / 'classes.txt'}")


# ==================== 数据处理与训练任务 ====================

def extract_local_dataset(local_zip_path: str, extract_folder: str):
    """解压本地 ZIP 文件到指定目录"""
    extract_path = Path(extract_folder)
    extract_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(local_zip_path):
        raise FileNotFoundError(f"本地 ZIP 文件不存在: {local_zip_path}")

    logger.info(f"📦 开始解压: {local_zip_path} -> {extract_folder}")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    logger.info(f"✅ 解压完成: {extract_folder}")


def organize_data_and_create_yaml(extracted_dir: str, dataset_dir: Path):
    """
    组织数据并生成 data.yaml
    1. 找到解压后的数据源目录（可能是子目录）
    2. 复制图片到 datasets/{task_id}/images
    3. 使用 batch_convert 生成 YOLO 标签到 datasets/{task_id}/labels
    4. 生成 data.yaml (使用绝对路径)
    """
    extracted_base = Path(extracted_dir)

    # --- 关键：自动识别数据源目录 ---
    sub_dirs = [p for p in extracted_base.iterdir() if p.is_dir()]
    if sub_dirs:
        source_dir = sub_dirs[0]  # 取第一个子目录作为源
        logger.info(f"📁 检测到数据源子目录: {source_dir}")
    else:
        source_dir = extracted_base  # 如果没有子目录，使用解压根目录
        logger.info(f"📁 使用解压根目录作为数据源: {source_dir}")

    # 创建目标目录
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # --- 复制所有图片 ---
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    copied_count = 0
    for item in source_dir.iterdir():
        if item.suffix.lower() in img_extensions:
            shutil.copy(item, images_dir / item.name)
            copied_count += 1
    logger.info(f"✅ 已复制 {copied_count} 张图片到 {images_dir}")

    # --- 执行批量转换生成 YOLO 标签 ---
    batch_convert(
        input_dir=str(source_dir),
        output_detect_dir=str(labels_dir),
        output_segment_dir=str(dataset_dir / "segmentation")  # 如果需要分割标签
    )

    # --- 生成 data.yaml (使用绝对路径) ---
    classes_file = labels_dir / 'classes.txt'
    if not classes_file.exists():
        raise FileNotFoundError("转换后未找到 classes.txt")

    with open(classes_file, 'r', encoding='utf-8-sig') as f:
        class_names = [line.strip() for line in f.readlines()]

    # ✅ 使用绝对路径，这是解决 YOLOv8 找不到图像的关键
    images_abs_path = str((dataset_dir / "images").resolve())

    data_config = {
        'train': images_abs_path,
        'val': images_abs_path,
        'test': images_abs_path,
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = dataset_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(f"✅ data.yaml 已生成: {yaml_path}")
    logger.info(f"📌 数据集图像路径（绝对路径）: {images_abs_path}")
    return str(yaml_path)


@celery.task(bind=True)
def run_training(self, model_path: str, dataset_config: str, epochs: int,
                 imgsz: int, batch_size: int, device: str, project: str, name: str):
    """执行 YOLOv8 训练任务，支持暂停/停止"""
    try:
        logger.info(f"开始训练任务: model={model_path}, dataset={dataset_config}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not Path(dataset_config).exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {dataset_config}")

        model = YOLO(model_path)
        logger.info("模型加载成功")

        def on_train_epoch_end(trainer):
            current_epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            progress = (current_epoch / total_epochs) * 100

            import time
            
            # 🔎 检查任务控制状态
            action = get_task_action(self.request.id)
            if action == "paused":
                logger.info(f"⏸️ 任务 {self.request.id} 暂停中...")
                while get_task_action(self.request.id) == "paused":
                    time.sleep(5)
                logger.info(f"▶️ 任务 {self.request.id} 已恢复运行")
            elif action == "stopped":
                logger.info(f"⏹️ 任务 {self.request.id} 已停止")
                trainer.stop = True  # 结束训练

            # 更新 Celery 状态
            self.update_state(
                state='PROGRESS',
                meta={
                    'stage': 'training',
                    'current_epoch': current_epoch,
                    'total_epochs': total_epochs,
                    'progress': round(progress, 1)
                }
            )

        def on_train_start(trainer):
            self.update_state(
                state='PROGRESS',
                meta={'stage': 'training_started', 'current_epoch': 0, 'total_epochs': epochs, 'progress': 0}
            )

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_train_start", on_train_start)

        results = model.train(
            data=dataset_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            verbose=True,
            workers=0,
        )

        weights_path = Path(project) / name / "weights" / "best.pt"
        if not weights_path.exists():
            weights_path = Path(project) / name / "weights" / "last.pt"

        metrics = results.results_dict if hasattr(results, 'results_dict') else {}
        final_metrics = {
            "mAP50": metrics.get('metrics/mAP50(B)', 0),
            "mAP50-95": metrics.get('metrics/mAP50-95(B)', 0),
            "precision": metrics.get('metrics/precision(B)', 0),
            "recall": metrics.get('metrics/recall(B)', 0),
            "f1_score": 2 * (metrics.get('metrics/precision(B)', 0) * metrics.get('metrics/recall(B)', 0)) /
                        (metrics.get('metrics/precision(B)', 0) + metrics.get('metrics/recall(B)', 0) + 1e-6)
        }

        # ************* 相对路径 *************
        # output = {
        #     "status": "success",
        #     "weights_path": str(weights_path) if weights_path.exists() else "",
        #     "project_path": str(Path(project) / name),
        #     "task_id": self.request.id,
        #     "final_metrics": final_metrics
        # }
        
        # ✅ 修改：使用绝对路径
        output = {
            "status": "success",
            "weights_path": str(weights_path.resolve()) if weights_path.exists() else "",
            "project_path": str((Path(project) / name).resolve()),
            "task_id": self.request.id,
            "final_metrics": final_metrics
        }

        # 在训练任务完成后调用该函数, 在 run_training 的 return output 之前，插入以下代码
        # 构造推送内容
        push_payload = {
            "f1_score": final_metrics["f1_score"],
            "mAP50": final_metrics["mAP50"],
            "mAP50-95": final_metrics["mAP50-95"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "project_path": str(Path(project) / name),
            "status": "success",
            "task_id": self.request.id,
            "weights_path": str(weights_path) if weights_path.exists() else ""
        }

        # 推送给前端
        push_model_to_frontend(push_payload)

        return output

    except Exception as e:
        logger.error(f"训练任务失败: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise e


# 添加推送函数
def push_model_to_frontend(payload: dict):
    """将训练完成信息推送给前端"""
    url = "http://60.168.144.127:8000/pushModel"
    try:
        # 构造 form-data 格式
        response = requests.post(url, files={key: (None, str(value)) for key, value in payload.items()}, timeout=10)
        if response.status_code == 200:
            logger.info("✅ 成功推送模型信息到前端")
        else:
            logger.warning(f"⚠️ 推送失败，状态码: {response.status_code}, 响应: {response.text}")
    except Exception as e:
        logger.error(f"❌ 推送模型信息失败: {e}")


# ==================== Flask 路由 ====================
@app.route('/train', methods=['POST'])
def train():
    """启动训练任务（保持之前的增量训练逻辑不变）"""
    data = request.get_json()
    raw_data_url = data.get('DatasetUrl')
    if not raw_data_url:
        return jsonify({"error": "缺少 DatasetUrl"}), 400
    local_zip_path = urlparse(raw_data_url).path

    task_id = data.get('TaskId')
    epochs = data.get('Epochs', 20)
    imgsz = data.get('ImageSize', 640)
    batch_size = data.get('BatchSize', 16)
    device = data.get('Device', "")
    project = data.get('project', "runs/train")
    name = data.get('name', task_id)
    premodel = data.get('PreTrainingModel', 'yolov8s.pt')
    increment_training = data.get('IncrementTraining', False)
    base_task_id = data.get('BaseTaskId')

    if not task_id:
        return jsonify({"error": "缺少 TaskId"}), 400

    try:
        epochs, imgsz, batch_size = int(epochs), int(imgsz), int(batch_size)
    except ValueError:
        return jsonify({"error": "参数必须为整数"}), 400

    downloads_dir = Path("downloads")
    dataset_dir = Path("datasets") / task_id
    model_path = Path("models") / premodel

    if increment_training and base_task_id:
        prev_model_path = Path(project) / base_task_id / "weights" / "best.pt"
        if prev_model_path.exists():
            model_path = prev_model_path
            logger.info(f"🔄 增量训练启用，加载历史模型: {model_path}")

    try:
        extract_local_dataset(local_zip_path, downloads_dir)
        yaml_path = organize_data_and_create_yaml(str(downloads_dir), dataset_dir)

        if not model_path.exists():
            return jsonify({"error": f"模型文件未找到: {model_path}"}), 404

        task = run_training.delay(
            model_path=str(model_path),
            dataset_config=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch_size=batch_size,
            device=device,
            project=project,
            name=name
        )

        # 默认写入 running 状态
        set_task_action(task.id, "running")

        return jsonify({
            "code": 0,
            "message": "成功",
            "data": {"TaskId": task.id}
        }), 200

    except Exception as e:
        logger.error(f"训练准备失败: {str(e)}")
        return jsonify({"error": f"任务初始化失败: {str(e)}"}), 500


@app.route('/pause/<task_id>', methods=['POST'])
def pause_task(task_id):
    set_task_action(task_id, "paused")
    return jsonify({"code": 0, "message": f"任务 {task_id} 已暂停"}), 200

@app.route('/resume/<task_id>', methods=['POST'])
def resume_task(task_id):
    set_task_action(task_id, "running")
    return jsonify({"code": 0, "message": f"任务 {task_id} 已恢复"}), 200

@app.route('/stop/<task_id>', methods=['POST'])
def stop_task(task_id):
    set_task_action(task_id, "stopped")
    return jsonify({"code": 0, "message": f"任务 {task_id} 已停止"}), 200


@app.route('/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = celery.AsyncResult(task_id)
    response = {"code": 0, "message": "", "data": {"task_id": task_id, "status": task.state.lower()}}

    if task.state == 'PENDING':
        response["message"] = "任务等待中"
        
    elif task.state == 'PROGRESS':
        # 正常是json格式的  根据前后端需求，改成form-data格式
        response["message"] = "正在训练"
        progress_info = task.info or {}
        response["data"]["result"] = {
            "stage": progress_info.get('stage', 'training'),
            "current_epoch": progress_info.get('current_epoch', 0),
            "total_epochs": progress_info.get('total_epochs', 0),
            "progress": f"{progress_info.get('progress', 0)}%"
        }
        
    elif task.state == 'SUCCESS':
        response["message"] = "训练完成"
        response["data"]["result"] = task.result
        
    elif task.state == 'FAILURE':
        response["code"] = 1
        response["message"] = "训练失败"
        response["data"]["error"] = str(task.info.get('error', '未知错误'))
    else:
        response["message"] = f"未知状态: {task.state}"

    return jsonify(response), 200 if task.state in ['SUCCESS', 'PROGRESS', 'PENDING'] else 500


@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "YOLOv8 API服务运行正常", "status": "healthy"})


# ------------------- 启动 -------------------
if __name__ == '__main__':
    # 确保必要目录存在
    for dir_name in ["models", "datasets", "downloads", "runs"]:
        Path(dir_name).mkdir(exist_ok=True)
    app.run(host='0.0.0.0', port=8000, debug=True)

    # 启动 Redis 服务器。                               redis-server
    # 在项目根目录下启动 Flask 应用：                     python app.py
    # 在另一个终端窗口启动 Celery Worker：                celery -A app.celery worker --loglevel=info --pool=solo
    # 最后启动界面UI程序:                                 streamlit run streamlit_app.py
    # 运行测试脚本：                                    python test_inference_flask.py 或 python test_train_flask.py

