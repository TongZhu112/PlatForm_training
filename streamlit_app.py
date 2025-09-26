import os
import io
import json
import token

import requests
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from lxml import etree as ET
import time
import base64
from datetime import datetime

# -----------------------------
# 系统配置和初始化
# -----------------------------
st.set_page_config(
    page_title="AI标注与训练平台 - 标注系统",
    layout="wide",
    page_icon="✏️"
)

# 注入全局CSS样式
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stCanvas {
        border: 1px solid #ddd;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .status-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-active {
        background-color: #d4edda;
        color: #155724;
    }
    .status-pending {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-completed {
        background-color: #cce5ff;
        color: #004085;
    }
    .annotation-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .annotation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .annotation-title {
        font-weight: bold;
        color: #1e3c72;
    }
    .annotation-actions {
        display: flex;
        gap: 0.5rem;
    }
    .btn-sm {
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
    }
    .system-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background-color: #e9ecef;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .system-status-icon {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    .status-connected {
        background-color: #28a745;
    }
    .status-disconnected {
        background-color: #dc3545;
    }
    .training-progress {
        margin-top: 1rem;
    }
    .training-progress .stProgress > div > div > div {
        background-color: #1e3c72;
    }
    .model-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .model-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .model-title {
        font-weight: bold;
        color: #1e3c72;
    }
    .model-meta {
        font-size: 0.85rem;
        color: #6c757d;
    }
    .model-actions {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# 全局配置和API连接
# -----------------------------
API_BASE_URL = "http://60.168.144.142:8000"  # 您的后端API地址
APP_VERSION = "1.2.0"

# 初始化会话状态
if "user" not in st.session_state:
    st.session_state.user = None
if "api_connected" not in st.session_state:
    st.session_state.api_connected = False
if "current_project" not in st.session_state:
    st.session_state.current_project = None
if "projects" not in st.session_state:
    st.session_state.projects = []
if "annotation_mode" not in st.session_state:
    st.session_state.annotation_mode = "single"  # single or batch
if "current_image_index" not in st.session_state:
    st.session_state.current_image_index = 0
if "auto_save" not in st.session_state:
    st.session_state.auto_save = True
if "last_save_time" not in st.session_state:
    st.session_state.last_save_time = None
if "training_tasks" not in st.session_state:
    st.session_state.training_tasks = []
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "models" not in st.session_state:
    st.session_state.models = []


# 检查API连接
def check_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=3)
        if response.status_code == 200:
            st.session_state.api_connected = True
            return True
    except:
        st.session_state.api_connected = False
    return False


# -----------------------------
# 用户认证相关函数   admin   123
# -----------------------------
def login():
    st.sidebar.subheader("用户登录")
    username = st.sidebar.text_input("用户名", key="login_username")
    password = st.sidebar.text_input("密码", type="password", key="login_password")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("登录", use_container_width=True):
            # 实际应用中应调用API验证
            if username and password:
                st.session_state.user = {
                    "username": username,
                    "role": "admin" if username == "admin" else "annotator",
                    "projects": ["项目A", "项目B"] if username == "admin" else ["项目A"]
                }
                st.success(f"欢迎回来, {username}!")
                st.rerun()
            else:
                st.sidebar.error("请输入用户名和密码")

    with col2:
        if st.button("注册", use_container_width=True):
            st.sidebar.info("请联系管理员注册账号")


def logout():
    st.session_state.user = None
    st.session_state.current_project = None
    st.session_state.projects = []
    st.success("已安全登出")
    st.rerun()


def render_user_info():
    if st.session_state.user:
        with st.sidebar:
            st.markdown(f"**👤 {st.session_state.user['username']}**")
            st.markdown(f"**角色:** {st.session_state.user['role']}")
            st.markdown("---")
            if st.button("登出", use_container_width=True):
                logout()
    else:
        login()


# -----------------------------
# 项目管理功能
# -----------------------------
def load_projects():
    """从API加载项目列表"""
    if not st.session_state.user:
        return []

    try:
        # # 实际应用中应调用API
        # response = requests.get(f"{API_BASE_URL}/projects", headers={"Authorization": f"Bearer {token}"})
        # if response.status_code == 200:
        #     return response.json()

        # 模拟数据
        return [
            {"id": "proj_001", "name": "交通标志检测", "description": "道路标志识别项目", "created_at": "2023-01-15",
             "image_count": 1250, "label_count": 4500},
            {"id": "proj_002", "name": "工业零件质检", "description": "生产线零件缺陷检测", "created_at": "2023-02-20",
             "image_count": 850, "label_count": 3200},
            {"id": "proj_003", "name": "野生动物监测", "description": "森林保护区动物识别", "created_at": "2023-03-10",
             "image_count": 2100, "label_count": 7800}
        ]
    except:
        return []


def select_project():
    """项目选择界面"""
    st.sidebar.header("项目管理")

    if not st.session_state.projects:
        st.session_state.projects = load_projects()

    project_names = [p["name"] for p in st.session_state.projects]
    selected_project = st.sidebar.selectbox(
        "选择项目",
        options=project_names,
        index=0 if st.session_state.current_project else 0,
        key="project_selector"
    )

    if selected_project:
        idx = project_names.index(selected_project)
        st.session_state.current_project = st.session_state.projects[idx]
        st.sidebar.success(f"当前项目: {selected_project}")

    if st.sidebar.button("刷新项目列表", use_container_width=True):
        st.session_state.projects = load_projects()
        st.rerun()

    return st.session_state.current_project


# -----------------------------
# 模型管理功能
# -----------------------------
def load_models():
    """从API加载模型列表"""
    try:
        # # 实际应用中应调用API
        # response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        # if response.status_code == 200:
        #     return response.json()

        # 模拟数据
        return [
            {"id": "model_001", "name": "交通标志检测模型 v1.0", "project": "proj_001", "created_at": "2023-01-25",
             "metrics": {"mAP50": 0.85, "mAP50-95": 0.65}, "status": "completed"},
            {"id": "model_002", "name": "交通标志检测模型 v1.1", "project": "proj_001", "created_at": "2023-02-05",
             "metrics": {"mAP50": 0.88, "mAP50-95": 0.68}, "status": "completed"},
            {"id": "model_003", "name": "工业零件质检模型 v1.0", "project": "proj_002", "created_at": "2023-02-28",
             "metrics": {"mAP50": 0.78, "mAP50-95": 0.52}, "status": "training", "progress": 65}
        ]
    except:
        return []


def render_model_management():
    """渲染模型管理界面"""
    st.header("模型管理")

    if not st.session_state.models:
        st.session_state.models = load_models()

    # 过滤当前项目的模型
    current_models = [m for m in st.session_state.models
                      if not st.session_state.current_project or m["project"] == st.session_state.current_project["id"]]

    if not current_models:
        st.info("当前项目没有可用的模型")
        return

    # 模型卡片展示
    for model in current_models:
        with st.container():
            st.markdown(f'<div class="model-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="model-header">', unsafe_allow_html=True)
            st.markdown(f'<div class="model-title">{model["name"]}</div>', unsafe_allow_html=True)

            # 状态标签
            status_class = "status-completed" if model["status"] == "completed" else "status-active" if model[
                                                                                                            "status"] == "training" else "status-pending"
            status_text = "已完成" if model["status"] == "completed" else "训练中" if model[
                                                                                          "status"] == "training" else "等待中"
            st.markdown(f'<span class="status-badge {status_class}">{status_text}</span>', unsafe_allow_html=True)
            st.markdown(f'</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="model-meta">项目: {st.session_state.current_project["name"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="model-meta">创建时间: {model["created_at"]}</div>', unsafe_allow_html=True)

            # 训练进度条（如果正在训练）
            if model["status"] == "training" and "progress" in model:
                st.markdown(f'<div class="training-progress">', unsafe_allow_html=True)
                st.progress(model["progress"] / 100)
                st.markdown(f'<div class="model-meta">进度: {model["progress"]}%</div>', unsafe_allow_html=True)
                st.markdown(f'</div>', unsafe_allow_html=True)

            # 模型指标
            if "metrics" in model and model["metrics"]:
                st.markdown("**性能指标:**")
                col1, col2 = st.columns(2)
                col1.metric("mAP50", f"{model['metrics']['mAP50']:.2f}")
                col2.metric("mAP50-95", f"{model['metrics']['mAP50-95']:.2f}")

            # 操作按钮
            st.markdown(f'<div class="model-actions">', unsafe_allow_html=True)
            if st.button(f"使用 {model['name']}", key=f"use_{model['id']}", use_container_width=True):
                st.session_state.current_model = model
                st.success(f"已选择模型: {model['name']}")
            if st.button(f"查看详细信息", key=f"details_{model['id']}", use_container_width=True):
                st.session_state.current_model = model
                st.info(f"模型详情: {model['name']}")
            st.markdown(f'</div>', unsafe_allow_html=True)
            st.markdown(f'</div>', unsafe_allow_html=True)

    # 模型训练控制
    if st.session_state.current_model and st.session_state.current_model["status"] == "training":
        st.subheader("训练控制")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("暂停训练", use_container_width=True):
                # 调用API暂停训练
                st.warning("训练已暂停")
        with col2:
            if st.button("继续训练", use_container_width=True):
                # 调用API继续训练
                st.success("训练已恢复")
        with col3:
            if st.button("停止训练", use_container_width=True):
                # 调用API停止训练
                st.error("训练已停止")


# -----------------------------
# 工具函数
# -----------------------------
def ensure_dirs(base_dir: Path):
    (base_dir / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / "labels").mkdir(parents=True, exist_ok=True)
    (base_dir / "annotations_voc").mkdir(parents=True, exist_ok=True)
    (base_dir / "annotations_json").mkdir(parents=True, exist_ok=True)
    return base_dir


def load_image_bytes_to_pil(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def yxyx_to_yolo(bbox_xyxy, img_w, img_h):
    """xyxy -> YOLO (xc, yc, w, h) normalized to [0,1]"""
    x1, y1, x2, y2 = bbox_xyxy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return xc / img_w, yc / img_h, bw / img_w, bh / img_h


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """YOLO normalized -> xyxy (int)"""
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    x1 = int(round(xc - w / 2))
    y1 = int(round(yc - h / 2))
    x2 = int(round(xc + w / 2))
    y2 = int(round(yc + h / 2))
    return [x1, y1, x2, y2]


def save_yolo_txt(label_path: Path, boxes: List[dict], class_to_id: dict, img_w: int, img_h: int):
    lines = []
    for b in boxes:
        cls = b["label"]
        x1, y1, x2, y2 = b["xyxy"]
        xc, yc, w, h = yxyx_to_yolo([x1, y1, x2, y2], img_w, img_h)
        lines.append(f"{class_to_id[cls]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_classes_txt(classes_path: Path, classes: List[str]):
    with open(classes_path, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")


def load_classes_txt(classes_path: Path) -> List[str]:
    if classes_path.exists():
        return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def save_voc_xml(xml_path: Path, image_path: Path, image_w: int, image_h: int, boxes: List[dict]):
    # VOC 结构
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = str(image_path.parent.name)
    ET.SubElement(annotation, "filename").text = image_path.name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_w)
    ET.SubElement(size, "height").text = str(image_h)
    ET.SubElement(size, "depth").text = "3"
    for b in boxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = b["label"]
        bnd = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = b["xyxy"]
        ET.SubElement(bnd, "xmin").text = str(max(1, x1))
        ET.SubElement(bnd, "ymin").text = str(max(1, y1))
        ET.SubElement(bnd, "xmax").text = str(max(1, x2))
        ET.SubElement(bnd, "ymax").text = str(max(1, y2))
    tree = ET.ElementTree(annotation)
    tree.write(str(xml_path), encoding="utf-8", xml_declaration=True, pretty_print=True)


def save_labelme_like_json(json_path: Path, image_path: Path, image_w: int, image_h: int, boxes: List[dict]):
    data = {
        "version": "5.0.1",
        "flags": {},
        "imagePath": image_path.name,
        "imageHeight": image_h,
        "imageWidth": image_w,
        "shapes": []
    }
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        shape = {
            "label": b["label"],
            "shape_type": "rectangle",
            "points": [[int(x1), int(y1)], [int(x2), int(y2)]],
            "flags": {},
        }
        data["shapes"].append(shape)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yolo_txt(label_path: Path, id_to_class: dict, img_w: int, img_h: int) -> List[dict]:
    """读取 YOLO txt 返回 [{'label': str, 'xyxy': [x1,y1,x2,y2]}]"""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        xyxy = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
        boxes.append({"label": id_to_class.get(cls_id, str(cls_id)), "xyxy": xyxy})
    return boxes


def fit_canvas_size(img_w: int, img_h: int, max_w: int, max_h: int, enlarge_small: bool) -> Tuple[int, int, float]:
    """限制画布最大尺寸，返回 (cw, ch, scale)。支持放大小图片。"""
    scale = min(max_w / img_w, max_h / img_h)
    if not enlarge_small:
        scale = min(scale, 1.0)
    return int(img_w * scale), int(img_h * scale), scale


def save_to_api(image_path: Path, boxes: List[dict], class_to_id: dict, img_w: int, img_h: int):
    """将标注数据保存到后端API"""
    if not st.session_state.api_connected:
        return False

    try:
        # 准备YOLO格式的标注数据
        yolo_lines = []
        for b in boxes:
            cls = b["label"]
            x1, y1, x2, y2 = b["xyxy"]
            xc, yc, w, h = yxyx_to_yolo([x1, y1, x2, y2], img_w, img_h)
            yolo_lines.append(f"{class_to_id[cls]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # 读取图片数据
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()

        # 准备API请求
        files = {
            'image': (image_path.name, img_bytes, 'image/jpeg'),
            'label': ('label.txt', "\n".join(yolo_lines), 'text/plain')
        }
        data = {
            'project_id': st.session_state.current_project["id"] if st.session_state.current_project else None,
            'classes': json.dumps(list(class_to_id.keys()))
        }

        # 发送请求
        response = requests.post(f"{API_BASE_URL}/upload_annotation", files=files, data=data)

        if response.status_code == 200:
            st.session_state.last_save_time = datetime.now().strftime("%H:%M:%S")
            return True
        else:
            st.error(f"保存失败: {response.text}")
            return False

    except Exception as e:
        st.error(f"API连接错误: {str(e)}")
        return False


def load_from_api(project_id: str, image_name: str = None) -> tuple:
    """从后端API加载标注数据"""
    if not st.session_state.api_connected or not project_id:
        return [], None

    try:
        params = {"project_id": project_id}
        if image_name:
            params["image_name"] = image_name

        response = requests.get(f"{API_BASE_URL}/annotations", params=params)

        if response.status_code == 200:
            data = response.json()
            # 转换API返回的数据为本地格式
            boxes = []
            for ann in data.get("annotations", []):
                boxes.append({
                    "label": ann["class"],
                    "xyxy": [ann["x1"], ann["y1"], ann["x2"], ann["y2"]]
                })
            classes = data.get("classes", [])
            return boxes, classes
        else:
            st.warning(f"加载标注数据失败: {response.text}")
            return [], []

    except Exception as e:
        st.warning(f"API连接错误: {str(e)}")
        return [], []


def auto_save_enabled():
    """检查是否启用自动保存"""
    return st.session_state.auto_save and st.session_state.api_connected and st.session_state.current_project


# -----------------------------
# 标注质量检查功能
# -----------------------------
def check_annotation_quality(boxes: List[dict], img_w: int, img_h: int) -> dict:
    """检查标注质量并返回问题报告"""
    report = {
        "total_boxes": len(boxes),
        "small_boxes": 0,
        "large_boxes": 0,
        "overlapping_boxes": 0,
        "invalid_boxes": 0,
        "issues": []
    }

    # 检查无效框
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["xyxy"]
        if x1 >= x2 or y1 >= y2:
            report["invalid_boxes"] += 1
            report["issues"].append(f"标注框 #{i + 1} 无效 (x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2})")

        # 检查小框
        area = (x2 - x1) * (y2 - y1)
        if area < 100:  # 小于100像素²
            report["small_boxes"] += 1
            report["issues"].append(f"标注框 #{i + 1} 太小 (面积: {area} 像素²)")

        # 检查大框
        img_area = img_w * img_h
        if area > 0.8 * img_area:  # 占图像80%以上
            report["large_boxes"] += 1
            report["issues"].append(f"标注框 #{i + 1} 太大 (占图像面积 {area / img_area:.1%})")

    # 检查重叠
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if check_overlap(boxes[i]["xyxy"], boxes[j]["xyxy"]):
                report["overlapping_boxes"] += 1
                report["issues"].append(f"标注框 #{i + 1} 和 #{j + 1} 重叠")

    return report


def check_overlap(box1, box2):
    """检查两个框是否重叠"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    if x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1:
        return False
    return True


def render_annotation_quality_check(boxes, img_w, img_h):
    """渲染标注质量检查结果"""
    if not boxes:
        return

    report = check_annotation_quality(boxes, img_w, img_h)

    # 创建质量报告卡片
    with st.expander("📊 标注质量检查", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总标注数", report["total_boxes"])
        col2.metric("小标注框", report["small_boxes"], delta_color="inverse")
        col3.metric("大标注框", report["large_boxes"], delta_color="inverse")
        col4.metric("重叠标注", report["overlapping_boxes"], delta_color="inverse")

        if report["issues"]:
            st.subheader("发现的问题")
            for issue in report["issues"][:5]:  # 只显示前5个问题
                st.warning(issue)
            if len(report["issues"]) > 5:
                st.caption(f"还有 {len(report['issues']) - 5} 个问题...")

            if st.button("修复常见问题", use_container_width=True):
                # 这里可以添加自动修复逻辑
                st.success("已尝试修复常见问题")
        else:
            st.success("标注质量良好！没有发现问题")


# -----------------------------
# 高级标注功能
# -----------------------------
def render_advanced_annotation_tools():
    """渲染高级标注工具"""
    st.sidebar.header("高级标注工具")

    # 智能标注建议
    with st.sidebar.expander("智能标注建议", expanded=True):
        st.info("基于当前模型的智能标注建议")
        if st.session_state.current_model:
            st.success(f"当前使用模型: {st.session_state.current_model['name']}")
            if st.button("使用模型预测标注", use_container_width=True):
                st.info("正在使用模型生成预测标注...")
                # 这里应该调用API获取预测结果
                time.sleep(1)
                st.success("预测标注已生成！")
        else:
            st.warning("请先选择一个模型")

    # 标注模板
    with st.sidebar.expander("标注模板", expanded=False):
        st.info("快速应用常用标注配置")
        templates = ["车辆检测模板", "行人检测模板", "交通标志模板"]
        selected_template = st.selectbox("选择模板", templates)
        if st.button("应用模板", use_container_width=True):
            # 这里应该加载模板对应的类别
            st.success(f"已应用 {selected_template}")

    # 标注历史
    with st.sidebar.expander("标注历史", expanded=False):
        st.info("最近标注操作历史")
        # 模拟历史记录
        history = [
            {"time": "10:25:30", "action": "添加", "class": "car", "image": "img_001.jpg"},
            {"time": "10:24:15", "action": "删除", "class": "person", "image": "img_002.jpg"},
            {"time": "10:22:40", "action": "修改", "class": "traffic_light", "image": "img_003.jpg"}
        ]
        for item in history[:5]:
            st.markdown(f"**{item['time']}** - {item['action']} `{item['class']}` 在 `{item['image']}`")


# -----------------------------
# 主程序
# -----------------------------
def main():
    # 检查API连接状态
    if check_api_connection():
        api_status = "已连接"
        status_class = "status-connected"
    else:
        api_status = "未连接"
        status_class = "status-disconnected"

    # 渲染顶部状态栏
    st.markdown(
        f"""
        <div class="main-header">
            <h1>✏️ AI标注与训练平台</h1>
            <div class="system-status">
                <span class="system-status-icon {status_class}"></span>
                API状态: {api_status} | 版本: {APP_VERSION}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 用户认证
    render_user_info()

    # 项目选择
    current_project = select_project()

    # 如果没有选择项目，显示提示
    if not current_project:
        st.info("请在左侧选择一个项目开始标注工作")
        st.stop()

    # 创建主选项卡
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 标注工作台",
        "📁 数据集管理",
        "🧠 模型管理",
        "🔍 目标检测"
    ])

    # 基础路径设置
    base_dir = Path("datasets") / current_project["id"]
    ensure_dirs(base_dir)

    # 画布配置
    st.sidebar.header("画布设置")
    max_canvas_width = st.sidebar.slider("最大画布宽度", 400, 1200, 700, 50)
    max_canvas_height = st.sidebar.slider("最大画布高度", 300, 800, 500, 50)
    enlarge_small_images = st.sidebar.checkbox("放大小图片以填充画布", value=True)
    st.sidebar.checkbox("启用自动保存", value=True, key="auto_save_checkbox",
                        on_change=lambda: setattr(st.session_state, 'auto_save', not st.session_state.auto_save))

    # 类别管理
    classes_path = base_dir / "classes.txt"
    if "classes" not in st.session_state:
        # 首先尝试从API加载
        _, classes_from_api = load_from_api(current_project["id"])
        if classes_from_api:
            st.session_state.classes = classes_from_api
        else:
            st.session_state.classes = load_classes_txt(classes_path)

    col_c1, col_c2 = st.sidebar.columns([2, 1])
    with col_c1:
        new_cls = st.text_input("新增类别名", key="new_class_input")
    with col_c2:
        if st.button("添加类别", use_container_width=True, key="add_class_btn"):
            if new_cls and new_cls not in st.session_state.classes:
                st.session_state.classes.append(new_cls)
                save_classes_txt(classes_path, st.session_state.classes)
                # 同时保存到API
                if st.session_state.api_connected:
                    try:
                        requests.post(f"{API_BASE_URL}/update_classes", json={
                            "project_id": current_project["id"],
                            "classes": st.session_state.classes
                        })
                    except:
                        st.warning("类别已本地保存，但API同步失败")
                st.rerun()

    if st.session_state.classes:
        st.sidebar.success(f"类别数：{len(st.session_state.classes)}")
        st.sidebar.write(st.session_state.classes)
    else:
        st.sidebar.warning("尚未添加类别。请添加类别后开始标注。")

    # 选择导出格式
    st.sidebar.header("导出设置")
    export_voc = st.sidebar.checkbox("同时导出 Pascal VOC XML（LabelImg 兼容）", value=True)
    export_json = st.sidebar.checkbox("同时导出 JSON（labelme 风格）", value=False)

    # 高级标注工具
    render_advanced_annotation_tools()

    # 标注工作台
    with tab1:
        st.subheader(f"标注工作台 - {current_project['name']}")

        # 图片来源：目录/上传
        img_tab1, img_tab2 = st.tabs(["项目图片库", "上传新图片"])

        image_files: List[Path] = []
        with img_tab1:
            img_dir = base_dir / "images"
            st.write(f"当前项目图片目录：`{img_dir}`")
            image_files = sorted(
                [p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
            st.write(f"检测到 {len(image_files)} 张图片。")

            # 批量操作
            if len(image_files) > 0:
                col_batch1, col_batch2, col_batch3 = st.columns(3)
                with col_batch1:
                    if st.button("批量标注模式", use_container_width=True):
                        st.session_state.annotation_mode = "batch"
                        st.success("已切换到批量标注模式")
                with col_batch2:
                    if st.button("单张标注模式", use_container_width=True):
                        st.session_state.annotation_mode = "single"
                        st.success("已切换到单张标注模式")
                with col_batch3:
                    st.info(f"当前模式: {st.session_state.annotation_mode}")

        with img_tab2:
            uploads = st.file_uploader("上传图片文件（可多选）", accept_multiple_files=True,
                                       type=["jpg", "jpeg", "png", "bmp"], key="uploader")
            if uploads:
                for up in uploads:
                    im = load_image_bytes_to_pil(up.read())
                    save_path = base_dir / "images" / up.name
                    im.save(save_path)
                st.success(f"已保存 {len(uploads)} 张到项目图片目录。")
                image_files = sorted([p for p in (base_dir / "images").glob("*") if
                                      p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

        if not image_files:
            st.warning("请在上方上传图片或确保项目图片目录中有图片文件。")
            st.stop()

        # 图片索引控制
        total_images = len(image_files)
        st.session_state.current_image_index = st.slider(
            "选择图片",
            0,
            total_images - 1,
            st.session_state.current_image_index,
            format="图片 #%d"
        )

        col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 4, 1])
        with col_nav1:
            if st.button("⬅ 上一张", use_container_width=True):
                st.session_state.current_image_index = max(0, st.session_state.current_image_index - 1)
                st.rerun()
        with col_nav2:
            if st.button("下一张 ➡", use_container_width=True):
                st.session_state.current_image_index = min(total_images - 1, st.session_state.current_image_index + 1)
                st.rerun()
        with col_nav3:
            st.write(
                f"共 {total_images} 张，当前第 {st.session_state.current_image_index + 1} 张：`{image_files[st.session_state.current_image_index].name}`")
        with col_nav4:
            st.metric("进度", f"{st.session_state.current_image_index + 1}/{total_images}")

        # 读取当前图片
        img_path = image_files[st.session_state.current_image_index]
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # 颜色与类别映射
        def palette(i):
            colors = [
                "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
                "#008080", "#e6beff", "#9A6324", "#fffac8", "#800000",
                "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#000000"
            ]
            return colors[i % len(colors)]

        class_to_id = {c: i for i, c in enumerate(st.session_state.classes)}
        id_to_class = {i: c for c, i in class_to_id.items()}

        # 读取已存在的 YOLO 标签（优先从API加载）
        existing_boxes, _ = load_from_api(current_project["id"], img_path.name)
        if not existing_boxes:  # 如果API没有，尝试本地加载
            existing_boxes = load_yolo_txt(base_dir / "labels" / (img_path.stem + ".txt"), id_to_class, img_w, img_h)

        # 画布尺寸与缩放
        canvas_w, canvas_h, scale = fit_canvas_size(img_w, img_h, max_canvas_width, max_canvas_height,
                                                    enlarge_small_images)

        # 当前类别（用于新框）
        current_cls = st.selectbox("当前选择的类别（用于新框）",
                                   options=st.session_state.classes or ["未设置类别"],
                                   index=0 if st.session_state.classes else 0,
                                   key="current_class_selector")

        # 将已有框转换为 canvas 初始对象（xyxy像素 -> 缩放后）
        initial_rects = []
        for b in existing_boxes:
            x1, y1, x2, y2 = b["xyxy"]
            initial_rects.append({
                "type": "rect",
                "left": x1 * scale,
                "top": y1 * scale,
                "width": (x2 - x1) * scale,
                "height": (y2 - y1) * scale,
                "fill": "rgba(0,0,0,0)",
                "stroke": palette(class_to_id.get(b["label"], 0)),
                "strokeWidth": 2,
                "name": b["label"],
            })

        # 标注画布
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color=palette(class_to_id.get(current_cls, 0)) if st.session_state.classes else "#000000",
            background_image=img.resize((canvas_w, canvas_h)),
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect",
            key=f"canvas_{img_path.name}",
            initial_drawing={"objects": initial_rects} if initial_rects else None,
            display_toolbar=True
        )

        # 解析 canvas 的 rect 对象，把缩放坐标还原到原图像素
        boxes_now: List[dict] = []
        if canvas_result and canvas_result.json_data is not None:
            for obj in canvas_result.json_data["objects"]:
                if obj.get("type") != "rect":
                    continue
                left_ = max(0, obj["left"]) / scale
                top_ = max(0, obj["top"]) / scale
                width_ = max(1, obj["width"]) / scale
                height_ = max(1, obj["height"]) / scale
                x1 = int(round(left_))
                y1 = int(round(top_))
                x2 = int(round(left_ + width_))
                y2 = int(round(top_ + height_))
                # 读取对象上的 name 作为类别；若无则用当前选择
                label_ = obj.get("name") or current_cls
                boxes_now.append({"label": label_, "xyxy": [x1, y1, x2, y2]})

        # 标注质量检查
        render_annotation_quality_check(boxes_now, img_w, img_h)

        # 渲染可编辑表格
        st.markdown("#### 已标注框（可编辑标签）")
        if len(boxes_now) == 0:
            st.info("暂无标注框。在左侧图片上拖拽绘制矩形即可。")
        else:
            # 一行一个框，显示 label 下拉、坐标与删除按钮
            to_delete = []
            for i, b in enumerate(boxes_now):
                x1, y1, x2, y2 = b["xyxy"]
                with st.container(border=True):
                    cols = st.columns([2, 2, 2, 2, 2, 1])
                    boxes_now[i]["label"] = cols[0].selectbox("类别",
                                                              options=st.session_state.classes or ["未设置类别"],
                                                              index=class_to_id.get(b["label"], 0) if b[
                                                                                                          "label"] in class_to_id else 0,
                                                              key=f"lbl_{i}")
                    boxes_now[i]["xyxy"][0] = cols[1].number_input("x1", value=int(x1), step=1, key=f"x1_{i}")
                    boxes_now[i]["xyxy"][1] = cols[2].number_input("y1", value=int(y1), step=1, key=f"y1_{i}")
                    boxes_now[i]["xyxy"][2] = cols[3].number_input("x2", value=int(x2), step=1, key=f"x2_{i}")
                    boxes_now[i]["xyxy"][3] = cols[4].number_input("y2", value=int(y2), step=1, key=f"y2_{i}")
                    if cols[5].button("🗑️", key=f"del_{i}"):
                        to_delete.append(i)
            if to_delete:
                boxes_now = [b for i, b in enumerate(boxes_now) if i not in to_delete]
                st.rerun()

        # 保存操作
        st.markdown("---")
        save_col1, save_col2, save_col3, save_col4 = st.columns(4)

        # 保存按钮
        if save_col1.button("💾 保存本张（本地）", use_container_width=True):
            # 保存到本地
            save_classes_txt(classes_path, st.session_state.classes)
            yolo_path = base_dir / "labels" / f"{img_path.stem}.txt"
            save_yolo_txt(yolo_path, boxes_now, class_to_id, img_w, img_h)

            # 可选导出
            if export_voc:
                voc_path = base_dir / "annotations_voc" / f"{img_path.stem}.xml"
                save_voc_xml(voc_path, img_path, img_w, img_h, boxes_now)
            if export_json:
                json_path = base_dir / "annotations_json" / f"{img_path.stem}.json"
                save_labelme_like_json(json_path, img_path, img_w, img_h, boxes_now)

            st.success("已本地保存！")

        if save_col2.button("💾 保存本张（云端）", use_container_width=True):
            if st.session_state.api_connected and st.session_state.current_project:
                # 保存到API
                if save_to_api(img_path, boxes_now, class_to_id, img_w, img_h):
                    st.success("已成功保存到云端！")
                else:
                    st.error("保存到云端失败，请检查连接")
            else:
                st.warning("API未连接或未选择项目，无法保存到云端")

        if save_col3.button("⬅ 保存并上一张", use_container_width=True):
            # 优先保存到云端
            if auto_save_enabled():
                save_to_api(img_path, boxes_now, class_to_id, img_w, img_h)
            else:
                # 本地保存
                save_classes_txt(classes_path, st.session_state.classes)
                yolo_path = base_dir / "labels" / f"{img_path.stem}.txt"
                save_yolo_txt(yolo_path, boxes_now, class_to_id, img_w, img_h)

                if export_voc:
                    voc_path = base_dir / "annotations_voc" / f"{img_path.stem}.xml"
                    save_voc_xml(voc_path, img_path, img_w, img_h, boxes_now)
                if export_json:
                    json_path = base_dir / "annotations_json" / f"{img_path.stem}.json"
                    save_labelme_like_json(json_path, img_path, img_w, img_h, boxes_now)

            st.session_state.current_image_index = max(0, st.session_state.current_image_index - 1)
            st.rerun()

        if save_col4.button("保存并下一张 ➡", use_container_width=True):
            # 优先保存到云端
            if auto_save_enabled():
                save_to_api(img_path, boxes_now, class_to_id, img_w, img_h)
            else:
                # 本地保存
                save_classes_txt(classes_path, st.session_state.classes)
                yolo_path = base_dir / "labels" / f"{img_path.stem}.txt"
                save_yolo_txt(yolo_path, boxes_now, class_to_id, img_w, img_h)

                if export_voc:
                    voc_path = base_dir / "annotations_voc" / f"{img_path.stem}.xml"
                    save_voc_xml(voc_path, img_path, img_w, img_h, boxes_now)
                if export_json:
                    json_path = base_dir / "annotations_json" / f"{img_path.stem}.json"
                    save_labelme_like_json(json_path, img_path, img_w, img_h, boxes_now)

            st.session_state.current_image_index = min(total_images - 1, st.session_state.current_image_index + 1)
            st.rerun()

        # 自动保存状态显示
        if auto_save_enabled():
            if st.session_state.last_save_time:
                st.caption(f"✅ 自动保存已启用 | 最后保存: {st.session_state.last_save_time}")
            else:
                st.caption("✅ 自动保存已启用 | 等待首次保存")
        else:
            st.caption("ℹ️ 自动保存已禁用 | 请手动保存")

    # 数据集管理
    with tab2:
        st.subheader("数据集管理")

        # 数据集概览
        st.markdown("### 数据集概览")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("图片总数", current_project["image_count"])
        with col2:
            st.metric("标注总数", current_project["label_count"])
        with col3:
            st.metric("类别数量", len(st.session_state.classes))

        # 数据集统计
        st.markdown("### 数据集统计")

        # 模拟数据分布
        class_counts = {cls: max(1, int(current_project["label_count"] * (i + 1) / len(st.session_state.classes)))
                        for i, cls in enumerate(st.session_state.classes)}

        # 创建图表
        st.bar_chart(class_counts)

        # 显示详细统计数据
        st.markdown("#### 每个类别的标注数量")
        for cls, count in class_counts.items():
            st.progress(min(count / max(class_counts.values()), 1.0), text=f"{cls}: {count} 个标注")

        # 数据集操作
        st.markdown("### 数据集操作")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("导出数据集", use_container_width=True):
                st.info("正在导出数据集...")
                # 这里应该调用API导出数据集
                time.sleep(1)
                st.success("数据集导出成功！")
        with col2:
            if st.button("划分训练/验证集", use_container_width=True):
                st.info("正在划分数据集...")
                # 这里应该调用API划分数据集
                time.sleep(1)
                st.success("数据集划分成功！训练: 70%, 验证: 30%")
        with col3:
            if st.button("数据增强", use_container_width=True):
                st.info("正在应用数据增强...")
                # 这里应该调用API进行数据增强
                time.sleep(1)
                st.success("数据增强完成！新增 200 张增强图片")

        # 数据集预览
        st.markdown("### 数据集预览")
        preview_count = min(9, len(image_files))
        cols = st.columns(3)
        for i in range(preview_count):
            with cols[i % 3]:
                img_preview = Image.open(image_files[i]).resize((200, 200))
                st.image(img_preview, caption=f"图片 #{i + 1}", use_column_width=True)
                # 显示标注数量
                boxes, _ = load_from_api(current_project["id"], image_files[i].name)
                if not boxes:
                    boxes = load_yolo_txt(base_dir / "labels" / (image_files[i].stem + ".txt"), id_to_class, img_w,
                                          img_h)
                st.caption(f"标注数量: {len(boxes)}")

    # 模型管理
    with tab3:
        render_model_management()

    # 目标检测
    with tab4:
        st.subheader("目标检测")

        if not st.session_state.current_model:
            st.info("请先在模型管理中选择一个模型")
            st.stop()

        st.markdown(f"**当前模型:** {st.session_state.current_model['name']}")
        st.markdown(f"**项目:** {current_project['name']}")

        # 检测配置
        st.markdown("### 检测配置")
        col1, col2, col3 = st.columns(3)
        with col1:
            confidence = st.slider("置信度阈值", 0.0, 1.0, 0.5, 0.05)
        with col2:
            iou = st.slider("IoU阈值", 0.0, 1.0, 0.45, 0.05)
        with col3:
            max_detections = st.number_input("最大检测数", 1, 100, 50, 1)

        # 检测输入
        st.markdown("### 检测输入")
        detect_tab1, detect_tab2 = st.tabs(["使用项目图片", "上传新图片"])

        with detect_tab1:
            st.info("从项目图片库中选择图片进行检测")
            detect_image = st.selectbox("选择图片", [img.name for img in image_files])
            if st.button("执行检测", use_container_width=True):
                st.info("正在执行目标检测...")
                # 这里应该调用API进行检测
                time.sleep(1)
                # 模拟检测结果
                st.success("检测完成！")
                # 显示结果
                result_img = Image.open(image_files[0]).convert("RGB")
                # 在图片上绘制检测框（简化版，实际应使用API返回的检测结果）
                st.image(result_img, caption="检测结果", use_column_width=True)
                # 显示检测统计
                st.markdown("### 检测结果统计")
                st.dataframe({
                    "类别": ["car", "person", "traffic_sign"],
                    "数量": [5, 3, 2],
                    "置信度": [0.92, 0.85, 0.78]
                })

        with detect_tab2:
            detect_upload = st.file_uploader("上传图片进行检测", type=["jpg", "jpeg", "png", "bmp"])
            if detect_upload:
                st.image(detect_upload, caption="上传的图片", use_column_width=True)
                if st.button("检测上传的图片", use_container_width=True):
                    st.info("正在执行目标检测...")
                    # 这里应该调用API进行检测
                    time.sleep(1)
                    # 模拟检测结果
                    st.success("检测完成！")
                    # 显示结果
                    result_img = Image.open(io.BytesIO(detect_upload.read())).convert("RGB")
                    st.image(result_img, caption="检测结果", use_column_width=True)
                    # 显示检测统计
                    st.markdown("### 检测结果统计")
                    st.dataframe({
                        "类别": ["car", "person"],
                        "数量": [2, 1],
                        "置信度": [0.88, 0.76]
                    })

        # 检测历史
        st.markdown("### 检测历史")
        # 模拟历史记录
        history = [
            {"time": "10:25:30", "image": "img_001.jpg", "detections": 8},
            {"time": "09:45:12", "image": "img_002.jpg", "detections": 5},
            {"time": "08:30:45", "image": "img_003.jpg", "detections": 12}
        ]
        for item in history:
            st.markdown(f"**{item['time']}** - `{item['image']}` - 检测到 {item['detections']} 个目标")


# -----------------------------
# 运行应用
# -----------------------------
if __name__ == "__main__":
    main()