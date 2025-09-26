import io
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from lxml import etree as ET

# NEW: Define constants for better maintainability
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}


# -----------------------------
# 工具函数
# -----------------------------
def ensure_dirs(base_dir: Path):
    """确保所有必要的子目录都存在。"""
    (base_dir / "images").mkdir(parents=True, exist_ok=True)
    (base_dir / "labels").mkdir(parents=True, exist_ok=True)
    (base_dir / "annotations_voc").mkdir(parents=True, exist_ok=True)
    (base_dir / "annotations_json").mkdir(parents=True, exist_ok=True)
    return base_dir


def load_image_bytes_to_pil(file_bytes: bytes) -> Image.Image:
    """从字节流加载图片并转换为 RGB 格式的 PIL Image。"""
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


# OPTIMIZED: Renamed function for clarity
def xyxy_to_yolo(bbox_xyxy: List[int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    将 [x1, y1, x2, y2] 格式的边界框转换为 YOLO 格式 (xc, yc, w, h)，并归一化。
    """
    x1, y1, x2, y2 = bbox_xyxy

    # 确保坐标在图像范围内
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

    # 确保 x1 < x2, y1 < y2
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    bw = x_max - x_min
    bh = y_max - y_min
    xc = x_min + bw / 2
    yc = y_min + bh / 2

    return xc / img_w, yc / img_h, bw / img_w, bh / img_h


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> List[int]:
    """将归一化的 YOLO 格式边界框转换为 [x1, y1, x2, y2] 像素坐标。"""
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    x1 = int(round(xc - w / 2))
    y1 = int(round(yc - h / 2))
    x2 = int(round(xc + w / 2))
    y2 = int(round(yc + h / 2))

    # NEW: 确保坐标在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)

    return [x1, y1, x2, y2]


def save_yolo_txt(label_path: Path, boxes: List[dict], class_to_id: dict, img_w: int, img_h: int):
    """将标注框保存为 YOLO txt 格式。"""
    lines = []
    for b in boxes:
        cls = b["label"]
        if cls not in class_to_id:
            continue  # Skip if class is not in the list
        xc, yc, w, h = xyxy_to_yolo(b["xyxy"], img_w, img_h)
        lines.append(f"{class_to_id[cls]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")


def save_classes_txt(classes_path: Path, classes: List[str]):
    """保存类别列表到 classes.txt。"""
    classes_path.write_text("\n".join(classes), encoding="utf-8")


def load_classes_txt(classes_path: Path) -> List[str]:
    """从 classes.txt 加载类别列表。"""
    if classes_path.exists():
        return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def save_voc_xml(xml_path: Path, image_path: Path, image_w: int, image_h: int, boxes: List[dict]):
    """将标注框保存为 Pascal VOC XML 格式。"""
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
        ET.SubElement(bnd, "xmin").text = str(max(1, min(x1, x2)))
        ET.SubElement(bnd, "ymin").text = str(max(1, min(y1, y2)))
        ET.SubElement(bnd, "xmax").text = str(max(1, max(x1, x2)))
        ET.SubElement(bnd, "ymax").text = str(max(1, max(y1, y2)))

    tree = ET.ElementTree(annotation)
    tree.write(str(xml_path), encoding="utf-8", xml_declaration=True, pretty_print=True)


def save_labelme_like_json(json_path: Path, image_path: Path, image_w: int, image_h: int, boxes: List[dict]):
    """将标注框保存为类 LabelMe 的 JSON 格式。"""
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
            "points": [[float(min(x1, x2)), float(min(y1, y2))], [float(max(x1, x2)), float(max(y1, y2))]],
            "flags": {},
        }
        data["shapes"].append(shape)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yolo_txt(label_path: Path, id_to_class: dict, img_w: int, img_h: int) -> List[dict]:
    """从 YOLO txt 加载标注框。"""
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
        boxes.append({"label": id_to_class.get(cls_id, f"class_{cls_id}"), "xyxy": xyxy})
    return boxes


def fit_canvas_size(img_w: int, img_h: int, max_w: int, max_h: int, enlarge_small: bool) -> Tuple[int, int, float]:
    """计算适应画布的尺寸和缩放比例。"""
    scale = min(max_w / img_w, max_h / img_h)
    if not enlarge_small:
        scale = min(scale, 1.0)
    return int(img_w * scale), int(img_h * scale), scale


# NEW: Refactored saving logic to a single function
def save_annotations(base_dir: Path, img_path: Path, boxes: List[dict], classes: List[str], img_w: int, img_h: int,
                     export_voc: bool, export_json: bool):
    """保存当前图片的所有格式标注。"""
    if not classes:
        st.error("无法保存，请先在侧边栏添加至少一个类别。")
        return False

    class_to_id = {c: i for i, c in enumerate(classes)}

    # 保存 classes.txt
    save_classes_txt(base_dir / "classes.txt", classes)

    # 保存 YOLO .txt
    yolo_path = base_dir / "labels" / f"{img_path.stem}.txt"
    save_yolo_txt(yolo_path, boxes, class_to_id, img_w, img_h)

    # 可选导出
    if export_voc:
        voc_path = base_dir / "annotations_voc" / f"{img_path.stem}.xml"
        save_voc_xml(voc_path, img_path, img_w, img_h, boxes)
    if export_json:
        json_path = base_dir / "annotations_json" / f"{img_path.stem}.json"
        save_labelme_like_json(json_path, img_path, img_w, img_h, boxes)

    st.toast(f"已保存标注: {img_path.name}", icon="💾")
    return True


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="YOLO 在线标注工具", layout="wide")

# 注入 CSS
st.markdown("""
<style>
[data-testid="column"]:first-child { display: flex; justify-content: center; align-items: flex-start; }
.stCanvas { border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# 初始化 session_state
if "classes" not in st.session_state:
    st.session_state.classes = []
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0
if "image_files" not in st.session_state:
    st.session_state.image_files = []
if "current_img" not in st.session_state:
    st.session_state.current_img = None
if "current_boxes" not in st.session_state:
    st.session_state.current_boxes = []

# --- 侧边栏 ---
with st.sidebar:
    st.header("数据集与类别")
    base_dir_str = st.text_input("数据集根目录", "dataset")
    base_dir = Path(base_dir_str).resolve()
    ensure_dirs(base_dir)

    st.header("画布设置")
    max_canvas_width = st.number_input("最大画布宽度 (px)", 400, 2000, 700, 100)
    max_canvas_height = st.number_input("最大画布高度 (px)", 300, 2000, 500, 100)
    enlarge_small_images = st.checkbox("放大小图片以填充画布", True)

    st.header("类别管理")
    classes_path = base_dir / "classes.txt"
    # 仅在应用启动时加载一次
    if not st.session_state.classes:
        st.session_state.classes = load_classes_txt(classes_path)

    col_c1, col_c2 = st.columns([2, 1])
    new_cls = col_c1.text_input("新增类别名", label_visibility="collapsed", placeholder="输入新类别后点击添加")
    if col_c2.button("添加", use_container_width=True):
        if new_cls and new_cls not in st.session_state.classes:
            st.session_state.classes.append(new_cls)
            save_classes_txt(classes_path, st.session_state.classes)
            st.rerun()

    if st.session_state.classes:
        st.success(f"当前类别数：{len(st.session_state.classes)}")
        st.write(st.session_state.classes)
    else:
        st.info("尚未添加类别，请在上方添加。")

    export_voc = st.checkbox("同时导出 Pascal VOC XML", True)
    export_json = st.checkbox("同时导出类 LabelMe JSON", False)
    st.markdown("---")

# --- 主界面 ---

# 图片来源
tab1, tab2 = st.tabs(["从目录读取", "上传图片（保存到 images/）"])
with tab1:
    img_dir = base_dir / "images"
    st.write(f"当前图片目录：`{img_dir}`")
    st.session_state.image_files = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in SUPPORTED_IMAGE_FORMATS])

with tab2:
    uploads = st.file_uploader("上传图片文件", accept_multiple_files=True, type=list(SUPPORTED_IMAGE_FORMATS))
    if uploads:
        for up in uploads:
            save_path = base_dir / "images" / up.name
            with open(save_path, "wb") as f:
                f.write(up.getbuffer())
        st.success(f"已保存 {len(uploads)} 张图片。")
        st.session_state.image_files = sorted(
            [p for p in (base_dir / "images").glob("*") if p.suffix.lower() in SUPPORTED_IMAGE_FORMATS])
        st.rerun()

if not st.session_state.image_files:
    st.warning("请在左侧设置数据集目录，并在上方『上传图片』或将图片手动放入 `dataset/images/` 目录。")
    st.stop()

st.session_state.idx = min(st.session_state.idx, len(st.session_state.image_files) - 1)


def go_prev():
    st.session_state.idx = (st.session_state.idx - 1) % len(st.session_state.image_files)
    st.session_state.canvas_version += 1
    st.session_state.current_img = None
    st.toast("加载上一张图片", icon="🔄")


def go_next():
    st.session_state.idx = (st.session_state.idx + 1) % len(st.session_state.image_files)
    st.session_state.canvas_version += 1
    st.session_state.current_img = None
    st.toast("加载下一张图片", icon="🔄")


col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 6])
col_nav1.button("⬅ 上一张", use_container_width=True, on_click=go_prev)
col_nav2.button("下一张 ➡", use_container_width=True, on_click=go_next)

col_nav3.write(
    f"共 {len(st.session_state.image_files)} 张，当前第 {st.session_state.idx + 1} 张：`{st.session_state.image_files[st.session_state.idx].name}`")

labeled_count = sum((base_dir / "labels" / (p.stem + ".txt")).exists() for p in st.session_state.image_files)
st.progress(labeled_count / len(st.session_state.image_files) if st.session_state.image_files else 0)
st.caption(f"标注进度: {labeled_count} / {len(st.session_state.image_files)}")

# --- 标注核心区 ---
img_path = st.session_state.image_files[st.session_state.idx]
img = Image.open(img_path).convert("RGB")
img_w, img_h = img.size

class_to_id = {c: i for i, c in enumerate(st.session_state.classes)}
id_to_class = {i: c for c, i in class_to_id.items()}


def palette(i):
    colors = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
              "#fabebe", "#008080", "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808080", "#ffd8b1",
              "#000075", "#808080"]
    return colors[i % len(colors)]


def palette_fill(i):
    fills = ["rgba(230,25,75,0.2)", "rgba(60,180,75,0.2)", "rgba(255,225,25,0.2)", "rgba(67,99,216,0.2)",
             "rgba(245,130,49,0.2)", "rgba(145,30,180,0.2)", "rgba(70,240,240,0.2)", "rgba(240,50,230,0.2)",
             "rgba(188,246,12,0.2)",
             "rgba(250,190,190,0.2)", "rgba(0,128,128,0.2)", "rgba(230,190,255,0.2)", "rgba(154,99,36,0.2)",
             "rgba(255,250,200,0.2)", "rgba(128,0,0,0.2)", "rgba(170,255,195,0.2)", "rgba(128,128,0,0.2)",
             "rgba(255,216,177,0.2)",
             "rgba(0,0,117,0.2)", "rgba(128,128,128,0.2)"]
    return fills[i % len(fills)]


color_to_class = {palette(i): c for i, c in id_to_class.items()}

if st.session_state.current_img != img_path.name:
    label_path = base_dir / "labels" / (img_path.stem + ".txt")
    st.session_state.current_boxes = load_yolo_txt(label_path, id_to_class, img_w, img_h)
    st.session_state.current_img = img_path.name
    st.session_state.canvas_version += 1

canvas_w, canvas_h, scale = fit_canvas_size(img_w, img_h, max_canvas_width, max_canvas_height, enlarge_small_images)

st.markdown("### 标注区")
left, right = st.columns([3, 2])

with left:
    if not st.session_state.classes:
        st.error("请先在左侧边栏添加类别才能开始标注！")
        st.stop()

    current_cls = st.selectbox("当前选择的类别（用于新框）", options=st.session_state.classes, index=0)
    drawing_mode_option = st.selectbox("操作模式", ["绘制新矩形", "编辑现有矩形"])
    drawing_mode = "rect" if drawing_mode_option == "绘制新矩形" else "transform"
    stroke_color = palette(class_to_id.get(current_cls, 0))

    initial_rects = []
    # FIX: 确保画布始终使用 session state 中的最新数据
    for b in st.session_state.current_boxes:
        x1, y1, x2, y2 = b["xyxy"]
        initial_rects.append({
            "type": "rect", "left": min(x1, x2) * scale, "top": min(y1, y2) * scale,
            "width": abs(x2 - x1) * scale, "height": abs(y2 - y1) * scale,
            "fill": palette_fill(class_to_id.get(b["label"], 0)),
            "stroke": palette(class_to_id.get(b["label"], 0)), "strokeWidth": 2,
        })

    canvas_key = f"canvas_{st.session_state.current_img}_{st.session_state.canvas_version}"
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)", stroke_width=2, stroke_color=stroke_color,
        background_image=img.resize((canvas_w, canvas_h)),
        update_streamlit=True, height=canvas_h, width=canvas_w,
        drawing_mode=drawing_mode, key=canvas_key,
        initial_drawing={"objects": initial_rects}
    )

    # FIX: 当画布数据有变化时，立即更新 session state
    if canvas_result and canvas_result.json_data and canvas_result.json_data["objects"]:
        boxes_from_canvas = []
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] != "rect": continue
            x1 = obj["left"] / scale
            y1 = obj["top"] / scale
            x2 = (obj["left"] + obj["width"]) / scale
            y2 = (obj["top"] + obj["height"]) / scale

            label = color_to_class.get(obj.get("stroke"), current_cls)
            boxes_from_canvas.append({"label": label, "xyxy": [int(x1), int(y1), int(x2), int(y2)]})

        # 仅当实际数据有变化时才更新，防止不必要的重新渲染
        if boxes_from_canvas != st.session_state.current_boxes:
            st.session_state.current_boxes = boxes_from_canvas

with right:
    st.markdown("#### 已标注框（实时编辑）")
    st.caption("在左侧画布拖拽即可实时更新数据。")

    if not st.session_state.current_boxes:
        st.info("暂无标注框。在左侧图片上拖拽绘制新框。")
    else:
        img_key_prefix = img_path.stem.replace(".", "_")

        # FIX: 移除 st.form，实现实时更新
        for i, b in enumerate(st.session_state.current_boxes):
            with st.container(border=True):
                cols = st.columns([2, 2, 2, 2, 2, 1])
                current_idx = st.session_state.classes.index(b["label"]) if b[
                                                                                "label"] in st.session_state.classes else 0
                new_label = cols[0].selectbox("类别", options=st.session_state.classes, index=current_idx,
                                              key=f"{img_key_prefix}_lbl_{i}")
                new_x1 = cols[1].number_input("x1", value=b["xyxy"][0], key=f"{img_key_prefix}_x1_{i}")
                new_y1 = cols[2].number_input("y1", value=b["xyxy"][1], key=f"{img_key_prefix}_y1_{i}")
                new_x2 = cols[3].number_input("x2", value=b["xyxy"][2], key=f"{img_key_prefix}_x2_{i}")
                new_y2 = cols[4].number_input("y2", value=b["xyxy"][3], key=f"{img_key_prefix}_y2_{i}")

                # NEW: 为每个框添加删除按钮
                if cols[5].button("删", key=f"{img_key_prefix}_del_{i}"):
                    st.session_state.current_boxes.pop(i)
                    st.session_state.canvas_version += 1
                    st.toast(f"已删除第 {i + 1} 个标注", icon="🗑️")
                    st.rerun()

                # NEW: 监听输入框变化，并更新 session_state
                if new_label != b["label"] or new_x1 != b["xyxy"][0] or new_y1 != b["xyxy"][1] or new_x2 != b["xyxy"][
                    2] or new_y2 != b["xyxy"][3]:
                    st.session_state.current_boxes[i]["label"] = new_label
                    # 自动修正坐标
                    x1_val, x2_val = min(new_x1, new_x2), max(new_x1, new_x2)
                    y1_val, y2_val = min(new_y1, new_y2), max(new_y1, new_y2)
                    st.session_state.current_boxes[i]["xyxy"] = [x1_val, y1_val, x2_val, y2_val]
                    st.session_state.canvas_version += 1  # 强制画布刷新

    if st.button("❌ 清空当前所有标注", use_container_width=True, type="secondary"):
        st.session_state.current_boxes = []
        st.session_state.canvas_version += 1
        st.toast("已清空所有标注", icon="🗑️")
        st.rerun()

st.markdown("---")

save_col1, save_col2, save_col3 = st.columns(3)
# FIX: 保存时使用 session_state 中的最新数据
if save_col1.button("💾 保存本张", use_container_width=True, type="primary"):
    save_annotations(base_dir, img_path, st.session_state.current_boxes, st.session_state.classes, img_w, img_h,
                     export_voc, export_json)

if save_col2.button("⬅️ 保存并上一张", use_container_width=True):
    if save_annotations(base_dir, img_path, st.session_state.current_boxes, st.session_state.classes, img_w, img_h,
                        export_voc, export_json):
        go_prev()
        st.rerun()

if save_col3.button("保存并下一张 ➡️", use_container_width=True):
    if save_annotations(base_dir, img_path, st.session_state.current_boxes, st.session_state.classes, img_w, img_h,
                        export_voc, export_json):
        go_next()
        st.rerun()

st.caption("提示：在'编辑'模式下，选中画布中的框后按键盘 `Delete` 键也可删除。")