# 📘 YOLOv8 API 接口文档

## 基本信息
- **服务地址**: `http://60.168.144.127:8000`
- **请求格式**: `application/json`
- **返回格式**: `application/json`
- **任务机制**:  
  - 推理和训练均为 **异步任务**  
  - 返回 `task_id`  
  - 前端需通过 `/task_status/{task_id}` 轮询获取任务状态与结果  

---

## 1. 健康检查接口

### 请求
```
GET http://60.168.144.127:8000/
```

### 返回示例
```json
{
  "message": "YOLOv8 API服务运行正常",
  "status": "healthy"
}
```

---

## 2. 执行推理任务

### 请求
```
POST http://60.168.144.127:8000/inference
Content-Type: application/json
```

### 参数说明
| 参数名        | 类型   | 是否必填 | 说明 |
|---------------|--------|----------|------|
| model_path    | string | ✅       | 模型路径 (`.pt` 文件)，如：`models/yolov8n.pt` |
| image_path    | string | ✅       | 待检测图片路径，如：`uploads/test.jpg` |
| confidence    | float  | ❌       | 置信度阈值，范围 `[0,1]`，默认 `0.25` |
| save_output   | bool   | ❌       | 是否保存结果图，默认 `true` |

### 请求示例
```json
{
  "model_path": "models/yolov8n.pt",
  "image_path": "uploads/sample.jpg",
  "confidence": 0.4,
  "save_output": true
}
```

### 返回示例
```json
{
  "task_id": "4df01b84-67db-43c2-87f8-9dc9ef2cfc58",
  "message": "推理任务已启动",
  "status_url": "/task_status/4df01b84-67db-43c2-87f8-9dc9ef2cfc58"
}
```

---

## 3. 执行训练任务

### 请求
```
POST http://60.168.144.127:8000/train
Content-Type: application/json
```

### 参数说明
| 参数名          | 类型   | 是否必填 | 说明 |
|-----------------|--------|----------|------|
| model_path      | string | ✅       | 预训练模型路径，例如：`models/yolov8n.pt` |
| dataset_config  | string | ✅       | 数据集配置文件路径 (`.yaml`) |
| epochs          | int    | ❌       | 训练轮数，默认 `100` |
| imgsz           | int    | ❌       | 图像尺寸，必须为 32 的倍数，默认 `640` |
| batch_size      | int    | ❌       | 批次大小，默认 `16` |
| device          | string | ❌       | 设备，`"0"` 表示 GPU0，`""` 表示 CPU |
| project         | string | ❌       | 训练结果保存目录，默认 `"runs/train"` |
| name            | string | ❌       | 实验名称，默认 `"exp"` |

### 请求示例
```json
{
  "model_path": "models/yolov8n.pt",
  "dataset_config": "datasets/mydata.yaml",
  "epochs": 50,
  "imgsz": 640,
  "batch_size": 16,
  "device": "0",
  "project": "runs/train",
  "name": "exp1"
}
```

### 返回示例
```json
{
  "task_id": "7bda6c7c-2c6d-4cb0-b6f8-75d9f9c0e3d2",
  "message": "训练任务已启动",
  "status_url": "/task_status/7bda6c7c-2c6d-4cb0-b6f8-75d9f9c0e3d2"
}
```

---

## 4. 查询任务状态

### 请求
```
GET http://60.168.144.127:8000/task_status/<task_id>
```

### 状态说明
- `pending`: 任务等待中
- `progress`: 任务进行中
- `success`: 任务完成
- `failure`: 任务失败

### 返回示例 - 推理成功
```json
{
  "task_id": "4df01b84-67db-43c2-87f8-9dc9ef2cfc58",
  "status": "success",
  "result": {
    "total_detections": 2,
    "classes": ["person", "dog"],
    "detections": [
      {
        "class_name": "person",
        "confidence": 0.92,
        "bbox": {"x1": 34, "y1": 58, "x2": 186, "y2": 305}
      },
      {
        "class_name": "dog",
        "confidence": 0.88,
        "bbox": {"x1": 200, "y1": 120, "x2": 360, "y2": 300}
      }
    ],
    "output_image_path": "runs/detect/predict/sample.jpg",
    "task_id": "4df01b84-67db-43c2-87f8-9dc9ef2cfc58"
  }
}
```

### 返回示例 - 训练任务进度
```json
{
  "task_id": "7bda6c7c-2c6d-4cb0-b6f8-75d9f9c0e3d2",
  "status": "progress",
  "progress": {
    "stage": "training",
    "current_epoch": 10,
    "total_epochs": 50,
    "progress": 20.0,
    "metrics": {
      "epoch": 10,
      "loss": 0.345
    }
  }
}
```

### 返回示例 - 任务失败
```json
{
  "task_id": "xxxxxx",
  "status": "failure",
  "error": "模型文件不存在: models/invalid.pt"
}
```

---

## 5. 获取可用模型

### 请求
```
GET http://60.168.144.127:8000/models
```

### 返回示例
```json
{
  "models": [
    "models/yolov8n.pt",
    "models/yolov8s.pt"
  ]
}
```

---

## ⚠️ 注意事项
1. 图片和数据集文件需预先上传到服务器 `uploads/`、`datasets/` 目录下。  
2. 所有推理/训练任务都是异步执行，必须通过 `/task_status/{task_id}` 查询结果。  
3. 训练和推理结果默认保存在 `runs/` 目录下。  
