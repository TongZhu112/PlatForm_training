# 任务状态查询接口文档

## 接口概述
- **URL**: `/task_status/<task_id>`
- **方法**: `GET`
- **描述**: 查询 YOLOv8 训练任务的状态，返回任务的当前状态、进度信息或最终训练结果（包括性能指标）。
- **Content-Type**: `application/json`

## 请求参数
| 参数名  | 类型   | 必填 | 描述                     | 示例                             |
|---------|--------|------|--------------------------|----------------------------------|
| task_id | String | 是   | 训练任务的唯一标识符      | `e125f230-d11f-4edc-9e37-1c58a9cedf55` |

## 返回格式
返回数据为 JSON 格式，包含以下字段：

| 字段名       | 类型   | 描述                                                                 |
|--------------|--------|----------------------------------------------------------------------|
| code         | Integer| 返回码，`0` 表示成功，`1` 表示失败                                    |
| message      | String | 状态描述信息（如 "任务等待中"、"正在训练"、"训练完成"）               |
| data         | Object | 任务详细信息，包含任务状态和相关数据                                  |
| data.task_id | String | 任务的唯一标识符，与请求中的 `task_id` 一致                           |
| data.status  | String | 任务状态，可能值：`pending`, `progress`, `success`, `failure`          |
| data.result  | Object | 任务进度或结果信息，仅在 `status` 为 `progress` 或 `success` 时存在   |
| data.error   | String | 错误信息，仅在 `status` 为 `failure` 时存在                           |

### `data.result` 字段（当 `status` 为 `progress` 或 `success` 时）
- **当 `status` 为 `progress` 时**：

  | 字段名         | 类型   | 描述                           | 示例         |
  |----------------|--------|--------------------------------|--------------|
  | stage          | String | 当前训练阶段                   | `training`   |
  | current_epoch  | Integer| 当前训练轮次                   | `5`          |
  | total_epochs   | Integer| 总训练轮次                     | `20`         |
  | progress       | String | 训练进度百分比（带 `%` 符号）  | `25.0%`      |

- **当 `status` 为 `success` 时**：

  | 字段名         | 类型   | 描述                           | 示例                                |
  |----------------|--------|--------------------------------|-------------------------------------|
  | status         | String | 任务状态（固定为 `success`）   | `success`                           |
  | task_id        | String | 任务ID                         | `e125f230-d11f-4edc-9e37-1c58a9cedf55` |
  | weights_path   | String | 训练后模型权重文件路径         | `runs/train/1234567/weights/best.pt` |
  | project_path   | String | 训练输出项目目录               | `runs/train/1234567`                |
  | final_metrics  | Object | 训练完成的性能指标             | 见下表                              |

  - **final_metrics 子字段**：
  
    | 字段名      | 类型   | 描述                           | 示例      |
    |-------------|--------|--------------------------------|-----------|
    | mAP50       | Float  | mAP@0.5（平均精度，IoU=0.5）   | `0.3317`  |
    | mAP50-95    | Float  | mAP@0.5:0.95（平均精度范围）   | `0.2322`  |
    | precision   | Float  | 精确率                         | `0.9957`  |
    | recall      | Float  | 召回率                         | `0.3333`  |
    | f1_score    | Float  | F1 分数                        | `0.4995`  |

### `data.error` 字段（当 `status` 为 `failure` 时）

| 字段名 | 类型   | 描述                           | 示例                     |
|--------|--------|--------------------------------|--------------------------|
| error  | String | 错误描述信息                   | `模型文件不存在`          |

## HTTP 状态码
- `200`: 请求成功（任务状态为 `pending`、`progress` 或 `success`）
- `500`: 请求失败（任务状态为 `failure` 或其他异常）

## 返回示例

### 1. 任务等待中（pending）
```json
{
  "code": 0,
  "message": "任务等待中",
  "data": {
    "task_id": "e125f230-d11f-4edc-9e37-1c58a9cedf55",
    "status": "pending"
  }
}
```

### 2. 任务进行中（progress）
```json
{
  "code": 0,
  "message": "正在训练",
  "data": {
    "task_id": "e125f230-d11f-4edc-9e37-1c58a9cedf55",
    "status": "progress",
    "result": {
      "stage": "training",
      "current_epoch": 5,
      "total_epochs": 20,
      "progress": "25.0%"
    }
  }
}
```

### 3. 任务成功（success）
```json
{
  "code": 0,
  "message": "训练完成",
  "data": {
    "task_id": "e125f230-d11f-4edc-9e37-1c58a9cedf55",
    "status": "success",
    "result": {
      "status": "success",
      "task_id": "e125f230-d11f-4edc-9e37-1c58a9cedf55",
      "weights_path": "runs/train/1234567/weights/best.pt",
      "project_path": "runs/train/1234567",
      "final_metrics": {
        "mAP50": 0.3317,
        "mAP50-95": 0.2322,
        "precision": 0.9957,
        "recall": 0.3333,
        "f1_score": 0.4995
      }
    }
  }
}
```

### 4. 任务失败（failure）
```json
{
  "code": 1,
  "message": "训练失败",
  "data": {
    "task_id": "e125f230-d11f-4edc-9e37-1c58a9cedf55",
    "status": "failure",
    "error": "模型文件不存在"
  }
}
```

## 使用说明
1. **调用频率**：建议每 3-5 秒轮询一次 `/task_status/<task_id>` 接口，以获取最新的任务状态和进度。
2. **进度展示**：
   - 当 `data.status` 为 `progress` 时，前端可使用 `data.result.progress` 显示进度条（例如 `25.0%`）。
   - 当 `data.status` 为 `success` 时，前端可展示 `data.result.final_metrics` 中的性能指标（如 mAP50、Precision 等）。
3. **错误处理**：
   - 检查 `code` 字段，`code=1` 表示失败，需展示 `data.error` 的错误信息。
   - 若 HTTP 状态码为 500，需提示用户服务器异常。
4. **超时处理**：建议前端设置最大等待时间（如 1 小时），避免无限轮询。

## 注意事项
- 确保 `task_id` 有效且由 `/train` 接口返回。
- 如果数据集路径或模型文件路径错误，可能导致 `failure` 状态，需检查 `data.error` 信息。
- 性能指标（如 mAP50、mAP50-95）仅在 `success` 状态下返回，且值可能因数据集和训练配置而异。