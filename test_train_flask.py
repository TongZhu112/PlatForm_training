import requests
import time
import json
from pathlib import Path

BASE_URL = "http://60.168.144.130:8000"

def wait_for_task(task_id):
    """轮询任务状态直到结束，并实时打印进度"""
    start_time = time.time()
    while True:
        try:
            status_response = requests.get(f"{BASE_URL}/task_status/{task_id}", timeout=30)
            status_data = status_response.json()
        except Exception as e:
            print(f"💥 状态查询失败: {e}")
            time.sleep(3)
            continue

        if status_data.get("code") != 0:
            print(f"❌ 查询失败: {status_data.get('message', '未知错误')}")
            break

        data = status_data.get("data", {})
        status = data.get("status", "unknown").lower()
        elapsed_time = time.time() - start_time

        print(f"🔄 任务 {task_id} 状态: {status} (耗时 {elapsed_time:.1f}秒)")

        if status == "success":
            print("🎉 训练完成！")
            print(json.dumps(status_data, indent=2, ensure_ascii=False))  # 输出json格式的接口内容

            # 🎉 训练完成！
            # {
            #   "code": 0,
            #   "data": {
            #     "result": {
            #       "final_metrics": {
            #         "f1_score": 0.4997984130177494,
            #         "mAP50": 0.33166666666666667,
            #         "mAP50-95": 0.2985,
            #         "precision": 0.9983922475442673,
            #         "recall": 0.3333333333333333
            #       },
            #       "project_path": "runs/train/task_002",
            #       "status": "success",
            #       "task_id": "ce8cb5d0-e83a-4df1-af69-2f681d1034f4",
            #       "weights_path": "runs/train/task_002/weights/best.pt"
            #     },
            #     "status": "success",
            #     "task_id": "ce8cb5d0-e83a-4df1-af69-2f681d1034f4"
            #   },
            #   "message": "训练完成"
            
            result_data = data.get("result", {})
            print("🏆 训练结果:")
            print(f"   权重路径: {result_data.get('weights_path', 'N/A')}")
            print(f"   项目路径: {result_data.get('project_path', 'N/A')}")
            metrics = result_data.get("final_metrics", {})
            print("📈 性能指标:")
            print(f"   mAP50: {metrics.get('mAP50', 0):.4f}")
            print(f"   mAP50-95: {metrics.get('mAP50-95', 0):.4f}")
            print(f"   Precision: {metrics.get('precision', 0):.4f}")
            print(f"   Recall: {metrics.get('recall', 0):.4f}")
            print(f"   F1-score: {metrics.get('f1_score', 0):.4f}")
            break

        elif status == "failure":
            print(f"❌ 训练失败: {data.get('error', '未知错误')}")
            break

        elif status == "progress":
            progress_info = data.get("result", {})
            stage = progress_info.get("stage", "unknown")
            current_epoch = progress_info.get("current_epoch", 0)
            total_epochs = progress_info.get("total_epochs", 0)
            progress = progress_info.get("progress", "0%")
            print(f"   阶段: {stage}")
            print(f"   进度: {current_epoch}/{total_epochs} 轮 ({progress})")

        elif status == "pending":
            print(f"   当前状态详情: {status_data.get('message', '任务等待中')}")

        else:
            print(f"   未知状态: {status}")

        time.sleep(3)


def first_training():
    """第一次正常训练"""
    training_data = {
        "DatasetUrl": "/home/tongzhu112/data/M8_Yolo.zip",  # 改为你的本地路径
        "TaskId": "task_001",
        "Epochs": 20,
        "ImageSize": 640,
        "BatchSize": 16,
        "Device": "0",
        "PreTrainingModel": "yolov8s.pt",
        "IncrementTraining": False
    }

    response = requests.post(f"{BASE_URL}/train", json=training_data, headers={"Content-Type": "application/json"})
    result = response.json()
    print("🚀 第一次训练启动:", result)
    # 获取训练进度
        
    return result["data"]["TaskId"]


def incremental_training(base_task_id):
    """基于第一次训练的增量训练"""
    training_data = {
        "DatasetUrl": "/home/tongzhu112/data/M8_Yolo_IncrementTraining.zip",  # 数据集同样可以更新
        "TaskId": "task_002",
        "Epochs": 20,
        "ImageSize": 640,
        "BatchSize": 16,
        "Device": "0",
        "PreTrainingModel": "yolov8s.pt",
        "IncrementTraining": True,
        "BaseTaskId": base_task_id
    }

    response = requests.post(f"{BASE_URL}/train", json=training_data, headers={"Content-Type": "application/json"})
    result = response.json()
    print("🚀 第二次增量训练启动:", result)
    
    # 获取训练进度
    
    
    return result["data"]["TaskId"]


if __name__ == "__main__":
    print("🔬 YOLOv8 训练测试 (Flask)")
    print("=" * 50)

    # 第一次训练
    first_id = first_training()
    first_result = wait_for_task(first_id)

    # 第二次增量训练
    second_id = incremental_training("task_001")
    second_result = wait_for_task(second_id)

# # ********************* 客户端代码示例，最简单的接口测试代码 *******************************
# import requests
# 
# base_url = "http://60.168.144.138:8000"
# training_data = {
#     # "data_url": "http://60.168.144.134:8000/home/tongzhu112/data/M8_Yolo.zip",
#     "DatasetUrl": "http://60.168.144.138:8000/home/tongzhu112/data/M8_Yolo.zip",
#     "TaskId": "1234567",
#     "Epochs": 20,
#     "ImageSize": 640,
#     "BatchSize": 16,
#     "Device": "0",
#     # "project": "runs/train",
#     # "name": "1234567",
#     "IncrementTraining":True,
#     "PreTrainingModel": "yolov8s.pt"
# }
# 
# response = requests.post(f"{base_url}/train", json=training_data)
# print(response.json())
