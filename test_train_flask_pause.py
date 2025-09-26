# test_train_flask.py
import requests
import time
import json
from pathlib import Path

BASE_URL = "http://60.168.144.131:8000"

def wait_for_task(task_id, auto_demo=False, auto_stop=False):
    """轮询任务状态直到结束，并实时打印进度"""
    start_time = time.time()
    paused_once = False
    stopped_once = False
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
            # print(json.dumps(status_data, indent=2, ensure_ascii=False))  # 输出json格式的接口内容
            
            progress_info = data.get("result", {})
            stage = progress_info.get("stage", "unknown")
            current_epoch = progress_info.get("current_epoch", 0)
            total_epochs = progress_info.get("total_epochs", 0)
            progress = progress_info.get("progress", "0%")
            print(f"   阶段: {stage}")
            print(f"   进度: {current_epoch}/{total_epochs} 轮 ({progress})")

            # 自动演示暂停/恢复
            if auto_demo and not paused_once and current_epoch >= 2:
                pause_task(task_id)
                time.sleep(10)
                resume_task(task_id)
                paused_once = True

            # 自动演示停止
            if auto_stop and not stopped_once and current_epoch >= 3:
                stop_task(task_id)
                stopped_once = True

        elif status == "pending":
            print(f"   当前状态详情: {status_data.get('message', '任务等待中')}")

        else:
            print(f"   未知状态: {status}")

        time.sleep(3)

def pause_task(task_id):
    resp = requests.post(f"{BASE_URL}/pause/{task_id}")
    print(f"⏸️ 暂停任务 {task_id}: {resp.json()}")

def resume_task(task_id):
    resp = requests.post(f"{BASE_URL}/resume/{task_id}")
    print(f"▶️ 恢复任务 {task_id}: {resp.json()}")

def stop_task(task_id):
    resp = requests.post(f"{BASE_URL}/stop/{task_id}")
    print(f"⏹️ 停止任务 {task_id}: {resp.json()}")

def first_training():
    """第一次正常训练"""
    training_data = {
        "DatasetUrl": "/home/tongzhu112/data/M8_Yolo.zip",
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
    return result["data"]["TaskId"]

def incremental_training(base_task_id, new_task_id="task_002"):
    """基于上一次的增量训练"""
    training_data = {
        "DatasetUrl": "/home/tongzhu112/data/M8_Yolo.zip",
        "TaskId": new_task_id,
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
    print(f"🚀 增量训练 {new_task_id} 启动:", result)
    return result["data"]["TaskId"]

if __name__ == "__main__":
    print("🔬 YOLOv8 训练测试 (Flask)")
    print("=" * 50)

    # 第一次训练，演示暂停/恢复
    first_id = first_training()
    wait_for_task(first_id, auto_demo=True)

    # # 第二次增量训练，演示停止
    # second_id = incremental_training("task_001", "task_002")
    # wait_for_task(second_id, auto_stop=True)
    # 
    # # 第三次训练：基于被停止的任务继续
    # third_id = incremental_training("task_002", "task_003")
    # wait_for_task(third_id)
