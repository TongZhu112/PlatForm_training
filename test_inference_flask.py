# test_inference_flask.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 推理测试脚本 (Flask 版)
"""
import requests
import time
import json
from pathlib import Path

def test_inference():
    """测试推理接口"""
    base_url = "http://localhost:8000"
    inference_data = {
        "model_path": "models/yolov8s.pt",
        "image_path": "uploads/bus.jpg",
        "confidence": 0.75,
        "save_output": True
    }

    print("🚀 开始测试推理接口...")
    try:
        response = requests.post(
            f"{base_url}/inference",
            json=inference_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            result = response.json()
            task_id = result["task_id"]
            print(f"✅ 推理任务已启动，任务ID: {task_id}")
            print(f"📊 状态查询URL: {base_url}/task_status/{task_id}")

            while True:
                status_response = requests.get(f"{base_url}/task_status/{task_id}")
                status_data = status_response.json()
                print(f"🔄 任务状态: {status_data['status']}")
                if status_data['status'] == 'success':
                    print("🎉 推理完成！")
                    print("🔍 检测结果:")
                    result_data = status_data['result']
                    print(f"   检测到 {result_data['total_detections']} 个目标")
                    print(f"   类别: {result_data['classes']}")
                    print(f"   输出图像路径: {result_data.get('output_image_path', 'N/A')}")
                    break
                elif status_data['status'] == 'failure':
                    print(f"❌ 推理失败: {status_data['error']}")
                    break
                time.sleep(2)
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"📝 错误信息: {response.text}")
    except Exception as e:
        print(f"💥 发生异常: {str(e)}")

def prepare_test_data():
    """准备测试数据"""
    directories = ["models", "datasets", "uploads", "runs"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 确保目录存在: {dir_name}")


if __name__ == "__main__":
    print("🔬 YOLOv8 推理测试 (Flask)")
    print("=" * 50)
    prepare_test_data()
    test_inference()
    
    