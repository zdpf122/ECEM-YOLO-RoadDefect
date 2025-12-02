from ultralytics import YOLO

# 加载YOLOv11模型
model = YOLO("yolo11s-ECEM.yaml")
model.load("yolo11s.pt")

# 开始训练
results = model.train(data="data.yaml")
metrics = model.val()  # 使用验证集评估模型
print(f"Validation mAP: {metrics.box.map}")
