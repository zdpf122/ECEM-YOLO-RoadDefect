from ultralytics import YOLO

# 加载YOLO模型
model = YOLO("runs/detect/RDDn/weights/best.pt")  # 根据实际路径调整

# 在验证集上评估模型
val_metrics = model.val(data="data.yaml", split="val")

# 在测试集上评估模型
test_metrics = model.val(data="data.yaml", split="test")


# 计算F1-score（F1 = 2*(Precision*Recall)/(Precision+Recall)）
def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# 计算验证集和测试集的F1-score
val_f1 = calculate_f1(val_metrics.box.mp, val_metrics.box.mr)
test_f1 = calculate_f1(test_metrics.box.mp, test_metrics.box.mr)


# 打印验证集指标
print("\n===== 验证集指标 =====")
print(f"Validation Precision (mAP): {val_metrics.box.mp:.4f}")
print(f"Validation Recall (mAR): {val_metrics.box.mr:.4f}")
print(f"Validation F1-score: {val_f1:.4f}")
print(f"Validation mAP50: {val_metrics.box.map50:.4f}")

# 打印测试集指标（含FPS）
print("\n===== 测试集指标 =====")
print(f"Test Precision (mAP): {test_metrics.box.mp:.4f}")
print(f"Test Recall (mAR): {test_metrics.box.mr:.4f}")
print(f"Test F1-score: {test_f1:.4f}")
print(f"Test mAP50: {test_metrics.box.map50:.4f}")
