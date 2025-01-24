import torch
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO

# 1. 初始化 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 使用 YOLOv8 Nano 模型，适合实时视频处理

# 2. Transformer 用于目标追踪
import torch.nn as nn

class TrackerTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super(TrackerTransformer, self).__init__()

        # 确保 input_dim 能被 num_heads 整除
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )

    def forward(self, x):
        return self.transformer(x)

# 3. 目标追踪器
class MultiObjectTracker:
    def __init__(self, input_dim, num_heads, num_layers):
        self.transformer = TrackerTransformer(input_dim, num_heads, num_layers)
        self.tracked_objects = []

    def track(self, detections):
        # 将检测到的目标进行 Transformer 处理
        tracked = self.transformer(detections)
        return tracked

# 4. KNN 匹配算法
def knn_tracking(previous_frame, current_frame):
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn.fit(previous_frame)
    distances, indices = knn.kneighbors(current_frame)
    return indices, distances

# 5. 目标检测与追踪
def track_objects_in_video(video_path):
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    tracker = MultiObjectTracker(input_dim=4, num_heads=2, num_layers=2)  # 输入维度为4（x_center, y_center, w, h）

    previous_frame = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 进行目标检测
        results = model(frame)

        # results[0] 是一个包含检测框结果的列表
        detections = []

        for result in results[0].boxes:  # results[0] 中包含一个名为 boxes 的属性
            x_min, y_min, x_max, y_max = result.xyxy[0].cpu().numpy()  # 获取框的坐标 (x_min, y_min, x_max, y_max)
            confidence = result.conf[0].cpu().numpy()  # 获取置信度
            class_id = result.cls[0].cpu().numpy()  # 获取类别

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min

            detections.append([x_center, y_center, w, h])

        if len(previous_frame) > 0:
            # 使用 KNN 进行目标匹配
            indices, distances = knn_tracking(np.array(previous_frame), np.array(detections))
            print(f"Distances: {distances}")
            print(f"Matched indices: {indices}")

        # 更新 previous_frame
        previous_frame = detections

        # 在帧上绘制目标框
        for det in detections:
            x_center, y_center, w, h = det
            cv2.rectangle(frame, (int(x_center - w / 2), int(y_center - h / 2)),
                          (int(x_center + w / 2), int(y_center + h / 2)), (0, 255, 0), 2)

        # 显示视频帧
        cv2.imshow("Tracked Objects", frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 运行目标追踪
track_objects_in_video('test.mp4')  # 替换为你的视频文件路径
