import torch
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO
import torch.nn as nn

# 1. 初始化 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 使用 YOLOv8 模型

# 2. Transformer 用于目标追踪
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

# 5. 使用 OpenCV 绘制英文标签
def draw_text(img, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
    # 在图像上绘制文本
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# 6. 目标检测与追踪
def track_objects_in_video(rtsp_url):
    # 打开视频流（RTSP 流或摄像头）
    cap = cv2.VideoCapture(rtsp_url)  # 可以替换为摄像头地址或 RTSP 流地址
    tracker = MultiObjectTracker(input_dim=4, num_heads=2, num_layers=2)  # 输入维度为4（x_center, y_center, w, h）

    previous_frame = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # 进行目标检测
        results = model(frame)

        # results[0] 是一个包含检测框结果的列表
        detections = []

        for result in results[0].boxes:  # results[0] 中包含一个名为 boxes 的属性
            x_min, y_min, x_max, y_max = result.xyxy[0].cpu().numpy()  # 获取框的坐标 (x_min, y_min, x_max, y_max)
            confidence = result.conf[0].cpu().numpy()  # 获取置信度
            class_id = result.cls[0].cpu().numpy()  # 获取类别
            class_name = results[0].names[int(class_id)]  # 获取类别名称

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min

            detections.append([x_center, y_center, w, h, class_name, confidence])

            # 在图像上绘制目标框和类别标签
            label = f"{class_name} ({confidence:.2f})"

            frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            # 绘制英文标签
            draw_text(frame, label, (int(x_min), int(y_min) - 10))

        if len(previous_frame) > 0:
            # 使用 KNN 进行目标匹配
            indices, distances = knn_tracking(np.array(previous_frame), np.array([det[:4] for det in detections]))
            print(f"Distances: {distances}")
            print(f"Matched indices: {indices}")

        # 更新 previous_frame
        previous_frame = [det[:4] for det in detections]

        # 显示实时视频流
        cv2.imshow("Tracked Objects", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 运行目标追踪（替换为 RTSP 流或摄像头地址）
track_objects_in_video('rtsp://184.72.239.149/vod/mp4:BigBuckBunny_115k.mov')  # 替换为你的视频流地址或摄像头地址
