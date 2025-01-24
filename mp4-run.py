import torch
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO
import torch.nn as nn
from filterpy.kalman import KalmanFilter
import time

# 1. 初始化 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 使用 YOLOv8 模型


# 2. 卡尔曼滤波器类
class KalmanTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])  # 状态转移矩阵
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 观测矩阵
        self.kf.P *= 1000.  # 初始误差协方差
        self.tracks = []  # 跟踪列表

    def update(self, detection):
        self.kf.predict()
        self.kf.update(detection)
        return self.kf.x[:2]  # 返回 x, y 位置

    def add(self, detection):
        self.tracks.append(detection)

    def get_tracks(self):
        return [track[:2] for track in self.tracks]  # 返回当前跟踪的目标位置


# 3. KNN 匹配目标
def knn_tracking(previous_frame, current_frame):
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn.fit(previous_frame)
    distances, indices = knn.kneighbors(current_frame)
    return indices, distances


# 4. 目标检测与追踪
def process_frame(frame, tracker, previous_frame, valid_classes):
    results = model(frame)
    detections = []

    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = result.xyxy[0].cpu().numpy()
        confidence = result.conf[0].cpu().numpy()
        class_id = result.cls[0].cpu().numpy()
        class_name = results[0].names[int(class_id)]

        if class_name not in valid_classes:
            continue

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        detections.append([x_center, y_center, w, h, class_name, confidence])

        label = f"{class_name} ({confidence:.2f})"
        frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        draw_text(frame, label, (int(x_min), int(y_min) - 10))

    if len(previous_frame) > 0:
        indices, distances = knn_tracking(np.array(previous_frame), np.array([det[:4] for det in detections]))
        print(f"Distances: {distances}")
        print(f"Matched indices: {indices}")

        # 更新卡尔曼滤波器的目标
        for i, index in enumerate(indices.flatten()):
            tracker.update(detections[i][:2])  # 使用检测到的中心位置更新卡尔曼滤波器

    previous_frame = [det[:4] for det in detections]
    return frame, detections, previous_frame


# 5. 使用 OpenCV 绘制英文标签
def draw_text(img, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


# 6. 目标追踪主函数（加入帧率控制）
def track_objects_in_video(video_path, frame_skip=3):
    cap = cv2.VideoCapture(video_path)
    tracker = KalmanTracker()  # 使用卡尔曼滤波器进行目标追踪
    previous_frame = []
    valid_classes = ['person', 'car']  # 只追踪人和车

    frame_count = 0  # 帧计数
    start_time = time.time()  # 计算处理时间

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue  # 跳过多余的帧，不进行处理

        # 开始处理这一帧
        frame, detections, previous_frame = process_frame(frame, tracker, previous_frame, valid_classes)

        # 显示视频帧
        cv2.imshow("Tracked Objects", frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 每秒处理的帧数
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            print(f"FPS: {frame_count / elapsed_time:.2f}")

    cap.release()
    cv2.destroyAllWindows()


# 运行目标追踪
track_objects_in_video('test.mp4', frame_skip=3)  # 替换为你的视频文件路径
