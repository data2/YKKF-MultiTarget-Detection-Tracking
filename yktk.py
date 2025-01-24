import torch
import cv2
import numpy as np
from sklearn.neighbors import KDTree
from ultralytics import YOLO
import torch.nn as nn
from filterpy.kalman import KalmanFilter
import logging
from concurrent.futures import ThreadPoolExecutor

# 配置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

# 1. 初始化 YOLOv8 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用 GPU 或 CPU
logging.debug(f"Using device: {device}")
model = YOLO("yolov8n.pt")  # 使用轻量级的 YOLOv8 模型
model.to(device)

# 2. 卡尔曼滤波器类
class KalmanTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)  # 状态维度（x, y, vx, vy, ax, ay）和观测维度（x, y）

        # 状态转移矩阵（考虑了加速度）
        self.kf.F = np.array([[1, 0, 1, 0, 0.5, 0],
                              [0, 1, 0, 1, 0, 0.5],
                              [0, 0, 1, 0, 1, 0],
                              [0, 0, 0, 1, 0, 1],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])

        # 观测矩阵
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0]])

        # 初始误差协方差
        self.kf.P *= 1000.

        self.tracks = []  # 跟踪目标列表

    def update(self, detection):
        """更新目标位置"""
        self.kf.predict()  # 预测下一时刻的位置
        self.kf.update(detection)  # 用检测结果更新目标
        logging.debug("KalmanTracker updating")
        return self.kf.x[:2]  # 返回 x, y 位置

    def add(self, detection):
        """添加新目标到跟踪列表"""
        self.tracks.append(detection)

    def get_tracks(self):
        """获取当前所有目标的位置"""
        return [track[:2] for track in self.tracks]

# 3. Transformer 用于目标追踪（包含自注意力机制）
class SelfAttentionTracker(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super(SelfAttentionTracker, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, input_dim)
        self.num_layers = num_layers

    def forward(self, x):
        # 自注意力机制
        attn_output, _ = self.attn(x, x, x)
        output = attn_output + x  # 残差连接
        output = self.fc(output)  # 线性变换
        logging.debug("SelfAttentionTracker forward")
        return output

# 4. 使用 OpenCV 绘制英文标签
def draw_text(img, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# 5. KNN匹配目标优化（使用 KDTree）
def knn_tracking(previous_frame, current_frame):
    knn = KDTree(previous_frame)
    distances, indices = knn.query(current_frame, k=1)
    return indices, distances

# 6. 目标检测与追踪
def process_frame(frame, tracker, kalman_tracker, previous_frame, valid_classes, track_ids, frame_skip=2,
                  frame_count=0):
    # 确保 track_ids 始终为字典类型
    if not isinstance(track_ids, dict):
        track_ids = {}

    # 目标检测
    results = model(frame)  # 推理
    detections = []

    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = result.xyxy[0].cpu().numpy()
        confidence = result.conf[0].cpu().numpy()
        class_id = result.cls[0].cpu().numpy()
        class_name = results[0].names[int(class_id)]

        if class_name not in valid_classes:
            continue

        # 如果目标已存在，保持 ID 不变
        detection_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        target_id = None
        for track_id, track in track_ids.items():
            prev_center = track[:2]
            distance = np.linalg.norm(np.array(prev_center) - np.array(detection_center))
            if distance < 50:  # 匹配阈值，距离较近则为同一个目标
                target_id = track_id
                break

        # 如果是新目标，则分配一个新的 ID
        if target_id is None:
            target_id = len(track_ids) + 1

        track_ids[target_id] = [detection_center[0], detection_center[1], x_max - x_min, y_max - y_min]

        label_with_id = f"{class_name} #{target_id} ({confidence:.2f})"
        detections.append(
            [detection_center[0], detection_center[1], x_max - x_min, y_max - y_min, class_name, confidence, target_id])

        # 绘制目标框和标签
        frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        draw_text(frame, label_with_id, (int(x_min), int(y_min) - 10))

    # 进行 KNN 匹配
    if len(previous_frame) > 0:
        logging.debug("knn ")
        indices, distances = knn_tracking(np.array(previous_frame), np.array([det[:4] for det in detections]))

        # 使用 Kalman 滤波器更新目标
        for i, index in enumerate(indices.flatten()):
            kalman_tracker.update(detections[i][:2])  # 使用卡尔曼滤波器更新目标位置

        # 使用 Transformer 进行上下文理解（只对目标进行更新）
        if len(detections) > 0:
            transformer_input = torch.tensor(np.array([det[:4] for det in detections]), dtype=torch.float32).to(device)
            transformer_output = tracker(transformer_input.unsqueeze(1))  # 使用 Transformer 更新目标状态
            logging.debug("Transformer output: %s", transformer_output)

    previous_frame = [det[:4] for det in detections]
    return frame, detections, previous_frame, track_ids

# 7. 目标追踪主程序（动态调整帧率）
def track_objects_in_video(video_path, frame_skip=2):
    cap = cv2.VideoCapture(video_path)
    kalman_tracker = KalmanTracker()  # 使用卡尔曼滤波器
    tracker = SelfAttentionTracker(input_dim=4, num_heads=2, num_layers=2).to(device)  # 使用自注意力机制
    previous_frame = []
    track_ids = {}  # 跟踪 ID
    valid_classes = ['person', 'car']  # 只追踪人和车

    frame_count = 0

    # 创建线程池
    with ThreadPoolExecutor() as executor:
        futures = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔一定帧数处理一次
            if frame_count % frame_skip == 0:
                futures.append(executor.submit(process_frame, frame, tracker, kalman_tracker, previous_frame,
                                               valid_classes, track_ids, frame_skip, frame_count))

            frame_count += 1

        # 获取并显示处理后的帧
        for future in futures:
            frame, detections, previous_frame, track_ids = future.result()
            cv2.imshow("Tracked Objects", frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# 运行目标追踪
track_objects_in_video('test.mp4')  # 替换为你的视频文件路径
