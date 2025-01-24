# YKTK-MultiTarget-Detection-Tracking

1. **Efficient Object Detection (YOLOv8)**:
   - Using YOLOv8 for object detection provides fast and accurate results, especially in dense environments. This allows the model to handle real-time video streams effectively.
   - By using a lightweight version of YOLOv8 (`yolov8n.pt`), it operates efficiently even on devices with limited resources.

2. **Multi-Object Tracking**:
   - The **Kalman filter** is used for predicting and updating object positions. This enhances tracking stability, particularly in scenarios with fast-moving objects or occlusions.
   - **KNN (KDTree)** is utilized to match objects between frames, optimizing target matching between consecutive frames, which helps reduce ID confusion.
   - **Transformer Self-Attention** is applied to capture contextual information, which improves tracking stability, especially when objects undergo significant state changes.

3. **Multithreading for Performance**:
   - The **ThreadPoolExecutor** is used to parallelize frame processing, which enhances performance and enables the model to handle high-frame-rate videos more efficiently.

4. **Dynamic Frame Rate Adjustment**:
   - With the `frame_skip` parameter, the model can adapt to various scenarios by adjusting the frequency of object detection and tracking, making it flexible for different real-time processing needs.

5. **Scalable Design**:
   - The model is modular, allowing easy replacement or addition of other detection or tracking methods. For instance, YOLOv8 can be swapped for a different detection model, and the Transformer network can be customized.

### todo:

1. **High Resource Consumption**:
   - Although YOLOv8 is a lightweight model, the inclusion of **Transformer Self-Attention** and **Kalman filter** adds computational overhead. This can strain resources on low-end GPUs or CPUs, leading to potential performance degradation.

2. **KNN Matching May Cause Misidentification**:
   - KNN matching using KDTree helps link objects across frames, but in highly dense or occluded scenarios, it may lead to incorrect ID assignments, especially when objects are very close to each other.

3. **Kalman Filter Prediction Errors**:
   - The Kalman filter may produce inaccurate predictions in situations where the object’s movement is unpredictable or changes rapidly. The filter might not be sensitive enough to sudden shifts in trajectory.

4. **Transformer Model Latency**:
   - While Transformer’s self-attention mechanism enhances contextual understanding, its high computational complexity, especially when the number of targets increases, could lead to delays. Despite multithreading, Transformer’s complexity may still cause performance bottlenecks.

5. **Fixed Object Classes**:
   - The model is designed to track only specific classes (e.g., `person` and `car`). Expanding to additional object categories would require modifying or extending the detection and tracking components.
