import cv2

rtsp_url = "rtsp://admin:abc12345@10.202.160.12:554/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open video stream.")
else:
    print("Successfully opened video stream.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break
        print(f"Frame shape: {frame.shape}")  # 输出每帧的尺寸

        # 显示视频流
        cv2.imshow("Video Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
