import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

cap = cv2.VideoCapture(0)
FRAME_W, FRAME_H = 640, 480

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, classes=[0])  

    if results[0].boxes:
        box = results[0].boxes.xyxy[0]
        x1, y1, x2, y2 = map(int, box)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        crop_w, crop_h = 400, 400

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(frame.shape[1], cx + crop_w // 2)
        y2 = min(frame.shape[0], cy + crop_h // 2)

        frame = cv2.resize(frame[y1:y2, x1:x2], (FRAME_W, FRAME_H))

    cv2.imshow("Auto Framing (YOLO)", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

