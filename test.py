from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture("video/file1.mp4")
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("detect.pt")

while True:
    peopleCounter = 0

    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            # 0 = human
            if cls == 0:
                peopleCounter += 1

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0]*100))/100

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, "person", org,
                            font, fontScale, color, thickness)

    cv2.putText(img, f"People in picture: {peopleCounter}",
                [10, 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()