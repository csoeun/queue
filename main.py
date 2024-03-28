from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import math

app = Flask(__name__)
model = YOLO("detect.pt") 

video_path = "video/file1.mp4"  

def gen_frames(): 
    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()  
        if not success:
            break
        else:
            results = model(frame, stream=True)

            people_counter = 0

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    cls = int(box.cls[0])
                    # 0 = человек
                    if cls == 0:
                        people_counter += 1

                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        confidence = math.ceil((box.conf[0]*100))/100
                        print("Person found, Confidence --->", confidence)

                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(frame, "person", org,
                                    font, fontScale, color, thickness)

            cv2.putText(frame, f"People in picture: {people_counter}",
                        [10, 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
