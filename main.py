from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model = YOLO("detect.pt")

video_folder = "video"
video_files = os.listdir(video_folder)
video_paths = [os.path.join(video_folder, file) for file in video_files]

people_counter = 0


def gen_frames(video_path):
    global people_counter

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        local_counter = 0
        success, frame = cap.read()
        if success:
            results = model(frame, stream=True, verbose=False)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    cls = int(box.cls[0])
                    # 0 - target
                    if cls == 0:
                        local_counter += 1

                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (0, 0, 255), 2)

                        cv2.putText(frame, f"People in picture: {people_counter}",
                                    [10, 20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break

        people_counter = local_counter

    cap.release()


@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    video_index = request.args.get('video', default=0, type=int)
    if video_index < 0 or video_index >= len(video_paths):
        video_index = 0
    video_path = video_paths[video_index]
    return Response(gen_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    global people_counter
    return render_template('index.html', video_files=video_files, cucu=people_counter)


@app.route('/people_counter')
def get_people_counter():
    global people_counter
    return jsonify({'people_counter': people_counter})


if __name__ == '__main__':
    app.run(debug=True)
