from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import atexit

app = Flask(__name__)

# ================= LOAD MODEL =================
model = load_model("new_resnet_model.h5")

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ================= LOAD DATA =================
df = pd.read_csv("songs_with_emotion_youtube.csv")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# ================= GLOBAL VARIABLES =================
current_emotion = "neutral"
current_confidence = 0
frame_count = 0

# ================= SONG RECOMMENDER =================
def recommend_songs(emotion):
    emotion = emotion.lower()
    # normalize dataset
    df['Emotion'] = df['Emotion'].str.lower()
    filtered = df[df['Emotion'] == emotion]

    if len(filtered) == 0:
        return []

    return filtered.sample(min(6, len(filtered))).to_dict(orient='records')


# ================= CAMERA =================
camera = cv2.VideoCapture(0)

def generate_frames():
    global current_emotion, current_confidence, frame_count

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1

        # ⚡ Skip frames for performance
        if frame_count % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # ❗ If no face detected
        if len(faces) == 0:
            current_emotion = "neutral"
            current_confidence = 0

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            # preprocess
            roi = cv2.resize(roi, (75, 75))
            roi = roi / 255.0
            roi = np.reshape(roi, (1, 75, 75, 1))

            # prediction
            preds = model.predict(roi, verbose=0)
            max_index = np.argmax(preds)

            current_emotion = emotion_labels[max_index]
            current_confidence = float(np.max(preds)) * 100

            # draw box + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame,
                        f"{current_emotion} ({int(current_confidence)}%)",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

        # encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ================= ROUTES =================

# Main page
@app.route('/')
def index():
    return render_template("index.html")


# Video stream
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Emotion API
@app.route('/get_emotion')
def get_emotion():
    return jsonify({
        "emotion": current_emotion,
        "confidence": round(current_confidence, 2)
    })


# Songs API (only on button click)
@app.route('/get_songs')
def get_songs():
    emotion = request.args.get("emotion", "neutral")
    songs = recommend_songs(emotion)
    return jsonify(songs)


# ================= CLEANUP =================
def release_camera():
    camera.release()

atexit.register(release_camera)


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)