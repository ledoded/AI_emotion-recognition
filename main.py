import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import sys
import os

# Пути
def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Загрузка TFLite модели
MODEL_PATH = 'emotion_model.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# Детектор лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

COLORS = {
    'angry': (0, 0, 255),
    'happy': (0, 255, 0),
    'neutral': (128, 128, 128),
    'sad': (255, 0, 0),
    'surprise': (0, 255, 255)
}

SMOOTH_FRAMES = 15
DIST_THRESH = 60
track_id_counter = 0
tracks = {}

def get_smoothed_probs(history_deque):
    if not history_deque:
        return None
    return np.mean(history_deque, axis=0)

def preprocess_face(roi_gray):
    roi_resized = cv2.resize(roi_gray, (48, 48))
    roi_float = roi_resized.astype('float32')
    roi_rgb = np.stack((roi_float,) * 3, axis=-1)
    roi_input = np.expand_dims(roi_rgb, axis=0)
    return roi_input

def predict_tflite(face_input):
    interpreter.set_tensor(input_details[0]['index'], face_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return tf.nn.softmax(output[0]).numpy()

# Захват видео
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка камеры")
    exit()

print("Нажмите 'q' или ESC для выхода")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE
    )

    face_data = []
    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        face_data.append((x, y, w, h, cx, cy))

    new_tracks = {}
    matched_indices = set()
    unmatched_indices = list(range(len(face_data)))

    if len(tracks) > 0 and len(face_data) > 0:
        for tid, track in tracks.items():
            best_dist = DIST_THRESH
            best_idx = -1
            for i, (_, _, _, _, cx, cy) in enumerate(face_data):
                if i in matched_indices:
                    continue
                dist = np.sqrt((track['pos'][0] - cx) ** 2 + (track['pos'][1] - cy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx >= 0:
                new_tracks[tid] = {
                    'history': track['history'],
                    'pos': (face_data[best_idx][4], face_data[best_idx][5])
                }
                matched_indices.add(best_idx)
                if best_idx in unmatched_indices:
                    unmatched_indices.remove(best_idx)

    for idx in unmatched_indices:
        x, y, w, h, cx, cy = face_data[idx]
        tid = track_id_counter
        track_id_counter += 1
        new_tracks[tid] = {
            'history': deque(maxlen=SMOOTH_FRAMES),
            'pos': (cx, cy)
        }

    tracks = new_tracks

    for tid, track in tracks.items():
        x = y = w = h = None
        for (fx, fy, fw, fh, cx, cy) in face_data:
            if (cx, cy) == track['pos']:
                x, y, w, h = fx, fy, fw, fh
                break

        if x is None:
            continue

        roi_gray = gray[y:y + h, x:x + w]
        roi_input = preprocess_face(roi_gray)
        probs = predict_tflite(roi_input)

        track['history'].append(probs)
        smooth_probs = get_smoothed_probs(track['history'])

        max_idx = np.argmax(smooth_probs)
        '''sad_index = EMOTIONS.index('sad')
        if smooth_probs[sad_index] > 0.15:
            max_idx = sad_index'''

        emotion = EMOTIONS[max_idx]
        confidence = smooth_probs[max_idx]

        # Отрисовка
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS[emotion], 2)

        text = f"{emotion} ({confidence:.2f})"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - text_h - 15), (x + text_w + 10, y), (0, 0, 0), -1)
        cv2.putText(frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, COLORS[emotion], 2)

        bar_h = 12
        bar_max_width = w
        start_y = y + h + 5

        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x - 65, start_y - 5),
                      (x + bar_max_width + 50, start_y + len(EMOTIONS) * bar_h + 5),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, (emo, prob) in enumerate(zip(EMOTIONS, smooth_probs)):
            bar_width = int(prob * bar_max_width)
            color = COLORS[emo]

            cv2.rectangle(frame,
                          (x, start_y + i * bar_h),
                          (x + bar_max_width, start_y + (i + 1) * bar_h - 2),
                          (60, 60, 60), -1)

            if bar_width > 0:
                cv2.rectangle(frame,
                              (x, start_y + i * bar_h),
                              (x + bar_width, start_y + (i + 1) * bar_h - 2),
                              color, -1)

            cv2.putText(frame, emo[:3],
                        (x - 60, start_y + i * bar_h + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.putText(frame, f"{prob:.2f}",
                        (x + bar_max_width + 5, start_y + i * bar_h + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.namedWindow('Emotion Recognition', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Emotion Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Emotion Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()