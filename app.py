from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import unquote

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


def _response(start_response, status: str, body: bytes, content_type: str) -> list[bytes]:
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ]
    start_response(status, headers)
    return [body]


def _serve_file(start_response, file_path: Path) -> list[bytes]:
    if not file_path.exists() or not file_path.is_file():
        return _response(start_response, "404 Not Found", b"Not Found", "text/plain; charset=utf-8")

    content_type, _ = mimetypes.guess_type(str(file_path))
    body = file_path.read_bytes()
    return _response(start_response, "200 OK", body, content_type or "application/octet-stream")


def app(environ, start_response):
    """Minimal WSGI app for Vercel that serves the FaceDetect frontend."""

    path = unquote(environ.get("PATH_INFO", "/"))

    if path in {"", "/"}:
        return _serve_file(start_response, FRONTEND_DIR / "index.html")

    if path == "/health":
        return _response(
            start_response,
            "200 OK",
            b'{"status":"ok","service":"FaceDetect Studio"}',
            "application/json; charset=utf-8",
        )

    if path.startswith("/frontend/"):
        relative_path = path.removeprefix("/frontend/")
        return _serve_file(start_response, FRONTEND_DIR / relative_path)

    if path in {"/index.html", "/app.js", "/styles.css"}:
        return _serve_file(start_response, FRONTEND_DIR / path.lstrip("/"))

    return _response(start_response, "404 Not Found", b"Not Found", "text/plain; charset=utf-8")


def run_webcam() -> None:
    import cv2
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model("models/best_emotion_model.keras")
    print("✅ Emotion model loaded successfully!")

    emotion_labels = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    ]

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("❌ Error loading face detector.")
        return

    print("✅ Face detector loaded successfully!")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("✅ Webcam started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            if w < 80 or h < 80:
                continue

            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)
            confidence = float(np.max(prediction))
            emotion_index = int(np.argmax(prediction))

            if confidence >= 0.60:
                emotion = emotion_labels[emotion_index]
                color = (0, 255, 0)
            else:
                emotion = "Unknown"
                color = (0, 0, 255)

            text = f"{emotion} | {confidence * 100:.1f}%"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )
            cv2.rectangle(
                frame,
                (x, y - 35),
                (x + text_width + 10, y),
                color,
                -1,
            )
            cv2.putText(
                frame,
                text,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.imshow("FaceDetect - Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
