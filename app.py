import cv2
import numpy as np
import tensorflow as tf

# ===========================
# Load Trained Model
# ===========================

model = tf.keras.models.load_model("models/best_emotion_model.keras")
print("✅ Emotion model loaded successfully!")

# ===========================
# Emotion Labels
# ===========================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# ===========================
# Load Face Detector
# ===========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("❌ Error loading face detector.")
    exit()

print("✅ Face detector loaded successfully!")

# ===========================
# Open Webcam
# ===========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("✅ Webcam started.")

# ===========================
# Real-Time Detection Loop
# ===========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # Ignore very small faces
        if w < 80 or h < 80:
            continue

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize to model input
        face = cv2.resize(face, (48, 48))

        # Normalize
        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Predict
        prediction = model.predict(face, verbose=0)

        confidence = float(np.max(prediction))
        emotion_index = np.argmax(prediction)

        # Confidence Threshold
        if confidence >= 0.60:
            emotion = emotion_labels[emotion_index]
            color = (0, 255, 0)      # Green
        else:
            emotion = "Unknown"
            color = (0, 0, 255)      # Red

        text = f"{emotion} | {confidence*100:.1f}%"

        # Draw Face Rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        # Text Background
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        cv2.rectangle(
            frame,
            (x, y-35),
            (x + text_width + 10, y),
            color,
            -1
        )

        # Draw Text
        cv2.putText(
            frame,
            text,
            (x+5, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("FaceDetect - Emotion Recognition", frame)

    # Press Q to Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===========================
# Cleanup
# ===========================

cap.release()
cv2.destroyAllWindows()