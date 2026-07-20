import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, filedialog
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
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("❌ Error loading face detector.")
    exit()

print("✅ Face detector loaded successfully!")

# ===========================
# Select Image
# ===========================

Tk().withdraw()

image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png")
    ]
)

if not image_path:
    print("No image selected.")
    exit()

    # ===========================
# Read Image
# ===========================

image = cv2.imread(image_path)

if image is None:
    print("❌ Could not load image.")
    exit()

print("✅ Image loaded successfully!")
# ===========================
# Convert to Grayscale
# ===========================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ===========================
# Detect Faces
# ===========================

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

print(f"Faces detected: {len(faces)}")

if len(faces) == 0:
    print("❌ No face detected.")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# ===========================
# Predict Emotion
# ===========================

for (x, y, w, h) in faces:

    # Crop the face
    face = gray[y:y+h, x:x+w]

    # Resize to model input size
    face = cv2.resize(face, (48, 48))

    # Normalize
    face = face.astype("float32") / 255.0

    # Add channel dimension
    face = np.expand_dims(face, axis=-1)

    # Add batch dimension
    face = np.expand_dims(face, axis=0)

prediction = model.predict(face, verbose=0)

emotion_index = np.argmax(prediction)

emotion = emotion_labels[emotion_index]

confidence = prediction[0][emotion_index] * 100

# Draw rectangle around face
cv2.rectangle(
    image,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    2
)

# Display emotion and confidence
text = f"{emotion} ({confidence:.1f}%)"

cv2.putText(
    image,
    text,
    (x, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)
# ===========================
# Display Result
# ===========================

cv2.imshow("Emotion Prediction", image)

cv2.waitKey(0)

cv2.destroyAllWindows()