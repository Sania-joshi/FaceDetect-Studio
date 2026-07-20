# 😀 FaceDetect - AI Emotion Recognition System

FaceDetect is a Computer Vision project that detects human facial emotions using Deep Learning. It can recognize emotions in **real-time through a webcam** as well as **from uploaded images**.

The Python project is built using **TensorFlow**, **OpenCV**, and a **Convolutional Neural Network (CNN)** trained on the FER2013 facial expression dataset.

I also added a separate **Vercel-ready frontend** under `frontend/` so you can present the project as a polished web app. The frontend is designed as the user experience layer; if you want live predictions in the browser, you will connect it to a model API or convert the model to a browser-friendly format later.

---

## 📌 Features

- 🎥 Real-time emotion detection using webcam
- 🖼 Emotion prediction from uploaded images
- 😀 Detects 7 different facial emotions
- 📊 Displays prediction confidence
- 🟢 Real-time face detection using Haar Cascade
- ❓ Shows "Unknown" when confidence is low
- 📈 Model evaluation using:
  - Accuracy
  - Confusion Matrix
  - Classification Report

---

## 🧠 Emotions Detected

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😀 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

---

## 🛠 Technologies Used

- Python 3.10
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## 📂 Project Structure

FaceDetect/
│
├── dataset/
│ ├── train/
│ └── val/
│
├── models/
│ ├── best_emotion_model.keras
│ └── emotion_model.keras
│
├── src/
│ ├── model.py
│ └── evaluate.py
│
├── frontend/
│ ├── index.html
│ ├── app.js
│ └── styles.css
│
├── train.py
├── app.py
├── image_predict.py
├── requirements.txt
├── README.md
└── screenshots/

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/FaceDetect.git
```

Go into the project folder

```bash
cd FaceDetect
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model

Run

```bash
python train.py
```

The trained model will be saved inside

```
models/
```

---

## 📈 Evaluating the Model

Run

```bash
python src/evaluate.py
```

This generates

- Accuracy
- Confusion Matrix
- Classification Report

---

## 🎥 Run Webcam Emotion Detection

```bash
python app.py
```

The application

- Detects faces
- Predicts emotions
- Displays confidence score in real time

Press **Q** to exit.

---

## 🌐 Frontend for Vercel

If you want a web application experience, use the frontend in `frontend/`.

What it gives you:

- A modern landing page for FaceDetect
- Webcam and image-upload UI
- Emotion dashboard layout
- Confidence and status panels
- A Vercel-friendly static structure

Important note:

- The current Python model runs locally through OpenCV/TensorFlow.
- Vercel is best used for the frontend.
- Real browser inference needs either a prediction API or a browser-compatible model export.

Deployment idea:

1. Deploy `frontend/` as the Vercel root directory.
2. Keep the Python training/inference scripts in this repo for local development.
3. Connect the frontend to a backend endpoint later if you want live predictions from the trained model.

---

## 🖼 Predict Emotion from an Image

```bash
python image_predict.py
```

Choose an image from your computer and the application predicts the emotion.

---

## 📊 Model Performance

Current Accuracy

```
58.82%
```

Performance may vary depending on

- lighting
- image quality
- facial angle
- facial expression

---

## 📷 Screenshots

### Webcam Detection

(Add webcam screenshot here)

---

### Image Detection

(Add image prediction screenshot here)

---

## 🔮 Future Improvements

- Better CNN architecture
- Transfer Learning (MobileNet / EfficientNet)
- Improved recognition for Sad, Fear and Disgust
- GUI Desktop Application
- Probability chart for all emotions
- Video file emotion detection
- FaceDetect executable (.exe)

---

## 👩‍💻 Author

**Sania Joshi**

B.E. Computer Science Engineering (AI & ML)

Chandigarh University

---

## ⭐ If you like this project

Please consider giving this repository a ⭐ on GitHub.