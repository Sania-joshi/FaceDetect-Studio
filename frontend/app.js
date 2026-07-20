const video = document.getElementById("video");
const imagePreview = document.getElementById("imagePreview");
const imageInput = document.getElementById("imageInput");
const startCameraBtn = document.getElementById("startCameraBtn");
const cameraStatus = document.getElementById("cameraStatus");
const emotionName = document.getElementById("emotionName");
const confidenceLabel = document.getElementById("confidenceLabel");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceValue = document.getElementById("confidenceValue");
const resultText = document.getElementById("resultText");
const faceStatus = document.getElementById("faceStatus");
const scanMode = document.getElementById("scanMode");
const sourceLabel = document.getElementById("sourceLabel");

let stream = null;

function setResult({ emotion, confidence, message, mode, source, face }) {
  const percent = Math.max(0, Math.min(100, confidence));

  emotionName.textContent = emotion;
  confidenceLabel.textContent = `${percent.toFixed(0)}%`;
  confidenceValue.textContent = `${percent.toFixed(1)}%`;
  confidenceBar.style.width = `${percent}%`;
  resultText.textContent = message;
  scanMode.textContent = mode;
  sourceLabel.textContent = source;
  faceStatus.textContent = face;
}

function setIdleState() {
  setResult({
    emotion: "Awaiting scan",
    confidence: 0,
    message:
      "Start the camera or upload an image to prepare the interface for prediction output.",
    mode: "Frontend demo",
    source: "Browser UI",
    face: "No face analyzed",
  });
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    video.srcObject = stream;
    video.hidden = false;
    imagePreview.hidden = true;
    cameraStatus.textContent = "Camera live";
    setResult({
      emotion: "Camera active",
      confidence: 0,
      message:
        "The frontend is ready. Connect a prediction endpoint or browser model to produce live emotion output.",
      mode: "Live webcam",
      source: "Webcam stream",
      face: "Awaiting analysis",
    });
  } catch (error) {
    cameraStatus.textContent = "Camera blocked";
    setResult({
      emotion: "Camera unavailable",
      confidence: 0,
      message:
        "The browser could not access the webcam. Use image upload or connect the app to a hosted inference service.",
      mode: "Fallback mode",
      source: "Browser UI",
      face: error?.message || "Permission denied",
    });
  }
}

function showUploadedImage(file) {
  const reader = new FileReader();

  reader.onload = () => {
    imagePreview.src = reader.result;
    imagePreview.hidden = false;
    video.hidden = true;
    cameraStatus.textContent = "Image loaded";

    setResult({
      emotion: "Image selected",
      confidence: 0,
      message:
        "The upload flow is ready for a prediction endpoint. Replace this state with model output when the backend is connected.",
      mode: "Image upload",
      source: file.name,
      face: "Preview ready",
    });
  };

  reader.readAsDataURL(file);
}

startCameraBtn.addEventListener("click", startCamera);
imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files || [];

  if (!file) {
    return;
  }

  showUploadedImage(file);
});

setIdleState();