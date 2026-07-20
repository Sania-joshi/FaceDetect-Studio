const video = document.getElementById("video");
const imagePreview = document.getElementById("imagePreview");
const imageInput = document.getElementById("imageInput");
const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const cameraStatus = document.getElementById("cameraStatus");
const emotionName = document.getElementById("emotionName");
const confidenceLabel = document.getElementById("confidenceLabel");
const confidenceBar = document.getElementById("confidenceBar");
const confidenceValue = document.getElementById("confidenceValue");
const resultText = document.getElementById("resultText");
const faceStatus = document.getElementById("faceStatus");
const scanMode = document.getElementById("scanMode");
const sourceLabel = document.getElementById("sourceLabel");
const captureCanvas = document.getElementById("captureCanvas");

const MODEL_URL = "https://justadudewhohacks.github.io/face-api.js/models";

let stream = null;
let detectionTimer = null;
let modelsReady = false;

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

function updateControls(running) {
  startCameraBtn.disabled = running;
  stopCameraBtn.disabled = !running;
}

function stopCamera() {
  if (detectionTimer) {
    clearInterval(detectionTimer);
    detectionTimer = null;
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  video.srcObject = null;
  video.hidden = true;
  imagePreview.hidden = true;
  cameraStatus.textContent = "Camera stopped";
  updateControls(false);
  setIdleState();
}

async function loadModels() {
  if (modelsReady) {
    return;
  }

  cameraStatus.textContent = "Loading emotion model...";
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
  modelsReady = true;
}

function getTopExpression(expressions) {
  return Object.entries(expressions).reduce(
    (best, current) => (current[1] > best[1] ? current : best),
    ["neutral", 0],
  );
}

function prettyEmotion(name) {
  return name.charAt(0).toUpperCase() + name.slice(1);
}

function pickBestExpression(expressions) {
  let bestExpression = "neutral";
  let bestProbability = 0;

  for (const [expression, probability] of Object.entries(expressions)) {
    if (probability > bestProbability) {
      bestExpression = expression;
      bestProbability = probability;
    }
  }

  return [bestExpression, bestProbability];
}

function prepareCanvasForImage(image) {
  const maxSide = 512;
  const scale = Math.min(1, maxSide / Math.max(image.naturalWidth || image.width, image.naturalHeight || image.height));
  const width = Math.max(1, Math.round((image.naturalWidth || image.width) * scale));
  const height = Math.max(1, Math.round((image.naturalHeight || image.height) * scale));

  captureCanvas.width = width;
  captureCanvas.height = height;
  const context = captureCanvas.getContext("2d", { willReadFrequently: true });
  context.clearRect(0, 0, width, height);
  context.drawImage(image, 0, 0, width, height);

  return captureCanvas;
}

async function analyzeVideoFrame() {
  if (!stream || video.hidden) {
    return;
  }

  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.2 }))
    .withFaceExpressions();

  if (!detection) {
    setResult({
      emotion: "No face detected",
      confidence: 0,
      message: "Move your face into the frame so the browser model can analyze your expression.",
      mode: "Live webcam",
      source: "Webcam stream",
      face: "No face in frame",
    });
    return;
  }

  const [expression, probability] = pickBestExpression(detection.expressions);
  const confidence = probability * 100;

  setResult({
    emotion: prettyEmotion(expression),
    confidence,
    message: `The webcam detected ${prettyEmotion(expression).toLowerCase()} with browser-side emotion recognition.`,
    mode: "Live webcam",
    source: "Webcam stream",
    face: `Face detected (${detection.detection.score.toFixed(2)})`,
  });
}

async function startCamera() {
  try {
    await loadModels();

    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    video.srcObject = stream;
    video.hidden = false;
    imagePreview.hidden = true;
    cameraStatus.textContent = "Camera live";
    updateControls(true);

    await new Promise((resolve) => {
      if (video.readyState >= 2) {
        resolve();
        return;
      }

      video.onloadeddata = () => resolve();
    });

    setResult({
      emotion: "Camera active",
      confidence: 0,
      message:
        "The browser model is running. Detecting facial expression from the live webcam stream.",
      mode: "Live webcam",
      source: "Webcam stream",
      face: "Awaiting first detection",
    });

    detectionTimer = window.setInterval(() => {
      analyzeVideoFrame().catch((error) => {
        cameraStatus.textContent = "Detection error";
        setResult({
          emotion: "Detection failed",
          confidence: 0,
          message: error?.message || "The browser model could not process the frame.",
          mode: "Live webcam",
          source: "Webcam stream",
          face: "Model error",
        });
      });
    }, 800);
  } catch (error) {
    cameraStatus.textContent = "Camera blocked";
    updateControls(false);
    setResult({
      emotion: "Camera unavailable",
      confidence: 0,
      message:
        "The browser could not access the webcam in this session. Use image upload to get an emotion result right away.",
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
    updateControls(false);

    loadModels()
      .then(async () => {
        const image = imagePreview;
        await new Promise((resolve) => {
          if (image.complete && image.naturalWidth > 0) {
            resolve();
            return;
          }

          image.onload = () => resolve();
        });

        const detectionSource = prepareCanvasForImage(image);
        const detection = await faceapi
          .detectSingleFace(detectionSource, new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.2 }))
          .withFaceExpressions();

        if (!detection) {
          setResult({
            emotion: "No face detected",
            confidence: 0,
            message: "The uploaded image did not contain a detectable face.",
            mode: "Image upload",
            source: file.name,
            face: "No face detected",
          });
          return;
        }

        const [expression, probability] = pickBestExpression(detection.expressions);
        const confidence = probability * 100;

        setResult({
          emotion: prettyEmotion(expression),
          confidence,
          message: `The uploaded image was classified as ${prettyEmotion(expression).toLowerCase()} using browser-side emotion recognition.`,
          mode: "Image upload",
          source: file.name,
          face: `Face detected (${detection.detection.score.toFixed(2)})`,
        });
        cameraStatus.textContent = "Image analyzed";
      })
      .catch((error) => {
        setResult({
          emotion: "Image analysis failed",
          confidence: 0,
          message: error?.message || "The uploaded image could not be processed.",
          mode: "Image upload",
          source: file.name,
          face: "Model error",
        });
        cameraStatus.textContent = "Image analysis failed";
      });
  };

  reader.readAsDataURL(file);
}

startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);
imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files || [];

  if (!file) {
    return;
  }

  showUploadedImage(file);
});

setIdleState();
updateControls(false);