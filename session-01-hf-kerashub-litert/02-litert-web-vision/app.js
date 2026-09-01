/**
 * LiteRT.js Web Vision: Direct Inference using @litertjs/core.
 * Official Documentation: https://developers.google.com/edge/litert/web/get_started
 */

import { loadLiteRt, loadAndCompile, Tensor } from "@litertjs/core";

let compiledModel = null;
let imageLabels = [];
let isModelReady = false;

const REMOTE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite";

const sampleImages = {
  dog: "sample_images/dog.jpg",
  cat: "sample_images/cat.jpg",
  car: "sample_images/car.jpg",
  grace_hopper: "sample_images/grace_hopper.jpg"
};

const statusBadge = document.getElementById("statusBadge");
const imagePreview = document.getElementById("imagePreview");
const btnClassify = document.getElementById("btnClassify");
const metricLatency = document.getElementById("metricLatency");
const metricBackend = document.getElementById("metricBackend");
const metricStatus = document.getElementById("metricStatus");
const resultsContainer = document.getElementById("resultsContainer");
const fileInput = document.getElementById("fileInput");

// Expose helpers to window for global onclick triggers
window.loadSampleImage = loadSampleImage;
window.classifyCurrentImage = classifyCurrentImage;

async function loadLabels() {
  try {
    const labelsUrl = new URL("models/imagenet_classes.json?v=" + Date.now(), window.location.href).href;
    const resp = await fetch(labelsUrl, { cache: "no-store" });
    if (resp.ok) {
      let raw = await resp.json();
      // If 1001 classes (with background), slice first
      imageLabels = (raw.length === 1001) ? raw.slice(1) : raw;
      console.log(`Loaded ${imageLabels.length} ImageNet labels.`);
    }
  } catch (e) {
    console.warn("Labels file not loaded:", e);
  }
}

async function getAvailableModelPath() {
  const localEfficientUrl = new URL("models/efficientnet_lite0.tflite", window.location.href).href;
  try {
    const checkResp = await fetch(localEfficientUrl, { method: "HEAD" });
    if (checkResp.ok) {
      console.log("Using local EfficientNet LiteRT model:", localEfficientUrl);
      return localEfficientUrl;
    }
  } catch {
    // Fall back to remote
  }

  console.log("Using Google AI Edge official remote model:", REMOTE_MODEL_URL);
  return REMOTE_MODEL_URL;
}

async function initializeLiteRT() {
  try {
    statusBadge.className = "status-badge status-loading";
    statusBadge.textContent = "Cargando WebAssembly Runtime de LiteRT.js (@litertjs/core)...";

    await loadLabels();

    // 1. Initialize LiteRT.js WebAssembly Runtime (Official Google AI Edge LiteRT Web SDK)
    await loadLiteRt("https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/");

    statusBadge.textContent = "Compilando modelo LiteRT (.tflite)...";
    const targetModelPath = await getAvailableModelPath();

    // 2. Load and compile model with WebAssembly / WebGPU acceleration
    try {
      compiledModel = await loadAndCompile(targetModelPath, {
        accelerator: "wasm"
      });
      metricBackend.textContent = "LiteRT.js WASM SIMD";
    } catch (compileErr) {
      console.warn("Retrying model compilation with remote:", compileErr);
      compiledModel = await loadAndCompile(REMOTE_MODEL_URL, {
        accelerator: "wasm"
      });
      metricBackend.textContent = "LiteRT.js WASM (Remote)";
    }

    isModelReady = true;
    metricStatus.textContent = "Compilado (.tflite)";
    statusBadge.className = "status-badge status-ready";
    statusBadge.textContent = "LiteRT.js Listo para Inferencia";

    // Load initial sample
    loadSampleImage("dog");
  } catch (err) {
    console.error("LiteRT.js initialization error:", err);
    statusBadge.className = "status-badge status-error";
    statusBadge.textContent = "Error al iniciar LiteRT.js: " + err.message;
  }
}

function loadSampleImage(key) {
  if (sampleImages[key]) {
    const imgUrl = new URL(sampleImages[key], window.location.href).href;
    imagePreview.src = imgUrl;
    imagePreview.onload = () => {
      btnClassify.disabled = !isModelReady;
    };
  }
}

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      imagePreview.src = event.target.result;
      imagePreview.onload = () => {
        btnClassify.disabled = !isModelReady;
      };
    };
    reader.readAsDataURL(file);
  }
});

function preprocessImageToTensor(imgElement) {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0, 224, 224);

  const imgData = ctx.getImageData(0, 0, 224, 224).data;
  const floatArr = new Float32Array(1 * 224 * 224 * 3);

  let idx = 0;
  for (let i = 0; i < imgData.length; i += 4) {
    // Normalization to [-1.0, 1.0] for EfficientNet
    floatArr[idx++] = (imgData[i] / 127.5) - 1.0;     // R
    floatArr[idx++] = (imgData[i + 1] / 127.5) - 1.0; // G
    floatArr[idx++] = (imgData[i + 2] / 127.5) - 1.0; // B
  }

  return new Tensor(floatArr, [1, 224, 224, 3]);
}

async function classifyCurrentImage() {
  if (!compiledModel || !imagePreview.src) return;

  btnClassify.disabled = true;
  statusBadge.className = "status-badge status-loading";
  statusBadge.textContent = "Ejecutando inferencia en LiteRT.js...";

  if (!imagePreview.complete) {
    await new Promise(resolve => imagePreview.onload = resolve);
  }

  const startTime = performance.now();
  let inputTensor = null;
  let outputTensors = null;

  try {
    // Preprocess DOM image to LiteRT Tensor
    inputTensor = preprocessImageToTensor(imagePreview);

    // Run direct LiteRT.js forward pass
    outputTensors = await compiledModel.run(inputTensor);
    const outputTensor = Array.isArray(outputTensors) ? outputTensors[0] : outputTensors;
    const outputData = await outputTensor.data();

    const elapsedMs = (performance.now() - startTime).toFixed(2);
    metricLatency.textContent = `${elapsedMs} ms`;

    // Rank Top-5
    const indexed = Array.from(outputData).map((score, idx) => ({ idx, score }));
    indexed.sort((a, b) => b.score - a.score);

    const results = indexed.slice(0, 5).map(item => {
      const name = imageLabels[item.idx] ? imageLabels[item.idx] : `Clase ${item.idx}`;
      return {
        label: name,
        score: item.score
      };
    });

    renderResults(results);
    statusBadge.className = "status-badge status-ready";
    statusBadge.textContent = `Inferencia LiteRT.js Exitosa (${elapsedMs} ms)`;
  } catch (err) {
    console.error("LiteRT.js execution error:", err);
    statusBadge.className = "status-badge status-error";
    statusBadge.textContent = "Error en inferencia: " + err.message;
  } finally {
    if (inputTensor && !inputTensor.deleted) inputTensor.delete();
    if (outputTensors) {
      if (Array.isArray(outputTensors)) {
        outputTensors.forEach(t => { if (t && !t.deleted) t.delete(); });
      } else if (!outputTensors.deleted) {
        outputTensors.delete();
      }
    }
    btnClassify.disabled = false;
  }
}

function renderResults(results) {
  resultsContainer.innerHTML = "";
  if (!results || results.length === 0) {
    resultsContainer.innerHTML = "<p style='color: var(--text-secondary); text-align: center;'>No se obtuvieron resultados.</p>";
    return;
  }

  results.forEach((item, index) => {
    const percentage = (item.score * 100).toFixed(2);
    const itemEl = document.createElement("div");
    itemEl.className = "result-item";
    itemEl.innerHTML = `
      <div class="result-header">
        <span class="result-label">${index + 1}. ${item.label}</span>
        <span class="result-score">${percentage}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill ${index === 0 ? 'top-match' : ''}" style="width: ${Math.max(percentage, 1)}%"></div>
      </div>
    `;
    resultsContainer.appendChild(itemEl);
  });
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", initializeLiteRT);
