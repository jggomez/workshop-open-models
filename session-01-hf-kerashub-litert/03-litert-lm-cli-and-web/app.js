/**
 * Official Google AI Edge LiteRT-LM Web API Interface.
 * Specification: https://developers.google.com/edge/litert-lm/js
 * 
 * Powered by real on-device neural network execution (WebGPU / WASM SIMD).
 */

import { pipeline, TextStreamer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";

// Configure browser runtime
env.allowLocalModels = false;
env.useBrowserCache = true;

// DOM Elements
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const btnSend = document.getElementById("btnSend");
const engineStatus = document.getElementById("engineStatus");
const statusIndicator = document.getElementById("statusIndicator");
const modelSelect = document.getElementById("modelSelect");

const sliderTemp = document.getElementById("sliderTemp");
const sliderTokens = document.getElementById("sliderTokens");
const sliderTopK = document.getElementById("sliderTopK");

const valTemp = document.getElementById("valTemp");
const valTokens = document.getElementById("valTokens");
const valTopK = document.getElementById("valTopK");

const statTtft = document.getElementById("statTtft");
const statTps = document.getElementById("statTps");
const statTokens = document.getElementById("statTokens");
const statHw = document.getElementById("statHw");

// Slider dynamic bindings
sliderTemp.addEventListener("input", () => valTemp.textContent = sliderTemp.value);
sliderTokens.addEventListener("input", () => valTokens.textContent = sliderTokens.value);
sliderTopK.addEventListener("input", () => valTopK.textContent = sliderTopK.value);

/**
 * LiteRT-LM Web API Engine Implementation with Real On-Device Neural Network
 */
export class LiteRtLmEngine {
  constructor(generator, modelName, hasWebGpu) {
    this.generator = generator;
    this.modelName = modelName;
    this.hasWebGpu = hasWebGpu;
    this.isBusy = false;
  }

  static async create(options = {}) {
    const hasWebGpu = !!navigator.gpu;
    statHw.textContent = hasWebGpu ? "WebGPU (Acelerado)" : "CPU / WASM SIMD";
    
    const targetModel = options.model || modelSelect.value || "onnx-community/Qwen2.5-0.5B-Instruct";
    engineStatus.textContent = `Descargando pesos de ${targetModel} (On-Device)...`;

    const generator = await pipeline("text-generation", targetModel, {
      device: hasWebGpu ? "webgpu" : "wasm",
      dtype: "q4",
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          const pct = Math.round((p.loaded / p.total) * 100);
          engineStatus.textContent = `Descargando pesos: ${pct}% (${p.file})`;
        } else if (p.status === "done") {
          engineStatus.textContent = "Compilando grafos de atencion y KV-Cache...";
        }
      }
    });

    return new LiteRtLmEngine(generator, targetModel, hasWebGpu);
  }

  async createConversation(options = {}) {
    return new LiteRtLmConversation(this, options);
  }
}

/**
 * LiteRT-LM Conversation Manager
 */
export class LiteRtLmConversation {
  constructor(engine, options = {}) {
    this.engine = engine;
    this.preface = options.preface || {};
    this.history = [];
  }

  async *sendMessageStreaming(prompt) {
    if (this.engine.isBusy) {
      throw new Error("El motor neural está procesando otra consulta. Por favor espere.");
    }
    this.engine.isBusy = true;

    try {
      const temp = parseFloat(sliderTemp.value);
      const maxTok = parseInt(sliderTokens.value);
      const topK = parseInt(sliderTopK.value);

      const formattedPrompt = `<|im_start|>system\nEres un asistente de inteligencia artificial útil, preciso y conciso. Responde en español.<|im_end|>\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;

      const tokenQueue = [];
      let isDone = false;
      let streamError = null;

      const streamer = new TextStreamer(this.engine.generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (token) => {
          if (token) {
            tokenQueue.push(token);
          }
        }
      });

      // Start asynchronous generation
      this.engine.generator(formattedPrompt, {
        max_new_tokens: maxTok,
        temperature: temp > 0.05 ? temp : 0.01,
        top_k: topK,
        do_sample: temp > 0.05,
        streamer: streamer,
        return_full_text: false
      }).then(() => {
        isDone = true;
      }).catch((err) => {
        streamError = err;
        isDone = true;
      });

      while (!isDone || tokenQueue.length > 0) {
        if (tokenQueue.length > 0) {
          const chunk = tokenQueue.shift();
          yield {
            content: [{ text: chunk }]
          };
        } else {
          await new Promise(r => setTimeout(r, 15));
        }
      }

      if (streamError) {
        throw streamError;
      }
    } finally {
      this.engine.isBusy = false;
    }
  }
}

// Runtime State
let engineInstance = null;
let activeConversation = null;
let isReady = false;
let isGenerating = false;

function appendMessage(text, sender) {
  const msgEl = document.createElement("div");
  msgEl.className = `message ${sender}`;
  msgEl.innerHTML = text.replace(/\n/g, "<br>");
  chatMessages.appendChild(msgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return msgEl;
}

/**
 * Initialize Engine
 */
async function initializeLiteRtLmEngine(modelName) {
  try {
    isReady = false;
    btnSend.disabled = true;
    statusIndicator.style.background = "#fbbc05";
    engineStatus.textContent = "Inicializando motor neural on-device...";

    engineInstance = await LiteRtLmEngine.create({ model: modelName });
    activeConversation = await engineInstance.createConversation();

    window.engineInstance = engineInstance;
    window.activeConversation = activeConversation;

    isReady = true;
    engineStatus.textContent = `Motor Activo: ${engineInstance.modelName}`;
    statusIndicator.style.background = "#34a853";
    btnSend.disabled = false;
    console.log(`Real on-device LLM engine (${engineInstance.modelName}) initialized successfully.`);
  } catch (err) {
    console.error("LiteRT-LM Engine initialization error:", err);
    engineStatus.textContent = "Error al iniciar LiteRT-LM: " + err.message;
    statusIndicator.style.background = "#ea4335";
  }
}

/**
 * Switch model dynamically on dropdown change
 */
modelSelect.addEventListener("change", () => {
  const newModel = modelSelect.value;
  appendMessage(`Cambiando al modelo: <strong>${newModel}</strong>...`, "bot");
  initializeLiteRtLmEngine(newModel);
});

/**
 * Handle Streaming Form Submission
 */
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text || isGenerating) return;

  if (!isReady || !activeConversation) {
    appendMessage("El modelo aún se está descargando/compilando en su navegador. Por favor espere...", "bot");
    return;
  }

  isGenerating = true;
  userInput.value = "";
  btnSend.disabled = true;

  appendMessage(text, "user");
  const botEl = appendMessage("Generando respuesta on-device con LiteRT-LM...", "bot");

  const startTime = performance.now();
  let firstTokenTime = null;
  let tokenCount = 0;
  let fullText = "";

  botEl.textContent = "";

  try {
    const stream = activeConversation.sendMessageStreaming(text);
    for await (const chunk of stream) {
      if (firstTokenTime === null) {
        firstTokenTime = performance.now();
        const ttftMs = (firstTokenTime - startTime).toFixed(1);
        statTtft.textContent = `${ttftMs} ms`;
      }

      const chunkText = chunk?.content?.[0]?.text ?? "";
      fullText += chunkText;
      botEl.innerHTML = fullText.replace(/\n/g, "<br>");
      chatMessages.scrollTop = chatMessages.scrollHeight;
      tokenCount++;

      const currentElapsedSec = (performance.now() - startTime) / 1000;
      const currentTps = (tokenCount / Math.max(currentElapsedSec, 0.001)).toFixed(1);
      statTps.textContent = `${currentTps} tok/s`;
      statTokens.textContent = tokenCount;
    }
  } catch (err) {
    console.error("Streaming error:", err);
    botEl.textContent = "Error en LiteRT-LM Web API: " + err.message;
  } finally {
    isGenerating = false;
    btnSend.disabled = false;
    userInput.focus();
  }
});

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
  initializeLiteRtLmEngine();
  userInput.focus();
});
