# Uso Practico, Fine-Tuning y Serving de Modelos Open Source en Produccion

Repositorio oficial con los ejercicios practicos, codigo fuente, laboratorios interactivos y cuadernos Jupyter del workshop intensivo de modelos de codigo abierto (Open Source AI).

---

## Estructura del Workshop

El taller se divide en cuatro sesiones tematicas:

1. **Sesion 1: Hugging Face, KerasHub y el uso de LLMs con LiteRT / LiteRT-LM**
   - Modulo 1.1: Ecosistema Hugging Face (datasets, modelos, spaces) y KerasHub para vision y NLP.
   - Modulo 1.2: Descarga y gestion de modelos con Hugging Face y uso de LLMs con LiteRT-LM.
   - Hands-on Labs y Material Tecnico:
     * **Playbook de Ingenieria:** [Guia Teorico-Practica de Referencia (Sesion 1)](./session-01-hf-kerashub-litert/playbook-tecnico-sesion-1.md).
     * **Lab 1:** Cuadernos interactivos KerasHub (`01_kerashub_getting_started.ipynb`, `02_gemma4_multimodal_ai.ipynb`, `03_imagenet_classification_basics.ipynb`) en `01-kerashub-image-classification/`.
     * **Lab 2:** Inferencia web con LiteRT.js (`@litertjs/core`) en el navegador y cuaderno (`01_litert_interpreter_inspection.ipynb`) en `02-litert-web-vision/`.
     * **Lab 3:** Chat Web con la API oficial LiteRT-LM Web API y WebGPU (`03-litert-lm-cli-and-web/`).
     * **Lab 4:** Cuaderno interactivo de traduccion e inferencia multimodal VLM (`01_huggingface_translation_and_vlm.ipynb`) en `04-huggingface-translation/`.
     * **Lab 5:** Cuaderno interactivo de evaluacion de datasets de Hugging Face con Gemma Multimodal (`01_huggingface_datasets_gemma_multimodal.ipynb`) en `05-huggingface-gemma-datasets/`.

2. **Sesion 2: Transfer Learning con Keras y Hugging Face**
   - Modulo 2.1: Estrategias de Transfer Learning: Feature Extraction con cabezales personalizados, congelamiento de capas (layer freezing) y prevencion de olvido catastrofico.
   - Modulo 2.2: Fine-tuning progresivo en 2 etapas con tasas de aprendizaje discriminativas sobre arquitecturas convolucionales y Transformers.
   - Hands-on Labs:
     * **Lab 1 (Sencillo):** Transfer Learning en Vision por Computador con MobileNetV2 y clasificacion de flores (`01_transfer_learning_vision_basics.ipynb`) en `01-transfer-learning-vision-basics/`.
     * **Lab 2 (Avanzado):** Transfer Learning en NLP con BERT (`bert_tiny_en_uncased`), protocolo en 2 fases, matriz de confusion y analisis de sentimientos (`02_transfer_learning_nlp_advanced.ipynb`) en `02-transfer-learning-nlp-advanced/`.

3. **Sesion 3: Fine-Tuning de LLMs con Hugging Face y Unsloth (SFT, DPO y GGUF)**
   - Modulo 3.1: Supervised Fine-Tuning (SFT) y adaptacion parametrica eficiente con LoRA (`peft` y `trl`).
   - Modulo 3.2: Alineacion de preferencias (Preference Alignment) con DPO (Direct Preference Optimization).
   - Modulo 3.3: Post-entrenamiento acelerado con kernels Triton de Unsloth y exportacion a GGUF para Ollama/vLLM.
   - Hands-on Labs:
     * **Lab 1 (SFT):** Supervised Fine-Tuning con Hugging Face y LoRA (`01_sft_lora_huggingface.ipynb`) en `01-sft-lora-huggingface/`.
     * **Lab 2 (DPO):** Alineacion de preferencias con DPO (`02_preference_alignment_dpo.ipynb`) en `02-preference-alignment-dpo/`.
     * **Lab 3 (Unsloth):** Fine-Tuning ultra-rapido con Unsloth y exportacion a GGUF para Ollama (`03_fast_finetuning_unsloth_gguf.ipynb`) en `03-fast-finetuning-unsloth-gguf/`.

4. **Sesion 4: Serving en Produccion con Ollama, vLLM y GCP (Vertex AI Model Garden)**
   - Modulo 4.1: Serving serverless y perimetral con Ollama y Cloud Storage FUSE en Cloud Run.
   - Modulo 4.2: Serving de alta concurrencia con vLLM, PagedAttention, GPU NVIDIA L4 y Model Armor.
   - Modulo 4.3: Ingestion y hosting administrado en la nube con Vertex AI Model Garden (MaaS vs Dedicated Endpoints).
   - Hands-on Labs:
     * **Lab 1:** Inferencia local con Ollama (GGUF de Sesion 3) y despliegue a Cloud Run con Cloud Storage FUSE (`01-ollama-cloudrun-gcsfuse/`).
     * **Lab 2:** Serving empresarial con vLLM y Gemma en Cloud Run GPU, Model Armor y observabilidad (`02-vllm-gemma-cloudrun-production/`).
     * **Lab 3:** Despliegue de DeepSeek y Gemma con Vertex AI Model Garden (`03-vertex-ai-model-garden/`).

---

## Ejecucion de los Laboratorios Web

Para interactuar con los laboratorios interactivos y las aplicaciones web on-device (Lab 2 y Lab 3), simplemente inicie el portal interactivo del workshop desde la raiz del repositorio:

```bash
# Iniciar portal interactivo (puerto 3000)
python3 -m http.server 3000
```

Luego abra en su navegador: **[http://localhost:3000](http://localhost:3000)**

Desde este portal centralizado podra acceder y ejecutar con un solo clic:
- **Lab 3:** Chat Web On-Device con LiteRT-LM Web API y aceleracion por hardware WebGPU.
- **Lab 2:** Inferencia de Clasificacion de Vision en el navegador con LiteRT.js (`@litertjs/core`).
- Acceso directo y enlaces a la documentacion de los cuadernos de cada sesion.

---

## Requisitos Previos e Instalacion

### 1. Requisitos del Sistema
- Python 3.10 o superior (para ejecutar los cuadernos interactivos y el servidor HTTP local).
- Navegador moderno con soporte WebGPU y WebAssembly (Google Chrome, Microsoft Edge, Safari Tech Preview).
- Git.

*(Nota: Los laboratorios web se ejecutan completamente en el navegador del cliente mediante WebGPU, WebAssembly y modulos ES importados via CDN, por lo que no requieren instalar Node.js ni paquetes npm).*

### 2. Ejecucion de los Cuadernos
Todos los cuadernos interactivos (`.ipynb`) incluyen en su primera celda los comandos de instalacion correspondientes (`!pip install ...`), por lo que estan completamente listos para ejecutarse con un solo clic en Google Colab o en su entorno local con Jupyter Lab / VS Code.

---

## Navegacion de Sesiones

- [Sesion 1: Hugging Face, KerasHub y LiteRT / LiteRT-LM](./session-01-hf-kerashub-litert/README.md)
- [Sesion 2: Transfer Learning con Keras y Hugging Face](./session-02-transfer-learning/README.md)
- [Sesion 3: Fine-Tuning de LLMs con Hugging Face y Unsloth (SFT, DPO y GGUF)](./session-03-fine-tuning-llms/README.md)
- [Sesion 4: Serving en Produccion con Ollama, vLLM y GCP](./session-04-production-serving/README.md)

---

## Referencias Oficiales y Documentacion de Frameworks

- **Google AI Edge LiteRT (Documentacion Oficial):** [https://ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)
- **LiteRT.js Web Guide:** [https://developers.google.com/edge/litert/web/get_started](https://developers.google.com/edge/litert/web/get_started)
- **LiteRT-LM Web API Reference:** [https://developers.google.com/edge/litert-lm/js](https://developers.google.com/edge/litert-lm/js)
- **KerasHub / Keras 3 (Documentacion Oficial):** [https://keras.io/keras_hub/](https://keras.io/keras_hub/)
- **Hugging Face Transformers:** [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
- **Hugging Face Datasets:** [https://huggingface.co/docs/datasets/](https://huggingface.co/docs/datasets/)
- **Google Gemma Open Models:** [https://ai.google.dev/gemma](https://ai.google.dev/gemma)
- **Ollama Documentation:** [https://ollama.com/](https://ollama.com/)
- **vLLM Project:** [https://docs.vllm.ai/](https://docs.vllm.ai/)
