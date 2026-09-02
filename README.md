# Uso Practico, Fine-Tuning y Serving de Modelos Open Source en Produccion

Repositorio oficial con los ejercicios practicos, codigo fuente, laboratorios interactivos y cuadernos Jupyter del workshop intensivo de modelos de codigo abierto (Open Source AI).

---

## Estructura del Workshop

El taller se divide en cuatro sesiones tematicas:

1. **Sesion 1: Hugging Face, KerasHub y el uso de LLMs con LiteRT / LiteRT-LM**
   - Modulo 1.1: Ecosistema Hugging Face (datasets, modelos, spaces) y KerasHub para vision y NLP.
   - Modulo 1.2: Descarga y gestion de modelos con Hugging Face y uso de LLMs con LiteRT-LM.
   - Hands-on Labs:
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

3. **Sesion 3: Fine-Tuning de LLMs (Gemma) con el Ecosistema Hugging Face**
   - Modulo 3.1: Adaptacion parametrica eficiente (PEFT y LoRA).
   - Modulo 3.2: Formateo de datasets de instrucciones, chat templates y configuracion de hiperparametros de SFT (Supervised Fine-Tuning).
   - Hands-on Lab: Fine-tuning supervisado de Gemma con Transformers, TRL y PEFT.

4. **Sesion 4: Serving en Produccion con Ollama, vLLM y GCP (Vertex AI Model Garden)**
   - Modulo 4.1: Serving de alto rendimiento con Ollama y vLLM (PagedAttention).
   - Modulo 4.2: Despliegue en la nube con Google Cloud Platform (GCP) y Vertex AI Model Garden.
   - Hands-on Lab: Arquitectura de serving hibrido (local y cloud).

---

## Ejecucion de los Laboratorios Web

Para interactuar con los laboratorios web on-device (Lab 2 y Lab 3), puede iniciar el portal interactivo del workshop:

```bash
# Iniciar portal interactivo (puerto 3000)
python3 -m http.server 3000
# y abrir en el navegador: http://localhost:3000
```

O ejecutar cada laboratorio directamente con npm o Python:
```bash
# Laboratorio 3 (Chat Web On-Device con LiteRT-LM Web API / WebGPU en http://localhost:3001)
npm run lab3
# o con Python:
npm run lab3:py

# Laboratorio 2 (Clasificacion de Vision en Navegador con LiteRT.js en http://localhost:3000)
npm run lab2
# o con Python:
npm run lab2:py
```

---

## Requisitos Previos e Instalacion

### 1. Requisitos del Sistema
- Python 3.10 o superior.
- Node.js 18 o superior (para los laboratorios web con JavaScript).
- Git.
- Navegador moderno con soporte WebGPU (Google Chrome, Microsoft Edge, Safari Tech Preview).

### 2. Ejecucion de los Cuadernos
Todos los cuadernos interactivos (`.ipynb`) incluyen en su primera celda los comandos de instalacion correspondientes (`!pip install ...`), por lo que estan completamente listos para ejecutarse con un solo clic en Google Colab o en su entorno local con Jupyter Lab / VS Code.

---

## Navegacion de Sesiones

- [Sesion 1: Hugging Face, KerasHub y LiteRT / LiteRT-LM](./session-01-hf-kerashub-litert/README.md)
- [Sesion 2: Transfer Learning con Keras y Hugging Face](./session-02-transfer-learning/README.md)
- [Sesion 3: Fine-Tuning de LLMs (Gemma) con Hugging Face](./session-03-fine-tuning-llms/README.md)
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
