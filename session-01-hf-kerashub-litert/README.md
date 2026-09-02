# Sesion 1: Hugging Face, KerasHub y el uso de LLMs con LiteRT / LiteRT-LM

Esta sesion proporciona los fundamentos practicos para la descarga, gestion, preprocesamiento e inferencia de modelos de Deep Learning y Modelos de Lenguaje (LLMs/SLMs) utilizando los frameworks mas adoptados de la industria: Hugging Face, KerasHub y Google LiteRT.

---

## Modulos Teorico-Practicos

### Modulo 1.1: Ecosistema Hugging Face y KerasHub
- **Hugging Face Hub:** Navegacion y uso de Model Cards, Dataset Cards y Spaces. Autenticacion con tokens y organizacion de repositorios.
- **KerasHub:** Filosofia de diseno modular basada en Backbones, Preprocessors y Task Classifiers. Compatibilidad multi-backend (TensorFlow, PyTorch, JAX).
- **Consumo de Modelos:** Descarga automatica de presets pre-entrenados para vision por computador, procesamiento de lenguaje natural y modelos multimodales (Gemma 4).

### Modulo 1.2: LiteRT y LiteRT-LM para Ejecucion On-Device
- **LiteRT (anteriormente TensorFlow Lite):** Arquitectura de inferencia de baja latencia para dispositivos perimetrales, moviles y navegadores web.
- **LiteRT-LM:** Runtime especializado de inferencia para Grandes Modelos de Lenguaje con optimizacion de memoria KV-cache, cuantizacion y aceleracion por hardware (WebGPU).
- **Inferencia en el Cliente:** Reduccion de costos de servidor y preservacion de privacidad procesando datos directamente en el cliente con JavaScript y WebAssembly.

---

## Indice de Hands-on Labs

| Laboratorio | Directorio | Formato | Descripcion Tecnica | Tecnologias |
|---|---|---|---|---|
| **Lab 1** | `01-kerashub-image-classification/` | Notebooks Interactivos | Vision con ResNet-50, LLMs con Gemma 3, fine-tuning Cats vs Dogs y multimodalidad con Gemma 4. | Python, KerasHub, JAX, Gemma 4 |
| **Lab 2** | `02-litert-web-vision/` | Web App + Notebook | Inferencia de vision en el navegador mediante LiteRT.js (`@litertjs/core`). | JavaScript, LiteRT.js, WebAssembly, ai-edge-litert |
| **Lab 3** | `03-litert-lm-cli-and-web/` | Web App | Inferencia de SLMs/LLMs on-device con la API oficial LiteRT-LM Web API. | JavaScript, LiteRT-LM Web API, WebGPU |
| **Lab 4** | `04-huggingface-translation/` | Notebook Interactivo | Traduccion con `pipeline()` e inferencia multimodal con `AutoModelForMultimodalLM`. | Python, Transformers, MarianMT, SmolVLM |
| **Lab 5** | `05-huggingface-gemma-datasets/` | Notebook Interactivo | Descarga de datasets con `datasets` e inferencia multimodal con Gemma (`AutoModelForMultimodalLM`). | Python, Datasets, Transformers, Gemma |

---

## Referencias Oficiales y Sitios Web

- **Google AI Edge LiteRT:** [https://ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)
- **LiteRT.js Web Guide:** [https://developers.google.com/edge/litert/web/get_started](https://developers.google.com/edge/litert/web/get_started)
- **LiteRT-LM Web API Reference:** [https://developers.google.com/edge/litert-lm/js](https://developers.google.com/edge/litert-lm/js)
- **KerasHub:** [https://keras.io/keras_hub/](https://keras.io/keras_hub/)
- **Hugging Face Transformers:** [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
- **Hugging Face Datasets:** [https://huggingface.co/docs/datasets/](https://huggingface.co/docs/datasets/)
- **Google Gemma Open Models:** [https://ai.google.dev/gemma](https://ai.google.dev/gemma)
