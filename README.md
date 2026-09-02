# Uso Practico, Fine-Tuning y Serving de Modelos Open Source en Produccion

Repositorio oficial con los ejercicios practicos, codigo fuente, laboratorios interactivos y cuadernos Jupyter del workshop intensivo de modelos de codigo abierto (Open Source AI).

---

## Estructura del Workshop

El taller se divide en cuatro sesiones tematicas:

1. **Sesion 1: Hugging Face, KerasHub y el uso de LLMs con LiteRT / LiteRT-LM**
   - Modulo 1.1: Ecosistema Hugging Face (datasets, modelos, spaces) y KerasHub para vision y NLP.
   - Modulo 1.2: Descarga y gestion de modelos con Hugging Face y uso de LLMs con LiteRT-LM.
   - Hands-on Labs:
     * **Lab 1:** Cuadernos interactivos KerasHub (`Keras_Hub.ipynb`, `Gemma4.ipynb`, `notebook.ipynb`) en `01-kerashub-image-classification/`.
     * **Lab 2:** Inferencia web con LiteRT.js (`@litertjs/core`) en el navegador (`02-litert-web-vision/`).
     * **Lab 3:** Chat Web con la API oficial LiteRT-LM Web API y WebGPU (`03-litert-lm-cli-and-web/`).
     * **Lab 4:** Cuaderno interactivo de traduccion e inferencia multimodal VLM (`04-huggingface-translation/`).
     * **Lab 5:** Cuaderno interactivo de evaluacion de datasets de Hugging Face con Gemma Multimodal (`05-huggingface-gemma-datasets/`).

2. **Sesion 2: Transfer Learning con Keras y Hugging Face**
   - Modulo 2.1: Estrategias de Transfer Learning: Personalizacion de cabezales (heads), congelamiento de capas (layer freezing) e integracion de modelos desde Hugging Face usando Keras.
   - Hands-on Lab: Pipeline completo de adaptacion y entrenamiento supervisado sobre un caso real.

3. **Sesion 3: Fine-Tuning de LLMs (Gemma) con el Ecosistema Hugging Face**
   - Modulo 3.1: Adaptacion parametrica eficiente (PEFT y LoRA).
   - Modulo 3.2: Formateo de datasets de instrucciones, chat templates y configuracion de hiperparametros de SFT (Supervised Fine-Tuning).
   - Hands-on Lab: Fine-tuning supervisado de Gemma con Transformers, TRL y PEFT.

4. **Sesion 4: Serving en Produccion con Ollama, vLLM y GCP (Vertex AI Model Garden)**
   - Modulo 4.1: Serving de alto rendimiento con Ollama y vLLM (PagedAttention).
   - Modulo 4.2: Despliegue en la nube con Google Cloud Platform (GCP) y Vertex AI Model Garden.
   - Hands-on Lab: Arquitectura de serving hibrido (local y cloud).

---

## Ejecucion Rapida de Laboratorios Web (1 Clic)

El repositorio incluye scripts ejecutables directos que inician el servidor y abren automaticamente el navegador:

```bash
# Ejecutar Laboratorio 3 (Chat Web On-Device con LiteRT-LM Web API / WebGPU)
./run_lab3.sh

# Ejecutar Laboratorio 2 (Clasificacion de Vision en Navegador con LiteRT.js)
./run_lab2.sh
```

Tambien puede abrir el portal interactivo de navegacion ejecutando:
```bash
python3 -m http.server 3000
# y abrir en el navegador: http://localhost:3000
```

---

## Requisitos Previos e Instalacion

### 1. Requisitos del Sistema
- Python 3.10 o superior.
- Node.js 18 o superior (para los laboratorios web con JavaScript).
- Git.
- Navegador moderno con soporte WebGPU (Google Chrome, Microsoft Edge, Safari Tech Preview).

### 2. Configuracion del Entorno Virtual de Python

```bash
# Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd workshop-open-source-models

# Crear y activar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias requeridas
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Limpieza del Entorno (Cleanup)

Para restablecer el entorno de trabajo y no dejar archivos temporales, imagenes descargadas ni modelos pesados en el disco local:

```bash
# Limpieza rapida (archivos temporales, imagenes y binarios descargados)
./cleanup.sh

# Limpieza completa (incluyendo eliminacion del entorno virtual .venv)
./cleanup.sh --venv

# Limpieza total (incluyendo caches globales de Hugging Face y Keras)
./cleanup.sh --all
```

Tambien puede ejecutarse mediante el script de Python multiplataforma:
```bash
python3 scripts/cleanup.py --all
```

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
