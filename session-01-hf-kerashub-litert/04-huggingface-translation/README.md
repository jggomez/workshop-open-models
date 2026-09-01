# Hands-on Lab 4: Hugging Face Transformers - Traduccion Automatica y Modelos Multimodales (VLM)

Este laboratorio interactivo ensena a implementar y ejecutar modelos de lenguaje y vision en el ecosistema **Hugging Face Transformers** mediante un cuaderno Jupyter optimizado para **Google Colab** y ejecucion local en dos partes:
1. Inferencia de alto nivel para Sequence-to-Sequence (traduccion automatica) utilizando **`from transformers import pipeline`**.
2. Procesamiento conjunto de vision y texto en Vision-Language Models (VLM) mediante **`from transformers import AutoProcessor, AutoModelForMultimodalLM`**.

---

## 1. Estructura del Laboratorio

```text
04-huggingface-translation/
├── README.md                 # Esta guia tecnica y explicativa
├── notebook.ipynb            # Cuaderno interactivo optimizado para Google Colab y Jupyter
└── requirements.txt          # Dependencias de Python para este laboratorio
```

---

## 2. Contenido del Cuaderno (`notebook.ipynb`)

### Parte A: Traduccion Automatica con `from transformers import pipeline`
- Carga de tokenizador y modelo Sequence-to-Sequence (`Helsinki-NLP/opus-mt-es-en`).
- Creacion del pipeline de inferencia `pipeline("text2text-generation", ...)`.
- Evaluacion de inferencia individual y por lotes (*Batch Processing*) con analisis de latencia.

### Parte B: Modelos Multimodales con `AutoProcessor` y `AutoModelForMultimodalLM`
- Carga de `AutoProcessor` y `AutoModelForMultimodalLM` con soporte de hardware acelerado.
- Descarga y visualizacion de imagenes de muestra con `PIL` y `matplotlib`.
- Formateo de plantillas conversacionales multimodales (*Chat Templates*) con tokens `<image>`.
- Generacion y decodificacion de respuestas para *Visual Question Answering* (VQA).

### Parte C: Limpieza de Memoria
- Liberacion explicita de tensores en VRAM/RAM (`del model, processor`, `gc.collect()`, `torch.cuda.empty_cache()`).

---

## 3. Guia de Ejecucion

### Opcion 1: En Google Colab (Recomendada)
Haga clic en el siguiente enlace para abrir y ejecutar directamente en Colab con GPU gratuita:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jggomez/workshop-open-source-models/blob/main/session-01-hf-kerashub-litert/04-huggingface-translation/notebook.ipynb)

### Opcion 2: En Entorno Local
```bash
# 1. Activar entorno virtual
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r session-01-hf-kerashub-litert/04-huggingface-translation/requirements.txt

# 3. Iniciar Jupyter Lab o VS Code Notebook
jupyter lab session-01-hf-kerashub-litert/04-huggingface-translation/notebook.ipynb
```

---

## 4. Paso Final: Limpieza del Entorno (Cleanup)

```bash
# Limpiar caches de Hugging Face
rm -rf ~/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-es-en

# O ejecutar el script de limpieza general
./cleanup.sh --cache
```

---

## 5. Referencias y Documentacion Oficial

- **Hugging Face Transformers Pipelines:** [https://huggingface.co/docs/transformers/main_classes/pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
- **Hugging Face AutoModelForMultimodalLM Reference:** [https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForMultimodalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForMultimodalLM)
- **Hugging Face AutoProcessor Reference:** [https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor)
- **Modelos Helsinki-NLP Opus-MT:** [https://huggingface.co/Helsinki-NLP](https://huggingface.co/Helsinki-NLP)
