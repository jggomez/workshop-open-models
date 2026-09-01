# Hands-on Lab 5: Hugging Face - Datasets e Inferencia con LLMs y Modelos Multimodales (Gemma)

Este laboratorio interactivo ensena a combinar la biblioteca **`datasets`** de Hugging Face con arquitecturas generativas multimodales (**`AutoProcessor`** y **`AutoModelForMultimodalLM`**) de la familia **Google Gemma** mediante un cuaderno interactivo optimizado para **Google Colab** y ejecucion local.

---

## 1. Estructura del Laboratorio

```text
05-huggingface-gemma-datasets/
├── README.md                 # Esta guia tecnica y explicativa
├── notebook.ipynb            # Cuaderno interactivo optimizado para Google Colab y Jupyter
└── requirements.txt          # Dependencias de Python para este laboratorio
```

---

## 2. Contenido del Cuaderno (`notebook.ipynb`)

### Paso 1: Descarga de Datasets desde Hugging Face Hub (`datasets.load_dataset`)
- Descarga automatica de conjuntos de datos de instrucciones directamente del repositorio central de Hugging Face.
- Inspeccion de esquemas de columnas, caracteristicas (*features*) y filtrado de muestras.

### Paso 2: Carga del Modelo con `AutoProcessor` y `AutoModelForMultimodalLM`
- Carga de la arquitectura multimodal (`google/gemma-4-12B-it`, `google/paligemma-3b-pt-224` o modelo compacto VLM).
- Uso de `device_map="auto"` para distribuir los tensores de pesos entre la GPU y la memoria RAM.

### Paso 3: Evaluacion sobre el Dataset con Chat Templates
- Aplicacion de plantillas de chat estandarizadas (`processor.apply_chat_template`) sobre las instrucciones del dataset.
- Inferencia automatizada, medicion de latencia de respuesta y calculo de velocidad de generacion (*tokens por segundo*).

### Paso 4: Razonamiento Multimodal (Imagen + Prompt)
- Procesamiento conjunto de tensores de pixeles y secuencias de texto para *Visual Question Answering*.

### Paso 5: Limpieza de Memoria
- Liberacion explicita de tensores y recoleccion de basura (`del model, processor`, `gc.collect()`, `torch.cuda.empty_cache()`).

---

## 3. Guia de Ejecucion

### Opcion 1: En Google Colab (Recomendada con GPU T4 / A100)
Haga clic en el siguiente enlace para abrir y ejecutar el cuaderno en Google Colab:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jggomez/workshop-open-source-models/blob/main/session-01-hf-kerashub-litert/05-huggingface-gemma-datasets/notebook.ipynb)

### Opcion 2: En Entorno Local
```bash
# 1. Activar entorno virtual
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r session-01-hf-kerashub-litert/05-huggingface-gemma-datasets/requirements.txt

# 3. Iniciar Jupyter Lab o VS Code Notebook
jupyter lab session-01-hf-kerashub-litert/05-huggingface-gemma-datasets/notebook.ipynb
```

---

## 4. Paso Final: Limpieza del Entorno (Cleanup)

```bash
# Limpiar caches del laboratorio
rm -rf ~/.cache/huggingface/datasets
rm -rf ~/.cache/huggingface/hub

# O ejecutar el script de limpieza general
./cleanup.sh --cache
```

---

## 5. Referencias y Documentacion Oficial

- **Biblioteca Hugging Face Datasets:** [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)
- **Modelos Google Gemma en Hugging Face:** [https://huggingface.co/google](https://huggingface.co/google)
- **Hugging Face AutoModelForMultimodalLM:** [https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForMultimodalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForMultimodalLM)
- **Hugging Face AutoProcessor:** [https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoProcessor)
