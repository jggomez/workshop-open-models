# Hands-on Lab 1: KerasHub - Vision, LLMs y Modelos Multimodales (Gemma)

Este laboratorio practico ensena a utilizar **KerasHub** sobre **Keras 3** para la descarga, inspeccion, inferencia y fine-tuning de modelos de deep learning de vision por computador, modelos de lenguaje (LLMs) y arquitecturas multimodales de ultima generacion (**Google Gemma 4**).

---

## 1. Estructura del Laboratorio

```text
01-kerashub-image-classification/
├── README.md                              # Esta guia tecnica y explicativa
├── 01_kerashub_getting_started.ipynb      # Cuaderno oficial KerasHub (Vision, LLMs y Fine-Tuning)
├── 02_gemma4_multimodal_ai.ipynb          # Cuaderno multimodal completo con Google Gemma 4
└── 03_imagenet_classification_basics.ipynb # Cuaderno interactivo complementario de clasificacion
```

---

## 2. Cuadernos Interactivos

### Cuaderno 1: `01_kerashub_getting_started.ipynb` (Getting Started con KerasHub)
- **Autor:** Juan Guillermo Gomez
- **Badge Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jggomez/workshop-open-models/blob/main/session-01-hf-kerashub-litert/01-kerashub-image-classification/01_kerashub_getting_started.ipynb)
- **Contenido y Secciones:**
  1. **Configuracion Multi-Backend:** Ejecucion acelerada con **JAX** (`KERAS_BACKEND="jax"`), compilacion XLA y gestion de memoria.
  2. **Vision por Computador:** Clasificacion de imagenes con el preset pre-entrenado `resnet_50_imagenet` sobre ImageNet-1k, deconstruccion en *Preprocessor* y *Backbone*, y decodificacion de etiquetas con `decode_imagenet_predictions`.
  3. **Generacion de Texto con LLMs:** Carga del modelo causal `gemma3_1b`, generacion individual y por lotes (*batched generation*), configuracion de muestreo estocastico (`TopKSampler(k=10, temperature=2.0)`), e inspeccion del tokenizador de subpalabras.
  4. **Transfer Learning de Extremo a Extremo:** Adaptacion de ResNet-50 para clasificacion binaria en el dataset Microsoft Cats vs Dogs, pipeline de aumentacion (`Rescaling`, `RandomFlip`, `RandomCrop`), entrenamiento con optimizador Adam, serializacion de preset local (`save_to_preset`) y publicacion al Model Hub de Kaggle (`upload_preset`).

### Cuaderno 2: `02_gemma4_multimodal_ai.ipynb` (Inteligencia Artificial Multimodal con Gemma 4)
- **Badge Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jggomez/workshop-open-models/blob/main/session-01-hf-kerashub-litert/01-kerashub-image-classification/02_gemma4_multimodal_ai.ipynb)
- **Contenido y Secciones:**
  1. **Carga del Modelo Unificado:** Carga del preset `gemma4_instruct_2b` con precision `bfloat16`.
  2. **Generacion de Texto:** Dialogo conversacional estructurado con turnos `<|turn>user` y `<|turn>model`.
  3. **Vision Multimodal (Image Captioning):** Descripcion de imagenes pasando el token `<|image|>`.
  4. **Grounding Visual / Deteccion de Objetos:** Localizacion de objetos en la escena devolviendo cajas delimitadoras normalizadas `[ymin, xmin, ymax, xmax]` en escala 0-1000.
  5. **Comprension de Audio Nativa:** Transcripcion automatica de habla (ASR) procesando ondas sonoras directamente con el token `<|audio|>`.
  6. **Llamada a Herramientas (Function Calling):** Declaracion de esquemas JSON en el sistema y generacion de llamadas a herramientas (`<|tool>call:...`).
  7. **Generacion de Codigo:** Sintesis de algoritmos en Python con validacion de casos borde.
  8. **Modo Pensamiento y Flujos Agentic:** Razonamiento reflexivo interno con etiquetas `<|thought|>` para ejecucion autonoma de acciones paso a paso.
  9. **Comparacion Multi-Imagen:** Razonamiento visual cruzado entre multiples imagenes simultaneas.
  10. **OCR, Traduccion y Extraccion de Entidades:** Lectura de texto en senalizaciones y estructuracion de datos.
  11. **Planificacion de Viajes:** Reconocimiento de destinos a partir de fotografias y diseno de itinerarios.

### Cuaderno 3: `03_imagenet_classification_basics.ipynb` (Clasificacion de Imagenes en ImageNet)
- **Badge Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jggomez/workshop-open-models/blob/main/session-01-hf-kerashub-litert/01-kerashub-image-classification/03_imagenet_classification_basics.ipynb)
- Cuaderno introductorio para clasificacion de imagenes con MobileNetV3 sobre ImageNet.

---

## 3. Guia de Ejecucion

### Opcion 1: En Google Colab (Recomendada con Acelerador GPU/TPU)
Haga clic en cualquiera de los badges superiores para abrir directamente el cuaderno en el entorno en la nube de Google Colab.

### Opcion 2: En Entorno Local
```bash
# 1. Instalar dependencias consolidadas del workshop (desde la raiz)
pip install -r requirements.txt

# 2. Iniciar Jupyter Lab
jupyter lab session-01-hf-kerashub-litert/01-kerashub-image-classification/
```

---

## 4. Paso Final: Limpieza del Entorno (Cleanup)

```bash
# Limpiar caches y archivos temporales
./cleanup.sh
```

---

## 5. Referencias y Documentacion Oficial

- **Sitio Web Oficial de KerasHub:** [https://keras.io/keras_hub/](https://keras.io/keras_hub/)
- **Documentacion de Keras 3 Multi-Backend:** [https://keras.io/getting_started/](https://keras.io/getting_started/)
- **Catalogo de Modelos y Presets de KerasHub:** [https://keras.io/api/keras_hub/models/](https://keras.io/api/keras_hub/models/)
- **Modelos Google Gemma en KerasHub:** [https://keras.io/api/keras_hub/models/gemma/](https://keras.io/api/keras_hub/models/gemma/)
- **Kaggle Model Hub:** [https://www.kaggle.com/models](https://www.kaggle.com/models)
