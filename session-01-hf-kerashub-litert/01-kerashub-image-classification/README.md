# Hands-on Lab 1: Clasificacion de Imagenes con KerasHub

Este laboratorio guia paso a paso en el uso de **KerasHub** y **Keras 3** para la descarga, inspeccion e inferencia de modelos de vision por computador pre-entrenados sobre el dataset ImageNet.

---

## 1. Fundamentos Teoricos

### Arquitectura de KerasHub
KerasHub organiza los modelos en tres capas desacopladas:
1. **Backbone:** Red neuronal base (ej. ResNet, MobileNet, EfficientNet) que extrae representaciones vectoriales densas a partir de imagenes o secuencias.
2. **Preprocessor:** Pipeline de transformacion determinista que aplica redimensionamiento, normalizacion de pixeles y empaquetado en tensores.
3. **Task Head / Classifier:** Capa final de proyeccion que mapea las caracteristicas extraidas al espacio de etiquetas objetivo (1,000 clases en el caso de ImageNet).

### Flexibilidad Multi-Backend
Keras 3 desacopla la definicion del grafo del motor de computo subyacente. Puede ejecutarse de forma intercambiable sobre:
- PyTorch (`KERAS_BACKEND=torch`)
- JAX (`KERAS_BACKEND=jax`)
- TensorFlow (`KERAS_BACKEND=tensorflow`)

---

## 2. Estructura del Laboratorio

```text
01-kerashub-image-classification/
├── README.md                    # Esta guia paso a paso
├── classify.py                  # Script principal de inferencia por linea de comandos
├── download_sample_images.py    # Utilidad para descargar imagenes de prueba
├── requirements.txt             # Dependencias de Python para este laboratorio
├── notebook.ipynb               # Cuaderno interactivo listo para Google Colab
└── sample_images/               # Directorio con imagenes de prueba
    ├── dog.jpg
    ├── cat.jpg
    └── car.jpg
```

---

## 3. Guia Paso a Paso de Ejecucion

### Paso 1: Activar el Entorno Virtual e Instalar Dependencias
Desde la raiz del repositorio:

```bash
source .venv/bin/activate
pip install -r session-01-hf-kerashub-litert/01-kerashub-image-classification/requirements.txt
```

### Paso 2: Descargar las Imagenes de Prueba
Ejecute el script de descarga para obtener imagenes de muestra de alta resolucion:

```bash
python3 session-01-hf-kerashub-litert/01-kerashub-image-classification/download_sample_images.py
```

### Paso 3: Ejecutar la Inferencia por Consola
Ejecute el script `classify.py` pasando una de las imagenes descargadas:

```bash
export KERAS_BACKEND=torch
python3 session-01-hf-kerashub-litert/01-kerashub-image-classification/classify.py --image session-01-hf-kerashub-litert/01-kerashub-image-classification/sample_images/dog.jpg --top_k 5
```

Salida esperada:
```text
Loading vision classifier preset: 'mobilenet_v3_small_imagenet' (Backend: torch)...
Model loaded successfully in 1.42 seconds.

Loading image from: session-01-hf-kerashub-litert/01-kerashub-image-classification/sample_images/dog.jpg
Running inference...
Inference completed in 18.54 ms.

--- Top 5 Predictions ---
1. Golden Retriever               Confidence: 55.34% (ID: n02099712)
2. Labrador Retriever             Confidence: 10.74% (ID: n02099601)
3. Tennis Ball                    Confidence: 2.82%  (ID: n04409515)
4. Kuvasz                         Confidence: 1.45%  (ID: n02104029)
5. Cocker Spaniel                 Confidence: 1.12%  (ID: n02102318)
```

---

## 4. Ejecucion en Google Colab
Abra [notebook.ipynb](file:///Users/jggomez/Documents/jggomez/code/workshop-open-source-models/session-01-hf-kerashub-litert/01-kerashub-image-classification/notebook.ipynb) en Google Colab con el badge interactivo incluido en la cabecera.

---

## 5. Paso Final: Limpieza del Entorno (Cleanup)

Para eliminar las imagenes de prueba descargadas y liberar memoria:

```bash
# Eliminar carpeta de imagenes de prueba descargadas
rm -rf session-01-hf-kerashub-litert/01-kerashub-image-classification/sample_images

# O ejecutar el script de limpieza general desde la raiz
./cleanup.sh
```

---

## 6. Referencias y Documentacion Oficial

- **Sitio Web Oficial de KerasHub:** [https://keras.io/keras_hub/](https://keras.io/keras_hub/)
- **Documentacion de Keras 3 Multi-Backend:** [https://keras.io/getting_started/](https://keras.io/getting_started/)
- **Catalogo de Modelos y Presets de KerasHub:** [https://keras.io/api/keras_hub/models/](https://keras.io/api/keras_hub/models/)
- **Repositorio de KerasHub en GitHub:** [https://github.com/keras-team/keras-hub](https://github.com/keras-team/keras-hub)
- **Keras Applications (ImageNet Classifiers):** [https://keras.io/api/applications/](https://keras.io/api/applications/)
