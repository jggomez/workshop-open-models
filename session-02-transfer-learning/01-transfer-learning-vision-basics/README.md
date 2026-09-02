# Hands-on Lab 1 (Sesion 2): Transfer Learning Basico para Vision por Computador

Este laboratorio ensena los conceptos fundamentales del **Aprendizaje por Transferencia (Transfer Learning)** adaptando una arquitectura pre-entrenada en ImageNet (**MobileNetV2**) a un nuevo dominio de clasificacion de imagenes (**Flores: 5 clases**) mediante la tecnica de **Extraccion de Caracteristicas (Feature Extraction)** con Keras 3.

---

## 1. Fundamentos Teoricos: Feature Extraction

En la estrategia de **Feature Extraction**:
1. Tomamos una red neuronal convolucional previamente entrenada sobre un dataset masivo (ImageNet-1k con 1.4 millones de imagenes).
2. **Congelamos el backbone** (`base_model.trainable = False`), conservando intactos los filtros que ya saben detectar bordes, texturas, formas y patrones semanticos generales.
3. Descartamos la capa densa superior original de 1,000 clases y acoplamos un nuevo cabezal de clasificacion:
   - `GlobalAveragePooling2D()`
   - `Dropout(0.2)`
   - `Dense(num_classes=5, activation="softmax")`
4. Entrenamos unicamente los pesos del nuevo cabezal, logrando una alta precision en pocas epocas y con una fraccion minima del costo computacional de un entrenamiento completo desde cero.

---

## 2. Estructura del Laboratorio

```text
01-transfer-learning-vision-basics/
├── README.md                                # Esta guia paso a paso
├── 01_transfer_learning_vision_basics.ipynb  # Cuaderno interactivo listo para Google Colab y Jupyter
└── requirements.txt                         # Dependencias de Python para ejecucion local
```

---

## 3. Contenido del Cuaderno (`01_transfer_learning_vision_basics.ipynb`)

- **Paso 1:** Instalacion y configuracion del entorno multi-backend de Keras 3.
- **Paso 2:** Descarga automatica y estructuracion del dataset `flower_photos` (margaritas, dientes de leon, rosas, girasoles, tulipanes).
- **Paso 3:** Creacion de pipelines `image_dataset_from_directory` con split 80/20 y prefetching.
- **Paso 4:** Aumentacion de datos en tiempo de entrenamiento (`RandomFlip`, `RandomRotation`, `RandomZoom`).
- **Paso 5:** Instanciacion de `MobileNetV2` sin cabezal (`include_top=False`), congelamiento de pesos y ensamblaje de la red.
- **Paso 6:** Compilacion con optimizador Adam (`learning_rate=1e-3`) y entrenamiento supervisado.
- **Paso 7:** Generacion de curvas de convergencia (Accuracy y Loss) para diagnosticar overfitting.
- **Paso 8:** Inferencia cualitativa sobre imagenes reales con porcentaje de confianza.
- **Paso 9:** Liberacion de memoria (`gc.collect()`).

---

## 4. Guia de Ejecucion

### Opcion 1: En Google Colab (Recomendada con Acelerador GPU)
Haga clic en el siguiente enlace para abrir directamente el cuaderno en Google Colab:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jggomez/workshop-open-models/blob/main/session-02-transfer-learning/01-transfer-learning-vision-basics/01_transfer_learning_vision_basics.ipynb)

### Opcion 2: En Entorno Local
```bash
# 1. Instalar dependencias
pip install -r session-02-transfer-learning/01-transfer-learning-vision-basics/requirements.txt

# 2. Iniciar Jupyter Lab
jupyter lab session-02-transfer-learning/01-transfer-learning-vision-basics/01_transfer_learning_vision_basics.ipynb
```

---

## 5. Referencias Oficiales

- **Guia Oficial de Transfer Learning en Keras 3:** [https://keras.io/guides/transfer_learning/](https://keras.io/guides/transfer_learning/)
- **Keras Applications (MobileNetV2):** [https://keras.io/api/applications/mobilenet/](https://keras.io/api/applications/mobilenet/)
