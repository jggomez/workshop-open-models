# Hands-on Lab 2 (Sesion 2): Transfer Learning Avanzado en NLP con Fine-Tuning Progresivo

Este laboratorio avanzado explora tecnicas de **Transfer Learning en Procesamiento de Lenguaje Natural (NLP)** adaptando un modelo Transformer bidireccional (**BERT** mediante el preset `bert_tiny_en_uncased` de KerasHub) para tareas de clasificacion de texto y analisis de sentimientos, implementando un protocolo de **Fine-Tuning Progresivo en 2 Fases** con **tasas de aprendizaje discriminativas**.

---

## 1. Fundamentos Teoricos: Olvido Catastrofico y Protocolo en 2 Fases

### Por que el Transfer Learning en Transformers requiere un protocolo en dos fases?
En los modelos de lenguaje basados en Transformers, inicializar un nuevo cabezal de clasificacion con pesos aleatorios y entrenar inmediatamente todo el modelo con una tasa de aprendizaje estandar (ej. `1e-3`) provoca que los gradientes iniciales desestabilicen y destruyan los pesos pre-entrenados del Transformer (*Catastrophic Forgetting*).

### Protocolo de 2 Fases:
1. **Fase 1 (Warm-up del Cabezal - Feature Extraction):**
   - Se congela el Transformer (`classifier.backbone.trainable = False`).
   - Se entrena unicamente el cabezal de clasificacion a una tasa de aprendizaje de `1e-3` durante 3 epocas.
   - El cabezal aprende a proyectar los vectores de salida del Transformer hacia las clases objetivo sin distorsionar el encoder.
2. **Fase 2 (Fine-Tuning Progresivo de Extremo a Extremo):**
   - Se descongela el Transformer (`classifier.backbone.trainable = True`).
   - Se reduce la tasa de aprendizaje 50 veces (`learning_rate=2e-5`).
   - Se permite que las matrices de atencion y capas densas intermedias se ajusten suavemente a las sutilezas lexicas del nuevo dataset sin descalibrar el conocimiento linguistico general.

---

## 2. Estructura del Laboratorio

```text
02-transfer-learning-nlp-advanced/
├── README.md                                # Esta guia paso a paso
└── 02_transfer_learning_nlp_advanced.ipynb  # Cuaderno interactivo listo para Google Colab y Jupyter
```

---

## 3. Contenido del Cuaderno (`02_transfer_learning_nlp_advanced.ipynb`)

- **Paso 1:** Configuracion de KerasHub y Keras 3 con backend JAX / TensorFlow.
- **Paso 2:** Creacion de datasets balanceados de entrenamiento y validacion con variabilidad sintactica y semantica.
- **Paso 3:** Carga del preset `bert_tiny_en_uncased` con tokenizador integrado (WordPiece con `[CLS]` y `[SEP]`).
- **Paso 4:** Fase 1: Congelamiento del backbone Transformer e inspeccion de parametros entrenables vs no entrenables.
- **Paso 5:** Entrenamiento de Warm-up para estabilizar el cabezal de clasificacion.
- **Paso 6:** Fase 2: Descongelamiento progresivo con tasa de aprendizaje conservadora (`2e-5`).
- **Paso 7:** Fine-tuning de extremo a extremo monitoreando la convergencia.
- **Paso 8:** Evaluacion cientifica rigurosa: Matriz de Confusion y reporte con Precision, Recall y F1-Score con `scikit-learn`.
- **Paso 9:** Inferencia cualitativa sobre oraciones complejas con negaciones, sarcasmo y ambiguedades.
- **Paso 10:** Liberacion de memoria y limpieza de tensores.

---

## 4. Guia de Ejecucion

### Opcion 1: En Google Colab (Recomendada con Acelerador GPU)
Haga clic en el siguiente enlace para abrir directamente el cuaderno en Google Colab:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jggomez/workshop-open-models/blob/main/session-02-transfer-learning/02-transfer-learning-nlp-advanced/02_transfer_learning_nlp_advanced.ipynb)

### Opcion 2: En Entorno Local
```bash
# 1. Instalar dependencias consolidadas (desde la raiz)
pip install -r requirements.txt

# 2. Iniciar Jupyter Lab
jupyter lab session-02-transfer-learning/02-transfer-learning-nlp-advanced/02_transfer_learning_nlp_advanced.ipynb
```

---

## 5. Referencias Oficiales

- **KerasHub Text Classification Guide:** [https://keras.io/api/keras_hub/models/bert/bert_classifier/](https://keras.io/api/keras_hub/models/bert/bert_classifier/)
- **KerasHub Presets Catalog:** [https://keras.io/api/keras_hub/models/](https://keras.io/api/keras_hub/models/)
- **BERT: Pre-training of Deep Bidirectional Transformers:** [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
