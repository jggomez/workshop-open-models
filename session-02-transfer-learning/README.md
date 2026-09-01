# Sesion 2: Transfer Learning con Keras y Hugging Face

Esta sesion cubre tecnicas avanzadas de Transfer Learning para adaptar modelos pre-entrenados de vision y procesamiento de lenguaje natural a tareas especificas de negocio.

---

## Contenido de la Sesion

### Modulo 2.1: Estrategias de Transfer Learning
- Conceptos fundamentales: Feature Extraction vs. Fine-Tuning progresivo.
- Congelamiento de capas (layer freezing) y descongelamiento selectivo.
- Diseno y acople de nuevos cabezales de clasificacion (custom task heads) con Keras 3.
- Carga e integracion de backbones pre-entrenados provenientes de Hugging Face Hub y KerasHub.

### Hands-on Lab 2: Pipeline Completo de Transfer Learning
- Carga de un dataset especializado de clasificacion.
- Construccion del pipeline de aumentacion y preprocesamiento de datos.
- Entrenamiento en dos fases:
  1. Entrenamiento de cabezal personalizado manteniendo el backbone congelado.
  2. Ajuste fino (fine-tuning) con tasa de aprendizaje reducida en las capas superiores.
- Evaluacion de metricas de rendimiento (Precision, Recall, F1-Score, Matriz de Confusion) y exportacion del modelo final.

---

## Estructura de Materiales

- `01-feature-extraction-vision/`
- `02-fine-tuning-nlp-backbone/`
- `notebooks/`

---

## Referencias Oficiales

- **Keras 3 Transfer Learning Guide:** [https://keras.io/guides/transfer_learning/](https://keras.io/guides/transfer_learning/)
- **KerasHub Vision Backbones:** [https://keras.io/api/keras_hub/models/](https://keras.io/api/keras_hub/models/)
- **Hugging Face Hub Integration:** [https://huggingface.co/docs/hub/](https://huggingface.co/docs/hub/)
