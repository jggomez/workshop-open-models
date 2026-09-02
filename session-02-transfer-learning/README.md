# Sesion 2: Transfer Learning con Keras y Hugging Face

Esta sesion proporciona las tecnicas metodologicas y arquitecturas avanzadas de **Transfer Learning (Aprendizaje por Transferencia)** para adaptar modelos pre-entrenados de vision por computador y procesamiento de lenguaje natural (NLP) a problemas especificos de negocio con minima necesidad de datos y computo.

---

## Modulos Teorico-Practicos

### Modulo 2.1: Fundamentos de Transfer Learning y Feature Extraction
- **Mecanismos de Transferencia:** Por que las representaciones jerarquicas de redes neuronales profundas generalizan eficazmente a dominios no vistos.
- **Congelamiento de Capas (Layer Freezing):** Preservacion de pesos pre-entrenados y desacoplamiento de gradientes.
- **Diseno de Cabezales Personalizados (Custom Heads):** Pooling espacial, capas de regularizacion (*Dropout*) y normalizaciones para adaptacion a nuevos espacios de etiquetas.

### Modulo 2.2: Fine-Tuning Progresivo en Modelos Complejos y Transformers
- **Prevencion del Olvido Catastrofico (Catastrophic Forgetting):** Como evitar que los gradientes iniciales descalibren los pesos de atencion pre-entrenados.
- **Protocolo de 2 Fases:** Warm-up con backbone congelado seguido de fine-tuning progresivo con descongelamiento selectivo.
- **Tasas de Aprendizaje Discriminativas (Discriminative Learning Rates):** Uso de tasas de aprendizaje 50x a 100x menores en capas base pre-entrenadas vs capas superiores.

---

## Indice de Hands-on Labs de la Sesion 2

| Laboratorio | Directorio | Cuaderno Principal | Enfoque Tecnico | Dominio |
|---|---|---|---|---|
| **Lab 1 (Sencillo)** | `01-transfer-learning-vision-basics/` | `01_transfer_learning_vision_basics.ipynb` | Extraccion de caracteristicas con backbone convolucional pre-entrenado congelado (**MobileNetV2** en ImageNet) y cabezal personalizado. | Vision por Computador (Flores) |
| **Lab 2 (Avanzado)** | `02-transfer-learning-nlp-advanced/` | `02_transfer_learning_nlp_advanced.ipynb` | Protocolo en 2 fases con **BERT** (`bert_tiny_en_uncased`), descongelamiento progresivo, matriz de confusion y analisis de sentimientos. | NLP / Lenguaje Natural (Texto) |

---

## Estructura de Materiales

```text
session-02-transfer-learning/
├── README.md                                # Esta guia general de la sesion
├── 01-transfer-learning-vision-basics/      # Lab 1: Vision - Feature Extraction con MobileNetV2
│   ├── README.md
│   └── 01_transfer_learning_vision_basics.ipynb
└── 02-transfer-learning-nlp-advanced/       # Lab 2: NLP - Fine-Tuning Progresivo con BERT
    ├── README.md
    └── 02_transfer_learning_nlp_advanced.ipynb
```

---

## Referencias Oficiales y Sitios Web

- **Keras 3 Transfer Learning Guide:** [https://keras.io/guides/transfer_learning/](https://keras.io/guides/transfer_learning/)
- **Keras Applications:** [https://keras.io/api/applications/](https://keras.io/api/applications/)
- **KerasHub Text Classifiers:** [https://keras.io/api/keras_hub/models/bert/bert_classifier/](https://keras.io/api/keras_hub/models/bert/bert_classifier/)
- **KerasHub Model Presets:** [https://keras.io/api/keras_hub/models/](https://keras.io/api/keras_hub/models/)
