# Hands-on Lab 3 (Sesion 3): Fine-Tuning Ultra-Rapido con Unsloth y Exportacion a GGUF para Ollama

Este laboratorio practico avanzado aborda tecnicas de alto rendimiento en el post-entrenamiento de Modelos de Lenguaje Abiertos empleando **Unsloth**, aceleracion por hardware con **kernels manuales en OpenAI Triton**, salida estructurada en **formato JSON**, y exportacion directa a binario cuantizado **GGUF** con manifiesto **`Modelfile` para Ollama**, sirviendo como puente de integracion con la **Sesion 4 (Serving en Produccion)**.

---

## 1. Fundamentos Tecnologicos: Unsloth, Triton y Formato GGUF

### Que hace a Unsloth tan eficiente?
**Unsloth** no es simplemente un envoltorio sobre PyTorch; reescribe las operaciones matematicas mas criticas de la arquitectura Transformer directamente en codigo **OpenAI Triton**:
1. **Kernels de Atencion Optimizados:** Computa la atencion manual y RoPE sin crear tensores intermedios que saturen la memoria de video.
2. **Backpropagation Manual:** Calcula los gradientes analiticos exactos de Cross-Entropy y RMSNorm ahorrando hasta un **70% de VRAM**.
3. **Throughput de Entrenamiento:** Logra entre **2x y 5x mayor velocidad** frente a pipelines convencionales de Hugging Face, manteniendo exactamente el **0% de perdida en precision**.

### El Formato GGUF y el Ecosistema Ollama / vLLM
El formato **GGUF (Georgi Gerganov Unified Format)** es el estandar universal de la comunidad de codigo abierto para inferencia perimetral y serving de alto rendimiento:
- Permite empaquetar pesos cuantizados (e.g. `q4_k_m`) y tokenizadores en un unico archivo autocontenido.
- Es el formato nativo para motores como **Ollama**, **vLLM** y **llama.cpp**.
- Permite desplegar el modelo resultante en servidores locales o en la nube con un simple `Modelfile`.

---

## 2. Estructura del Laboratorio

```text
03-fast-finetuning-unsloth-gguf/
├── README.md                              # Esta guia explicativa y metodologica
└── 03_fast_finetuning_unsloth_gguf.ipynb  # Cuaderno interactivo optimizado para Google Colab y Jupyter
```

---

## 3. Contenido del Cuaderno (`03_fast_finetuning_unsloth_gguf.ipynb`)

- **Paso 1:** Instalacion optimizada de Unsloth con soporte CUDA para entornos de GPU en la nube (Google Colab).
- **Paso 2:** Carga del modelo base `unsloth/gemma-2-2b-it-bnb-4bit` mediante `FastLanguageModel.from_pretrained` (contexto de hasta 2048 tokens en menos de 4 GB de VRAM).
- **Paso 3:** Inyeccion de adaptadores LoRA sobre todos los modulos lineales (`q`, `k`, `v`, `o`, `gate`, `up`, `down`) con $r=16$.
- **Paso 4:** Estructuracion de un dataset corporativo para clasificacion de incidentes con **salida en formato JSON estricto**.
- **Paso 5:** Entrenamiento acelerado con `trl.SFTTrainer` y optimizaciones de gradientes de Unsloth.
- **Paso 6:** Inferencia token por token de alta velocidad activando `FastLanguageModel.for_inference(model)`.
- **Paso 7:** Exportacion del modelo a formato binario cuantizado **GGUF** (`q4_k_m`).
- **Paso 8:** Generacion automatizada del archivo `Modelfile` y comandos para servir el modelo con **Ollama** (`ollama create ...` y `ollama run ...`).
- **Paso 9:** Liberacion de tensores y memoria VRAM.

---

## 4. Guia de Ejecucion

### Opcion 1: En Google Colab (Recomendada con Acelerador GPU T4/A100)
Haga clic en el siguiente enlace para abrir directamente el cuaderno en Google Colab:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jggomez/workshop-open-models/blob/main/session-03-fine-tuning-llms/03-fast-finetuning-unsloth-gguf/03_fast_finetuning_unsloth_gguf.ipynb)

### Opcion 2: En Entorno Local (Requiere GPU NVIDIA con soporte CUDA y drivers actualizados)
```bash
# Iniciar Jupyter Lab
jupyter lab session-03-fine-tuning-llms/03-fast-finetuning-unsloth-gguf/03_fast_finetuning_unsloth_gguf.ipynb
```

---

## 5. Referencias Oficiales

- **Repositorio Oficial de Unsloth:** [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- **Documentacion de Unsloth:** [https://docs.unsloth.ai/](https://docs.unsloth.ai/)
- **Documentacion Oficial de Ollama:** [https://ollama.com/](https://ollama.com/)
- **Especificacion del Formato GGUF (llama.cpp):** [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
