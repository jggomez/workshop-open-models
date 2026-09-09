# Hands-on Lab 3 (Sesion 3): Fine-Tuning Ultra-Rapido con Unsloth y Exportacion a GGUF para Ollama

Este laboratorio practico avanzado aborda tecnicas de alto rendimiento en el post-entrenamiento de Modelos de Lenguaje Abiertos empleando **Unsloth**, aceleracion por hardware con **kernels manuales en OpenAI Triton**, salida estructurada en **formato JSON**, y exportacion directa a binario cuantizado **GGUF** con manifiesto **`Modelfile` para Ollama**, sirviendo como puente de integracion directo con la [Sesion 4 - Lab 1: Serving con Ollama y GCS FUSE](../../session-04-production-serving/01-ollama-cloudrun-gcsfuse/README.md).

---

## 1. Fundamentos Tecnologicos: Unsloth, Triton y Formato GGUF

### Que hace a Unsloth tan eficiente?
**Unsloth** no es simplemente un envoltorio sobre PyTorch; reescribe las operaciones matematicas mas criticas de la arquitectura Transformer directamente en codigo **OpenAI Triton**:
1. **Kernels de Atencion Optimizados:** Computa la atencion manual y RoPE sin crear tensores intermedios que saturen la memoria de video.
2. **Backpropagation Manual:** Calcula los gradientes analiticos exactos de Cross-Entropy y RMSNorm ahorrando hasta un **70% de VRAM**.
3. **Throughput de Entrenamiento:** Logra entre **2x y 5x mayor velocidad** frente a pipelines convencionales de Hugging Face, manteniendo exactamente el **0% de perdida en precision**.

### El Formato GGUF y el Ecosistema Ollama / vLLM
El formato **GGUF (Georgi Gerganov Unified Format)**, concebido originalmente por Georgi Gerganov para `llama.cpp` y adoptado de forma nativa por Hugging Face, es el estandar universal de la comunidad de codigo abierto para inferencia perimetral y serving de alto rendimiento:

1. **GGUF frente a Safetensors:**
   A diferencia de formatos que almacenan exclusivamente tensores numericos puros como `safetensors` (que requieren archivos independientes como `tokenizer.json`, `config.json` y plantillas Jinja), **GGUF codifica tanto los tensores cuantizados como un conjunto estandarizado de metadatos** (arquitectura del modelo, vocabulario completo, hiperparametros y plantillas de chat) en un unico archivo binario autocontenido optimizado para carga instantanea via `mmap`.

2. **Esquema de Cuantizacion K-quants (`q4_k_m`):**
   Utiliza super-bloques con factores de escala diferenciados (`block_scale` y `block_min`), alcanzando ~4.5 bits reales por parametro. A diferencia de cuantizaciones heredadas de 4 bits (`q4_0` o `q4_1`), `q4_k_m` preserva la fidelidad de razonamiento del modelo pre-entrenado reduciendo el consumo de VRAM y RAM en mas del 70%.

3. **Ecosistema Integrado y Soporte en Hugging Face Hub:**
   El Hugging Face Hub dispone de soporte nativo para inspeccionar la metadata y tensores de archivos GGUF en su visor web, exploracion por etiqueta (`library=gguf`), integracion con bibliotecas cliente como `@huggingface/gguf` y herramientas de conversion automatizada (`ggml-org/gguf-my-repo`). Los motores de serving **Ollama**, **vLLM** y **llama.cpp** consumen este binario directamente mediante un archivo `Modelfile`.

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
- **Guia Oficial del Formato GGUF en Hugging Face Hub:** [https://huggingface.co/docs/hub/gguf](https://huggingface.co/docs/hub/gguf)
- **Especificacion del Formato GGUF (llama.cpp):** [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
