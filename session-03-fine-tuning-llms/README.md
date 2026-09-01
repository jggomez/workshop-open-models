# Sesion 3: Fine-Tuning de LLMs (Gemma) con Hugging Face

Esta sesion profundiza en el ajuste fino supervisado (SFT) de Grandes Modelos de Lenguaje (LLMs) empleando tecnicas parametricamente eficientes (PEFT / LoRA).

---

## Contenido de la Sesion

### Modulo 3.1: Adaptacion Parametrica Eficiente (PEFT y LoRA)
- Limitaciones del fine-tuning completo (Full Fine-Tuning) en recursos de computo y memoria.
- Fundamentos matematicos de LoRA (Low-Rank Adaptation) y QLoRA (Quantized LoRA).
- Inyeccion de adaptadores de bajo rango en matrices de proyeccion de atencion (`q_proj`, `v_proj`).
- Evaluacion de parametros entrenables vs. parametros congelados.

### Modulo 3.2: Datasets de Instrucciones, Chat Templates e Hiperparametros
- Formateo de datasets estilo instruccional (User / Assistant / System).
- Estandarizacion con Jinja Chat Templates (`apply_chat_template`).
- Configuracion del `SFTTrainer` (biblioteca `trl` de Hugging Face).
- Seleccion optima de hiperparametros: learning rate, warmup ratio, scheduler, gradient accumulation steps y precision mixta (`bf16`/`fp16`).

### Hands-on Lab 3: SFT con Gemma y PEFT
- Preparacion del dataset instruccional en formato JSONL / Hugging Face Datasets.
- Configuracion de `LoraConfig` y cuantizacion de 4 bits (`BitsAndBytesConfig`).
- Ejecucion del pipeline de entrenamiento con validacion periodica y logging.
- Fusion de adaptadores LoRA (merge_and_unload) y guardado del artefacto final en Hugging Face Hub / formato local.

---

## Estructura de Materiales

- `01-peft-lora-configuration/`
- `02-instruction-dataset-formatting/`
- `03-sft-gemma-training/`
- `notebooks/`

---

## Referencias Oficiales

- **Hugging Face PEFT Documentation:** [https://huggingface.co/docs/peft/](https://huggingface.co/docs/peft/)
- **Hugging Face TRL (Transformer Reinforcement Learning):** [https://huggingface.co/docs/trl/](https://huggingface.co/docs/trl/)
- **Google Gemma Model Fine-Tuning:** [https://ai.google.dev/gemma/docs/lora_tuning](https://ai.google.dev/gemma/docs/lora_tuning)
