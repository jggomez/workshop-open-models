# Hands-on Lab 1 (Sesion 3): Fine-Tuning Supervisado con Hugging Face (SFT y LoRA)

Este laboratorio practico ensena la metodologia estandar de la industria para el **Ajuste Fino Supervisado (Supervised Fine-Tuning - SFT)** de Modelos de Lenguaje Abiertos (LLMs) empleando la biblioteca **`trl` (Transformer Reinforcement Learning)** y adaptadores parametricamente eficientes **LoRA (`peft`)**.

---

## 1. Fundamentos Teoricos: SFT y LoRA

### Que es el Supervised Fine-Tuning (SFT)?
A diferencia del pre-entrenamiento auto-supervisado (donde el modelo solo aprende a predecir la siguiente palabra en texto crudo), el **SFT** entrena al modelo sobre pares estructurados de instrucciones y respuestas. Esto permite:
1. Ensenar al modelo el formato conversacional (turnos `user` y `assistant`).
2. Adaptar el estilo, tono y terminologia a un dominio corporativo o tecnico cerrado.
3. Reducir respuestas evasivas o genericas incorporando respuestas institucionales verificables.

### Descomposicion de Bajo Rango con LoRA
Para evitar actualizar los miles de millones de parametros del modelo base (lo que causaria desbordamiento de memoria VRAM), **LoRA** congela la matriz original $W_0$ y aprende dos matrices de bajo rango $A$ y $B$:

$$W = W_0 + \frac{\alpha}{r} (B \times A)$$

- Solo se entrena menos del **0.5%** de los parametros totales.
- El adaptador resultante ocupa menos de **25 MB** de almacenamiento.
- Se puede fusionar con los pesos base en produccion mediante `merge_and_unload()`.

---

## 2. Estructura del Laboratorio

```text
01-sft-lora-huggingface/
├── README.md                      # Esta guia explicativa y metodologica
└── 01_sft_lora_huggingface.ipynb  # Cuaderno interactivo optimizado para Google Colab y Jupyter
```

---

## 3. Contenido del Cuaderno (`01_sft_lora_huggingface.ipynb`)

- **Paso 1:** Instalacion automatizada de dependencias (`transformers`, `trl`, `peft`, `bitsandbytes`, `accelerate`).
- **Paso 2:** Deteccion de hardware acelerado (CUDA / Apple MPS).
- **Paso 3:** Carga del modelo base (`google/gemma-2-2b-it` o `Qwen/Qwen2.5-1.5B-Instruct`) con precision `bfloat16` y cuantizacion 4-bit opcional (`BitsAndBytesConfig`).
- **Paso 4:** Evaluacion de referencia inicial (*baseline*) demostrando las respuestas genericas del modelo antes del fine-tuning.
- **Paso 5:** Estructuracion de un dataset instruccional en formato de mensajes (`[{"role": "user", ...}, {"role": "assistant", ...}]`).
- **Paso 6:** Inyeccion de adaptadores LoRA con `LoraConfig` ($r=8$, $\alpha=16$, $target\_modules=["q\_proj", "v\_proj", "k\_proj", "o\_proj"]$).
- **Paso 7:** Entrenamiento con `SFTTrainer` y `SFTConfig` utilizando optimizador AdamW paginado.
- **Paso 8:** Evaluacion cualitativa comparativa demostrando la adopcion del conocimiento y estilo institucional.
- **Paso 9:** Guardado de adaptadores y explicacion del metodo `merge_and_unload()`.
- **Paso 10:** Liberacion explicita de tensores y memoria VRAM.

---

## 4. Guia de Ejecucion

### Opcion 1: En Google Colab (Recomendada con GPU T4 Gratuita)
Haga clic en el siguiente enlace para abrir directamente el cuaderno en el entorno en la nube de Google Colab:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jggomez/workshop-open-models/blob/main/session-03-fine-tuning-llms/01-sft-lora-huggingface/01_sft_lora_huggingface.ipynb)

### Opcion 2: En Entorno Local
```bash
# Iniciar Jupyter Lab (las dependencias se auto-instalan en la primera celda del cuaderno)
jupyter lab session-03-fine-tuning-llms/01-sft-lora-huggingface/01_sft_lora_huggingface.ipynb
```

---

## 5. Referencias Oficiales

- **Hugging Face TRL SFTTrainer:** [https://huggingface.co/docs/trl/sft_trainer](https://huggingface.co/docs/trl/sft_trainer)
- **Hugging Face PEFT Documentation:** [https://huggingface.co/docs/peft/](https://huggingface.co/docs/peft/)
- **Paper Original de LoRA (Hu et al., 2021):** [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
