# Sesion 3: Fine-Tuning de LLMs con Hugging Face y Unsloth

Esta sesion profundiza en el ciclo completo de post-entrenamiento de Modelos de Lenguaje Abiertos (LLMs/SLMs), desde el **Ajuste Fino Supervisado (SFT)** con adaptadores **LoRA**, pasando por la **Alineacion de Preferencias (Preference Alignment)** con **DPO (Direct Preference Optimization)**, hasta el post-entrenamiento acelerado por hardware con **Unsloth** y su exportacion directa a **GGUF** para serving en **Ollama y vLLM**.

- **Documento Teorico y Playbook de Arquitectura:** [Playbook Tecnico: LoRA, QLoRA y DPO (Sesion 3)](./playbook-tecnico-sesion-3.md)

---

## Modulos Teorico-Practicos

### Modulo 3.1: Supervised Fine-Tuning (SFT) y Adaptacion Parametrica Eficiente (PEFT / LoRA)
- **Del Pre-entrenamiento al Post-entrenamiento:** Por que los modelos base requieren afinamiento supervisado para adquirir capacidad instruccional y conocimiento de dominio.
- **Fundamentos de LoRA:** Descomposicion de bajo rango ($W = W_0 + \frac{\alpha}{r} BA$), evaluacion de parametros entrenables (< 0.5%) y reduccion del consumo de VRAM.
- **Plantillas de Chat (Chat Templates):** Estandarizacion conversacional con Jinja (`apply_chat_template`) y enmascaramiento de perdida (*response masking*).
- **Entrenamiento con TRL:** Configuracion de `SFTTrainer` y `SFTConfig` con optimizadores paginados AdamW.

### Modulo 3.2: Alineacion de Preferencias con Direct Preference Optimization (DPO)
- **Limitaciones del SFT:** Por que el SFT no puede penalizar alucinaciones ni modular sutilezas de estilo.
- **La Revolucion DPO vs RLHF Clasico:** Eliminacion del *Reward Model* y de la inestabilidad de PPO mediante una clasificacion binaria cerrada sobre razones de verosimilitud logaritmica.
- **Estructuracion de Preferencias:** Datasets ternarios con `prompt`, `chosen` (respuesta factual y concisa) y `rejected` (respuesta alucinada o verbosa).
- **Entrenamiento con `DPOTrainer`:** Control de divergencia KL con $\beta=0.1$ y tasas de aprendizaje conservadoras ($5\times 10^{-6}$).
- **Evaluacion Cuantitativa Rigurosa:** Medicion de *Reward Accuracy* intrinseca (split reservado) y protocolo ciego *LLM-as-a-Judge* con control de sesgo posicional (*Win Rate*).

### Modulo 3.3: Post-Entrenamiento Acelerado con Unsloth y Pipeline a Produccion (GGUF / Ollama)
- **Arquitectura de Kernels Triton:** Como Unsloth duplica la velocidad de calculo y ahorra hasta un 70% de VRAM reescribiendo atencion y derivadas analiticas.
- **LoRA Extendido:** Inyeccion de adaptadores en todas las matrices lineales (`q`, `k`, `v`, `o`, `gate`, `up`, `down`).
- **Salida Estructurada:** Entrenamiento para generacion y extraccion de datos en formato JSON estricto.
- **Exportacion a GGUF y Modelfiles:** Conversion a binarios cuantizados `q4_k_m` y creacion de manifiestos para despliegue inmediato en Ollama y vLLM (conexion con la Sesion 4).

---

## Indice de Hands-on Labs de la Sesion 3

| Laboratorio | Directorio | Cuaderno Principal | Enfoque Tecnico | Dominio / Tarea |
|---|---|---|---|---|
| **Lab 1 (SFT)** | [`01-sft-lora-huggingface/`](./01-sft-lora-huggingface/README.md) | [`01_sft_lora_huggingface.ipynb`](./01-sft-lora-huggingface/01_sft_lora_huggingface.ipynb) | Supervised Fine-Tuning con Hugging Face `trl` (`SFTTrainer`) y `peft` (LoRA). | Soporte Tecnico / Respuestas de Dominio |
| **Lab 2 (DPO)** | [`02-preference-alignment-dpo/`](./02-preference-alignment-dpo/README.md) | [`02_preference_alignment_dpo.ipynb`](./02-preference-alignment-dpo/02_preference_alignment_dpo.ipynb) | Preference Alignment con DPO y evaluacion cuantitativa dual (Reward Accuracy y LLM-as-a-Judge Win Rate con mitigacion de sesgo posicional). | Mitigacion de Alucinaciones y Concision |
| **Lab 3 (Unsloth)** | [`03-fast-finetuning-unsloth-gguf/`](./03-fast-finetuning-unsloth-gguf/README.md) | [`03_fast_finetuning_unsloth_gguf.ipynb`](./03-fast-finetuning-unsloth-gguf/03_fast_finetuning_unsloth_gguf.ipynb) | Fine-Tuning de alto rendimiento con kernels Triton de Unsloth y exportacion a binario GGUF con `Modelfile` para Ollama/vLLM. | Extraccion Estructurada en JSON / Serving |

---

## Estructura de Materiales

```text
session-03-fine-tuning-llms/
├── README.md                                      # Esta guia general de la sesion
├── playbook-tecnico-sesion-3.md                   # Playbook tecnico de LoRA, QLoRA y DPO
├── 01-sft-lora-huggingface/                       # Lab 1: SFT y LoRA con Hugging Face
│   ├── README.md
│   └── 01_sft_lora_huggingface.ipynb
├── 02-preference-alignment-dpo/                   # Lab 2: Alineacion DPO con Hugging Face TRL
│   ├── README.md
│   └── 02_preference_alignment_dpo.ipynb
└── 03-fast-finetuning-unsloth-gguf/               # Lab 3: Unsloth QLoRA y Exportacion GGUF para Ollama
    ├── README.md
    └── 03_fast_finetuning_unsloth_gguf.ipynb
```

---

## Referencias Oficiales y Documentacion

- **Hugging Face TRL (Transformer Reinforcement Learning):** [https://huggingface.co/docs/trl/](https://huggingface.co/docs/trl/)
- **Hugging Face PEFT (Parameter-Efficient Fine-Tuning):** [https://huggingface.co/docs/peft/](https://huggingface.co/docs/peft/)
- **Paper Original DPO (Rafailov et al., Stanford 2023):** [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- **Documentacion de Unsloth:** [https://docs.unsloth.ai/](https://docs.unsloth.ai/)
- **Guia Oficial del Formato GGUF en Hugging Face Hub:** [https://huggingface.co/docs/hub/gguf](https://huggingface.co/docs/hub/gguf)
- **Ecosistema Ollama:** [https://ollama.com/](https://ollama.com/)
