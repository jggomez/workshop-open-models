# Hands-on Lab 2 (Sesion 3): Alineacion de Preferencias en LLMs con DPO (Direct Preference Optimization)

Este laboratorio practico avanzado aborda la frontera de post-entrenamiento de Modelos de Lenguaje Abiertos: la **Alineacion de Preferencias (Preference Alignment)** empleando el algoritmo **DPO (Direct Preference Optimization)** a traves de la biblioteca **`trl` (Transformer Reinforcement Learning)** de Hugging Face.

---

## 1. Fundamentos Teoricos: DPO vs RLHF Clasico

### Por que la Alineacion de Preferencias es Necesaria?
El fine-tuning supervisado (SFT) entrena al modelo a imitar texto, pero no puede penalizar comportamientos indeseados. Cuando el modelo se enfrenta a situaciones ambiguas, puede:
1. Alucinar informacion con total seguridad sintactica.
2. Responder de manera excesivamente verbosa e ineficiente.
3. Producir recomendaciones inseguras o contradictorias.

### La Formulacion Matematica de DPO
En lugar de entrenar un *Reward Model* separado y optimizar mediante tecnicas complejas e inestables de Aprendizaje por Refuerzo como PPO (que requieren 4 modelos simultaneos en VRAM), **DPO (Rafailov et al., Stanford 2023)** demuestra que la perdida de alineacion puede calcularse directamente mediante una clasificacion binaria cerrada:

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

- $x$: Instruccion o prompt.
- $y_w$ (*chosen*): Respuesta deseada, factual y concisa.
- $y_l$ (*rejected*): Respuesta defectuosa o alucinada.
- $\beta$: Coeficiente de penalizacion de divergencia KL (tipicamente $0.1$).

### Evaluacion Cuantitativa Rigurosa (Superando la Inspeccion Subjetiva)
Para validar cientificamente que el modelo aprendio las preferencias deseadas, implementamos dos niveles de medicion cuantitativa reproducible:
1. **Nivel 1: Metricas Intrinsecas de Recompensa (Reward Margin & Accuracy):**
   Calcula la recompensa implicita $r_\theta(x, y) = \beta (\log \pi_\theta(y|x) - \log \pi_{ref}(y|x))$ y el margen $\Delta r = r_\theta(x, y_w) - r_\theta(x, y_l)$ sobre un conjunto de prueba reservado. Mide el **Reward Accuracy (%)**, que pasa de un ~50% azaroso en el modelo base a >80%-100% en el modelo DPO.
2. **Nivel 2: Protocolo LLM-as-a-Judge con Control de Sesgos (Win Rate):**
   Enfrentamiento ciego pareado (*pairwise blind comparison*) entre el modelo base y el modelo DPO, mitigando activamente el **sesgo posicional (Position Bias)** mediante evaluacion cruzada (`[A, B]` y `[B, A]`) y penalizando la verborrea innecesaria (*Length Bias*).

---

## 2. Estructura del Laboratorio

```text
02-preference-alignment-dpo/
├── README.md                          # Esta guia explicativa y metodologica
└── 02_preference_alignment_dpo.ipynb  # Cuaderno interactivo optimizado para Google Colab y Jupyter
```

---

## 3. Contenido del Cuaderno (`02_preference_alignment_dpo.ipynb`)

- **Paso 1:** Instalacion de `trl`, `transformers`, `peft`, `accelerate` y `bitsandbytes`.
- **Paso 2:** Verificacion de hardware y soporte CUDA.
- **Paso 3:** Carga del modelo base (`google/gemma-2-2b-it` o `Qwen/Qwen2.5-1.5B-Instruct`) cuantizado en 4-bit (`nf4`).
- **Paso 4:** Creacion del dataset de preferencias ternarias con split formal 80/20 (`train_dpo_dataset` y `eval_dpo_dataset`).
- **Paso 5:** Configuracion de adaptadores LoRA mediante `peft.LoraConfig`.
- **Paso 6:** Entrenamiento de alineacion con `trl.DPOTrainer` y `trl.DPOConfig` ($\beta=0.1$, $lr=5\times 10^{-6}$).
- **Paso 7:** **Evaluacion Cuantitativa Nivel 1:** Calculo determinista de log-probabilidades, margen de recompensa $\Delta r$ y *Reward Accuracy (%)* en el split reservado.
- **Paso 8:** **Evaluacion Cuantitativa Nivel 2:** Protocolo ciego *LLM-as-a-Judge*, mitigacion de sesgo posicional cruzado A/B y calculo de *Win Rate (%)*.
- **Paso 9:** Evaluacion cualitativa comparativa de respuestas lado a lado (Base vs DPO).
- **Paso 10:** Exportacion y guardado de los adaptadores LoRA alineados.
- **Paso 11:** Liberacion de memoria y tensores VRAM.

---

## 4. Guia de Ejecucion

### Opcion 1: En Google Colab (Recomendada con GPU T4 Gratuita)
Haga clic en el siguiente enlace para abrir directamente el cuaderno en el entorno en la nube de Google Colab:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jggomez/workshop-open-models/blob/main/session-03-fine-tuning-llms/02-preference-alignment-dpo/02_preference_alignment_dpo.ipynb)

### Opcion 2: En Entorno Local
```bash
# Iniciar Jupyter Lab (las dependencias se auto-instalan en la primera celda del cuaderno)
jupyter lab session-03-fine-tuning-llms/02-preference-alignment-dpo/02_preference_alignment_dpo.ipynb
```

---

## 5. Referencias Oficiales

- **Paper Original de DPO (Rafailov et al., 2023):** [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- **Hugging Face TRL DPOTrainer:** [https://huggingface.co/docs/trl/dpo_trainer](https://huggingface.co/docs/trl/dpo_trainer)
- **Hugging Face Alignment Handbook:** [https://github.com/huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook)
- **AlpacaEval (LLM-as-a-Judge Benchmark):** [https://github.com/tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)
- **MT-Bench (Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena):** [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)
