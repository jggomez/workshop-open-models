# Sesion 4: Serving en Produccion con Ollama, vLLM y GCP Vertex AI

Esta sesion esta orientada a la puesta en produccion, despliegue de alta concurrencia y arquitectura de serving en la nube para modelos de lenguaje y vision de codigo abierto, abarcando desde la inferencia perimetral y serverless con **Ollama** y **Google Cloud Run**, pasando por clusters de alto rendimiento con **vLLM**, hasta plataformas administradas con **Vertex AI Model Garden**.

- **Playbook de Ingenieria:** [Playbook de Ingenieria de Inferencia (Sesion 4)](./playbook-ingenieria-inferencia.md)
- **Prerrequisito y Fundamentos de Post-Entrenamiento:** [Playbook Tecnico de Fine-Tuning y Alineacion (Sesion 3)](../session-03-fine-tuning-llms/playbook-tecnico-sesion-3.md)

---

## Modulos Teorico-Practicos

### Modulo 4.1: Serving Ligero y Serverless con Ollama y Cloud Storage FUSE
- **Enlace de Artefactos:** Consumo e importacion de modelos cuantizados (GGUF) y manifiestos `Modelfile` generados tras el post-entrenamiento (Sesion 3).
- **Inferencia Local y APIs:** Interaccion via linea de comandos, endpoints REST nativos (`/api/generate`, `/api/chat`) e integracion con clientes compatibles con OpenAI (`/v1/chat/completions`).
- **Arquitectura Serverless en Cloud Run:** Desacoplamiento de datos y computo mediante **Cloud Storage Volume Mounts (GCS FUSE)**, evitando imagenes de contenedor pesadas y optimizando tiempos de despliegue.

### Modulo 4.2: Serving de Alto Rendimiento con vLLM y Seguridad Perimetral
- **Fundamentos Algoritmicos de vLLM:** Optimizacion de memoria de video mediante **PagedAttention** (reduccion de fragmentacion de KV-Cache) y **Continuous Batching** (procesamiento dinamico a nivel de token).
- **Arquitectura SRE en Google Cloud:** Despliegue de vLLM sobre Cloud Run con aceleracion por hardware **GPU NVIDIA L4** y pesos montados via GCS FUSE.
- **Seguridad Perimetral con Model Armor:** Intercepcion de trafico en el Load Balancer regional mediante Service Extensions para mitigar jailbreaks, prompt injections y fuga de datos (PII).
- **Observabilidad Empresarial:** Patron sidecar con Prometheus para recoleccion continua de latencia (TTFT), velocidad (TPS) y ocupacion de KV-Cache.

### Modulo 4.3: Ingestion y Hosting con Vertex AI Model Garden
- **Catalogo Unificado de Modelos:** Descubrimiento, evaluacion y seleccion de modelos de vanguardia (DeepSeek-R1 / V3, Google Gemma 2, Llama 3).
- **Modalidades de Despliegue:** Analisis comparativo entre **Model-as-a-Service (MaaS / Serverless API)** y **Dedicated Endpoints (Self-Deployed con GPUs dedicadas)**.
- **Invocacion Programatica:** Consumo mediante el SDK de Google Cloud (`google-cloud-aiplatform`) y autenticacion con Application Default Credentials (ADC).
- **Gobernanza de Costos:** Buenas practicas de escalado y desmantelamiento (*undeploy*) para optimizacion del gasto en GPU.

---

## Indice de Hands-on Labs de la Sesion 4

| Laboratorio | Directorio | Formato | Descripcion Tecnica | Tecnologias Clave |
|---|---|---|---|---|
| **Lab 1** | [`01-ollama-cloudrun-gcsfuse/`](./01-ollama-cloudrun-gcsfuse/README.md) | Shell + Docker + Python | Inferencia local con Ollama (GGUF de Sesion 3) y despliegue serverless a Cloud Run con Cloud Storage FUSE, Artifact Registry y protocolo de limpieza. | Ollama, GGUF, Cloud Run, GCS FUSE, Artifact Registry, Python |
| **Lab 2** | [`02-vllm-gemma-cloudrun-production/`](./02-vllm-gemma-cloudrun-production/README.md) | Blueprint SRE + Python Client | Serving de alta concurrencia con vLLM y Gemma 2 en Cloud Run GPU (L4), Model Armor y Prometheus sidecar. | vLLM, PagedAttention, Cloud Run GPU, Model Armor, Prometheus |
| **Lab 3** | [`03-vertex-ai-model-garden/`](./03-vertex-ai-model-garden/README.md) | Guia Markdown + Python SDK | Guia paso a paso para desplegar DeepSeek y Gemma en Vertex AI Model Garden (MaaS vs Dedicated Endpoints). | Vertex AI, Model Garden, DeepSeek, Gemma, Python SDK |

---

## Estructura de Materiales

```text
session-04-production-serving/
├── README.md                                # Esta guia general de la sesion
├── playbook-ingenieria-inferencia.md        # Playbook de Ingenieria de Inferencia (llama.cpp, Ollama, vLLM, Vertex AI)
├── 01-ollama-cloudrun-gcsfuse/              # Lab 1: Ollama Local y Despliegue en Cloud Run con GCS FUSE
│   ├── README.md
│   ├── Dockerfile
│   ├── deploy_cloud_run.sh
│   └── test_client.py
├── 02-vllm-gemma-cloudrun-production/       # Lab 2: Serving de Alto Rendimiento con vLLM y Gemma
│   ├── README.md
│   ├── architecture_blueprint.md
│   └── test_vllm_client.py
└── 03-vertex-ai-model-garden/               # Lab 3: Vertex AI Model Garden (DeepSeek y MaaS)
    ├── README.md
    └── test_vertex_model_garden.py
```

---

## Referencias Oficiales y Articulos Tecnicos

- **Repositorio de Automatizacion vLLM + Gemma:** [https://github.com/jggomez/AI-deploy-vllm-gemma](https://github.com/jggomez/AI-deploy-vllm-gemma)
- **Articulo Tecnico de Referencia (DeepSeek & Model Garden):** [Deploying Powerful Language Models with Ease: Leveraging DeepSeek and Model Garden on Vertex AI](https://jggomezt.medium.com/deploying-powerful-language-models-with-ease-leveraging-deepseek-and-model-garden-on-vertex-ai-2068dae099c1)
- **Documentacion de vLLM:** [https://docs.vllm.ai/](https://docs.vllm.ai/)
- **Documentacion de Ollama:** [https://ollama.com/](https://ollama.com/)
- **Google Cloud Run Volume Mounts:** [https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts](https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts)
- **Google Cloud Model Armor:** [https://cloud.google.com/security/products/model-armor](https://cloud.google.com/security/products/model-armor)
- **Google Cloud Vertex AI Model Garden:** [https://cloud.google.com/vertex-ai/docs/model-garden/explore-models](https://cloud.google.com/vertex-ai/docs/model-garden/explore-models)
