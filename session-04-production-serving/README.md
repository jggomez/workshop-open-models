# Sesion 4: Serving en Produccion con Ollama, vLLM y GCP Vertex AI

Esta sesion esta orientada a la puesta en produccion, despliegue de alta concurrencia y arquitectura de serving en la nube para modelos de lenguaje y vision.

---

## Contenido de la Sesion

### Modulo 4.1: Serving de Alto Rendimiento con Ollama y vLLM
- Arquitectura de inferencia para LLMs: Token generation, KV-cache y bottlenecks de memoria de video (VRAM).
- Optimizacion de concurrencia mediante PagedAttention y Continuous Batching con vLLM.
- Despliegue rapido local y perimetral con Ollama: ModelFiles, APIs REST y streaming.
- Benchmarking de latencia (Time To First Token - TTFT) y throughput (Tokens Per Second - TPS).

### Modulo 4.2: Despliegue en la Nube con Google Cloud Platform (GCP)
- Exploracion de Vertex AI Model Garden: Ingesta y hosting de modelos de codigo abierto (Llama, Gemma, Mistral).
- Empaquetado en contenedores personalizados (Custom Serving Containers) compatibles con Triton / vLLM.
- Configuracion de endpoints dedicados con auto-scaling basado en trafico y cola de peticiones.
- Monitoreo de metricas, seguridad de red y cuotas de hardware (GPUs NVIDIA L4/A100).

### Hands-on Lab 4: Serving Hibrido
- Levantamiento de un servidor vLLM local con soporte de API compatible con OpenAI (`/v1/chat/completions`).
- Despliegue de un endpoint administrado en Vertex AI GCP mediante `gcloud` / Python SDK de Vertex AI.
- Integracion de balanceo de carga y pruebas de estres con clientes concurrentes.

---

## Estructura de Materiales

- `01-ollama-local-serving/`
- `02-vllm-high-throughput/`
- `03-gcp-vertex-ai-deployment/`
- `notebooks/`

---

## Referencias Oficiales

- **vLLM Documentation:** [https://docs.vllm.ai/](https://docs.vllm.ai/)
- **Ollama Documentation:** [https://ollama.com/](https://ollama.com/)
- **Google Cloud Vertex AI Model Garden:** [https://cloud.google.com/vertex-ai/docs/model-garden/explore-models](https://cloud.google.com/vertex-ai/docs/model-garden/explore-models)
