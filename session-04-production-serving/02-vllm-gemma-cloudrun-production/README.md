# Hands-on Lab 2 (Sesion 4): Serving de Alta Concurrencia con vLLM y Gemma en Cloud Run GPU

Este laboratorio practico aborda la arquitectura de serving de nivel empresarial para modelos de lenguaje masivos (**Google Gemma**), basada en las especificaciones de ingenieria y automatizacion del repositorio oficial [`AI-deploy-vllm-gemma`](https://github.com/jggomez/AI-deploy-vllm-gemma).

---

## 1. Fundamentos de Arquitectura: Por que vLLM en Produccion?

A diferencia de servidores de inferencia basados puramente en Hugging Face o FastAPI simple, **vLLM** es el motor estandar de la industria para despliegues de alto rendimiento por dos pilares tecnicos fundamentales:

1. **PagedAttention:** Reduce el desperdicio de memoria de video (*VRAM fragmentation*) del 60%-80% a menos del 4%, almacenando tokens no contiguos en bloques discretos de memoria virtual análogos a las páginas del kernel del sistema operativo.
2. **Continuous Batching (In-Flight Batching):** Permite procesar de forma concurrente cientos de peticiones entrantes agregando y liberando tokens a nivel de ciclo de iteracion individual, aumentando el throughput de 10x a 24x frente a servidores estandares.

---

## 2. Arquitectura del Sistema (Blueprint SRE)

La infraestructura combina cinco componentes en Google Cloud Platform:
- **Regional External Application Load Balancer:** Punto de entrada seguro con direccion IP estatica y terminacion TLS (HTTPS).
- **Google Cloud Model Armor (Service Extension):** Intercepcion de trafico en el gateway de red para filtrar jailbreaks, inyecciones de prompts y fuga de PII antes de que el trafico llegue a la GPU.
- **Cloud Run con GPU NVIDIA L4:** Servicio serverless auto-escalable con aceleracion por hardware Ada Lovelace (24 GB VRAM).
- **Cloud Storage FUSE (Volume Mounts):** Montaje dinamico de los pesos del modelo Gemma desde un bucket regional en `/mnt/models`, eliminando imagenes Docker pesadas.
- **Sidecar de Observabilidad (Prometheus GMP):** Recoleccion continua de latencia TTFT (Time to First Token), Throughput (TPS) y ocupacion de KV-Cache.

> Para consultar la especificacion completa de componentes y matrices de seguridad, revise el documento [architecture_blueprint.md](./architecture_blueprint.md).

---

## 3. Estructura del Laboratorio

```text
02-vllm-gemma-cloudrun-production/
├── README.md                  # Esta guia tecnica y paso a paso
├── architecture_blueprint.md  # Especificacion detallada de arquitectura y seguridad
└── test_vllm_client.py        # Cliente de pruebas con streaming, medicion de TPS y test de seguridad
```

---

## 4. Guia Paso a Paso de Despliegue en Google Cloud

Los siguientes pasos describen la secuencia de provisionamiento automatizada implementada en [`AI-deploy-vllm-gemma`](https://github.com/jggomez/AI-deploy-vllm-gemma):

### Paso 1: Configuracion del Entorno e IAM
Configure las variables principales y cree el Service Account de ejecucion (`vllm-sa`) con permisos minimos estrictos:

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export MODEL_ID="google/gemma-2-2b-it"
export BUCKET_NAME="${PROJECT_ID}-vllm-models"

# Habilitar APIs requeridas
gcloud services enable \
  run.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  modelarmor.googleapis.com \
  networkservices.googleapis.com \
  secretmanager.googleapis.com
```

### Paso 2: Descarga Eficiente de Pesos Canónicos a GCS
En lugar de descargar los pesos en cada arranque del contenedor, se ejecuta un trabajo automatizado en **Cloud Build** utilizando la CLI de Hugging Face (`hf download`) para almacenar los pesos directamente en el bucket de GCS:

```bash
# Registrar token de Hugging Face en Secret Manager
echo -n "TU_HF_TOKEN" | gcloud secrets create hf-token --data-file=-

# Los pesos se descargan directamente a: gs://${BUCKET_NAME}/gemma/
```

### Paso 3: Despliegue del Contenedor vLLM con GPU y FUSE
El contenedor de vLLM se ejecuta en Cloud Run con acelerador NVIDIA L4 y el bucket montado en `/mnt/models`:

```bash
gcloud run deploy vllm-gemma \
  --image="vllm/vllm-openai:latest" \
  --platform=managed \
  --region="${REGION}" \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu=4 \
  --memory=16Gi \
  --no-cpu-throttling \
  --timeout=600 \
  --concurrency=16 \
  --execution-environment=gen2 \
  --add-volume="name=gemma-weights,type=cloud-storage,bucket=${BUCKET_NAME}" \
  --add-volume-mount="volume=gemma-weights,mount-path=/mnt/models" \
  --args="--model=/mnt/models/gemma,--port=8000,--gpu-memory-utilization=0.90,--max-model-len=4096"
```

### Paso 4: Gateway de Seguridad con Model Armor
Se configura un Serverless NEG apuntando al servicio Cloud Run y se adjunta un **Service Extension** en el Load Balancer para aplicar la plantilla de politicas de Model Armor (inspeccion de entrada y salida contra prompt injections y toxicidad).

---

## 5. Verificacion y Pruebas con el Cliente de Rendimiento

Utilice el script `test_vllm_client.py` provisto en este laboratorio para validar tanto el endpoint local como el servicio desplegado en Cloud Run:

```bash
# Invocacion del cliente contra el endpoint en produccion
python3 test_vllm_client.py https://<LOAD_BALANCER_IP_O_CLOUD_RUN_URL> google/gemma-2-2b-it
```

El script ejecuta automaticamente:
1. **Comprobacion de Salud:** Verificacion de endpoints `/health` y `/v1/models`.
2. **Inferencia con Streaming en Tiempo Real:** Decodificacion progresiva de tokens y medicion de:
   - **Time-to-First-Token (TTFT):** Latencia hasta el primer token emitido (tipicamente < 250 ms en GPU L4).
   - **Throughput (TPS):** Velocidad de generacion sostenida de tokens por segundo.
3. **Validacion de Proteccion Perimetral:** Envio de un vector de ataque de jailbreak para comprobar que Model Armor bloquea la peticion (HTTP 400/403) antes de tocar el cómputo de la GPU.

---

## 6. Referencias Oficiales y Repositorio Base

- **Repositorio de Automatizacion SRE:** [https://github.com/jggomez/AI-deploy-vllm-gemma](https://github.com/jggomez/AI-deploy-vllm-gemma)
- **Documentacion Oficial de vLLM:** [https://docs.vllm.ai/](https://docs.vllm.ai/)
- **PagedAttention Paper (Kwon et al., 2023):** [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **Google Cloud Model Armor:** [https://cloud.google.com/security/products/model-armor](https://cloud.google.com/security/products/model-armor)
- **Cloud Run con GPUs:** [https://cloud.google.com/run/docs/configuring/services/gpu](https://cloud.google.com/run/docs/configuring/services/gpu)
