# Hands-on Lab 1 (Sesion 4): Serving con Ollama (Local y Cloud Run con Cloud Storage FUSE)

Este laboratorio ensena el ciclo de puesta en produccion de modelos cuantizados empleando **Ollama**, desde la importacion local del modelo **GGUF** afinado en la Sesion 3 (Unsloth) hasta su despliegue serverless escalable en **Google Cloud Run** aplicando la mejor practica arquitectonica de **Cloud Storage FUSE (Volume Mounts)**.

---

## 1. Fundamentos Tecnologicos: Por que Ollama y por que GCS FUSE?

### El Rol de Ollama en Inferencia Ligera
**Ollama** empaqueta el motor de inferencia `llama.cpp` en una arquitectura de servicio que proporciona:
- Gestion automatizada de memoria y descompresion de pesos cuantizados (GGUF).
- API REST nativa (`/api/generate`, `/api/chat`, `/api/tags`).
- API compatible al 100% con el protocolo de OpenAI (`/v1/chat/completions`), lo que facilita la integracion inmediata con bibliotecas existentes como LangChain, LlamaIndex o SDKs oficiales de OpenAI.

### La Decision Arquitectonica: Empaquetado en Docker vs. Cloud Storage FUSE

Al desplegar modelos en contenedores sobre Google Cloud Run, surgen dos enfoques:

| Dimension | Opcion Ingenua: Empaquetar Pesos en Docker | Mejor Practica: Cloud Storage FUSE (Volume Mount) |
|---|---|---|
| **Tamano de Imagen** | 4 GB a 16 GB por imagen de contenedor. | **~500 MB a 1 GB** (solo el binario oficial de Ollama). |
| **Tiempo de Build** | 10 a 20 minutos en Cloud Build por cada cambio. | **< 1 minuto** de compilacion. |
| **Costo en Artifact Registry** | Alto almacenamiento y transferencia por cada version. | Minimo (los pesos residen en GCS a $0.02/GB/mes). |
| **Actualizacion de Pesos** | Requiere reconstruir y redesplegar el contenedor. | Desacoplado: basta con subir un nuevo archivo GGUF a GCS. |
| **Arranque en Cloud Run** | Descarga pesada de la capa de imagen en cada nodo. | Lectura bajo demanda por streaming / mmap mediante FUSE. |

Google Cloud Run provee soporte nativo de **Cloud Storage Volume Mounts** (respaldado por `gcsfuse`). El servicio monta el bucket directamente en `/root/.ollama`, permitiendo que Ollama lea los modelos persistidos sin incurrir en transferencias innecesarias.

---

## 2. Estructura del Laboratorio

```text
01-ollama-cloudrun-gcsfuse/
├── README.md             # Esta guia metodologica y paso a paso
├── Dockerfile            # Imagen ligera basada en ollama/ollama escuchando en puerto 8080
├── deploy_cloud_run.sh   # Script de automatizacion de infraestructura en GCP
└── test_client.py        # Cliente de inferencia en Python (API Nativa y protocolo OpenAI)
```

---

## 3. Parte A: Enlace con la Sesion 3 e Inferencia Local con Ollama

### Paso 1: Localizar los Artefactos de la Sesion 3
En el Laboratorio 3 de la Sesion 3 generamos dos artefactos clave:
1. El archivo binario cuantizado **GGUF** (`unsloth.Q4_K_M.gguf`).
2. El archivo de manifiesto **`Modelfile`** con la plantilla de turnos conversacionales y el prompt de sistema:

```dockerfile
FROM ./model_gguf/unsloth.Q4_K_M.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER stop "<end_of_turn>"

TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

SYSTEM """Eres un clasificador de incidentes corporativos de TechCloud Pro. Respondes exclusivamente en formato JSON valido con claves: categoria, urgencia, sentimiento y accion_sugerida."""
```

### Paso 2: Crear el Modelo en Ollama Local
Desde el directorio donde se encuentre el `Modelfile`:

```bash
# Registrar el modelo en el catalogo local de Ollama
ollama create techcloud-classifier -f Modelfile

# Verificar que figure en la lista de modelos
ollama list
```

### Paso 3: Probar la Inferencia Local
Prueba por linea de comandos:
```bash
ollama run techcloud-classifier "Alerta: El cluster de Redis esta al 99% de memoria y rechazando llaves."
```

Prueba programatica con Python mediante el cliente incluido:
```bash
python3 test_client.py http://localhost:11434 techcloud-classifier
```

---

## 4. Parte B: Despliegue en Google Cloud Run con GCS FUSE

### Paso 1: Subir los Modelos al Bucket de Cloud Storage
Cree un bucket regional en su proyecto de Google Cloud y copie los pesos:

```bash
export PROJECT_ID=$(gcloud config get-value project)
export BUCKET_NAME="${PROJECT_ID}-ollama-models"
export REGION="us-central1"

# Crear bucket con acceso uniforme
gcloud storage buckets create "gs://${BUCKET_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access

# Copiar el directorio de modelos de Ollama al bucket
# Ollama organiza sus modelos en /root/.ollama/models/ (manifests y blobs)
gcloud storage cp -r ~/.ollama/models "gs://${BUCKET_NAME}/"
```

### Paso 2: Desplegar el Servicio en Cloud Run
Ejecute el script automatizado provisto en este laboratorio:

```bash
# Opcion 1: Despliegue en CPU multi-core de bajo costo
./deploy_cloud_run.sh

# Opcion 2: Despliegue con aceleracion GPU NVIDIA L4
USE_GPU=true ./deploy_cloud_run.sh
```

El comando subyacente que ejecuta `deploy_cloud_run.sh` es:

```bash
gcloud run deploy ollama-service \
  --image="gcr.io/${PROJECT_ID}/ollama-service:latest" \
  --platform=managed \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --allow-unauthenticated \
  --cpu=4 \
  --memory=16Gi \
  --concurrency=8 \
  --timeout=600 \
  --execution-environment=gen2 \
  --add-volume="name=ollama-store,type=cloud-storage,bucket=${BUCKET_NAME}" \
  --add-volume-mount="volume=ollama-store,mount-path=/root/.ollama"
```

---

## 5. Parte C: Verificacion y Pruebas del Endpoint en Produccion

Una vez finalizado el despliegue, obtenga la URL publica del servicio:

```bash
export CLOUD_RUN_URL=$(gcloud run services describe ollama-service --region=us-central1 --format="value(status.url)")
echo "Servicio activo en: ${CLOUD_RUN_URL}"
```

### 1. Comprobar Version y Estado
```bash
curl -s "${CLOUD_RUN_URL}/api/version"
```

### 2. Ejecutar Inferencia de Prueba con Python
```bash
python3 test_client.py "${CLOUD_RUN_URL}" techcloud-classifier
```

### 3. Invocacion mediante cURL compatible con OpenAI
```bash
curl -s -X POST "${CLOUD_RUN_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "techcloud-classifier",
    "messages": [
      {"role": "user", "content": "Se cayo la pasarela de pagos con error 500."}
    ],
    "temperature": 0.1
  }' | jq .
```

---

## 6. Optimizacion Operativa y Mejores Practicas

1. **Mitigacion de Cold Starts:** Configure `--min-instances=1` si su aplicacion no tolera la latencia de inicializacion del contenedor y montaje del volumen FUSE. Para entornos de desarrollo o pruebas por lotes, `--min-instances=0` garantiza costo cero cuando no hay trafico.
2. **Concurrencia Adecuada:** En modelos de 2B a 7B ejecutados en CPU, configure `--concurrency=4` u `8`. En instancias con GPU NVIDIA L4, vLLM o Ollama pueden manejar hasta 16 o 32 peticiones concurrentes segun el tamano del contexto.
3. **Persistencia de Logs:** Las peticiones y metricas de latencia quedan registradas en Cloud Logging y Cloud Monitoring de forma nativa.

---

## 7. Referencias Oficiales

- **Cloud Run Volume Mounts (Cloud Storage FUSE):** [https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts](https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts)
- **Documentacion Oficial de Ollama:** [https://ollama.com/](https://ollama.com/)
- **API Reference de Ollama:** [https://github.com/ollama/ollama/blob/main/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md)
- **Especificacion GGUF (llama.cpp):** [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
