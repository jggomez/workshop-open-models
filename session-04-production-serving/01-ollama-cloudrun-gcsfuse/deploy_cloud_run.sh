#!/usr/bin/env bash
set -euo pipefail

# Script de automatizacion para el despliegue de Ollama en Google Cloud Run
# utilizando Cloud Storage FUSE (Volume Mounts) para almacenamiento persistente
# de modelos.
#
# Requisito previo: los modelos deben estar ya subidos al bucket.
#   gcloud storage cp -r ~/.ollama/models gs://BUCKET/
#   (Linux con systemd: /usr/share/ollama/.ollama/models)

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-ollama-service}"
BUCKET_NAME="${MODEL_BUCKET:-${PROJECT_ID}-ollama-models}"
ARTIFACT_REPO="${ARTIFACT_REPO:-ollama-repo}"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${SERVICE_NAME}:latest"
USE_GPU="${USE_GPU:-false}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Error: PROJECT_ID no esta configurado. Defina GCP_PROJECT_ID o configure gcloud config set project."
  exit 1
fi

echo "=== Configurando Despliegue de Ollama en Cloud Run ==="
echo "Proyecto:   ${PROJECT_ID}"
echo "Region:     ${REGION}"
echo "Servicio:   ${SERVICE_NAME}"
echo "Bucket GCS: ${BUCKET_NAME}"
echo "Imagen:     ${IMAGE_NAME}"
echo "Modo GPU:   ${USE_GPU}"
echo "======================================================"

# 1. Habilitar APIs requeridas de Google Cloud
echo "[1/6] Verificando y habilitando APIs de GCP..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT_ID}"

# 2. Crear bucket de Google Cloud Storage si no existe
echo "[2/6] Verificando existencia de bucket Cloud Storage..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Creando bucket regional gs://${BUCKET_NAME} en ${REGION}..."
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access
else
  echo "Bucket gs://${BUCKET_NAME} ya existe."
fi

# Un bucket vacio produce un despliegue "exitoso" cuyo /api/tags devuelve una
# lista vacia: fallo silencioso. Se comprueba antes de gastar tiempo en build.
echo "      Comprobando que el bucket contenga modelos..."
if ! gcloud storage ls "gs://${BUCKET_NAME}/models/manifests/" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo ""
  echo "ADVERTENCIA: el bucket no contiene modelos de Ollama."
  echo "Suba los modelos antes de desplegar:"
  echo "  gcloud storage cp -r ~/.ollama/models gs://${BUCKET_NAME}/"
  echo "  (Linux con systemd: /usr/share/ollama/.ollama/models)"
  echo ""
  read -r -p "Continuar de todas formas? [y/N] " REPLY
  [[ "${REPLY}" =~ ^[Yy]$ ]] || exit 1
else
  echo "      Modelos encontrados en el bucket."
fi

# 3. Crear repositorio de Artifact Registry si no existe
# gcr.io (Container Registry) fue reemplazado por Artifact Registry.
echo "[3/6] Verificando repositorio de Artifact Registry..."
if ! gcloud artifacts repositories describe "${ARTIFACT_REPO}" \
     --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Creando repositorio ${ARTIFACT_REPO} en ${REGION}..."
  gcloud artifacts repositories create "${ARTIFACT_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}"
else
  echo "Repositorio ${ARTIFACT_REPO} ya existe."
fi

# 4. Compilar imagen de contenedor ligera sin modelos embebidos
echo "[4/6] Compilando contenedor ligero con Cloud Build..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# El Dockerfile debe fijar OLLAMA_HOST=0.0.0.0:8080. Por defecto Ollama escucha
# en 11434 y el contenedor no responderia al puerto que sondea Cloud Run.
if [[ -f "${SCRIPT_DIR}/Dockerfile" ]] && ! grep -q "8080" "${SCRIPT_DIR}/Dockerfile"; then
  echo "ADVERTENCIA: el Dockerfile no menciona el puerto 8080."
  echo "Debe incluir: ENV OLLAMA_HOST=0.0.0.0:8080"
fi

gcloud builds submit "${SCRIPT_DIR}" \
  --tag="${IMAGE_NAME}" \
  --project="${PROJECT_ID}"

# 5. Asignar permisos IAM al Service Account
# objectAdmin (lectura + escritura): Ollama escribe en /root/.ollama, que es el
# punto de montaje del bucket. Solo lectura hace fallar el arranque.
echo "[5/6] Configurando permisos IAM de lectura/escritura sobre el bucket..."
SERVICE_ACCOUNT="$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --project="${PROJECT_ID}" --format="value(spec.template.spec.serviceAccountName)" 2>/dev/null || true)"
if [[ -z "${SERVICE_ACCOUNT}" ]]; then
  PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")"
  SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi
echo "      Service Account: ${SERVICE_ACCOUNT}"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin" \
  --project="${PROJECT_ID}"

# 6. Desplegar en Cloud Run con volumen FUSE
echo "[6/6] Desplegando en Google Cloud Run..."
DEPLOY_CMD=(
  gcloud run deploy "${SERVICE_NAME}"
  --image="${IMAGE_NAME}"
  --platform=managed
  --region="${REGION}"
  --project="${PROJECT_ID}"
  --port=8080
  --allow-unauthenticated
  --concurrency=8
  --timeout=600
  --execution-environment=gen2
  --add-volume="name=ollama-store,type=cloud-storage,bucket=${BUCKET_NAME}"
  --add-volume-mount="volume=ollama-store,mount-path=/root/.ollama"
)

if [[ "${USE_GPU}" == "true" ]]; then
  echo "Configurando acelerador de hardware GPU NVIDIA L4..."
  # Los flags de GPU en Cloud Run cambian con frecuencia y su disponibilidad
  # depende de la region y la cuota del proyecto. Si el deploy falla por un
  # flag no reconocido, consultar:
  # https://cloud.google.com/run/docs/configuring/services/gpu
  DEPLOY_CMD+=(
    --gpu=1
    --gpu-type=nvidia-l4
    --no-gpu-zonal-redundancy
    --cpu=4
    --memory=16Gi
    --no-cpu-throttling
    --max-instances=3
  )
else
  echo "Configurando perfil de CPU multi-core de alto rendimiento..."
  DEPLOY_CMD+=(
    --cpu=4
    --memory=16Gi
    --no-cpu-throttling
  )
fi

"${DEPLOY_CMD[@]}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --project="${PROJECT_ID}" --format="value(status.url)")"
echo "======================================================"
echo "Despliegue completado con exito."
echo "URL del servicio: ${SERVICE_URL}"
echo ""
echo "Verificacion:"
echo "  1) Estado del servicio:"
echo "     curl -s ${SERVICE_URL}/api/version"
echo ""
echo "  2) Modelos visibles (confirma que el volumen FUSE se monto):"
echo "     curl -s ${SERVICE_URL}/api/tags | jq ."
echo "     Si devuelve una lista vacia, revisar el montaje o los permisos IAM."
echo ""
echo "NOTA: el servicio esta desplegado con --allow-unauthenticated y es"
echo "accesible publicamente. Cualquiera con la URL consume cuota."
echo "======================================================"