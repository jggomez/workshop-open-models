#!/usr/bin/env bash
set -euo pipefail

# Script de automatizacion para el despliegue de Ollama en Google Cloud Run
# utilizando Cloud Storage FUSE (Volume Mounts) para almacenamiento persistente de modelos.

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-ollama-service}"
BUCKET_NAME="${MODEL_BUCKET:-${PROJECT_ID}-ollama-models}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"
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
echo "Modo GPU:   ${USE_GPU}"
echo "======================================================"

# 1. Habilitar APIs requeridas de Google Cloud
echo "[1/5] Verificando y habilitando APIs de GCP..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT_ID}"

# 2. Crear bucket de Google Cloud Storage si no existe
echo "[2/5] Verificando existencia de bucket Cloud Storage..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Creando bucket regional gs://${BUCKET_NAME} en ${REGION}..."
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access
else
  echo "Bucket gs://${BUCKET_NAME} ya existe."
fi

# 3. Compilar imagen de contenedor ligera sin modelos embebidos
echo "[3/5] Compilando contenedor ligero con Cloud Build..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gcloud builds submit "${SCRIPT_DIR}" \
  --tag="${IMAGE_NAME}" \
  --project="${PROJECT_ID}"

# 4. Asignar permisos al Service Account predeterminado de Compute/Cloud Run
echo "[4/5] Configurando permisos IAM de lectura sobre el bucket..."
SERVICE_ACCOUNT="$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --project="${PROJECT_ID}" --format="value(spec.template.spec.serviceAccountName)" 2>/dev/null || true)"
if [[ -z "${SERVICE_ACCOUNT}" ]]; then
  PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")"
  SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin" \
  --project="${PROJECT_ID}"

# 5. Desplegar en Cloud Run con volumen FUSE
echo "[5/5] Desplegando en Google Cloud Run..."
DEPLOY_CMD=(
  gcloud run deploy "${SERVICE_NAME}"
  --image="${IMAGE_NAME}"
  --platform=managed
  --region="${REGION}"
  --project="${PROJECT_ID}"
  --allow-unauthenticated
  --concurrency=8
  --timeout=600
  --execution-environment=gen2
  --add-volume="name=ollama-store,type=cloud-storage,bucket=${BUCKET_NAME}"
  --add-volume-mount="volume=ollama-store,mount-path=/root/.ollama"
)

if [[ "${USE_GPU}" == "true" ]]; then
  echo "Configurando acelerador de hardware GPU NVIDIA L4..."
  DEPLOY_CMD+=(
    --gpu=1
    --gpu-type=nvidia-l4
    --cpu=4
    --memory=16Gi
    --no-cpu-throttling
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
echo "Prueba de salud: curl -s ${SERVICE_URL}/api/version"
echo "======================================================"
