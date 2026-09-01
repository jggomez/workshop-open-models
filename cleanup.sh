#!/bin/bash
# Workshop Workspace Cleanup Script
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Limpieza del Entorno de Trabajo del Workshop"
echo "============================================================"

# Limpiar archivos temporales y checkpoints
echo "[1/4] Eliminando caches de Python y Jupyter..."
find "$DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$DIR" -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
find "$DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$DIR" -type f -name ".DS_Store" -delete 2>/dev/null || true

# Limpiar imagenes y binarios descargados en los laboratorios
echo "[2/4] Eliminando imagenes de prueba y binarios de modelos..."
rm -rf "$DIR/session-01-hf-kerashub-litert/01-kerashub-image-classification/sample_images" 2>/dev/null || true
rm -f "$DIR/session-01-hf-kerashub-litert/02-litert-web-vision/models/efficientnet_lite0.tflite" 2>/dev/null || true
rm -f "$DIR/session-01-hf-kerashub-litert/02-litert-web-vision/models/mobilenet_quant.tflite" 2>/dev/null || true
rm -f "$DIR/session-01-hf-kerashub-litert/02-litert-web-vision/models/imagenet_classes.json" 2>/dev/null || true
rm -f "$DIR/session-01-hf-kerashub-litert/02-litert-web-vision/models/temp_model.tgz" 2>/dev/null || true

# Opcion de desinstalar entorno virtual si se especifica --venv o --all
if [ "$1" == "--venv" ] || [ "$1" == "--all" ]; then
    echo "[3/4] Eliminando entorno virtual (.venv)..."
    rm -rf "$DIR/.venv" 2>/dev/null || true
else
    echo "[3/4] Conservando entorno virtual (.venv). Use './cleanup.sh --venv' para eliminarlo."
fi

# Opcion de limpiar cache de Hugging Face y Keras si se especifica --cache o --all
if [ "$1" == "--cache" ] || [ "$1" == "--all" ]; then
    echo "[4/4] Purgando caches globales (~/.cache/huggingface y ~/.keras)..."
    rm -rf ~/.cache/huggingface 2>/dev/null || true
    rm -rf ~/.keras 2>/dev/null || true
else
    echo "[4/4] Conservando caches globales de Hugging Face y Keras."
fi

echo ""
echo "Limpieza completada exitosamente."
