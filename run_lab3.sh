#!/bin/bash
# Script de ejecucion directa para el Laboratorio 3 (LiteRT-LM Web API)
PORT=3001
DIR="session-01-hf-kerashub-litert/03-litert-lm-cli-and-web"

echo "Iniciando servidor web para el Laboratorio 3 en el puerto $PORT..."
echo "Directorio: $DIR"
echo "Abriendo navegador en http://localhost:$PORT..."

# Abrir el navegador en background en macOS
(sleep 1 && open "http://localhost:$PORT") &

# Ejecutar el servidor HTTP
python3 -m http.server $PORT --directory "$DIR"
