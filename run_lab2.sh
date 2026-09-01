#!/bin/bash
# Script de ejecucion directa para el Laboratorio 2 (LiteRT.js Web Vision)
PORT=3000
DIR="session-01-hf-kerashub-litert/02-litert-web-vision"

echo "Iniciando servidor web para el Laboratorio 2 en el puerto $PORT..."
echo "Directorio: $DIR"
echo "Abriendo navegador en http://localhost:$PORT..."

# Abrir el navegador en background en macOS
(sleep 1 && open "http://localhost:$PORT") &

# Ejecutar el servidor HTTP
python3 -m http.server $PORT --directory "$DIR"
