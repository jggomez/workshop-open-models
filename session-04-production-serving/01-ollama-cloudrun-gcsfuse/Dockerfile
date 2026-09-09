FROM ollama/ollama:latest

# Cloud Run escucha de forma predeterminada en el puerto 8080
ENV OLLAMA_HOST=0.0.0.0:8080

# Ruta donde se almacenan y leen los modelos (montada via Cloud Storage FUSE)
ENV OLLAMA_MODELS=/root/.ollama/models

# Permitir origenes cruzados para pruebas web
ENV OLLAMA_ORIGINS=*

EXPOSE 8080

ENTRYPOINT ["ollama", "serve"]
