FROM ollama/ollama:latest

# Cloud Run sondea el puerto 8080; Ollama escucha en 11434 por defecto.
ENV OLLAMA_HOST=0.0.0.0:8080

# Punto de montaje del bucket via Cloud Storage FUSE
ENV OLLAMA_MODELS=/root/.ollama/models

EXPOSE 8080

ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]