# Hands-on Lab 1 (Sesion 4): Serving con Ollama (Local y Cloud Run con Cloud Storage FUSE)

Este laboratorio ensena el ciclo de puesta en produccion de modelos cuantizados empleando **Ollama**, desde la importacion local del modelo **GGUF** afinado en la Sesion 3 (Unsloth) hasta su despliegue serverless escalable en **Google Cloud Run** aplicando la mejor practica arquitectonica de **Cloud Storage FUSE (Volume Mounts)**.

---

## 1. Fundamentos Tecnologicos: Por que Ollama y por que GCS FUSE?

### El Rol de Ollama en Inferencia Ligera
**Ollama** empaqueta el motor de inferencia `llama.cpp` en una arquitectura de servicio que proporciona:
- Gestion automatizada de memoria y descompresion de pesos cuantizados (GGUF).
- API REST nativa (`/api/generate`, `/api/chat`, `/api/tags`).
- API compatible con el protocolo de OpenAI (`/v1/chat/completions`), lo que facilita la integracion con bibliotecas existentes como LangChain, LlamaIndex o SDKs oficiales de OpenAI.

### La Decision Arquitectonica: Empaquetado en Docker vs. Cloud Storage FUSE

| Dimension | Opcion Ingenua: Empaquetar Pesos en Docker | Mejor Practica: Cloud Storage FUSE (Volume Mount) |
|---|---|---|
| **Tamano de Imagen** | 4 GB a 16 GB por imagen de contenedor. | **~500 MB a 1 GB** (solo el binario oficial de Ollama). |
| **Tiempo de Build** | 10 a 20 minutos en Cloud Build por cada cambio. | **< 1 minuto** de compilacion. |
| **Costo en Artifact Registry** | Alto almacenamiento y transferencia por cada version. | Minimo (los pesos residen en GCS a ~$0.02/GB/mes). |
| **Actualizacion de Pesos** | Requiere reconstruir y redesplegar el contenedor. | Desacoplado: basta con subir un nuevo archivo GGUF a GCS. |
| **Arranque en Cloud Run** | Descarga pesada de la capa de imagen en cada nodo. | Lectura bajo demanda por streaming / mmap mediante FUSE. |

Cloud Run provee soporte nativo de **Cloud Storage Volume Mounts** (respaldado por `gcsfuse`). El servicio monta el bucket en `/root/.ollama`, permitiendo que Ollama lea los modelos persistidos sin transferencias innecesarias.

---

## 2. Estructura del Laboratorio

```text
01-ollama-cloudrun-gcsfuse/
├── README.md             # Esta guia metodologica y paso a paso
├── Dockerfile            # Imagen ligera basada en ollama/ollama escuchando en 8080
├── deploy_cloud_run.sh   # Script de automatizacion de infraestructura en GCP
└── test_client.py        # Cliente de inferencia en Python (API Nativa y protocolo OpenAI)
```

---

## 3. Parte A: Enlace con la Sesion 3 e Inferencia Local con Ollama

### Paso 1: Localizar los Artefactos de la Sesion 3

En la Sesion 3 se generaron dos artefactos:

1. El binario cuantizado **GGUF**. Verificar el nombre real antes de continuar:
   ```bash
   ls -lh model_gguf_gguf/
   ```
   Unsloth anade el sufijo `_gguf` al directorio indicado y nombra el archivo
   segun el modelo base (por ejemplo `gemma-2-2b-it.Q4_K_M.gguf`), no
   `unsloth.Q4_K_M.gguf`.

2. El manifiesto **`Modelfile`**.

> **Importante:** Unsloth genera su propio `Modelfile` dentro de
> `model_gguf_gguf/`. Ese archivo trae `temperature 1.5` y **no incluye la
> instruccion de la tarea**, por lo que no reproduce el formato de
> entrenamiento. Usar el manifiesto de abajo en su lugar.

#### Manifiesto correcto

```dockerfile
FROM ./gemma-2-2b-it.Q4_K_M.gguf

PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"

TEMPLATE """<start_of_turn>user
Analiza el siguiente ticket de soporte y extrae la informacion en formato JSON con las claves categoria, urgencia, sentimiento y accion_sugerida:
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""
```

Diferencias frente al manifiesto de la version anterior del laboratorio:

| Cambio | Motivo |
|---|---|
| Sin bloque `SYSTEM` ni rama `{{ if .System }}` | Gemma-2 no define rol de sistema. No existe `<start_of_turn>system` y el modelo nunca lo vio durante el entrenamiento. |
| Instruccion dentro del `TEMPLATE` | El fine-tuning uso la instruccion delante de cada ticket. Si el usuario envia solo el ticket, el modelo recibe un formato que no vio. |
| `temperature 0` en lugar de `0.2` | La validacion se hizo con `do_sample=False` (greedy). Para extraccion de JSON se busca determinismo, no variedad. |
| Sin `top_p` | Con temperatura 0 no hay muestreo aleatorio que ajustar. |
| Stop adicional `<start_of_turn>` | Evita que el modelo abra un turno nuevo en lugar de cerrar. |
| Sin `<bos>` en la plantilla | `llama.cpp` lo inserta segun los metadatos del GGUF. Anadirlo a mano produce doble BOS y degeneracion de la salida. |

### Paso 2: Crear el Modelo en Ollama Local

El `Modelfile` y el `.gguf` deben estar en el **mismo directorio** (`FROM ./` es
una ruta relativa al manifiesto).

```bash
ollama create techcloud-classifier -f Modelfile
ollama list
```

### Paso 3: Probar la Inferencia Local

```bash
ollama run techcloud-classifier "Alerta: El cluster de Redis esta al 99% de memoria y rechazando llaves."
```

Se envia unicamente el texto del ticket: la instruccion la anade el `TEMPLATE`.

Verificar que el manifiesto quedo registrado como se esperaba:

```bash
ollama show techcloud-classifier --modelfile
```

Prueba programatica:

```bash
python3 test_client.py http://localhost:11434 techcloud-classifier
```

---

## 4. Parte B: Despliegue en Google Cloud Run con GCS FUSE

### Paso 0: Variables y APIs

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export BUCKET_NAME="${PROJECT_ID}-ollama-models"
export REPO="ollama-repo"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/ollama-service:latest"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com
```

### Paso 1: Subir los Modelos al Bucket

Ollama guarda los modelos en `manifests/` y `blobs/`. La ubicacion del
directorio depende de la instalacion:

| Instalacion | Ruta |
|---|---|
| macOS / Windows / Linux (usuario) | `~/.ollama/models` |
| Linux como servicio systemd | `/usr/share/ollama/.ollama/models` |

Confirmar antes de copiar:

```bash
ls ~/.ollama/models 2>/dev/null || ls /usr/share/ollama/.ollama/models
```

```bash
gcloud storage buckets create "gs://${BUCKET_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access

# Ajustar la ruta de origen segun la tabla anterior
gcloud storage cp -r ~/.ollama/models "gs://${BUCKET_NAME}/"
```

Verificar que la estructura quedo correcta:

```bash
gcloud storage ls "gs://${BUCKET_NAME}/models/"
# Debe listar: manifests/ y blobs/
```

### Paso 2: Dockerfile

```dockerfile
FROM ollama/ollama:latest

# Cloud Run inyecta PORT; Ollama escucha en 11434 por defecto.
ENV OLLAMA_HOST=0.0.0.0:8080
ENV OLLAMA_MODELS=/root/.ollama/models

EXPOSE 8080
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
```

### Paso 3: Construir y Publicar la Imagen

Este paso faltaba en la version anterior del laboratorio: `gcloud run deploy`
referencia una imagen que debe existir previamente en el registro.

```bash
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}"

gcloud builds submit --tag "${IMAGE}"
```

> `gcr.io` (Container Registry) fue reemplazado por Artifact Registry.
> Verificar el estado actual en la documentacion de GCP, ya que las fechas de
> retirada se han ido ajustando.

### Paso 4: Permisos del Service Account

La cuenta de servicio de Cloud Run necesita acceso al bucket. Ollama escribe en
`/root/.ollama`, por lo que **no basta con lectura**:

```bash
export SA=$(gcloud run services describe ollama-service \
  --region="${REGION}" --format="value(spec.template.spec.serviceAccountName)" 2>/dev/null)
export SA=${SA:-$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')-compute@developer.gserviceaccount.com}

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${SA}" \
  --role="roles/storage.objectAdmin"
```

### Paso 5: Desplegar el Servicio

```bash
# Opcion 1: CPU multi-core
./deploy_cloud_run.sh

# Opcion 2: GPU NVIDIA L4
USE_GPU=true ./deploy_cloud_run.sh
```

Comando subyacente (CPU):

```bash
gcloud run deploy ollama-service \
  --image="${IMAGE}" \
  --platform=managed \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --port=8080 \
  --cpu=4 \
  --memory=16Gi \
  --concurrency=8 \
  --timeout=600 \
  --execution-environment=gen2 \
  --add-volume="name=ollama-store,type=cloud-storage,bucket=${BUCKET_NAME}" \
  --add-volume-mount="volume=ollama-store,mount-path=/root/.ollama"
```

Variante con GPU. La version anterior del laboratorio ofrecia `USE_GPU=true`
pero el comando no incluia ningun flag de GPU, por lo que desplegaba en CPU:

```bash
gcloud run deploy ollama-service \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --port=8080 \
  --cpu=8 \
  --memory=32Gi \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-cpu-throttling \
  --max-instances=3 \
  --concurrency=16 \
  --timeout=600 \
  --execution-environment=gen2 \
  --add-volume="name=ollama-store,type=cloud-storage,bucket=${BUCKET_NAME}" \
  --add-volume-mount="volume=ollama-store,mount-path=/root/.ollama"
```

> La GPU en Cloud Run tiene restricciones de region, cuota y limite de
> instancias que cambian con frecuencia. Confirmar disponibilidad y flags
> vigentes en la documentacion oficial antes de ejecutar. Revisar tambien que
> `deploy_cloud_run.sh` implemente realmente la rama `USE_GPU`.

#### Sobre `--allow-unauthenticated`

La version anterior lo incluia por defecto. Eso deja el endpoint de inferencia
abierto a internet: cualquiera con la URL consume cuota y presupuesto. Para el
laboratorio es aceptable; fuera de el, omitirlo e invocar con token:

```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" ...
```

---

## 5. Parte C: Verificacion y Pruebas del Endpoint

```bash
export CLOUD_RUN_URL=$(gcloud run services describe ollama-service \
  --region="${REGION}" --format="value(status.url)")
echo "Servicio activo en: ${CLOUD_RUN_URL}"
```

### 1. Comprobar version y modelos disponibles

```bash
curl -s "${CLOUD_RUN_URL}/api/version"

# Confirma que el volumen FUSE se monto y Ollama ve los modelos.
# Si devuelve una lista vacia, el problema esta en el montaje o los permisos.
curl -s "${CLOUD_RUN_URL}/api/tags" | jq .
```

### 2. Inferencia con el cliente Python

```bash
python3 test_client.py "${CLOUD_RUN_URL}" techcloud-classifier
```

### 3. Invocacion compatible con OpenAI

```bash
curl -s -X POST "${CLOUD_RUN_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "techcloud-classifier",
    "messages": [
      {"role": "user", "content": "Se cayo la pasarela de pagos con error 500."}
    ],
    "temperature": 0
  }' | jq .
```

> `temperature` en la peticion **sobrescribe** el valor del `Modelfile`. La
> version anterior enviaba `0.1`, rompiendo el determinismo configurado. Enviar
> `0` o omitir el campo.

---

## 6. Optimizacion Operativa y Mejores Practicas

1. **Cold starts.** `--min-instances=1` evita la latencia de inicializacion del
   contenedor y montaje FUSE. Para desarrollo, `--min-instances=0` garantiza
   costo cero sin trafico.
2. **Concurrencia.** En modelos de 2B a 7B sobre CPU, usar `--concurrency=4` u
   `8`. Sobre NVIDIA L4 puede subirse a 16 o 32 segun el tamano de contexto.
3. **Primera peticion.** Con FUSE, la carga inicial del GGUF se lee por
   streaming desde GCS. La primera inferencia tras un cold start es
   notablemente mas lenta que las siguientes.
4. **Logs y metricas.** Quedan registrados de forma nativa en Cloud Logging y
   Cloud Monitoring.

---

## 7. Referencias Oficiales

- **Cloud Run Volume Mounts (Cloud Storage FUSE):** https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts
- **Cloud Run GPU:** https://cloud.google.com/run/docs/configuring/services/gpu
- **Artifact Registry:** https://cloud.google.com/artifact-registry/docs
- **Documentacion Oficial de Ollama:** https://ollama.com/
- **API Reference de Ollama:** https://github.com/ollama/ollama/blob/main/docs/api.md
- **Guia Oficial del Formato GGUF en Hugging Face Hub:** https://huggingface.co/docs/hub/gguf
- **Especificacion GGUF (llama.cpp):** https://github.com/ggerganov/llama.cpp