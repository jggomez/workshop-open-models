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

Verificar el nombre real del binario GGUF antes de continuar:

```bash
ls -lh model_gguf_gguf/
```

Unsloth anade el sufijo `_gguf` al directorio indicado y nombra el archivo segun el modelo base, por ejemplo `gemma-2-2b-it.Q4_K_M.gguf`.

> Unsloth genera tambien su propio `Modelfile` dentro de ese directorio. Ese archivo trae `temperature 1.5` y **no incluye la instruccion de la tarea**, por lo que no reproduce el formato de entrenamiento. Usar el manifiesto de abajo en su lugar.

### Paso 2: Escribir el Manifiesto (`Modelfile`)

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

Razones de cada decision:

| Elemento | Motivo |
|---|---|
| Sin bloque `SYSTEM` | Gemma-2 no define rol de sistema. No existe `<start_of_turn>system` y el modelo nunca lo vio durante el entrenamiento. |
| Instruccion dentro del `TEMPLATE` | El fine-tuning uso la instruccion delante de cada ticket. El usuario envia solo el ticket y la plantilla completa el resto. |
| `temperature 0` | La validacion se hizo con `do_sample=False` (greedy). Para extraccion de JSON se busca determinismo, no variedad. |
| Sin `top_p` ni `repeat_penalty` | Con temperatura 0 no hay muestreo aleatorio que ajustar. |
| Dos `stop` | `<end_of_turn>` cierra el turno; `<start_of_turn>` evita que el modelo abra uno nuevo en lugar de detenerse. |
| Sin `<bos>` en la plantilla | `llama.cpp` lo inserta segun los metadatos del GGUF. Anadirlo a mano produce doble BOS y degeneracion de la salida. |

### Paso 3: Crear el Modelo en Ollama Local

El `Modelfile` y el `.gguf` deben estar en el **mismo directorio**, ya que `FROM ./` es una ruta relativa al manifiesto.

```bash
ollama create techcloud-classifier -f Modelfile
ollama list
```

Confirmar que el manifiesto quedo registrado como se esperaba:

```bash
ollama show techcloud-classifier --modelfile
```

### Paso 4: Probar la Inferencia Local

```bash
ollama run techcloud-classifier "Alerta: El cluster de Redis esta al 99% de memoria y rechazando llaves."
```

Se envia unicamente el texto del ticket: la instruccion la anade el `TEMPLATE`.

Prueba programatica:

```bash
python3 test_client.py http://localhost:11434 techcloud-classifier
```

El cliente devuelve codigo de salida 0 si ambas APIs responden con JSON valido.

---

## 4. Parte B: Despliegue en Google Cloud Run con GCS FUSE

### Paso 1: Variables de Entorno

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export BUCKET_NAME="${PROJECT_ID}-ollama-models"
export ARTIFACT_REPO="ollama-repo"
```

### Paso 2: Subir los Modelos al Bucket

Ollama organiza su almacen en dos directorios: `blobs/` guarda los datos pesados identificados por hash, y `manifests/` guarda archivos JSON que describen cada modelo referenciando esos hashes. **Se copian ambos**, no el `.gguf` original.

La ubicacion depende de la instalacion:

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

Verificar la estructura resultante:

```bash
gcloud storage ls "gs://${BUCKET_NAME}/models/"
# Debe listar: manifests/ y blobs/
```

Sin los manifests, Ollama tiene los datos pero no sabe que exista ningun modelo registrado.

### Paso 3: Dockerfile

```dockerfile
FROM ollama/ollama:latest

# Cloud Run sondea el puerto 8080; Ollama escucha en 11434 por defecto.
ENV OLLAMA_HOST=0.0.0.0:8080

# Punto de montaje del bucket via Cloud Storage FUSE
ENV OLLAMA_MODELS=/root/.ollama/models

EXPOSE 8080

ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
```

Para builds reproducibles, sustituir `:latest` por el tag de version que se ejecuta en local (`ollama --version`).

### Paso 4: Desplegar

El script `deploy_cloud_run.sh` automatiza el ciclo completo: habilita APIs, crea el bucket y el repositorio de Artifact Registry, compila la imagen con Cloud Build, configura los permisos IAM y despliega el servicio.

```bash
chmod +x deploy_cloud_run.sh

# Opcion 1: CPU multi-core
./deploy_cloud_run.sh

# Opcion 2: GPU NVIDIA L4
USE_GPU=true ./deploy_cloud_run.sh
```

Comando subyacente en modo CPU:

```bash
gcloud run deploy ollama-service \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/ollama-service:latest" \
  --platform=managed \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --port=8080 \
  --cpu=4 \
  --memory=16Gi \
  --concurrency=8 \
  --timeout=600 \
  --execution-environment=gen2 \
  --no-cpu-throttling \
  --add-volume="name=ollama-store,type=cloud-storage,bucket=${BUCKET_NAME}" \
  --add-volume-mount="volume=ollama-store,mount-path=/root/.ollama"
```

Flags adicionales en modo GPU:

```bash
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --max-instances=3
```

> Los flags de GPU en Cloud Run, su disponibilidad por region y la cuota asociada cambian con frecuencia. Confirmar en la documentacion oficial antes de ejecutar.

### Paso 5: Permisos IAM

El script lo gestiona automaticamente, pero conviene entender el requisito: la cuenta de servicio necesita el rol `roles/storage.objectAdmin` sobre el bucket. **No basta con lectura**, porque Ollama escribe en `/root/.ollama`, que es el punto de montaje.

```bash
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin"
```

### Nota sobre `--allow-unauthenticated`

El script despliega con esta opcion, lo que deja el endpoint de inferencia abierto a internet: cualquiera con la URL consume cuota y presupuesto. Para el laboratorio es aceptable; fuera de el, omitirla e invocar con token:

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

### 1. Estado del servicio

```bash
curl -s "${CLOUD_RUN_URL}/api/version"
```

### 2. Modelos visibles

Esta es la verificacion clave del montaje FUSE:

```bash
curl -s "${CLOUD_RUN_URL}/api/tags" | jq .
```

Una lista vacia indica que el volumen no se monto o que faltan los manifests en el bucket. Revisar en ese orden: la ruta de montaje del deploy, el contenido de `gs://BUCKET/models/manifests/` y los permisos IAM de la cuenta de servicio.

### 3. Inferencia con el cliente Python

```bash
python3 test_client.py "${CLOUD_RUN_URL}" techcloud-classifier
```

La primera peticion tras un cold start puede tardar varios minutos: FUSE lee el GGUF completo por streaming desde GCS. Las siguientes son rapidas.

### 4. Invocacion compatible con OpenAI

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

> El campo `temperature` de la peticion **sobrescribe** el valor del `Modelfile`. Enviar `0` u omitirlo para conservar el determinismo. Del mismo modo, no enviar mensajes con `"role": "system"`: Gemma-2 no define ese rol.

---

## 6. Optimizacion Operativa y Mejores Practicas

1. **Cold starts.** `--min-instances=1` evita la latencia de inicializacion del contenedor y montaje FUSE. Para desarrollo, `--min-instances=0` garantiza costo cero sin trafico.
2. **Concurrencia.** En modelos de 2B a 7B sobre CPU, usar `--concurrency=4` u `8`. Sobre NVIDIA L4 puede subirse a 16 o 32 segun el tamano de contexto.
3. **Actualizacion de pesos.** Al estar desacoplados de la imagen, basta con subir el nuevo modelo al bucket y reiniciar el servicio. No hace falta reconstruir el contenedor.
4. **Logs y metricas.** Quedan registrados de forma nativa en Cloud Logging y Cloud Monitoring.
5. **Costos.** Revisar la facturacion al terminar el laboratorio. Cloud Build, Artifact Registry, el almacenamiento en GCS y las instancias con `--min-instances=1` o GPU generan cargos continuos.

---

## 7. Limpieza de Recursos (Teardown)

Ejecutar al terminar el laboratorio. Los recursos creados generan cargos continuos aunque no haya trafico: el almacenamiento en GCS, las imagenes en Artifact Registry y cualquier instancia con `--min-instances=1` o GPU.

> **Estas operaciones son irreversibles.** Verificar el proyecto activo antes de empezar:
> ```bash
> gcloud config get-value project
> ```

### Variables

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export SERVICE_NAME="ollama-service"
export BUCKET_NAME="${PROJECT_ID}-ollama-models"
export ARTIFACT_REPO="ollama-repo"
```

### 1. Eliminar el servicio de Cloud Run

Detiene toda facturacion de computo de forma inmediata. Si solo se hace un paso de limpieza, que sea este.

```bash
gcloud run services delete "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --quiet
```

### 2. Eliminar el bucket y los modelos

```bash
# Revisar el contenido antes de borrar
gcloud storage ls -r "gs://${BUCKET_NAME}/"

# Borrado recursivo del bucket completo
gcloud storage rm -r "gs://${BUCKET_NAME}" --project="${PROJECT_ID}"
```

El `.gguf` original permanece en la maquina local y en Ollama, asi que este borrado no pierde el modelo afinado.

### 3. Eliminar el repositorio de Artifact Registry

```bash
gcloud artifacts repositories delete "${ARTIFACT_REPO}" \
  --location="${REGION}" \
  --project="${PROJECT_ID}" \
  --quiet
```

Borra todas las imagenes que contiene. Para conservar el repositorio y eliminar solo la imagen del laboratorio:

```bash
gcloud artifacts docker images delete \
  "${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${SERVICE_NAME}" \
  --delete-tags --quiet
```

### 4. Eliminar artefactos de Cloud Build

Cloud Build deja los contextos de compilacion en un bucket propio que sigue facturando almacenamiento:

```bash
gcloud storage ls | grep -E "cloudbuild|_cloudbuild"
gcloud storage rm -r "gs://${PROJECT_ID}_cloudbuild" --project="${PROJECT_ID}"
```

El nombre del bucket varia segun el proyecto y la region. El `grep` confirma cual existe antes de borrarlo.

### 5. Revocar el permiso IAM

Solo si el bucket no se elimino en el paso 2. Al borrar el bucket desaparece tambien su politica IAM.

```bash
export PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
export SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud storage buckets remove-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin" \
  --project="${PROJECT_ID}"
```

### 6. Verificar que no queda nada

```bash
gcloud run services list --region="${REGION}" --project="${PROJECT_ID}"
gcloud storage ls --project="${PROJECT_ID}"
gcloud artifacts repositories list --location="${REGION}" --project="${PROJECT_ID}"
```

Confirmar tambien en la consola de facturacion que no hay cargos activos: los datos de facturacion tardan hasta 24 horas en reflejarse.

### Nota sobre las APIs habilitadas

El script habilito `run`, `artifactregistry`, `storage` y `cloudbuild`. **Habilitar una API no genera cargos por si misma**, solo su uso. No es necesario deshabilitarlas, y hacerlo puede afectar a otros servicios del mismo proyecto. Si aun asi se quiere revertir:

```bash
gcloud services disable run.googleapis.com --project="${PROJECT_ID}"
```

### Limpieza local (opcional)

```bash
# Eliminar el modelo del catalogo local de Ollama
ollama rm techcloud-classifier
ollama list
```

---

## 8. Referencias Oficiales

- **Cloud Run Volume Mounts (Cloud Storage FUSE):** https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts
- **Cloud Run GPU:** https://cloud.google.com/run/docs/configuring/services/gpu
- **Artifact Registry:** https://cloud.google.com/artifact-registry/docs
- **Documentacion Oficial de Ollama:** https://ollama.com/
- **API Reference de Ollama:** https://github.com/ollama/ollama/blob/main/docs/api.md
- **Guia Oficial del Formato GGUF en Hugging Face Hub:** https://huggingface.co/docs/hub/gguf
- **Especificacion GGUF (llama.cpp):** https://github.com/ggerganov/llama.cpp