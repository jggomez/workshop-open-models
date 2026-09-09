# Hands-on Lab 3 (Sesion 4): Despliegue de Modelos con Vertex AI Model Garden

Esta guia practica documenta la metodologia para descubrir, evaluar y desplegar modelos de lenguaje abiertos en produccion utilizando **Google Cloud Vertex AI Model Garden**.

> **Sobre las versiones de modelos:** el catalogo de Model Garden cambia con frecuencia y los modelos tienen ciclo de vida propio (fechas de retirada, restricciones para proyectos nuevos). Esta guia describe el **procedimiento**, que es estable; los nombres y versiones concretos deben verificarse en la consola al momento de ejecutar el laboratorio.

---

## 1. Que es Vertex AI Model Garden?

**Vertex AI Model Garden** es el catalogo unificado de Google Cloud para descubrir, probar, personalizar y desplegar modelos. Ofrece acceso curado a mas de **200 modelos** de Google, de socios y de codigo abierto.

Categorias oficiales del catalogo:

| Categoria | Descripcion |
|---|---|
| **Foundation models** | Modelos multitarea preentrenados, ajustables o personalizables desde Vertex AI Studio, la API o el SDK de Python. Incluye las familias Gemini, Imagen, Veo y Chirp. |
| **Fine-tunable models** | Modelos afinables mediante notebooks o pipelines. Incluye la familia Gemma y variantes como CodeGemma o PaliGemma. |
| **Task-specific solutions** | Modelos preconstruidos listos para usar, muchos personalizables con datos propios. |

Model Garden elimina la complejidad de configurar clusteres de computo manuales, aprovisionar drivers de GPU NVIDIA, compilar contenedores de inferencia o configurar redes, ofreciendo una experiencia estandarizada desde la exploracion hasta el endpoint en produccion.

### Filtros del catalogo

En el panel de filtros de la consola se puede acotar por:

- **Tasks:** la tarea que debe realizar el modelo.
- **Model collections:** modelos gestionados por Google, por socios o propios.
- **Providers:** proveedor del modelo.
- **Features:** caracteristicas requeridas.

### Ciclo de vida de los modelos

Los modelos tienen fechas de retirada y restricciones de disponibilidad. Un ejemplo concreto: desde el **29 de abril de 2025**, Gemini 1.5 Pro y Gemini 1.5 Flash dejaron de estar disponibles en proyectos sin uso previo de esos modelos, incluidos todos los proyectos nuevos.

Antes de construir sobre un modelo, revisar su ficha tecnica y la pagina de versiones y ciclo de vida.

---

## 2. Decision de Arquitectura: Model-as-a-Service (MaaS) vs. Dedicated Endpoints

| Dimension | Model-as-a-Service (MaaS / Serverless API) | Dedicated Endpoints (Self-Deployed) |
|---|---|---|
| **Gestion de Servidores** | Cero infraestructura: Google administra el cluster. | Servidores e infraestructura dedicados en su proyecto de GCP. |
| **Modelo de Costos** | **Pago por token consumido**. Costo cero sin trafico. | **Pago por hora de maquina/GPU asignada**, independientemente del uso. |
| **Personalizacion de Pesos** | Solo pesos base o variantes instruccionales estandar. | Permite desplegar pesos base, cuantizados o **adaptadores LoRA propios**. |
| **Tiempo de Despliegue** | **Inmediato**: las APIs ya estan disponibles. | **15 a 30 minutos** de aprovisionamiento y descarga de pesos. |
| **Latencia y SLA** | Concurrencia compartida; latencia sujeta a carga global. | **Latencia determinista y SLA garantizado** con cuotas dedicadas. |
| **Seguridad de Red** | Endpoint publico autenticado via Google Cloud IAM. | Integrable en **VPC Service Controls** y redes privadas. |
| **Caso de Uso Ideal** | Prototipado, pruebas de concepto, cargas esporadicas. | **Produccion de alto volumen**, SLAs criticos y datos aislados. |

**Relacion con el Lab 1 (Ollama en Cloud Run):** aquel enfoque sirve modelos cuantizados propios sobre infraestructura serverless de bajo costo. Model Garden con Dedicated Endpoint sirve modelos del catalogo sobre GPU dedicada con SLA. Son complementarios: el primero optimiza costo y control del artefacto; el segundo, latencia y garantias operativas.

---

## 3. Estructura del Laboratorio

```text
03-vertex-ai-model-garden/
├── README.md                     # Esta guia metodologica y paso a paso
└── test_vertex_model_garden.py   # Script de invocacion con google-cloud-aiplatform
```

---

## 4. Requisito Previo: Cuota de GPU

Este es el punto donde falla la mayoria de los despliegues, y conviene resolverlo **antes** de empezar. Los proyectos nuevos tienen cuota **cero** de GPU para Vertex AI: el asistente de despliegue permite seleccionar el hardware y el error aparece minutos despues, durante el aprovisionamiento.

Verificar la cuota disponible:

```bash
gcloud alpha services quota list \
  --service=aiplatform.googleapis.com \
  --consumer=projects/${PROJECT_ID} \
  --filter="custom_model_serving_nvidia_l4_gpus"
```

Tambien en la consola: **IAM & Admin > Quotas**, filtrando por `aiplatform.googleapis.com` y `nvidia_l4` o `nvidia_a100`.

Si la cuota es 0, solicitar aumento desde esa misma pantalla. La aprobacion puede tardar de horas a dias segun la region.

---

## 5. Guia Paso a Paso de Despliegue

### Paso 1: Acceso a Model Garden

1. Inicie sesion en [Google Cloud Console](https://console.cloud.google.com/).
2. Seleccione su proyecto activo con facturacion habilitada.
3. En el menu lateral, seleccione **Vertex AI > Model Garden**.

### Paso 2: Busqueda y Seleccion del Modelo

1. Use la barra de busqueda o los filtros del panel lateral.
2. Haga clic en la tarjeta del modelo deseado.
3. Revise la ficha tecnica (*Model Card*):
   - Arquitectura y ventana de contexto maxima.
   - Licencia de uso comercial o academica.
   - Hardware minimo recomendado.
   - **Fechas de retirada o restricciones de disponibilidad.**

### Paso 3: Configuracion del Despliegue

Haga clic en **Deploy** en la parte superior de la tarjeta.

1. **Endpoint:**
   - **Name:** identificador descriptivo (ej. `gemma-lab-endpoint`).
   - **Region:** una con cuota de GPU disponible (ej. `us-central1`, `us-east4`).

2. **Hardware:**

   | Tamano del modelo | Familia | Ejemplo de maquina |
   |---|---|---|
   | 2B a 9B | G2 (NVIDIA L4, 24 GB) | `g2-standard-8`, 1x L4 |
   | 27B a 70B | A2 (NVIDIA A100) | `a2-highgpu-1g`, `a2-ultragpu-1g` |

   La ficha del modelo indica la configuracion recomendada. Un modelo que no cabe en la VRAM disponible falla durante la carga.

3. **Contenedor de inferencia:** Vertex AI selecciona automaticamente una imagen optimizada basada en **vLLM** o **Text Generation Inference (TGI)** mantenida por Google.

4. **Escalado automatico:**
   - **Minimum nodes:** `1`.
   - **Maximum nodes:** un limite bajo para control de costos (ej. `2` o `3`).

   > Los Dedicated Endpoints con GPU **no escalan a cero**. El minimo de 1 nodo implica facturacion continua mientras el modelo permanezca desplegado. Para cargas esporadicas, MaaS es la opcion adecuada.

5. Haga clic en **Deploy**. El aprovisionamiento toma entre 10 y 25 minutos.

---

## 6. Pruebas de Inferencia

### Opcion A: Consola Web

1. Navegue a **Vertex AI > Online Prediction > Endpoints**.
2. Seleccione su endpoint activo.
3. En la pestana **Test & Use**, ingrese un prompt y haga clic en **Predict**.

### Opcion B: Python SDK

```bash
pip install google-cloud-aiplatform
gcloud auth application-default login

python3 test_vertex_model_garden.py \
  "${PROJECT_ID}" \
  "us-central1" \
  "${ENDPOINT_ID}" \
  "Explica brevemente en que consiste la cuantizacion de modelos de lenguaje."
```

Obtener el `ENDPOINT_ID` por linea de comandos:

```bash
gcloud ai endpoints list --region="${REGION}" --project="${PROJECT_ID}"
```

> **Formato del payload:** cada contenedor de inferencia espera un esquema distinto. Un endpoint servido con vLLM no acepta el mismo cuerpo que uno con TGI, y los modelos ajustados a chat requieren su plantilla de turnos correspondiente. El formato exacto aparece en la pestana **Sample request** de la ficha del modelo. Si la prediccion devuelve un error 400, revisar ahi antes que en el codigo del cliente.

---

## 7. Control de Costos y Teardown

> [!CAUTION]
> **Alerta Critica de Facturacion**
> Los **Dedicated Endpoints** cobran por cada hora que la instancia de GPU permanezca asignada, **incluso sin recibir peticiones**. Un `a2-ultragpu-1g` olvidado durante un fin de semana genera un cargo considerable.

### Desmantelamiento desde la Consola

1. Vaya a **Vertex AI > Online Prediction > Endpoints**.
2. Haga clic en el endpoint desplegado.
3. En la seccion de modelos asociados, menu de tres puntos > **Undeploy model from endpoint**.
4. Confirme. Esto libera las GPUs y detiene el costo horario de inmediato.
5. Elimine el recurso vacio del endpoint.

### Desmantelamiento por Linea de Comandos

El `undeploy-model` requiere el `DEPLOYED_MODEL_ID`, que **no es el mismo** que el `ENDPOINT_ID` ni que el ID del modelo en el catalogo. Obtenerlo primero:

```bash
export ENDPOINT_ID="..."
export REGION="us-central1"
export PROJECT_ID=$(gcloud config get-value project)

gcloud ai endpoints describe "${ENDPOINT_ID}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(deployedModels[].id)"
```

```bash
export DEPLOYED_MODEL_ID="..."

# 1. Desasociar el modelo: detiene la facturacion de GPU
gcloud ai endpoints undeploy-model "${ENDPOINT_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --deployed-model-id="${DEPLOYED_MODEL_ID}"

# 2. Eliminar el endpoint vacio
gcloud ai endpoints delete "${ENDPOINT_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --quiet
```

### Verificacion Final

```bash
# No debe quedar ningun endpoint activo
gcloud ai endpoints list --region="${REGION}" --project="${PROJECT_ID}"

# Revisar tambien otras regiones si se probaron varias
for R in us-central1 us-east4 europe-west4; do
  echo "== ${R} =="
  gcloud ai endpoints list --region="${R}" --project="${PROJECT_ID}"
done
```

Los datos de facturacion tardan hasta 24 horas en reflejarse, asi que revisar la consola justo despues del teardown seguira mostrando cargos ya detenidos.

---

## 8. Referencias Oficiales

- **Model Garden (portal):** https://cloud.google.com/model-garden
- **Explorar modelos:** https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models
- **Versiones y ciclo de vida de modelos:** https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
- **Cuotas y limites de Vertex AI:** https://cloud.google.com/vertex-ai/docs/quotas
- **Google Cloud AI Platform Python SDK:** https://cloud.google.com/python/docs/reference/aiplatform/latest