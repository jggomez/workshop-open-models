# Hands-on Lab 3 (Sesion 4): Despliegue de Modelos con Vertex AI Model Garden (DeepSeek y Gemma)

Esta guia practica paso a paso documenta la metodologia para descubrir, evaluar y desplegar modelos de lenguaje abiertos avanzados (**DeepSeek**, **Google Gemma**, **Llama 3**) en produccion utilizando **Google Cloud Vertex AI Model Garden**, basada en las directrices tecnicas del articulo [*Deploying Powerful Language Models with Ease: Leveraging DeepSeek and Model Garden on Vertex AI*](https://jggomezt.medium.com/deploying-powerful-language-models-with-ease-leveraging-deepseek-and-model-garden-on-vertex-ai-2068dae099c1).

---

## 1. Que es Vertex AI Model Garden?

**Vertex AI Model Garden** es el centro de mando y catalogo unificado de Google Cloud para el ciclo de vida de modelos fundacionales. Proporciona acceso curado a mas de 150 modelos de vanguardia que incluyen:
- Modelos propietarios de Google: Familia **Gemini** (Gemini 1.5 Pro, Flash) e **Imagen 3**.
- Modelos de codigo abierto de alta demanda: **Google Gemma 2**, **DeepSeek-R1**, **DeepSeek-V3**, **Meta Llama 3**, **Mistral**, **Qwen** y **Whisper**.
- Modelos de socios tecnologicos: **Anthropic Claude**.

Model Garden elimina la complejidad de configurar clústeres de cómputo manuales, aprovisionar drivers de GPU NVIDIA, compilar contenedores de inferencia o configurar redes complejas, ofreciendo una experiencia estandarizada desde la exploracion hasta el endpoint en produccion.

---

## 2. Decision de Arquitectura: Model-as-a-Service (MaaS) vs. Dedicated Endpoints

Al desplegar modelos abiertos en Vertex AI, la plataforma ofrece dos modalidades de consumo con caracteristicas operativas y de costo diferenciadas:

| Dimension | Model-as-a-Service (MaaS / Serverless API) | Dedicated Endpoints (Self-Deployed) |
|---|---|---|
| **Gestion de Servidores** | Cero infraestructura: Google administra el cluster en segundo plano. | Servidores e infraestructura dedicados en su proyecto de GCP. |
| **Modelo de Costos** | **Pago por token consumido** (Pay-as-you-go). Costo cero cuando no hay trafico. | **Pago por hora de máquina/GPU activa**, independientemente del uso. |
| **Personalizacion de Pesos** | Solo permite consumir los pesos base o variantes instruccionales estandar. | Permite desplegar pesos base, cuantizados o **adaptadores LoRA propios**. |
| **Tiempo de Despliegue** | **Inmediato (0 minutos)**: Las APIs ya estan disponibles. | **15 a 30 minutos** mientras se aprovisiona el nodo y se descargan pesos. |
| **Latencia y SLA** | Concurrencia compartida; latencia sujeta a variaciones de carga global. | **Latencia determinista y SLA garantizado** con cuotas dedicadas. |
| **Seguridad de Red** | Endpoint publico autenticado via Google Cloud IAM. | Posibilidad de integrar en **VPC Service Controls** y redes privadas. |
| **Caso de Uso Ideal** | Prototipado rapido, pruebas de concepto, cargas esporadicas. | **Produccion empresarial de alto volumen**, SLAs criticos y datos aislados. |

---

## 3. Estructura del Laboratorio

```text
03-vertex-ai-model-garden/
├── README.md                     # Esta guia metodologica y paso a paso
└── test_vertex_model_garden.py   # Script de invocacion en Python con google-cloud-aiplatform
```

---

## 4. Guia Paso a Paso de Despliegue en Google Cloud Console

Siga estos pasos en la consola web de Google Cloud para desplegar un modelo abierto (ej. **DeepSeek-R1-Distill-Llama-8B** o **Google Gemma 2 9B**):

### Paso 1: Acceso a Model Garden
1. Inicie sesion en [Google Cloud Console](https://console.cloud.google.com/).
2. Asegurese de seleccionar su proyecto activo de GCP con facturacion habilitada.
3. En el menu de navegacion lateral, seleccione **Vertex AI > Model Garden**.

### Paso 2: Busqueda y Seleccion del Modelo
1. En la barra de busqueda de Model Garden, escriba `DeepSeek` o `Gemma`.
2. Haga clic en la tarjeta del modelo deseado (ej. **DeepSeek-R1** o **Gemma 2**).
3. Revise la ficha tecnica (*Model Card*):
   - Descripcion de arquitectura y contexto maximo.
   - Licencia de uso comercial o academica.
   - Requisitos minimos de aceleracion por hardware recomendados.

### Paso 3: Opciones de Despliegue (Boton Deploy)
En la parte superior de la tarjeta del modelo, haga clic en el boton **Deploy** (Desplegar). Se abrira el asistente de configuracion:

1. **Configuracion del Endpoint:**
   - **Endpoint Name:** Asigne un nombre identificador (ej. `deepseek-r1-endpoint` o `gemma2-production`).
   - **Region:** Seleccione una region con disponibilidad de cuotas de GPU (ej. `us-central1` o `us-east4`).
2. **Seleccion de Hardware y Maquina:**
   - Para modelos de 2B a 9B: Seleccione la familia **G2** con aceleradores **NVIDIA L4** (ej. `g2-standard-8`, 1x NVIDIA L4 con 24 GB VRAM).
   - Para modelos de gran escala (>27B a 70B): Seleccione la familia **A2** con aceleradores **NVIDIA A100** (ej. `a2-highgpu-1g` o `a2-ultragpu-1g`).
3. **Contenedor de Inferencia Preconfigurado:**
   - Vertex AI selecciona automaticamente una imagen optimizada basada en **vLLM** o **Text Generation Inference (TGI)** mantenida por Google.
4. **Escalado Automatico:**
   - **Minimum number of nodes:** Establezca `1` para disponibilidad inmediata (o `0` si el tipo de máquina soporta escalado a cero).
   - **Maximum number of nodes:** Establezca un limite superior para control de costos (ej. `3`).
5. Haga clic en **Deploy**. El aprovisionamiento toma entre 10 y 25 minutos mientras Google asigna los nodos y monta los artefactos.

---

## 5. Pruebas de Inferencia

### Opcion A: Pruebas desde Vertex AI Studio (Consola Web)
1. Una vez completado el despliegue, navegue a **Vertex AI > Online Prediction > Endpoints**.
2. Seleccione su endpoint activo.
3. En la pestana **Test & Use**, ingrese un prompt de prueba en formato JSON y haga clic en **Predict**.

### Opcion B: Pruebas Programaticas con Python SDK
Utilice el script `test_vertex_model_garden.py` provisto en este directorio para realizar predicciones automatizadas:

```bash
# 1. Instalar biblioteca oficial de Google Cloud
pip install google-cloud-aiplatform

# 2. Autenticar credenciales locales con Application Default Credentials (ADC)
gcloud auth application-default login

# 3. Invocar el endpoint pasando Project ID, Region y Endpoint ID
python3 test_vertex_model_garden.py \
  "TU_PROJECT_ID" \
  "us-central1" \
  "TU_ENDPOINT_ID" \
  "Explica brevemente que ventajas tecnicas ofrece la arquitectura DeepSeek-R1 para razonamiento logico."
```

---

## 6. Control de Costos y Buenas Practicas de Teardown (Undeploy)

> [!CAUTION]
> **Alerta Critica de Facturacion:**  
> A diferencia de los servicios serverless tradicionales, los **Dedicated Endpoints** en Vertex AI cobran de forma continua por cada hora que la instancia de GPU permanezca asignada, **incluso si no esta recibiendo peticiones**.

### Como Desmantelar el Endpoint para Detener la Facturacion:
1. En la consola de Google Cloud, vaya a **Vertex AI > Online Prediction > Endpoints**.
2. Haga clic en el endpoint desplegado.
3. En la seccion de modelos asociados, haga clic en el menu de tres puntos y seleccione **Undeploy model from endpoint**.
4. Confirme la accion. Esto liberara las GPUs y detendra de inmediato el costo horario.
5. Opcionalmente, elimine el recurso vacio del endpoint.

Desmantelamiento por linea de comandos:
```bash
# Desasociar modelo del endpoint
gcloud ai endpoints undeploy-model "${ENDPOINT_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --deployed-model-id="${DEPLOYED_MODEL_ID}"

# Eliminar el endpoint
gcloud ai endpoints delete "${ENDPOINT_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --quiet
```

---

## 7. Referencias Oficiales y Articulo Tecnico

- **Articulo Original de Referencia:** [*Deploying Powerful Language Models with Ease: Leveraging DeepSeek and Model Garden on Vertex AI*](https://jggomezt.medium.com/deploying-powerful-language-models-with-ease-leveraging-deepseek-and-model-garden-on-vertex-ai-2068dae099c1)
- **Portal Oficial de Vertex AI Model Garden:** [https://cloud.google.com/vertex-ai/docs/model-garden/explore-models](https://cloud.google.com/vertex-ai/docs/model-garden/explore-models)
- **Google Cloud AI Platform Python SDK:** [https://cloud.google.com/python/docs/reference/aiplatform/latest](https://cloud.google.com/python/docs/reference/aiplatform/latest)
- **Modelos DeepSeek en Vertex AI:** [https://cloud.google.com/vertex-ai/docs/model-garden/explore-models#deepseek](https://cloud.google.com/vertex-ai/docs/model-garden/explore-models#deepseek)
