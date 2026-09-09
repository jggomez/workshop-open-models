# Blueprint de Arquitectura SRE: Serving de Produccion con vLLM, Gemma, Cloud Run GPU y Model Armor

Este documento detalla la especificacion de arquitectura para el despliegue de modelos de lenguaje abiertos (Google Gemma) en un entorno de produccion de nivel empresarial sobre **Google Cloud Platform (GCP)**, basado en la solucion de ingenieria del repositorio [`AI-deploy-vllm-gemma`](https://github.com/jggomez/AI-deploy-vllm-gemma).

---

## 1. Diagrama de Arquitectura del Sistema

```mermaid
flowchart TD
    Client([Cliente / Aplicacion Externa]) -->|HTTPS Port 443| LB["Regional External HTTP(S) Load Balancer"]
    
    subgraph "Perimetro de Seguridad y Mitigacion"
        LB -->|Intercept & Inspect| Ext["Service Extension (Wasm / Envoy)"]
        Ext -->|Validar Politicas de Seguridad| MA["Google Cloud Model Armor"]
        MA -->|Veredicto: Limpio| LB
        MA -.->|Veredicto: Bloqueado (400 Bad Request)| Client
    end

    subgraph "VPC Interna y Compute Serverless"
        LB -->|Serverless NEG| vLLM["Cloud Run vLLM Service (NVIDIA L4 GPU)"]
        Sidecar["Prometheus Sidecar Collector"] -.->|Scrape /metrics| vLLM
    end

    subgraph "Capa de Almacenamiento y Pesos"
        vLLM <-->|Montaje Local /mnt/models| FUSE["Cloud Storage FUSE Mount"]
        FUSE <-->|Streaming / Lectura mmap| GCS[("Bucket GCS: Pesos Gemma 2 / 4")]
    end

    subgraph "Observabilidad Corporativa"
        Sidecar -->|Push Metrics| Monitoring[("Google Cloud Monitoring / Logging")]
    end
```

---

## 2. Componentes Fundamentales y Responsabilidades

### 2.1. Motor de Inferencia: vLLM (Virtual Large Language Model)
Frente a implementaciones estandar de Hugging Face Transformers, vLLM proporciona un incremento de throughput entre 10x y 24x gracias a dos innovaciones algoritmicas:
1. **PagedAttention:** En los LLMs autorregresivos, la memoria de video (VRAM) se fragmenta severamente debido al crecimiento impredecible del KV-Cache (*Key-Value Cache*). PagedAttention gestiona el KV-Cache de forma analoga a la memoria virtual paginada de los sistemas operativos, almacenando tokens no contiguos en bloques discretos de memoria. Esto elimina el desperdicio de memoria del 60%-80% a menos del 4%.
2. **Continuous Batching (In-Flight Batching):** En lugar de esperar a que termine la generacion de todas las peticiones en un lote para iniciar el siguiente, vLLM itera a nivel de token individual: cuando una peticion termina, se libera su slot e inmediatamente se incorpora una nueva peticion de la cola sin penalizar a los demas usuarios.

### 2.2. Computo Serverless Acelerado: Google Cloud Run con GPU NVIDIA L4
- **Arquitectura de Hardware:** Instancias Cloud Run aprovisionadas con aceleradores NVIDIA Ada Lovelace L4 (24 GB VRAM GDDR6 con núcleos Tensor de cuarta generacion).
- **Escalado Automatico:** Escalado dinamico basado en concurrencia real (`--concurrency=16` o `32`), con capacidad de auto-escalar a 0 instancias en ventanas sin trafico para reducir costos, o mantener instancias minimas (`--min-instances=1`) para garantizar cero latencia de arranque.
- **Entorno de Ejecucion:** Entorno de segunda generacion de Cloud Run (`gen2`) con CPU no regulada (`--no-cpu-throttling`) para alimentar la GPU continuamente.

### 2.3. Desacoplamiento de Almacenamiento: Cloud Storage FUSE (Volume Mounts)
- **Persistencia Externa:** Los pesos de Gemma (de 4 GB a 10 GB segun cuantizacion y tamano de parametros) residen en un bucket regional de Google Cloud Storage (`gs://<PROJECT_ID>-vllm-models/gemma/`).
- **Montaje Declarativo:** Cloud Run mapea el bucket como un directorio POSIX local en `/mnt/models`. El contenedor de vLLM arranca de forma instantanea sin necesidad de contener los pesos en su imagen Docker ni descargarlos de internet en cada inicio de instancia.

### 2.4. Seguridad Perimetral e Inspeccion: Google Cloud Model Armor
Los modelos de lenguaje expuestos publicamente son susceptibles a vectores de ataque criticos:
- **Jailbreaks y Prompt Injections:** Intentos deliberados de anular las restricciones del sistema.
- **Fuga de Datos Sensibles (PII):** Extraccion de informacion confidencial (correos, numeros de tarjeta, claves).
- **Discurso Toxico y Danino:** Contenido que viola politicas de uso.

**Model Armor** se integra a nivel de red utilizando una **Service Extension** en el External Application Load Balancer. Cada payload entrante y saliente es inspeccionado antes de llegar al contenedor de vLLM. Si se detecta una violacion, el Load Balancer bloquea la peticion de inmediato, protegiendo tanto la seguridad como el presupuesto computacional de la GPU.

### 2.5. Observabilidad Granular: Sidecar de Prometheus
vLLM expone de forma nativa metricas detalladas en el endpoint interno `/metrics`. Para recolectarlas sin alterar la imagen de inferencia, se inyecta un contenedor sidecar (`us-docker.pkg.dev/cloud-ops-agents-artifacts/cloud-run-gmp-sidecar/cloud-run-gmp-sidecar:1.2.0`) administrado mediante Google Cloud Managed Service for Prometheus (GMP) con la anotacion `run.googleapis.com/container-dependencies: '{"collector":["app"]}'`:
- Mide el **Time To First Token (TTFT)**.
- Mide los **Tokens Per Second (TPS)** reales.
- Monitorea el porcentaje de ocupacion del KV-Cache.
- Exporta metricas directamente a Google Cloud Monitoring para configurar alertas SRE y tableros operativos.

---

## 3. Matriz de Seguridad y Permisos IAM (Principio de Menor Privilegio)

| Identidad / Servicio | Rol Asignado | Proposito |
|---|---|---|
| `vllm-sa@<PROJECT>.iam` | `roles/storage.objectViewer` | Lectura exclusiva de los pesos del modelo en el bucket GCS. |
| `vllm-sa@<PROJECT>.iam` | `roles/logging.logWriter` | Emision de logs de inferencia a Cloud Logging. |
| `vllm-sa@<PROJECT>.iam` | `roles/monitoring.metricWriter` | Envio de telemetria al agente de Cloud Monitoring. |
| `vllm-sa@<PROJECT>.iam` | `roles/secretmanager.secretAccessor` | Lectura segura del token de Hugging Face durante el build. |
| Load Balancer | `roles/networkservices.serviceExtensionUser` | Invocacion del pipeline de filtrado de Model Armor. |
