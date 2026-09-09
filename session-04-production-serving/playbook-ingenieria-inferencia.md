# Playbook de Ingenieria de Inferencia

## Servir modelos de lenguaje en produccion: llama.cpp, Ollama, vLLM y Vertex AI Model Garden

---

## Indice

1. [Que es la ingenieria de inferencia](#1-que-es-la-ingenieria-de-inferencia)
2. [Fundamentos: que ocurre al generar un token](#2-fundamentos-que-ocurre-al-generar-un-token)
3. [Metricas que importan](#3-metricas-que-importan)
4. [Cuantizacion](#4-cuantizacion)
5. [llama.cpp y el formato GGUF](#5-llamacpp-y-el-formato-gguf)
6. [Ollama](#6-ollama)
7. [vLLM](#7-vllm)
8. [Vertex AI Model Garden](#8-vertex-ai-model-garden)
9. [Comparativa y arbol de decision](#9-comparativa-y-arbol-de-decision)
10. [Patrones de despliegue](#10-patrones-de-despliegue)
11. [Operacion: benchmarking y costos](#11-operacion-benchmarking-y-costos)
12. [Errores frecuentes](#12-errores-frecuentes)
13. [Referencias](#13-referencias)

---

# 1. Que es la ingenieria de inferencia

## 1.1 Definicion

El entrenamiento de un modelo ocurre una vez. La inferencia ocurre millones de veces. La **ingenieria de inferencia** es la disciplina que se ocupa de esa segunda fase: tomar un modelo ya entrenado y servirlo de forma que cumpla objetivos de latencia, throughput, costo y disponibilidad.

Es una disciplina distinta del entrenamiento, con restricciones propias:

| Dimension | Entrenamiento | Inferencia |
|---|---|---|
| **Frecuencia** | Una vez (o periodica) | Continua, bajo demanda |
| **Patron de computo** | Lotes grandes, paralelismo masivo | Peticiones dispersas, tamano variable |
| **Cuello de botella** | Computo (FLOPs) | Memoria (ancho de banda) |
| **Metrica de exito** | Loss de validacion | Latencia percibida, costo por token |
| **Tolerancia a fallos** | Alta (reintentar el paso) | Baja (el usuario espera) |
| **Optimizacion** | Convergencia | Utilizacion del hardware |

Un modelo con excelente calidad que responde en 40 segundos es inutilizable en un chat. Un modelo que responde en 200 ms pero cuesta 3.000 dolares al mes en GPU ociosa es inviable para una carga esporadica. La ingenieria de inferencia consiste en navegar ese espacio de compromisos.

## 1.2 Las tres preguntas

Todo diseno de servicio de inferencia se reduce a tres preguntas:

**Cuantas peticiones simultaneas debo atender?** Determina si necesitas batching agresivo o si basta con procesar una a una. Un asistente interno para 20 personas y una API publica con 10.000 usuarios concurrentes requieren arquitecturas incompatibles.

**Cuanta latencia tolera el usuario?** Un chat interactivo necesita el primer token en menos de un segundo. Un pipeline nocturno de clasificacion de tickets puede tardar minutos por lote sin que nadie lo note.

**Que control necesito sobre el artefacto?** Servir un modelo del catalogo de un proveedor es distinto de servir tus propios adaptadores LoRA, y eso a su vez es distinto de servir un modelo cuantizado a medida que solo existe en tu disco.

Las respuestas a esas tres preguntas determinan la eleccion de herramienta antes que cualquier consideracion tecnica.

## 1.3 El espacio de herramientas

Este playbook cubre cuatro puntos de ese espacio:

- **llama.cpp**: el motor de inferencia en C++ que hace posible ejecutar modelos cuantizados en hardware modesto, incluido CPU puro.
- **Ollama**: una capa de gestion sobre llama.cpp que convierte el motor en un servicio con catalogo de modelos y API REST.
- **vLLM**: un motor de inferencia orientado a throughput sobre GPU, disenado para servir muchas peticiones concurrentes con alta utilizacion del hardware.
- **Vertex AI Model Garden**: la capa gestionada de Google Cloud, que abstrae la infraestructura por completo.

No son alternativas equivalentes. Ocupan nichos distintos y en muchas arquitecturas conviven.

---

# 2. Fundamentos: que ocurre al generar un token

Sin entender esta seccion, las decisiones de las siguientes son arbitrarias.

## 2.1 Las dos fases: prefill y decode

Generar una respuesta tiene dos fases con perfiles de computo radicalmente distintos.

### Prefill (procesamiento del prompt)

El modelo procesa **todos los tokens del prompt a la vez**. Si el prompt tiene 500 tokens, se calculan las 500 posiciones en paralelo en una sola pasada por la red.

Es una operacion **compute-bound**: hay mucho trabajo matematico (multiplicaciones matriz-matriz grandes) y los pesos se leen una sola vez para procesar muchos tokens. La GPU trabaja cerca de su capacidad teorica de FLOPs.

La duracion del prefill determina el **Time To First Token (TTFT)**.

### Decode (generacion autoregresiva)

Una vez procesado el prompt, el modelo genera un token, lo anade a la secuencia, y vuelve a pasar por toda la red para generar el siguiente. **Un token por pasada completa.**

Es una operacion **memory-bound**. En cada pasada hay que leer todos los pesos del modelo desde la memoria para calcular un unico token. Con un modelo de 7B en 16 bits, eso significa mover 14 GB de memoria para producir una palabra.

Aqui esta la asimetria fundamental:

```
Prefill:  500 tokens en 1 pasada  -> aprovecha el hardware
Decode:   1 token   en 1 pasada   -> desperdicia el hardware
```

## 2.2 Por que decode desperdicia el hardware

Una GPU moderna puede hacer del orden de 10^14 operaciones por segundo, pero solo puede leer del orden de 10^12 bytes por segundo desde su memoria. La proporcion entre ambas cifras es de aproximadamente 100 a 1.

Esto significa que por cada byte leido de memoria, el hardware podria hacer unas 100 operaciones antes de quedarse sin datos. A esa proporcion se le llama **intensidad aritmetica**.

En decode con una sola peticion, la intensidad aritmetica es aproximadamente 1: se lee un peso y se usa una vez. El resultado es que la GPU pasa la mayor parte del tiempo **esperando datos**, no calculando.

La consecuencia practica es contraintuitiva:

> Generar tokens para 32 peticiones simultaneas tarda casi lo mismo que generar tokens para una sola.

Porque el coste dominante es leer los pesos, y los pesos se leen una vez para todo el lote. Este hecho es la base de todas las tecnicas de batching, y explica por que vLLM existe.

## 2.3 El KV cache

Sin optimizacion, generar el token N requeriria recalcular la atencion sobre los N-1 tokens anteriores. Seria cuadratico y prohibitivo.

La solucion es el **KV cache**: guardar en memoria los vectores Key y Value calculados para cada token, y reutilizarlos en los pasos siguientes. Convierte el coste por token de cuadratico a lineal.

El precio es memoria. Su tamano se calcula asi:

```
tamano_kv = 2 x n_capas x n_cabezas_kv x dim_cabeza x longitud_seq x batch x bytes_por_valor
```

El factor 2 corresponde a Key y Value. Un ejemplo con un modelo de 7B:

```
32 capas, 32 cabezas KV, dim 128, 2 bytes (fp16)
Por token:  2 x 32 x 32 x 128 x 2 = 524.288 bytes  (~0,5 MB)
4.000 tokens de contexto:           ~2 GB
16 peticiones concurrentes:        ~32 GB
```

**El KV cache puede ocupar mas memoria que el propio modelo.** Esta es la restriccion central de cualquier servicio de inferencia con concurrencia, y el problema que PagedAttention resuelve.

### Grouped-Query Attention (GQA)

Las arquitecturas modernas reducen este coste compartiendo las cabezas Key/Value entre varias cabezas Query. Un modelo con 32 cabezas Query pero solo 8 cabezas KV reduce el KV cache a la cuarta parte, con impacto minimo en calidad.

Al evaluar la memoria necesaria, lo que cuenta es `n_cabezas_kv`, no `n_cabezas_query`. Es un error frecuente en el dimensionamiento.

## 2.4 Muestreo (sampling)

En cada paso el modelo produce un vector de logits, un numero por cada token del vocabulario. Convertirlo en un token concreto es el muestreo, y sus parametros determinan el comportamiento observable del sistema.

| Parametro | Efecto | Uso recomendado |
|---|---|---|
| `temperature` | Escala los logits antes del softmax. 0 = determinista (greedy). Valores altos aplanan la distribucion. | **0 para extraccion estructurada.** 0,7 a 1,0 para texto creativo. |
| `top_p` (nucleus) | Restringe a los tokens cuya probabilidad acumulada alcanza p. | 0,9 a 0,95 con temperatura > 0. Irrelevante con temperatura 0. |
| `top_k` | Restringe a los k tokens mas probables. | Alternativa mas burda a top_p. |
| `min_p` | Umbral relativo al token mas probable. | Alternativa moderna a top_p. |
| `repeat_penalty` | Penaliza tokens ya generados. | Con cuidado: puede degradar salidas estructuradas donde la repeticion es legitima. |

Regla practica: **para tareas de extraccion de datos, clasificacion o generacion de JSON, usar temperatura 0**. La variabilidad no aporta valor y si introduce riesgo de salida mal formada. Con temperatura 0, `top_p`, `top_k` y `min_p` no tienen efecto y deben omitirse.

Un peligro concreto de `repeat_penalty` con salidas estructuradas: si el modelo debe generar `{"a": 1, "b": 2}`, las comillas y las llaves se repiten legitimamente. Penalizarlas puede romper la sintaxis.

## 2.5 Tokens de control y plantillas de chat

Un modelo ajustado a instrucciones no espera texto plano, sino una secuencia con marcadores de turno especificos de su familia:

| Familia | Formato de turno |
|---|---|
| Gemma 2 / 3 | `<start_of_turn>user ... <end_of_turn>` |
| Llama 3 | `<\|start_header_id\|>user<\|end_header_id\|> ... <\|eot_id\|>` |
| Qwen / ChatML | `<\|im_start\|>user ... <\|im_end\|>` |
| Mistral | `[INST] ... [/INST]` |

Tres reglas operativas se derivan de esto:

**La plantilla de inferencia debe coincidir exactamente con la de entrenamiento.** Cualquier divergencia (un salto de linea de mas, un rol que el modelo no conoce) degrada la calidad de forma silenciosa. No produce error, produce respuestas peores.

**No todos los modelos tienen rol de sistema.** Gemma, por ejemplo, no define `<start_of_turn>system`. Enviar un mensaje con `"role": "system"` a un modelo Gemma lo obliga a colocarlo en algun sitio no previsto.

**El token BOS se anade una sola vez.** Muchas plantillas de chat ya incluyen `<bos>` al inicio, y muchos tokenizadores lo anaden automaticamente. Si ocurren ambas cosas, el modelo recibe `<bos><bos>` y algunas arquitecturas degeneran de forma espectacular ante ese patron.

---

# 3. Metricas que importan

## 3.1 Las cuatro metricas fundamentales

| Metrica | Definicion | Que la afecta |
|---|---|---|
| **TTFT** (Time To First Token) | Tiempo hasta el primer token de la respuesta. | Longitud del prompt, tiempo de cola, prefill. |
| **TPOT** (Time Per Output Token) | Tiempo medio entre tokens consecutivos. Tambien ITL (Inter-Token Latency). | Ancho de banda de memoria, tamano del lote, tamano del modelo. |
| **Latencia total** | `TTFT + (TPOT x n_tokens_salida)` | Ambas anteriores mas la longitud de la respuesta. |
| **Throughput** | Tokens por segundo agregados en todo el sistema. | Concurrencia, eficiencia del batching. |

## 3.2 El compromiso central

**Latencia y throughput estan en tension directa.**

Aumentar el tamano del lote mejora el throughput (mas tokens totales por segundo) pero empeora el TPOT de cada peticion individual (cada una espera a las demas). Reducir el lote hace lo contrario.

No existe una configuracion optima universal. Existe una configuracion optima **para un objetivo declarado**:

```
Chat interactivo:        optimizar TTFT y TPOT, aceptar throughput bajo
Procesamiento por lotes: optimizar throughput, aceptar latencia alta
API multiusuario:        maximizar throughput sujeto a TPOT < umbral
```

Este ultimo caso, el mas comun en produccion, es exactamente el problema que vLLM esta disenado para resolver.

## 3.3 Referencias de percepcion humana

| TPOT | Tokens/s | Percepcion |
|---|---|---|
| > 200 ms | < 5 | Doloroso, se percibe como bloqueo |
| 100 ms | 10 | Lento pero utilizable |
| 50 ms | 20 | Comodo, similar a lectura rapida |
| < 25 ms | > 40 | Mas rapido de lo que se puede leer |

Superar los 40 tokens por segundo en un chat aporta poco valor percibido. Ese presupuesto se aprovecha mejor en atender mas usuarios concurrentes.

Para TTFT, el umbral practico esta en torno a **1 segundo**. Por encima, el usuario percibe que "no ha pasado nada". El streaming de la respuesta mitiga esto: mostrar tokens conforme llegan hace que la latencia total importe mucho menos.

## 3.4 Metricas de costo

| Metrica | Formula | Aplicable a |
|---|---|---|
| Costo por millon de tokens | `(costo_hora / (tokens_seg x 3600)) x 10^6` | Comparar despliegues |
| Utilizacion | `tiempo_computando / tiempo_asignado` | Detectar GPU ociosa |
| Costo por peticion | `costo_hora x latencia_media / 3600` | Modelar carga real |

La **utilizacion** es la metrica que mas dinero desperdicia cuando se ignora. Una GPU dedicada con 5% de utilizacion cuesta lo mismo que una al 95%. Si la utilizacion medida es baja de forma persistente, la arquitectura correcta probablemente es serverless o pago por token, no una instancia mas pequena.

---

# 4. Cuantizacion

## 4.1 El problema

Un modelo de 7.000 millones de parametros en 16 bits ocupa 14 GB solo en pesos, sin contar el KV cache ni los buffers de activacion. Eso excede la VRAM de la mayoria de las GPU de consumo y de las instancias economicas en la nube.

La **cuantizacion** reduce la precision numerica de los pesos. De 16 bits a 4 bits, el mismo modelo pasa a ocupar unos 3,5 GB.

El beneficio no es solo caber en memoria. Como el decode es memory-bound, **leer menos bytes se traduce directamente en mas velocidad**. Un modelo cuantizado a 4 bits genera tokens aproximadamente al doble o triple de velocidad que el mismo modelo en 16 bits, en el mismo hardware.

## 4.2 Como funciona

La idea basica es agrupar los pesos en bloques y representarlos con menos bits mas un factor de escala por bloque:

```
Original (fp16):  [0.0234, -0.0187, 0.0891, ...]   16 bits cada uno
Cuantizado (4b):  escala=0.0056, [4, -3, 15, ...]   4 bits + escala por bloque
```

La precision perdida se compensa parcialmente porque las distribuciones de pesos en redes neuronales son aproximadamente gaussianas y centradas, y porque el error se promedia a lo largo de muchas capas.

## 4.3 Formatos principales

| Formato | Motor | Caracteristica | Uso tipico |
|---|---|---|---|
| **GGUF** | llama.cpp / Ollama | K-quants con precision mixta por tensor. Soporta CPU y offload parcial. | Local, edge, CPU, hardware heterogeneo |
| **GPTQ** | vLLM, ExLlama | Cuantizacion post-entrenamiento con datos de calibracion. Solo GPU. | Servir en GPU |
| **AWQ** | vLLM, AutoAWQ | Preserva los canales de activacion mas importantes. Solo GPU. | Servir en GPU, mejor calidad que GPTQ a igual tamano |
| **FP8** | vLLM, TensorRT-LLM | Formato nativo en hardware Hopper y Ada. Minima perdida. | GPU moderna, produccion |
| **NF4 / bitsandbytes** | Transformers, PEFT | Optimizado para distribuciones normales. Usado en QLoRA. | Entrenamiento, no serving |

Nota importante: **NF4 / bitsandbytes es un formato de entrenamiento, no de servicio**. Es lo que se usa durante el fine-tuning con QLoRA, pero para servir conviene convertir a GGUF (CPU/local) o a AWQ/GPTQ/FP8 (GPU). El rendimiento de inferencia de bitsandbytes es notablemente inferior.

## 4.4 Niveles de cuantizacion en GGUF

Los nombres siguen el patron `Q<bits>_<variante>`:

| Nivel | Bits efectivos | Tamano relativo | Perdida de calidad |
|---|---|---|---|
| `Q8_0` | 8 | ~53% | Practicamente nula |
| `Q6_K` | ~6,6 | ~41% | Muy baja |
| `Q5_K_M` | ~5,7 | ~35% | Baja |
| `Q4_K_M` | ~4,8 | ~30% | Aceptable, **punto dulce habitual** |
| `Q4_K_S` | ~4,6 | ~28% | Aceptable |
| `Q3_K_M` | ~3,9 | ~24% | Perceptible |
| `Q2_K` | ~3,3 | ~20% | Severa |

Las variantes `_K` son **k-quants**: no aplican la misma precision a todos los tensores. Las capas de atencion y las matrices mas sensibles conservan mas bits que las capas feed-forward. El sufijo `_S`, `_M` o `_L` indica cuanta precision extra se asigna.

### Reglas practicas

**`Q4_K_M` es el punto de partida razonable** para la mayoria de casos. Ofrece la mejor relacion tamano/calidad.

**Los modelos pequenos toleran peor la cuantizacion.** Un modelo de 70B en `Q4_K_M` pierde muy poco. Un modelo de 2B en el mismo nivel puede degradarse de forma notable, especialmente en tareas de salida estructurada donde un error de sintaxis invalida toda la respuesta.

**Si la salida estructurada empieza a fallar, subir de nivel antes que cambiar de modelo.** Pasar de `Q4_K_M` a `Q5_K_M` o `Q8_0` cuesta memoria pero suele resolver JSON mal formado.

**Un modelo grande muy cuantizado suele superar a uno pequeno sin cuantizar,** a igualdad de memoria. Un 13B en `Q4_K_M` (~7,5 GB) rinde generalmente mejor que un 7B en fp16 (~14 GB) y ademas ocupa la mitad.

## 4.5 Evaluar la degradacion

La cuantizacion degrada en silencio. No hay error, solo respuestas peores. El unico metodo fiable es medir:

1. **Antes de cuantizar**, guardar las salidas del modelo sobre un conjunto de casos representativos.
2. **Despues de cuantizar**, ejecutar los mismos casos.
3. Comparar de forma objetiva: tasa de JSON valido, exactitud de clasificacion, coincidencia con la referencia.

Sin ese control, la evaluacion se convierte en impresiones subjetivas.

---

# 5. llama.cpp y el formato GGUF

## 5.1 Que es

**llama.cpp** es un motor de inferencia escrito en C/C++ sin dependencias pesadas. Su proposito original era ejecutar modelos Llama en un portatil, y ese objetivo determino todo su diseno.

Es la capa sobre la que se construyen Ollama, LM Studio, GPT4All, Jan y buena parte del ecosistema de inferencia local.

## 5.2 Componentes

### ggml

La libreria de tensores subyacente. Implementa las operaciones (multiplicacion de matrices, atencion, normalizacion) con kernels optimizados para multiples backends:

- CPU con AVX2, AVX-512, NEON
- CUDA (NVIDIA)
- Metal (Apple Silicon)
- ROCm (AMD)
- Vulkan, SYCL

Que un mismo binario funcione en un Mac M-series, un servidor con NVIDIA y una Raspberry Pi es consecuencia directa de este diseno multi-backend.

### El formato GGUF

GGUF (GGML Universal Format) sucede al formato GGML original. Es un **archivo unico** que contiene:

- Cabecera con version y numero de tensores
- **Metadatos** en pares clave-valor: arquitectura, hiperparametros, vocabulario, plantilla de chat, tokens especiales
- Los tensores cuantizados

Que los metadatos vayan dentro del archivo tiene una consecuencia practica importante: **el GGUF sabe cual es su propia plantilla de chat y sus tokens especiales**. Por eso llama.cpp inserta el BOS automaticamente, y por eso anadirlo manualmente produce duplicados.

### Carga por mmap

llama.cpp mapea el archivo en memoria en lugar de copiarlo. Esto permite:

- Arranque casi instantaneo tras la primera carga
- Compartir el modelo entre varios procesos
- Que el sistema operativo gestione la paginacion

Es tambien lo que hace viable el patron de Cloud Storage FUSE: el archivo puede residir en un almacenamiento remoto y leerse bajo demanda.

### Offload por capas

El parametro `n_gpu_layers` controla cuantas capas se colocan en VRAM y cuantas quedan en RAM del sistema:

```
n_gpu_layers = 0    -> todo en CPU
n_gpu_layers = 20   -> 20 capas en GPU, el resto en CPU
n_gpu_layers = 99   -> todo en GPU (si cabe)
```

Esto permite ejecutar un modelo de 13B en una GPU de 8 GB, a costa de velocidad. Las capas en CPU son mucho mas lentas, pero el modelo corre. Es la caracteristica que hace a llama.cpp unico frente a motores solo-GPU.

## 5.3 Fortalezas y limites

**Fortalezas:**
- Corre practicamente en cualquier hardware, incluido CPU puro
- Consumo de memoria minimo
- Sin dependencias de Python ni CUDA
- Ecosistema de modelos GGUF muy amplio
- Arranque rapido

**Limites:**
- **Batching limitado.** Esta optimizado para una o pocas peticiones concurrentes. No compite con vLLM en throughput multiusuario.
- Sin gestion avanzada de memoria del KV cache
- Sin soporte nativo de multiples adaptadores LoRA en caliente
- El escalado horizontal debe resolverse fuera del motor

## 5.4 Cuando usarlo

Es la eleccion correcta cuando:

- El despliegue es local, en edge o en hardware sin GPU
- La concurrencia es baja (1 a 4 peticiones simultaneas)
- La privacidad exige que los datos no salgan de la maquina
- El presupuesto de memoria es ajustado
- Se necesita prototipar rapido sin infraestructura

Es la eleccion incorrecta cuando el objetivo es servir decenas de peticiones concurrentes con alta utilizacion de GPU.

---

# 6. Ollama

## 6.1 Que anade sobre llama.cpp

llama.cpp es un motor; Ollama es un **servicio**. La diferencia esta en todo lo que rodea a la generacion de tokens:

| Capacidad | llama.cpp | Ollama |
|---|---|---|
| Motor de inferencia | Si | Si (usa llama.cpp) |
| Gestion de catalogo de modelos | No | Si |
| API REST | Basica | Completa, con protocolo OpenAI |
| Carga y descarga automatica | No | Si |
| Almacenamiento con deduplicacion | No | Si |
| Configuracion declarativa | Flags de CLI | `Modelfile` |
| Ejecucion como daemon | Manual | Nativa |

En terminos practicos, Ollama convierte "compilar y ejecutar un binario con quince flags" en "un servicio que expone una API y gestiona modelos por nombre".

## 6.2 El Modelfile

Es el manifiesto declarativo de un modelo. Su sintaxis esta inspirada en Dockerfile:

```dockerfile
FROM ./modelo.Q4_K_M.gguf

PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"

TEMPLATE """<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""
```

### Directivas

| Directiva | Funcion |
|---|---|
| `FROM` | Origen del modelo: archivo GGUF local o modelo del catalogo |
| `PARAMETER` | Parametros de generacion y contexto |
| `TEMPLATE` | Plantilla de turnos con marcadores Go (`{{ .Prompt }}`, `{{ .Response }}`, `{{ .System }}`) |
| `SYSTEM` | Prompt de sistema por defecto |
| `ADAPTER` | Adaptador LoRA a aplicar |
| `LICENSE` | Texto de licencia |

### El TEMPLATE es lo critico

Es donde se producen la mayoria de los fallos de calidad. Tres reglas:

**Debe reproducir exactamente el formato de entrenamiento.** Si el modelo se afino con una instruccion fija delante de cada entrada, esa instruccion pertenece al `TEMPLATE`, no al mensaje del usuario. Asi el cliente envia solo el dato variable y el formato se reproduce siempre.

**No incluir `<bos>`.** llama.cpp lo inserta segun los metadatos del GGUF.

**Los `stop` no son opcionales.** Sin ellos el modelo no se detiene al terminar su turno y sigue generando texto.

### Nota sobre plantillas anidadas

Al generar un `Modelfile` desde Python con f-strings, las llaves de Go entran en conflicto con las de Python. Hay que duplicarlas:

```python
f'''TEMPLATE """...{{{{ .Prompt }}}}..."""'''
# produce: TEMPLATE """...{{ .Prompt }}..."""
```

## 6.3 Almacenamiento: manifests y blobs

Al ejecutar `ollama create`, Ollama descompone el modelo en un almacen de contenido direccionable:

```
~/.ollama/models/
├── manifests/
│   └── registry.ollama.ai/library/mi-modelo/latest    (JSON pequeno)
└── blobs/
    ├── sha256-a3f9c2...    (pesos, GB)
    ├── sha256-7d1e88...    (plantilla, bytes)
    └── sha256-c04b1a...    (parametros, bytes)
```

**Blobs** contiene los datos, identificados por el hash de su contenido. **Manifests** contiene un JSON que describe cada modelo referenciando esos hashes.

La ventaja es la deduplicacion: registrar dos variantes del mismo modelo con parametros distintos crea dos manifests que apuntan al mismo blob de pesos. Los gigabytes se almacenan una sola vez.

Esto tiene una implicacion operativa directa: **para mover modelos entre maquinas hay que copiar ambos directorios**, no el GGUF original. Sin manifests, Ollama tiene los datos pero no sabe que exista ningun modelo registrado.

## 6.4 API

### Nativa

```
POST /api/generate     Completado simple
POST /api/chat         Conversacion con historial
GET  /api/tags         Modelos disponibles
POST /api/pull         Descargar del catalogo
GET  /api/version      Estado del servicio
```

`/api/tags` es la comprobacion de diagnostico mas util: si devuelve una lista vacia, el problema esta en el almacenamiento o el montaje, no en la inferencia.

### Compatible con OpenAI

```
POST /v1/chat/completions
POST /v1/completions
GET  /v1/models
```

Permite reutilizar SDK y bibliotecas existentes cambiando solo la URL base. Con una salvedad importante: **los parametros enviados en la peticion sobrescriben los del `Modelfile`**. Si el manifiesto define `temperature 0` y el cliente envia `0.7`, gana el cliente.

## 6.5 Variables de entorno relevantes

| Variable | Funcion |
|---|---|
| `OLLAMA_HOST` | Direccion y puerto de escucha (por defecto `127.0.0.1:11434`) |
| `OLLAMA_MODELS` | Directorio del almacen de modelos |
| `OLLAMA_KEEP_ALIVE` | Tiempo que un modelo permanece cargado en memoria |
| `OLLAMA_NUM_PARALLEL` | Peticiones paralelas por modelo |
| `OLLAMA_MAX_LOADED_MODELS` | Modelos simultaneos en memoria |
| `OLLAMA_ORIGINS` | Origenes CORS permitidos |

`OLLAMA_KEEP_ALIVE` merece atencion en despliegues serverless: si el modelo se descarga de memoria entre peticiones, cada llamada paga el coste de recarga.

## 6.6 Cuando usarlo

**Adecuado para:**
- Desarrollo y prototipado local
- Despliegues internos de baja a media concurrencia
- Servir modelos propios cuantizados
- Entornos con CPU o GPU modesta
- Casos donde importa mas la simplicidad operativa que el throughput maximo

**No adecuado para:**
- APIs publicas de alto volumen
- Cargas que exigen maximizar la utilizacion de GPU cara
- Servir muchos adaptadores LoRA distintos simultaneamente

---

# 7. vLLM

## 7.1 El problema que resuelve

Recordemos el hecho central de la seccion 2: en decode, el coste dominante es leer los pesos desde memoria, y esos pesos se leen una sola vez para todo el lote. Procesar 32 peticiones simultaneas cuesta casi lo mismo que procesar una.

Un servidor ingenuo desperdicia esa oportunidad de dos formas:

**Batching estatico.** Agrupa N peticiones, las procesa juntas y espera a que **todas** terminen antes de admitir nuevas. Si una peticion genera 500 tokens y las demas 20, el lote entero queda bloqueado hasta la mas larga. La GPU trabaja al vacio la mayor parte del tiempo.

**Reserva contigua del KV cache.** Se reserva memoria para la longitud maxima posible de cada peticion. Si el limite es 4.096 tokens y la peticion usa 200, se desperdicia el 95% de la reserva.

vLLM ataca ambos problemas.

## 7.2 PagedAttention

Es la contribucion tecnica central de vLLM, presentada en el articulo *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., SOSP 2023).

La idea toma prestado el concepto de **memoria virtual paginada** de los sistemas operativos.

En lugar de reservar un bloque contiguo de memoria para el KV cache de cada secuencia, vLLM lo divide en **bloques de tamano fijo** (tipicamente 16 tokens) que pueden estar dispersos en la memoria fisica. Una tabla de bloques por secuencia mapea las posiciones logicas a los bloques fisicos.

```
Sin paginacion:
  [====secuencia A: reservado 4096====] (usa 200) <- 95% desperdiciado
  [====secuencia B: reservado 4096====] (usa 150) <- 96% desperdiciado

Con paginacion:
  Bloques fisicos: [A0][B0][A1][B1][A2][libre][libre]...
  Tabla A: [0, 2, 4]
  Tabla B: [1, 3]
  Se asignan bloques conforme se necesitan
```

Los beneficios:

**Desperdicio de memoria reducido a la fraccion del ultimo bloque parcial**, en el orden del 4% frente al 60-80% habitual con reserva contigua.

**Mas peticiones concurrentes con la misma VRAM.** La memoria liberada se traduce directamente en mayor concurrencia posible.

**Comparticion de bloques (copy-on-write).** Varias secuencias que comparten prefijo pueden apuntar a los mismos bloques fisicos. Es especialmente valioso cuando todas las peticiones comparten un prompt de sistema largo o cuando se generan varias respuestas para la misma entrada.

## 7.3 Continuous batching

Tambien llamado *iteration-level scheduling*. En lugar de esperar a que todo el lote termine, vLLM toma decisiones **en cada paso de generacion**:

```
Paso N:    procesando [A, B, C, D]
Paso N+1:  B termina -> se retira, entra E -> [A, C, D, E]
Paso N+2:  procesando [A, C, D, E]
Paso N+3:  A termina -> entra F -> [C, D, E, F]
```

Ninguna peticion espera a las demas. Una que llega tarde no aguarda al siguiente lote: se incorpora en la siguiente iteracion. El resultado combinado con PagedAttention es un incremento sustancial de throughput frente a servidores de batching estatico, manteniendo latencias por peticion competitivas.

## 7.4 Otras capacidades

### Prefix caching automatico

Si varias peticiones comparten el mismo prefijo (un prompt de sistema, instrucciones fijas, ejemplos few-shot), vLLM reutiliza el KV cache ya calculado en lugar de recalcularlo.

El impacto en el TTFT es grande cuando el prefijo comun es largo. En un servicio donde todas las peticiones llevan 800 tokens de instrucciones fijas y 50 de contenido variable, el prefill se reduce practicamente a esos 50 tokens.

Es la razon tecnica por la que conviene colocar la parte fija del prompt **al principio** y la variable al final.

### Servir multiples adaptadores LoRA

vLLM puede cargar un modelo base y servir varios adaptadores LoRA simultaneamente, seleccionando cual aplicar por peticion:

```bash
vllm serve modelo-base \
  --enable-lora \
  --lora-modules clasificador=/ruta/lora-1 resumidor=/ruta/lora-2 \
  --max-lora-rank 16
```

Esto tiene una implicacion arquitectonica notable: en lugar de fusionar cada LoRA en una copia completa del modelo (5 GB por variante), se mantiene una sola copia del base y adaptadores de unos pocos MB.

Ademas evita un problema de calidad: fusionar un LoRA entrenado sobre una base cuantizada a 4 bits y guardarlo en 16 bits arrastra el error de cuantizacion. Servir el adaptador sobre la base original en precision completa lo evita.

### Decodificacion estructurada

vLLM soporta restringir la generacion a una gramatica o esquema JSON, garantizando que la salida sea sintacticamente valida. Para tareas de extraccion estructurada elimina toda una categoria de errores en el post-procesado.

### Tensor parallelism

Reparte el modelo entre varias GPU con `--tensor-parallel-size N`. Necesario cuando el modelo no cabe en una sola tarjeta.

### API compatible con OpenAI

vLLM expone `/v1/chat/completions` y `/v1/completions`, con lo que sustituye a la API de OpenAI cambiando solo la URL base.

## 7.5 Parametros de despliegue relevantes

| Parametro | Funcion | Consideracion |
|---|---|---|
| `--gpu-memory-utilization` | Fraccion de VRAM que vLLM reserva (por defecto ~0,9) | Bajarlo si otros procesos comparten la GPU |
| `--max-model-len` | Longitud maxima de contexto | Reducirlo libera memoria para mas concurrencia |
| `--max-num-seqs` | Secuencias concurrentes maximas | Techo del batching |
| `--quantization` | `awq`, `gptq`, `fp8` | Debe coincidir con el formato del checkpoint |
| `--enable-prefix-caching` | Activa el cache de prefijos | Muy rentable con prompts de sistema largos |
| `--tensor-parallel-size` | GPUs entre las que repartir el modelo | Solo si el modelo no cabe en una |

`--max-model-len` es la palanca mas util cuando falta memoria: reducir el contexto maximo de 32.768 a 4.096 libera una cantidad considerable de KV cache y permite mas peticiones simultaneas.

## 7.6 Fortalezas y limites

**Fortalezas:**
- Throughput muy superior en escenarios multiusuario
- Gestion de memoria eficiente
- Soporte nativo de LoRA multi-adaptador
- Salida estructurada garantizada
- API estandar de facto

**Limites:**
- **Requiere GPU NVIDIA.** No es una opcion para CPU ni para hardware heterogeneo.
- Consumo de memoria base alto: reserva VRAM de forma agresiva al arrancar
- Arranque mas lento que llama.cpp
- No lee GGUF de forma eficiente; su ecosistema es AWQ, GPTQ y FP8
- Mayor complejidad operativa

## 7.7 Cuando usarlo

**Adecuado para:**
- APIs con concurrencia real (decenas o cientos de peticiones)
- Cuando existe GPU dedicada y el objetivo es maximizar su utilizacion
- Servir varios adaptadores LoRA sobre un mismo base
- Cargas donde el costo por token es la metrica principal
- Casos que exigen salida estructurada garantizada

**No adecuado para:**
- Despliegue local o en edge
- Hardware sin GPU NVIDIA
- Concurrencia baja (donde su ventaja no se materializa y su sobrecoste no compensa)

---

# 8. Vertex AI Model Garden

## 8.1 Que es

**Vertex AI Model Garden** es el catalogo unificado de Google Cloud para descubrir, probar, personalizar y desplegar modelos. Reune modelos propios de Google, de socios y de codigo abierto bajo una misma superficie de SDK, IAM y despliegue.

Su valor no esta en el motor de inferencia (por debajo usa vLLM o TGI), sino en **eliminar la capa de infraestructura**: no hay que aprovisionar clusteres, instalar drivers de GPU, compilar contenedores ni configurar redes.

## 8.2 Categorias del catalogo

| Categoria | Descripcion |
|---|---|
| **Foundation models** | Modelos multitarea preentrenados, ajustables desde Vertex AI Studio, la API o el SDK. Familias Gemini, Imagen, Veo, Chirp. |
| **Fine-tunable models** | Modelos afinables mediante notebooks o pipelines. Familia Gemma y variantes. |
| **Task-specific solutions** | Modelos preconstruidos listos para usar, muchos personalizables. |

## 8.3 Las dos modalidades de consumo

Esta es la decision arquitectonica principal al usar Model Garden.

| Dimension | Model-as-a-Service (MaaS) | Dedicated Endpoint |
|---|---|---|
| **Infraestructura** | Ninguna: Google gestiona el cluster | Nodos dedicados en tu proyecto |
| **Costo** | Por token consumido. **Cero sin trafico** | Por hora de GPU asignada, **use o no use** |
| **Personalizacion** | Solo pesos del catalogo | Pesos propios, cuantizados o **adaptadores LoRA** |
| **Disponibilidad** | Inmediata | 10 a 25 minutos de aprovisionamiento |
| **Latencia** | Compartida, variable segun carga global | Determinista, con SLA |
| **Red** | Endpoint publico con IAM | Integrable en VPC Service Controls |
| **Escalado a cero** | Implicito | **No disponible con GPU** |

La ultima fila es la que mas dinero cuesta ignorar. Un Dedicated Endpoint con GPU factura de forma continua desde que se despliega hasta que se desmantela, con independencia del trafico.

## 8.4 Que hay debajo

Al desplegar un Dedicated Endpoint, Vertex AI selecciona automaticamente una imagen de inferencia optimizada, tipicamente basada en **vLLM** o **Text Generation Inference (TGI)**, mantenida por Google.

Esto tiene una consecuencia practica: **el formato del payload depende del contenedor**, y no hay un esquema universal.

```
vLLM (raw)     {"prompt": "...", "max_tokens": 200}
vLLM (OpenAI)  {"messages": [{"role": "user", "content": "..."}]}
TGI            {"inputs": "...", "parameters": {"max_new_tokens": 200}}
```

Un error HTTP 400 al probar un endpoint casi siempre significa que el esquema no coincide, no que el codigo del cliente este mal. El formato correcto aparece en la pestana **Sample request** de la ficha del modelo.

## 8.5 Dimensionamiento del hardware

| Tamano del modelo | Familia | Ejemplo | VRAM |
|---|---|---|---|
| 2B a 9B | G2 (NVIDIA L4) | `g2-standard-8` | 24 GB |
| 27B a 70B | A2 (NVIDIA A100) | `a2-highgpu-1g`, `a2-ultragpu-1g` | 40 / 80 GB |

Regla de dimensionamiento: los pesos son solo una parte. Hay que sumar el KV cache, que con concurrencia alta puede superar al modelo. Una estimacion conservadora es reservar entre el 30% y el 50% de la VRAM para KV cache y buffers.

## 8.6 El requisito que bloquea el laboratorio

**Los proyectos nuevos tienen cuota cero de GPU para Vertex AI.**

El asistente de despliegue permite seleccionar el hardware sin advertencia, y el error aparece minutos despues durante el aprovisionamiento. Verificar antes:

```bash
gcloud alpha services quota list \
  --service=aiplatform.googleapis.com \
  --consumer=projects/${PROJECT_ID} \
  --filter="custom_model_serving_nvidia_l4_gpus"
```

O en la consola: **IAM & Admin > Quotas**, filtrando por `aiplatform.googleapis.com`. La aprobacion de un aumento puede tardar de horas a dias segun la region.

## 8.7 Ciclo de vida de los modelos

Los modelos del catalogo tienen fechas de retirada y restricciones de disponibilidad. Un ejemplo concreto: desde el **29 de abril de 2025**, Gemini 1.5 Pro y Gemini 1.5 Flash dejaron de estar disponibles en proyectos sin uso previo, incluidos todos los proyectos nuevos.

Consecuencia para el diseno: **no fijar dependencias duras a una version concreta de modelo** sin revisar su calendario de ciclo de vida. Un servicio construido sobre un modelo proximo a retirarse hereda una fecha de caducidad.

## 8.8 Cuando usarlo

**MaaS es adecuado para:**
- Prototipado y pruebas de concepto
- Cargas esporadicas o impredecibles
- Casos donde no se necesitan pesos propios
- Equipos sin capacidad de operar infraestructura

**Dedicated Endpoint es adecuado para:**
- Produccion de alto volumen sostenido
- Requisitos de latencia deterministas y SLA
- Necesidad de servir pesos o adaptadores propios
- Aislamiento de red mediante VPC Service Controls

**Ninguno de los dos es adecuado para:**
- Despliegue on-premise o en edge
- Casos donde el dato no puede salir de la organizacion
- Presupuestos donde el costo por hora de GPU dedicada es prohibitivo para un trafico bajo

---

# 9. Comparativa y arbol de decision

## 9.1 Tabla comparativa

| Dimension | llama.cpp | Ollama | vLLM | Model Garden |
|---|---|---|---|---|
| **Hardware** | CPU, GPU, Apple Silicon, ARM | Igual que llama.cpp | GPU NVIDIA | Gestionado por GCP |
| **Formato** | GGUF | GGUF | AWQ, GPTQ, FP8, safetensors | Catalogo + custom |
| **Concurrencia** | Baja (1-4) | Baja-media | **Alta (decenas-cientos)** | Alta |
| **Throughput** | Bajo | Bajo-medio | **Muy alto** | Alto |
| **Latencia (1 peticion)** | **Buena** | **Buena** | Buena | Media (red) |
| **Memoria** | **Minima** | Minima | Alta (reserva agresiva) | N/A |
| **Complejidad operativa** | Media | **Baja** | Alta | **Muy baja** |
| **Arranque** | **Rapido** | Rapido | Lento | 10-25 min |
| **Multi-LoRA** | No | Limitado | **Si** | Segun modelo |
| **Salida estructurada** | Basica | Basica | **Garantizada** | Segun contenedor |
| **Costo sin trafico** | **Cero** | **Cero** | GPU asignada | **Cero (MaaS)** / GPU (Dedicated) |
| **Datos fuera de la org.** | No | No | No | **Si** |

## 9.2 Arbol de decision

```
1. Los datos pueden salir de tu infraestructura?
   NO -> llama.cpp / Ollama / vLLM autogestionado
   SI -> continuar

2. Necesitas servir pesos propios (fine-tuning, cuantizacion a medida)?
   NO -> Model Garden en modo MaaS
   SI -> continuar

3. Cuantas peticiones concurrentes en pico?
   < 5   -> Ollama
   5-50  -> vLLM en una GPU
   > 50  -> vLLM con tensor parallelism, o Dedicated Endpoint

4. El trafico es continuo o esporadico?
   Esporadico -> Ollama sobre serverless (Cloud Run) con escalado a cero
   Continuo   -> vLLM sobre GPU dedicada o Dedicated Endpoint

5. Tienes equipo para operar infraestructura?
   NO -> Model Garden
   SI -> vLLM autogestionado (menor costo por token)
```

## 9.3 Casos de uso tipicos

| Escenario | Recomendacion | Motivo |
|---|---|---|
| Prototipo en portatil | Ollama | Simplicidad, sin GPU necesaria |
| Asistente interno, 20 usuarios | Ollama sobre Cloud Run | Escalado a cero, costo minimo |
| API publica, 500 req/min | vLLM sobre L4/A100 | Throughput, costo por token |
| Clasificacion nocturna por lotes | vLLM con lote grande | Maximizar throughput, latencia irrelevante |
| Chat con 10 variantes especializadas | vLLM multi-LoRA | Una copia del base, adaptadores de MB |
| Prueba de concepto sin infraestructura | Model Garden MaaS | Cero despliegue, pago por uso |
| Modelo de frontera con SLA | Model Garden Dedicated | Latencia garantizada, sin ops |
| Dispositivo sin conexion | llama.cpp | Unico que corre en edge |

## 9.4 Sobre combinaciones

Estas herramientas no son excluyentes. Arquitecturas habituales:

- **Ollama en desarrollo, vLLM en produccion.** El mismo modelo, distinto motor segun la etapa. Requiere mantener dos formatos (GGUF y AWQ/GPTQ) y verificar que la plantilla de chat coincide en ambos.
- **Model Garden para el modelo generalista, vLLM propio para el especializado.** Se paga por token en las tareas de baja frecuencia y se amortiza GPU propia en las de alto volumen.
- **llama.cpp en edge, vLLM en el centro.** Inferencia local para lo sensible o urgente, escalado central para el resto.

---

# 10. Patrones de despliegue

## 10.1 Patron: local con Ollama

El mas simple. Modelo cuantizado a GGUF, manifiesto declarativo, servicio local.

```bash
ollama create mi-modelo -f Modelfile
ollama run mi-modelo "entrada de prueba"
```

**Aplicable a:** desarrollo, demos, uso personal, entornos aislados.

**Limite:** no escala mas alla de la maquina.

## 10.2 Patron: serverless con almacenamiento desacoplado

Ollama en un contenedor sobre Cloud Run, con los pesos en Cloud Storage montado via FUSE.

```
[Cliente] -> [Cloud Run: contenedor Ollama ~1 GB]
                    |
                    v (volume mount)
             [GCS bucket: modelos]
```

La clave arquitectonica es **desacoplar los pesos de la imagen**:

| Dimension | Pesos en la imagen | Pesos en GCS + FUSE |
|---|---|---|
| Tamano de imagen | 4-16 GB | ~500 MB - 1 GB |
| Tiempo de build | 10-20 min | < 1 min |
| Actualizar el modelo | Reconstruir y redesplegar | Subir un archivo |
| Arranque | Descarga de capa completa | Lectura por streaming |

**Aplicable a:** cargas esporadicas o impredecibles, donde el escalado a cero ahorra mas de lo que cuesta el cold start.

**Limite:** el arranque en frio incluye la lectura del modelo desde el almacenamiento remoto. La primera peticion tras un periodo de inactividad puede tardar minutos.

**Mitigacion:** `--min-instances=1` elimina el cold start a cambio de facturacion continua. Es un intercambio explicito entre latencia y costo.

## 10.3 Patron: GPU dedicada con vLLM

```bash
vllm serve modelo-base \
  --enable-lora \
  --lora-modules tarea1=/ruta/lora-1 tarea2=/ruta/lora-2 \
  --max-lora-rank 16 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90
```

**Aplicable a:** produccion con trafico sostenido y concurrencia real.

**Ventaja del enfoque multi-LoRA:** una sola copia del modelo base en precision completa, mas adaptadores de pocos MB. Evita mantener N copias fusionadas de 5 GB y evita el error de cuantizacion que introduce fusionar un LoRA entrenado sobre una base de 4 bits.

**Limite:** la GPU factura mientras exista, haya trafico o no.

## 10.4 Patron: gestionado con Model Garden

Sin infraestructura propia. Despliegue por consola o SDK, consumo por API.

**Aplicable a:** equipos sin capacidad de operar GPU, o casos donde el time-to-market pesa mas que el costo por token.

**Limite:** menor control sobre el artefacto y sobre la ubicacion del dato.

## 10.5 Del fine-tuning al servicio

Un flujo completo, desde un modelo afinado con LoRA hasta produccion:

```
[Base cuantizada 4 bits] + [LoRA entrenado]
            |
            +-- Ruta GGUF (Ollama / llama.cpp)
            |     1. Fusionar LoRA -> pesos 16 bits
            |     2. Convertir a GGUF
            |     3. Cuantizar (Q4_K_M)
            |     4. Escribir Modelfile con la plantilla de entrenamiento
            |     5. ollama create
            |
            +-- Ruta vLLM (recomendada si hay GPU)
                  Opcion A: fusionar y servir en 16 bits
                  Opcion B: servir el adaptador sobre la base original
                            (mejor calidad, artefacto de MB)
```

**Nota de calidad sobre la ruta de fusion.** Si el fine-tuning se hizo con QLoRA sobre una base cuantizada a 4 bits, fusionar y guardar en 16 bits **no recupera la precision perdida**: arrastra el error de cuantizacion de la base. Servir el adaptador por separado sobre los pesos originales lo evita por completo.

**Verificacion obligatoria en ambas rutas.** Guardar las salidas del modelo antes de exportar y compararlas despues. La fusion y la cuantizacion degradan en silencio.

---

# 11. Operacion: benchmarking y costos

## 11.1 Como medir

Un benchmark util cumple cuatro condiciones:

**Carga representativa.** Las longitudes de prompt y de respuesta de tu caso real, no las del ejemplo de la documentacion. Un servicio con prompts de 2.000 tokens y respuestas de 50 tiene un perfil completamente distinto a uno con 50 y 500.

**Concurrencia realista.** Medir con una peticion a la vez no dice nada sobre el comportamiento con 50 simultaneas. Las curvas no son lineales.

**Fase de calentamiento.** Descartar las primeras peticiones: incluyen carga del modelo, compilacion de kernels y llenado de caches.

**Percentiles, no medias.** El p50 oculta el problema. El p95 y el p99 son lo que percibe el usuario descontento.

## 11.2 Que registrar

| Metrica | Por que |
|---|---|
| TTFT (p50, p95, p99) | Percepcion de respuesta |
| TPOT (p50, p95) | Fluidez de la generacion |
| Throughput agregado | Capacidad del sistema |
| Peticiones en cola | Indicador de saturacion |
| Utilizacion de GPU | Deteccion de recurso ocioso |
| Uso de VRAM | Margen antes de OOM |
| Tasa de error | Salud del servicio |

## 11.3 Modelo de costos

### Autogestionado (vLLM sobre GPU)

```
costo_por_millon_tokens = (costo_hora / (tokens_seg x 3600)) x 1.000.000
```

Ejemplo ilustrativo: una GPU a 1 USD/hora que sostiene 500 tokens/s agregados produce 1.800.000 tokens por hora, lo que da unos 0,56 USD por millon de tokens.

La conclusion clave: **el costo por token es inversamente proporcional a la utilizacion**. La misma GPU al 20% de utilizacion cuesta cinco veces mas por token.

### Serverless (Ollama sobre Cloud Run)

```
costo = (vCPU-segundos x precio_vcpu) + (GiB-segundos x precio_mem) + peticiones
```

Con escalado a cero, el costo sin trafico es cero. El punto de equilibrio frente a una instancia dedicada depende del ciclo de trabajo: por debajo de aproximadamente un 20-30% de utilizacion sostenida, serverless suele salir mas barato.

### Pago por token (MaaS)

Sin costo fijo. Predecible por peticion. Se vuelve mas caro que la infraestructura propia a partir de cierto volumen; ese punto de cruce debe calcularse con los precios vigentes y el trafico esperado.

## 11.4 Palancas de optimizacion

En orden de impacto habitual:

**1. Cuantizacion.** De 16 a 4 bits: aproximadamente 4x menos memoria y 2-3x mas velocidad. Es la palanca de mayor efecto.

**2. Reducir el contexto maximo.** `--max-model-len` de 32.768 a 4.096 libera KV cache proporcionalmente y multiplica la concurrencia posible.

**3. Prefix caching.** Con prompts de sistema largos y compartidos, reduce el TTFT de forma sustancial sin coste.

**4. Aumentar el tamano de lote.** Mejora el throughput a costa del TPOT. Ajustar hasta el limite que tolere el objetivo de latencia.

**5. Modelo mas pequeno.** Frecuentemente un modelo de 2B afinado para una tarea concreta iguala o supera a uno de 7B generalista, a una fraccion del costo. Es la optimizacion mas infravalorada.

**6. Speculative decoding.** Un modelo pequeno propone varios tokens y el grande los verifica en una sola pasada. Puede acelerar el decode, pero anade complejidad significativa.

## 11.5 Dimensionamiento de memoria

```
VRAM_necesaria = pesos + KV_cache + activaciones + overhead

pesos     = n_parametros x bytes_por_parametro
KV_cache  = 2 x capas x cabezas_kv x dim_cabeza x contexto x concurrencia x bytes
overhead  = ~1-2 GB (buffers, fragmentacion, runtime)
```

Ejemplo con un modelo de 7B en 4 bits, contexto 4.096, 16 peticiones concurrentes:

```
pesos:     ~3,5 GB
KV cache:  ~2 GB por secuencia x 16 = 32 GB   <- dominante
overhead:  ~1,5 GB
Total:     ~37 GB  -> no cabe en una L4 de 24 GB
```

Con contexto reducido a 2.048 y 8 peticiones concurrentes, el KV cache baja a unos 8 GB y el total a unos 13 GB, que si cabe. Este calculo debe hacerse **antes** de elegir el hardware, no despues del primer OOM.

---

# 12. Errores frecuentes

Los siguientes fallos comparten una caracteristica: **no producen error**. Producen resultados peores, lo que los hace dificiles de diagnosticar.

## 12.1 Desajuste de plantilla de chat

**Sintoma:** el modelo funciona pero responde peor de lo esperado. Ignora instrucciones, anade preambulos, no respeta el formato.

**Causa:** el formato de inferencia no coincide con el de entrenamiento. Tipicamente: una instruccion fija presente durante el fine-tuning y ausente en produccion, o un rol de sistema que el modelo no conoce.

**Diagnostico:** imprimir el `repr()` del prompt completo tal como llega al modelo y compararlo caracter a caracter con un ejemplo del dataset de entrenamiento.

**Prevencion:** colocar la parte fija del prompt en la plantilla del servidor (`TEMPLATE` en Ollama), no en el cliente.

## 12.2 Doble BOS

**Sintoma:** salida degenerada, tokens de relleno o caracteres sin sentido.

**Causa:** la plantilla de chat inserta `<bos>` y el tokenizador anade otro automaticamente.

**Diagnostico:** contar los tokens BOS en el texto tokenizado. Debe haber exactamente uno.

**Prevencion:** al tokenizar un texto que ya contiene la plantilla, usar `add_special_tokens=False`. En un `Modelfile`, no incluir `<bos>` manualmente.

## 12.3 Token de parada ausente

**Sintoma:** el modelo genera la respuesta correcta y despues continua indefinidamente, inventando turnos nuevos.

**Causa:** falta la configuracion de `stop`, o el token de parada del modelo no coincide con el configurado.

**Prevencion:** declarar todos los tokens de fin de turno de la familia. Para Gemma, tanto `<end_of_turn>` como `<start_of_turn>`.

## 12.4 Parametros de generacion sobrescritos

**Sintoma:** el modelo se comporta de forma distinta segun el cliente que lo invoca.

**Causa:** los parametros enviados en la peticion HTTP tienen prioridad sobre los del manifiesto del servidor. Un `Modelfile` con `temperature 0` no impide que un cliente envie `0.7`.

**Prevencion:** documentar los parametros esperados y validarlos en el cliente. Para tareas deterministas, enviar `temperature: 0` explicitamente.

## 12.5 Cuantizacion excesiva en modelos pequenos

**Sintoma:** JSON mal formado, claves inventadas, sintaxis rota. Con frecuencia intermitente.

**Causa:** un modelo de 2B en `Q4_K_M` o inferior pierde la precision necesaria para mantener estructura sintactica estricta.

**Prevencion:** en tareas de salida estructurada con modelos pequenos, empezar en `Q5_K_M` o `Q8_0`. Medir la tasa de JSON valido antes y despues de cuantizar.

## 12.6 Conflictos de version de dependencias

**Sintoma:** el modelo carga sin errores, ocupa la memoria esperada, genera texto. Y el texto es basura.

**Causa:** incompatibilidad entre la version de la libreria de carga y el formato del checkpoint. Algunas capas no encuentran sus pesos y se inicializan al azar, en silencio.

**Diagnostico:** medir la perdida del modelo recien cargado sobre texto trivial. Un modelo sano da entre 2 y 4. Un valor por encima de `ln(vocabulario)` (unos 12,4 para un vocabulario de 256.000) indica que el modelo esta **peor que adivinando al azar**, lo que solo ocurre si los pesos estan corruptos.

**Prevencion:** fijar versiones de dependencias. Ejecutar una prueba de salud tras cargar y antes de cualquier otra cosa.

## 12.7 Artefactos no verificados tras la exportacion

**Sintoma:** el modelo funcionaba en el entorno de entrenamiento y responde peor en produccion.

**Causa:** fusion de LoRA y cuantizacion degradan sin avisar.

**Prevencion:** guardar las salidas de un conjunto de casos representativos antes de exportar. Ejecutar los mismos casos sobre el artefacto final. Comparar de forma objetiva.

## 12.8 Recursos en la nube sin desmantelar

**Sintoma:** factura inesperada.

**Causa:** un endpoint con GPU factura de forma continua desde el despliegue hasta el desmantelamiento, con independencia del trafico. Es facil olvidar uno en una region que no se revisa.

**Prevencion:** procedimiento de teardown documentado, ejecutado al terminar. Verificar en **todas** las regiones utilizadas. Tener en cuenta que los datos de facturacion tardan hasta 24 horas en reflejarse.

---

# 13. Referencias

## Articulos fundacionales

- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. https://arxiv.org/abs/2309.06180
- Yu, G. et al. (2022). *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022. (Origen del continuous batching)
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. https://arxiv.org/abs/2305.14314
- Frantar, E. et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. https://arxiv.org/abs/2210.17323
- Lin, J. et al. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. https://arxiv.org/abs/2306.00978
- Ainslie, J. et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. https://arxiv.org/abs/2305.13245
- Leviathan, Y. et al. (2023). *Fast Inference from Transformers via Speculative Decoding*. https://arxiv.org/abs/2211.17192

## llama.cpp y GGUF

- Repositorio de llama.cpp: https://github.com/ggerganov/llama.cpp
- Especificacion del formato GGUF: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- GGUF en Hugging Face Hub: https://huggingface.co/docs/hub/gguf

## Ollama

- Sitio oficial: https://ollama.com/
- Referencia de la API: https://github.com/ollama/ollama/blob/main/docs/api.md
- Referencia del Modelfile: https://github.com/ollama/ollama/blob/main/docs/modelfile.md
- Compatibilidad con OpenAI: https://github.com/ollama/ollama/blob/main/docs/openai.md

## vLLM

- Documentacion oficial: https://docs.vllm.ai/
- Repositorio: https://github.com/vllm-project/vllm
- Guia de LoRA multi-adaptador: https://docs.vllm.ai/en/latest/models/lora.html
- Servidor compatible con OpenAI: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

## Google Cloud

- Model Garden (portal): https://cloud.google.com/model-garden
- Explorar modelos: https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models
- Versiones y ciclo de vida de modelos: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
- Cuotas de Vertex AI: https://cloud.google.com/vertex-ai/docs/quotas
- Cloud Run: volume mounts con Cloud Storage: https://cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts
- Cloud Run con GPU: https://cloud.google.com/run/docs/configuring/services/gpu
- SDK de Python para Vertex AI: https://cloud.google.com/python/docs/reference/aiplatform/latest

## Fine-tuning

- Unsloth: https://github.com/unslothai/unsloth
- PEFT (Hugging Face): https://huggingface.co/docs/peft
- TRL (Transformer Reinforcement Learning): https://huggingface.co/docs/trl

---

*Los enlaces y las cifras de rendimiento reflejan el estado del ecosistema en el momento de redaccion. Este campo evoluciona con rapidez: verificar la documentacion oficial antes de tomar decisiones de arquitectura.*