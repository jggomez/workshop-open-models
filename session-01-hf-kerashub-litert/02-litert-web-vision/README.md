# Hands-on Lab 2: Inferencia On-Device en el Navegador y Python con LiteRT.js

Este laboratorio ensena a compilar, inspeccionar y ejecutar modelos de vision por computador en formato **LiteRT** (`.tflite`) tanto en Python (`ai-edge-litert`) como directamente en el navegador web del cliente mediante JavaScript y WebAssembly usando la biblioteca oficial **LiteRT.js** (`@litertjs/core`) de Google AI Edge ([Get started with LiteRT.js](https://developers.google.com/edge/litert/web/get_started)).

---

## 1. Fundamentos Tecnologicos y Biblioteca Utilizada

### Que es LiteRT.js y de donde proviene la biblioteca?
**LiteRT.js** (publicado en npm bajo el paquete oficial **`@litertjs/core`** por el equipo de Google AI Edge) es el motor oficial de ejecucion de modelos de Deep Learning en el navegador, sucesor directo de TensorFlow Lite Web.

- **Paquete oficial en npm:** [`@litertjs/core`](https://www.npmjs.com/package/@litertjs/core)
- **Paquete oficial en Python:** [`ai-edge-litert`](https://pypi.org/project/ai-edge-litert/)
- **Guia oficial de Google LiteRT Web (Get Started):** [https://developers.google.com/edge/litert/web/get_started](https://developers.google.com/edge/litert/web/get_started)
- **Repositorio oficial en GitHub:** [https://github.com/google-ai-edge/litert](https://github.com/google-ai-edge/litert)

---

## 2. Analisis Detallado del Codigo de Inferencia en LiteRT.js (`app.js`)

La implementacion en `app.js` se compone de 5 bloques tecnicos esenciales que definen el ciclo de vida completo de un modelo de Deep Learning en el navegador:

### Bloque 1: Importacion de Simbolos del SDK
```javascript
import { loadLiteRt, loadAndCompile, Tensor } from "@litertjs/core";
```
- **`loadLiteRt`**: Funcion de arranque que descarga y enlaza los binarios WebAssembly (`.wasm`) de LiteRT.
- **`loadAndCompile`**: Funcion asincrona que obtiene el binario `.tflite` y compila su grafo de ejecucion.
- **`Tensor`**: Clase que envuelve los buffers de memoria en el espacio lineal de WebAssembly o buffers de WebGPU.

### Bloque 2: Inicializacion del Runtime WebAssembly (`loadLiteRt`)
```javascript
// Descarga e inicializa el runtime WebAssembly en segundo plano
await loadLiteRt("https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/");
```
- Antes de procesar modelos, LiteRT.js requiere instanciar el entorno C++ compilado a WebAssembly.
- Al apuntar a la ruta `/wasm/`, el runtime selecciona de forma transparente la compilacion mas eficiente segun la CPU del cliente (`threaded`, `simd` o `jspi`).

### Bloque 3: Carga y Compilacion del Grafo `.tflite` (`loadAndCompile`)
```javascript
compiledModel = await loadAndCompile("models/efficientnet_lite0.tflite", {
  accelerator: "wasm" // Opciones: 'wasm', 'webgpu', 'webnn'
});
```
- **Compilacion local:** LiteRT.js lee la estructura FlatBuffer del archivo `.tflite`, planifica la asignacion de tensores intermedios y configura los kernels optimizados de *XNNPACK* (para CPU) o shaders (para WebGPU).
- El objeto resultante `compiledModel` queda residente en memoria para ejecutar inferencias sucesivas con latencia minima.

### Bloque 4: Preprocesamiento y Creacion de Tensores (`new Tensor`)
```javascript
function preprocessImageToTensor(imgElement) {
  // 1. Redimensionar la imagen a 224x224 pixeles en un canvas HTML5
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0, 224, 224);

  // 2. Extraer los canales RGBA y normalizar los valores RGB a float32 [-1.0, 1.0]
  const imgData = ctx.getImageData(0, 0, 224, 224).data;
  const floatArr = new Float32Array(1 * 224 * 224 * 3);

  let idx = 0;
  for (let i = 0; i < imgData.length; i += 4) {
    floatArr[idx++] = (imgData[i] / 127.5) - 1.0;     // Canal Rojo
    floatArr[idx++] = (imgData[i + 1] / 127.5) - 1.0; // Canal Verde
    floatArr[idx++] = (imgData[i + 2] / 127.5) - 1.0; // Canal Azul
  }

  // 3. Instanciar el Tensor LiteRT con forma [Batch=1, Height=224, Width=224, Channels=3]
  return new Tensor(floatArr, [1, 224, 224, 3]);
}
```
- Transforma los pixeles del DOM en un tensor estructurado con tipo de dato `Float32Array` compatible con la firma de entrada del modelo.

### Bloque 5: Inferencia, Decodificacion y Liberacion de Memoria (`run` y `delete`)
```javascript
// 1. Crear tensor de entrada y ejecutar forward-pass
const inputTensor = preprocessImageToTensor(imagePreview);
const outputTensors = await compiledModel.run(inputTensor);

// 2. Extraer probabilidades de salida
const outputTensor = Array.isArray(outputTensors) ? outputTensors[0] : outputTensors;
const outputData = await outputTensor.data(); // Float32Array con 1,000 probabilidades

// 3. Liberar explicitamente la memoria lineal de WebAssembly
inputTensor.delete();
outputTensor.delete();
```
- **`compiledModel.run(inputTensor)`**: Ejecuta las capas convolucionales y de atencion directamente en el cliente.
- **`tensor.delete()`**: Al residir en la memoria de WebAssembly/C++, estos buffers no son recolectados por el Garbage Collector de JavaScript. Invocando `.delete()` se previene cualquier fuga de memoria tras clasificaciones continuas.

---

## 3. Estructura del Laboratorio

```text
02-litert-web-vision/
├── README.md                    # Esta guia paso a paso con explicacion tecnica
├── index.html                   # Interfaz web con Import Map para ES Modules
├── app.js                       # Controlador de inferencia con LiteRT.js (@litertjs/core)
├── styles.css                   # Hoja de estilos de la aplicacion
├── package.json                 # Dependencias npm (@litertjs/core)
├── 01_litert_interpreter_inspection.ipynb # Cuaderno de inspeccion de tensores con Python
└── models/                      # Binarios del modelo y metadatos
    ├── efficientnet_lite0.tflite # Modelo oficial Google AI Edge LiteRT (17.7 MB)
    ├── imagenet_classes.json    # Diccionario de 1,000 clases ImageNet
    └── download_model.py        # Script auxiliar de descarga
```

---

## 4. Guia Paso a Paso de Ejecucion

### Parte A: Inferencia en Python con `ai-edge-litert`

#### Paso 1: Instalacion de la biblioteca en el entorno virtual
```bash
source .venv/bin/activate
pip install ai-edge-litert pillow numpy requests
```

#### Paso 2: Descargar el modelo oficial LiteRT y las etiquetas
```bash
python3 session-01-hf-kerashub-litert/02-litert-web-vision/models/download_model.py
```

#### Paso 3: Inspeccionar tensores y ejecutar inferencia en Python
```bash
source .venv/bin/activate
python3 -c '
from ai_edge_litert.interpreter import Interpreter
import numpy as np
from PIL import Image
import json

interpreter = Interpreter(model_path="session-01-hf-kerashub-litert/02-litert-web-vision/models/efficientnet_lite0.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open("session-01-hf-kerashub-litert/01-kerashub-image-classification/sample_images/dog.jpg").convert("RGB").resize((224, 224))
img_arr = (np.array(img, dtype=np.float32) / 127.5) - 1.0
input_data = np.expand_dims(img_arr, axis=0)

interpreter.set_tensor(input_details[0]["index"], input_data)
interpreter.invoke()
out_data = interpreter.get_tensor(output_details[0]["index"])[0]

top5 = np.argsort(out_data)[::-1][:5]
with open("session-01-hf-kerashub-litert/02-litert-web-vision/models/imagenet_classes.json") as f:
    labels = json.load(f)

print("Top 5 Predicciones:")
for r, idx in enumerate(top5, 1):
    label_name = labels[idx] if idx < len(labels) else f"Clase {idx}"
    print(f"  {r}. {label_name:<30} {out_data[idx] * 100:.2f}%")
'
```

---

### Parte B: Inferencia en el Navegador con LiteRT.js (@litertjs/core)

#### Paso 1: Iniciar el servidor local
Con npm desde la raiz del repositorio:
```bash
npm run lab2
```
O con el servidor HTTP de Python:
```bash
python3 -m http.server 3000 --directory session-01-hf-kerashub-litert/02-litert-web-vision
```

#### Paso 2: Probar la aplicacion web
1. Abra su navegador en `http://localhost:3000`.
2. Observe el estado de inicializacion: `LiteRT.js Listo para Inferencia`.
3. Seleccione una imagen de muestra (*Perro*, *Gato*, *Auto*, *Grace Hopper*) o suba cualquier imagen desde su equipo.
4. Presione **Ejecutar Inferencia LiteRT.js On-Device**.
5. Analice la latencia y las probabilidades calculadas en el cliente directamente con `@litertjs/core` sobre el archivo `.tflite`.

---

## 5. Paso Final: Limpieza del Entorno

Para eliminar el modelo descargado y los archivos generados:

```bash
# Eliminar binarios locales de modelos
rm -f session-01-hf-kerashub-litert/02-litert-web-vision/models/efficientnet_lite0.tflite
rm -f session-01-hf-kerashub-litert/02-litert-web-vision/models/imagenet_classes.json
```

---

## 6. Referencias y Documentacion Oficial

- **Guia Oficial de Inicio de LiteRT.js (Google AI Edge):** [https://developers.google.com/edge/litert/web/get_started](https://developers.google.com/edge/litert/web/get_started)
- **Paquete LiteRT.js en npm (`@litertjs/core`):** [https://www.npmjs.com/package/@litertjs/core](https://www.npmjs.com/package/@litertjs/core)
- **Portal Oficial Google AI Edge LiteRT:** [https://ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)
- **Documentacion de la API de Python (`ai-edge-litert`):** [https://pypi.org/project/ai-edge-litert/](https://pypi.org/project/ai-edge-litert/)
- **Repositorio de Codigo Fuente de LiteRT:** [https://github.com/google-ai-edge/litert](https://github.com/google-ai-edge/litert)
