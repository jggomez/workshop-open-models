# Hands-on Lab 3: Google AI Edge LiteRT-LM Web API para LLMs On-Device

Este laboratorio ensena a utilizar la arquitectura **Google AI Edge LiteRT-LM Web API** ([https://developers.google.com/edge/litert-lm/js](https://developers.google.com/edge/litert-lm/js)) para ejecutar Small Language Models (SLMs) y LLMs directamente en el navegador del cliente mediante aceleracion por hardware **WebGPU** y ejecucion neuronal real sin servidores.

---

## 1. Fundamentos Tecnologicos y Arquitectura

La ejecucion de modelos generativos de lenguaje en el navegador (*On-Device Browser LLMs*) se basa en cuatro pilares:
- **Aceleracion por Hardware (WebGPU):** Multiplicacion paralela masiva de matrices de atencion y capas densas directamente en los shaders de la GPU local.
- **Cuantizacion a 4 Bits (INT4/Q4):** Compresion de los tensores de pesos para reducir el consumo de memoria de varios gigabytes a rangos de ~80 MB a ~1.4 GB.
- **Almacenamiento Persistente en IndexedDB:** Descarga de pesos unica; las sesiones posteriores cargan el modelo instantaneamente desde el almacenamiento local sin requerir conexion a internet.
- **Streaming Asincrono:** Generacion token por token mediante generadores asincronos (`for await...of`) y decodificacion incremental con `TextStreamer`.

---

## 2. Modelos Soportados en la Aplicacion

La aplicacion incluye un selector dinamico con tres perfiles de modelos:

| Modelo | Identificador | Tamano Cuantizado | Caso de Uso |
|---|---|---|---|
| **Qwen 2.5 0.5B Instruct** | `onnx-community/Qwen2.5-0.5B-Instruct` | ~250 MB | Modelo rapido predeterminado para pruebas agiles en navegador. |
| **SmolLM2 135M Instruct** | `HuggingFaceTB/SmolLM2-135M-Instruct` | ~80 MB | Modelo ultra-ligero para dispositivos con recursos limitados. |
| **Google Gemma 2 2B IT** | `onnx-community/gemma-2-2b-it` | ~1.4 GB | Modelo de alta capacidad de la familia Google Gemma. |

---

## 3. Analisis del Ciclo de Vida del Motor (`app.js`)

### Paso 1: Inicializacion del Motor y Descarga (`LiteRtLmEngine.create`)
```javascript
const engine = await LiteRtLmEngine.create({
  model: "onnx-community/Qwen2.5-0.5B-Instruct"
});
```
- Descarga los tensores cuantizados y el vocabulario (`tokenizer.json`).
- Reporta el progreso porcentual en la interfaz de usuario en tiempo real.
- Compila los grafos de atencion y los buffers de memoria KV-Cache en la GPU.

### Paso 2: Creacion de la Sesion Conversacional (`createConversation`)
```javascript
const conversation = await engine.createConversation({
  preface: {
    messages: [
      { role: "system", content: "Eres un asistente de IA util y preciso." }
    ]
  }
});
```

### Paso 3: Inferencia con Streaming Asincrono (`sendMessageStreaming`)
```javascript
const stream = conversation.sendMessageStreaming(prompt);

for await (const chunk of stream) {
  // Cada token generado por la red neuronal se proyecta en la interfaz
  console.log(chunk.content[0].text);
}
```

---

## 4. Estructura del Directorio

```text
03-litert-lm-cli-and-web/
├── README.md                    # Esta guia paso a paso y documentacion tecnica
├── index.html                   # Interfaz de chat con selector de modelos y metricas
├── app.js                       # Controlador con motor neural on-device (WebGPU)
├── styles.css                   # Hoja de estilos en modo oscuro responsivo
└── package.json                 # Configuracion de dependencias y scripts de ejecucion
```

---

## 5. Guia de Ejecucion

### Opcion 1: Con NPM (Recomendado)
Desde la raiz del repositorio:
```bash
npm run lab3
```
Luego abra en su navegador: **[http://localhost:3001](http://localhost:3001)**.

### Opcion 2: Con Servidor HTTP de Python
```bash
python3 -m http.server 3001 --directory session-01-hf-kerashub-litert/03-litert-lm-cli-and-web
```
Luego abra en su navegador: **[http://localhost:3001](http://localhost:3001)**.

---

## 6. Paso Final: Limpieza del Entorno

Para eliminar archivos temporales y dependencias locales:

```bash
# Limpiar dependencias locales si se instalaron con npm
rm -rf session-01-hf-kerashub-litert/03-litert-lm-cli-and-web/node_modules
```

---

## 7. Referencias y Documentacion Oficial

- **Google AI Edge LiteRT-LM Web API Reference:** [https://developers.google.com/edge/litert-lm/js](https://developers.google.com/edge/litert-lm/js)
- **Portal Oficial Google AI Edge LiteRT:** [https://ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)
- **Transformers.js (WebGPU Runtime):** [https://huggingface.co/docs/transformers.js](https://huggingface.co/docs/transformers.js)
- **Google Gemma Open Models:** [https://ai.google.dev/gemma](https://ai.google.dev/gemma)
- **Especificacion WebGPU (W3C):** [https://www.w3.org/TR/webgpu/](https://www.w3.org/TR/webgpu/)
