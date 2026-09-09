#!/usr/bin/env python3
"""
Cliente de prueba para inferencia con Ollama (Local y Google Cloud Run).
Demuestra invocaciones mediante la API nativa de Ollama y el protocolo
compatible con OpenAI.

Uso:
    python3 test_client.py [URL] [MODELO]
    python3 test_client.py http://localhost:11434 techcloud-classifier
    python3 test_client.py https://ollama-service-xxx.run.app techcloud-classifier

Codigo de salida 0 si todas las pruebas pasan, 1 en caso contrario.
"""

import json
import os
import sys
import time
import urllib.request

DEFAULT_HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "techcloud-classifier")

# El cold start de Cloud Run con FUSE incluye la lectura del GGUF (~1.6 GB)
# por streaming desde GCS. La primera peticion puede tardar varios minutos.
TIMEOUT_INFERENCIA = 300

# IMPORTANTE: solo el texto del ticket.
# La instruccion ("Analiza el siguiente ticket...") la inserta el TEMPLATE del
# Modelfile. Incluirla tambien aqui la duplica y produce un formato que el
# modelo no vio durante el fine-tuning.
TICKET_PRUEBA = (
    "Alerta critica: La base de datos Spanner rechaza conexiones por agotamiento "
    "de sesiones. Nuestros microservicios de cobro estan fallando en el 80% de "
    "los carritos de compra."
)


def _post(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _validar_json(texto):
    """La tarea entrenada produce JSON. Verificarlo es el objetivo de la prueba."""
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError as e:
        print(f"    [FALLO] La respuesta no es JSON valido: {e}")
        return False

    esperadas = {"categoria", "urgencia", "sentimiento", "accion_sugerida"}
    faltan = esperadas - set(datos.keys())
    if faltan:
        print(f"    [AVISO] Faltan claves: {sorted(faltan)}")
    else:
        print("    [OK] JSON valido con las 4 claves esperadas.")
    return True


def check_health(base_url):
    url = f"{base_url.rstrip('/')}/api/version"
    print(f"[1] Verificando salud y conectividad con {url}...")
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        print(f"    Ollama version {data.get('version', 'desconocida')}")
        return True
    except Exception as e:
        print(f"    Error de conexion: {e}")
        return False


def list_models(base_url):
    url = f"{base_url.rstrip('/')}/api/tags"
    print(f"\n[2] Consultando catalogo de modelos en {url}...")
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"    Error listando modelos: {e}")
        return []

    modelos = data.get("models", [])
    print(f"    Modelos registrados ({len(modelos)}):")
    for m in modelos:
        print(f"    - {m.get('name')} ({m.get('size', 0) / (1024 ** 2):.1f} MB)")

    if not modelos:
        print("\n    [FALLO] El servidor no tiene ningun modelo registrado.")
        print("    En Cloud Run esto indica que el volumen FUSE no se monto o")
        print("    que faltan los manifests en el bucket. Verificar:")
        print("      gcloud storage ls gs://BUCKET/models/manifests/")

    return [m.get("name", "") for m in modelos]


def resolver_modelo(solicitado, disponibles):
    """Ollama devuelve los nombres con tag ('modelo:latest'). No hay fallback a
    otro modelo a proposito: probar uno distinto daria metricas de aspecto
    normal para un despliegue roto."""
    if solicitado in disponibles:
        return solicitado
    for nombre in disponibles:
        if nombre.split(":")[0] == solicitado.split(":")[0]:
            print(f"    Resuelto '{solicitado}' -> '{nombre}'")
            return nombre
    print(f"\n[FALLO] El modelo '{solicitado}' no esta en el servidor.")
    print(f"        Disponibles: {disponibles or 'ninguno'}")
    return None


def query_native_ollama_api(base_url, model, prompt_text):
    url = f"{base_url.rstrip('/')}/api/generate"
    print(f"\n[3] Inferencia con API nativa de Ollama ({url})...")
    print("    (la primera peticion tras un cold start puede tardar minutos)")

    # Sin bloque 'options': el Modelfile define temperature 0. Sobrescribirlo
    # aqui probaria una configuracion distinta a la validada.
    payload = {"model": model, "prompt": prompt_text, "stream": False}

    inicio = time.time()
    try:
        res = _post(url, payload, TIMEOUT_INFERENCIA)
    except Exception as e:
        print(f"    [FALLO] Error en la inferencia nativa: {e}")
        return False

    texto = res.get("response", "").strip()
    tokens = res.get("eval_count", 0)
    dur_ns = res.get("eval_duration", 0)
    tps = tokens / (dur_ns / 1e9) if dur_ns else 0

    print(f"    Latencia total: {time.time() - inicio:.2f} s")
    print(f"    Throughput:     {tps:.2f} tokens/s ({tokens} tokens)")
    print(f"    Respuesta:\n{texto}")
    return _validar_json(texto)


def query_openai_compatible_api(base_url, model, user_message):
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    print(f"\n[4] Inferencia con protocolo compatible OpenAI ({url})...")

    # Sin mensaje 'system': Gemma-2 no define rol de sistema y el modelo nunca
    # lo vio en entrenamiento. temperature 0 para mantener el determinismo.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": 0,
        "stream": False,
    }

    inicio = time.time()
    try:
        res = _post(url, payload, TIMEOUT_INFERENCIA)
    except Exception as e:
        print(f"    [FALLO] Error en protocolo OpenAI: {e}")
        return False

    choices = res.get("choices", [])
    if not choices:
        print(f"    [FALLO] Respuesta sin 'choices': {res}")
        return False

    texto = choices[0]["message"]["content"].strip()
    print(f"    Latencia total: {time.time() - inicio:.2f} s")
    print(f"    Respuesta:\n{texto}")
    return _validar_json(texto)


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    modelo = sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME

    print("======================================================")
    print("  Cliente de Inferencia Ollama (Local / Cloud Run)")
    print(f"  Servidor: {host}")
    print(f"  Modelo:   {modelo}")
    print("======================================================")

    if not check_health(host):
        print("\nPara ejecutar localmente:")
        print("  1. Inicie Ollama:   ollama serve")
        print("  2. Cree el modelo:  ollama create techcloud-classifier -f Modelfile")
        print("  3. Ejecute:         python3 test_client.py http://localhost:11434")
        return 1

    disponibles = list_models(host)
    activo = resolver_modelo(modelo, disponibles)
    if activo is None:
        return 1

    ok_nativa = query_native_ollama_api(host, activo, TICKET_PRUEBA)
    ok_openai = query_openai_compatible_api(host, activo, TICKET_PRUEBA)

    print("\n======================================================")
    print(f"  API nativa:       {'OK' if ok_nativa else 'FALLO'}")
    print(f"  Protocolo OpenAI: {'OK' if ok_openai else 'FALLO'}")
    print("======================================================")
    return 0 if (ok_nativa and ok_openai) else 1


if __name__ == "__main__":
    sys.exit(main())