#!/usr/bin/env python3
"""
Cliente de invocacion para modelos desplegados desde Vertex AI Model Garden.

Prueba el endpoint por dos vias: el SDK de Python y REST autenticado. Si el SDK
falla y REST funciona, el problema esta en las credenciales ADC y no en el
endpoint; esa comparacion es el objetivo de ejecutar ambas.

Uso:
    python3 test_vertex_model_garden.py <PROJECT_ID> <LOCATION> <ENDPOINT_ID> [PROMPT]

Codigo de salida 0 si alguna via obtiene respuesta, 1 si ninguna.
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

TIMEOUT = 300  # un endpoint recien desplegado tarda en aceptar la primera peticion

# ---------------------------------------------------------------------------
# FORMATO DEL PAYLOAD
#
# No existe un formato unico. Cada contenedor de inferencia espera un esquema
# distinto, y este es el motivo mas frecuente de un error 400:
#
#   vLLM (raw)        {"prompt": "...", "max_tokens": 200}
#   vLLM (OpenAI)     {"messages": [{"role": "user", "content": "..."}]}
#   TGI               {"inputs": "...", "parameters": {"max_new_tokens": 200}}
#
# Ademas, los modelos ajustados a chat requieren su plantilla de turnos
# (<start_of_turn>, <|im_start|>, [INST], etc.) dentro del texto.
#
# El esquema correcto aparece en la pestana "Sample request" de la ficha del
# modelo en Model Garden. Ajustar PAYLOAD_BASE segun lo que indique.
# ---------------------------------------------------------------------------

def construir_instancia(prompt_text, max_tokens=200):
    return {
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.95,
    }


def extraer_texto(pred):
    """El formato de salida varia igual que el de entrada. Se prueban las
    claves conocidas antes de caer al volcado completo."""
    if not isinstance(pred, dict):
        return str(pred)
    for clave in ("generated_text", "text", "output", "content", "prediction"):
        if clave in pred:
            return str(pred[clave])
    if "choices" in pred and pred["choices"]:
        c = pred["choices"][0]
        if isinstance(c, dict):
            return str(c.get("text") or c.get("message", {}).get("content", c))
    return json.dumps(pred, indent=2, ensure_ascii=False)


def diagnosticar(error):
    """Un 404, un 403 y un 400 requieren acciones distintas."""
    msg = str(error)
    if "404" in msg or "not found" in msg.lower():
        return ("El endpoint no existe en esta region o el ID es incorrecto.\n"
                "  gcloud ai endpoints list --region=REGION")
    if "403" in msg or "permission" in msg.lower():
        return ("Permisos insuficientes. La cuenta necesita el rol\n"
                "  roles/aiplatform.user sobre el proyecto.")
    if "400" in msg or "invalid" in msg.lower():
        return ("Payload mal formado: el esquema no coincide con el que espera\n"
                "  el contenedor. Ver 'Sample request' en la ficha del modelo\n"
                "  y ajustar construir_instancia().")
    if "503" in msg or "unavailable" in msg.lower():
        return ("El endpoint no esta listo. El aprovisionamiento tarda de 10 a\n"
                "  25 minutos tras el despliegue.")
    if "credential" in msg.lower() or "default" in msg.lower():
        return "Credenciales no configuradas:\n  gcloud auth application-default login"
    return "Revisar el mensaje completo arriba."


def query_sdk(project_id, location, endpoint_id, prompt_text):
    print("\n[1] Invocacion con el SDK de Python")
    try:
        from google.cloud import aiplatform
    except ImportError:
        print("    La biblioteca 'google-cloud-aiplatform' no esta instalada.")
        print("    pip install google-cloud-aiplatform")
        return False

    try:
        aiplatform.init(project=project_id, location=location)
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
    except Exception as e:
        print(f"    [FALLO] No se pudo abrir el endpoint: {e}")
        print(f"    {diagnosticar(e)}")
        return False

    print(f"    Prompt: {prompt_text[:70]}...")
    inicio = time.time()
    try:
        respuesta = endpoint.predict(
            instances=[construir_instancia(prompt_text)],
            timeout=TIMEOUT,
        )
    except Exception as e:
        print(f"    [FALLO] {e}")
        print(f"    {diagnosticar(e)}")
        return False

    print(f"    Latencia: {time.time() - inicio:.2f} s")
    print("    ------------------------------------------------------")
    for pred in respuesta.predictions:
        print(f"    {extraer_texto(pred)}")
    print("    ------------------------------------------------------")
    return True


def query_rest(project_id, location, endpoint_id, prompt_text):
    print("\n[2] Invocacion REST con token OAuth2")

    try:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            stderr=subprocess.PIPE,
        ).decode("utf-8").strip()
    except FileNotFoundError:
        print("    'gcloud' no esta en el PATH. Instalar Google Cloud CLI.")
        return False
    except subprocess.CalledProcessError as e:
        # No se silencia stderr: el motivo real del fallo esta ahi.
        print(f"    No se pudo obtener el token: {e.stderr.decode('utf-8').strip()}")
        print("    gcloud auth login")
        return False

    url = (f"https://{location}-aiplatform.googleapis.com/v1/projects/"
           f"{project_id}/locations/{location}/endpoints/{endpoint_id}:predict")

    req = urllib.request.Request(
        url,
        data=json.dumps({"instances": [construir_instancia(prompt_text)]}).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
    )

    inicio = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            datos = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        cuerpo = e.read().decode("utf-8", errors="replace")
        print(f"    [FALLO] HTTP {e.code}")
        print(f"    {cuerpo[:500]}")
        print(f"    {diagnosticar(e.code)}")
        return False
    except Exception as e:
        print(f"    [FALLO] {e}")
        print(f"    {diagnosticar(e)}")
        return False

    print(f"    Latencia: {time.time() - inicio:.2f} s")
    print("    ------------------------------------------------------")
    for pred in datos.get("predictions", []):
        print(f"    {extraer_texto(pred)}")
    print("    ------------------------------------------------------")
    return True


def main():
    if len(sys.argv) < 4:
        print("Uso:")
        print("  python3 test_vertex_model_garden.py <PROJECT_ID> <LOCATION> <ENDPOINT_ID> [PROMPT]")
        print("\nEjemplo:")
        print("  python3 test_vertex_model_garden.py mi-proyecto us-central1 1234567890123456789")
        print("\nObtener el ENDPOINT_ID:")
        print("  gcloud ai endpoints list --region=us-central1")
        return 1

    proyecto, region, endpoint_id = sys.argv[1], sys.argv[2], sys.argv[3]
    prompt = (sys.argv[4] if len(sys.argv) > 4 else
              "Explica brevemente en que consiste la cuantizacion de modelos de lenguaje.")

    print("======================================================")
    print("  Invocacion de Endpoint Dedicado en Vertex AI")
    print(f"  Proyecto:    {proyecto}")
    print(f"  Region:      {region}")
    print(f"  Endpoint ID: {endpoint_id}")
    print("======================================================")

    ok_sdk = query_sdk(proyecto, region, endpoint_id, prompt)
    ok_rest = query_rest(proyecto, region, endpoint_id, prompt)

    print("\n======================================================")
    print(f"  SDK de Python: {'OK' if ok_sdk else 'FALLO'}")
    print(f"  REST OAuth2:   {'OK' if ok_rest else 'FALLO'}")
    if ok_rest and not ok_sdk:
        print("\n  REST funciona y el SDK no: el endpoint esta bien.")
        print("  Revisar las credenciales ADC del SDK:")
        print("    gcloud auth application-default login")
    print("======================================================")
    return 0 if (ok_sdk or ok_rest) else 1


if __name__ == "__main__":
    sys.exit(main())