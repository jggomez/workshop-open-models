#!/usr/bin/env python3
"""
Cliente de invocacion para modelos desplegados desde Vertex AI Model Garden (GCP).
Demuestra la invocacion programatica de endpoints dedicados (e.g. DeepSeek, Gemma, Llama)
utilizando el SDK oficial de Google Cloud y llamadas REST autenticadas.
"""

import sys
import os
import time
import json

def query_vertex_endpoint(project_id: str, location: str, endpoint_id: str, prompt_text: str):
    """
    Invoca un endpoint dedicado de Vertex AI utilizando google-cloud-aiplatform.
    """
    print("======================================================")
    print("  Invocacion de Endpoint Dedicado en Vertex AI")
    print(f"  Proyecto:    {project_id}")
    print(f"  Region:      {location}")
    print(f"  Endpoint ID: {endpoint_id}")
    print("======================================================")

    try:
        from google.cloud import aiplatform
    except ImportError:
        print("\nError: La biblioteca 'google-cloud-aiplatform' no esta instalada.")
        print("Instalela ejecutando: pip install google-cloud-aiplatform\n")
        return False

    aiplatform.init(project=project_id, location=location)
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)

    print(f"\n[1] Preparando payload de inferencia para el modelo...")
    # Formato estandar para contenedores vLLM / TGI preconstruidos en Vertex AI
    instances = [
        {
            "prompt": prompt_text,
            "max_tokens": 200,
            "temperature": 0.2,
            "top_p": 0.95
        }
    ]

    print(f"    Prompt: {prompt_text}")
    print(f"\n[2] Enviando solicitud de prediccion a Vertex AI...")
    
    t_start = time.time()
    try:
        response = endpoint.predict(instances=instances)
        elapsed = time.time() - t_start
        print(f"    Latencia de respuesta: {elapsed:.2f} s")
        print("\n[3] Respuesta recibida del modelo:")
        print("------------------------------------------------------")
        for pred in response.predictions:
            if isinstance(pred, dict):
                print(pred.get("generated_text", json.dumps(pred, indent=2)))
            else:
                print(pred)
        print("------------------------------------------------------")
        return True
    except Exception as e:
        print(f"\nError durante la prediccion en Vertex AI: {e}")
        print("Verifique que sus credenciales esten activas (gcloud auth application-default login)")
        return False

def query_vertex_rest_api(project_id: str, location: str, endpoint_id: str, prompt_text: str):
    """
    Ejemplo de invocacion mediante REST nativo utilizando un token de acceso OAuth2.
    """
    import urllib.request
    import subprocess

    print("\n[Opcion REST Alternativa]")
    try:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        print("No se pudo obtener el access token mediante gcloud CLI.")
        return

    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "instances": [{"prompt": prompt_text, "max_tokens": 150}]
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("Respuesta REST exitosa:")
            print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Aviso en llamada REST: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso:")
        print("  python3 test_vertex_model_garden.py <PROJECT_ID> <LOCATION> <ENDPOINT_ID> [PROMPT]")
        print("\nEjemplo:")
        print("  python3 test_vertex_model_garden.py mi-proyecto-gcp us-central1 1234567890123456789")
        sys.exit(1)

    proj = sys.argv[1]
    loc = sys.argv[2]
    ep_id = sys.argv[3]
    prompt = sys.argv[4] if len(sys.argv) > 4 else "Explica que ventajas ofrece Vertex AI Model Garden frente a desplegar infraestructura propia."

    query_vertex_endpoint(proj, loc, ep_id, prompt)
