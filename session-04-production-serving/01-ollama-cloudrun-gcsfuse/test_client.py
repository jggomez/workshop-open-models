#!/usr/bin/env python3
"""
Cliente de prueba para inferencia con Ollama (Local y Google Cloud Run).
Demuestra invocaciones mediante la API nativa de Ollama y el protocolo compatible con OpenAI.
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error

DEFAULT_HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "techcloud-classifier")

def check_health(base_url: str):
    url = f"{base_url.rstrip('/')}/api/version"
    print(f"[1] Verificando salud y conectividad con {url}...")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            print(f"    Respuesta: Ollama Version {data.get('version', 'unknown')}")
            return True
    except Exception as e:
        print(f"    Error de conexion: {e}")
        return False

def list_models(base_url: str):
    url = f"{base_url.rstrip('/')}/api/tags"
    print(f"\n[2] Consultando catalogo de modelos disponibles en {url}...")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = data.get("models", [])
            print(f"    Modelos registrados ({len(models)}):")
            for m in models:
                size_mb = m.get("size", 0) / (1024 * 1024)
                print(f"    - {m.get('name')} ({size_mb:.1f} MB)")
            return [m.get("name") for m in models]
    except Exception as e:
        print(f"    Error listando modelos: {e}")
        return []

def query_native_ollama_api(base_url: str, model: str, prompt_text: str):
    url = f"{base_url.rstrip('/')}/api/generate"
    print(f"\n[3] Inferencia con API Nativa de Ollama ({url})...")
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.95
        }
    }
    
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - start_time
            response_text = result.get("response", "").strip()
            total_eval = result.get("eval_count", 0)
            eval_duration_ns = result.get("eval_duration", 1)
            tps = (total_eval / (eval_duration_ns / 1e9)) if eval_duration_ns else 0
            
            print(f"    Latencia total: {elapsed:.2f} s")
            print(f"    Throughput:     {tps:.2f} tokens/s ({total_eval} tokens)")
            print(f"    Respuesta generada:\n{response_text}")
            return response_text
    except Exception as e:
        print(f"    Fallo en la inferencia nativa: {e}")
        return None

def query_openai_compatible_api(base_url: str, model: str, user_message: str):
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    print(f"\n[4] Inferencia con Protocolo Compatible OpenAI ({url})...")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Responde en formato JSON estricto."},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.1,
        "stream": False
    }
    
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - start_time
            choices = result.get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""
            
            print(f"    Latencia total: {elapsed:.2f} s")
            print(f"    Respuesta estructurada (OpenAI format):\n{content.strip()}")
            return content
    except Exception as e:
        print(f"    Fallo en protocolo OpenAI: {e}")
        return None

if __name__ == "__main__":
    target_host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    target_model = sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME
    
    print("======================================================")
    print("  Cliente de Inferencia Ollama (Local / Cloud Run)")
    print(f"  Servidor objetivo: {target_host}")
    print(f"  Modelo objetivo:   {target_model}")
    print("======================================================")
    
    if check_health(target_host):
        available_models = list_models(target_host)
        
        # Prueba de clasificacion con ticket de soporte (tarea entrenada en Sesion 3)
        sample_ticket = (
            "Analiza el siguiente ticket de soporte y extrae la informacion en formato JSON "
            "con las claves categoria, urgencia, sentimiento y accion_sugerida:\n"
            "Alerta critica: La base de datos Spanner rechaza conexiones por agotamiento de sesiones. "
            "Nuestros microservicios de cobro estan fallando en el 80% de los carritos de compra."
        )
        
        # Si el modelo personalizado no esta cargado pero hay otros, usar el primero disponible
        active_model = target_model
        if active_model not in available_models and available_models:
            print(f"\nAviso: {target_model} no esta en el servidor. Usando modelo disponible: {available_models[0]}")
            active_model = available_models[0]
            
        query_native_ollama_api(target_host, active_model, sample_ticket)
        query_openai_compatible_api(target_host, active_model, sample_ticket)
    else:
        print("\nPara ejecutar localmente:")
        print("  1. Inicie Ollama: ollama serve")
        print("  2. Cree el modelo: ollama create techcloud-classifier -f ../session-03-fine-tuning-llms/03-fast-finetuning-unsloth-gguf/Modelfile")
        print("  3. Ejecute este script: python3 test_client.py http://localhost:11434 techcloud-classifier")
