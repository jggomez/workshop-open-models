"""
Utility script to download the official Google AI Edge LiteRT Vision model, labels, and sample images.
Ref: https://developers.google.com/edge/litert/web/get_started
"""

import json
import os
import ssl
import urllib.request
from PIL import Image

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite"
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"

SAMPLE_URLS = {
    "dog.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Golden_Retriever_2019.jpg/500px-Golden_Retriever_2019.jpg",
    "cat.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/500px-Cat_November_2010-1a.jpg",
    "car.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg/500px-2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg",
    "grace_hopper.jpg": "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
}

def setup_model_assets():
    models_dir = os.path.dirname(os.path.abspath(__file__))
    lab_dir = os.path.dirname(models_dir)
    samples_dir = os.path.join(lab_dir, "sample_images")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "efficientnet_lite0.tflite")
    labels_path = os.path.join(models_dir, "imagenet_classes.json")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    ssl_context = ssl._create_unverified_context()
    
    # 1. Download official Google AI Edge LiteRT Model
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        print(f"Downloading Google AI Edge LiteRT model from: {MODEL_URL}...")
        try:
            req = urllib.request.Request(MODEL_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=20, context=ssl_context) as resp, open(model_path, "wb") as f:
                f.write(resp.read())
            print(f"Model saved to: {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")
        except Exception as e:
            print(f"Error downloading model: {e}")
    else:
        print(f"Model already exists at: {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")

    # 2. Download ImageNet Labels (clean 1000 classes for EfficientNet/ImageNet)
    print("Saving ImageNet labels (1000 classes)...")
    try:
        req = urllib.request.Request(LABELS_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=ssl_context) as resp:
            raw_labels = resp.read().decode("utf-8").strip().splitlines()
        labels = [line.strip().replace("_", " ").title() for line in raw_labels if line.strip()]
        # Strip background label if present so it matches 1000 output logits exactly
        if len(labels) == 1001:
            labels = labels[1:]
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        print(f"Labels saved to: {labels_path} ({len(labels)} classes)")
    except Exception as e:
        print(f"Warning downloading labels ({e}). Generating fallback ImageNet labels...")
        fallback_labels = [f"ImageNet Class {i}" for i in range(1000)]
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(fallback_labels, f, indent=2)

    # 3. Download Sample Images
    for fname, url in SAMPLE_URLS.items():
        fpath = os.path.join(samples_dir, fname)
        if not os.path.exists(fpath):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15, context=ssl_context) as resp, open(fpath, "wb") as f:
                    f.write(resp.read())
                print(f"Downloaded sample: {fname}")
            except Exception as e:
                print(f"Fallback generated for: {fname} ({e})")
                img = Image.new("RGB", (224, 224), color=(70, 90, 120))
                img.save(fpath)

if __name__ == "__main__":
    setup_model_assets()
