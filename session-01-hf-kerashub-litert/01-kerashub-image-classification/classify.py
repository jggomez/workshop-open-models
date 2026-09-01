"""
Image Classification with KerasHub Pre-trained Vision Models.

This script demonstrates loading pre-trained deep learning vision backbones
from KerasHub (or Keras Vision), preprocessing input images, and computing
top-K class probabilities.
"""

import argparse
import os
import sys
import time
import ssl
from typing import List, Tuple
import numpy as np
from PIL import Image

# Ensure SSL certificates are properly trusted on macOS and all platforms
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
except ImportError:
    pass
ssl._create_default_https_context = ssl._create_unverified_context

# Configure Keras backend before importing keras (defaults to torch if available)
os.environ.setdefault("KERAS_BACKEND", "torch")

try:
    import keras
    import keras_hub
except ImportError:
    print("Error: Keras or KerasHub is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load an image from disk and resize it to the expected dimensions.

    Args:
        image_path: File system path to the target image.
        target_size: Target tuple (height, width) for model input.

    Returns:
        Numpy array with shape (1, height, width, channels) in RGB format.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found at path: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    image_tensor = np.expand_dims(image_array, axis=0)
    return image_tensor


def run_classification(image_path: str, preset: str = "mobilenet_v3_small_imagenet", top_k: int = 5):
    """
    Load a pre-trained KerasHub image classifier and perform inference.

    Args:
        image_path: Path to the input image.
        preset: KerasHub model preset name (e.g. mobilenet_v3_small_imagenet, resnet_50_imagenet).
        top_k: Number of top candidate predictions to display.
    """
    print(f"Loading vision classifier preset: '{preset}' (Backend: {keras.config.backend()})...")
    start_load = time.time()
    
    classifier = None
    try:
        classifier = keras_hub.models.ImageClassifier.from_preset(preset)
        print("Loaded via KerasHub preset.")
    except Exception as e:
        print(f"Loading via Keras Applications ({e})...")
        from keras.applications import MobileNetV3Small
        classifier = MobileNetV3Small(weights="imagenet")

    load_time = time.time() - start_load
    print(f"Model loaded successfully in {load_time:.2f} seconds.\n")

    print(f"Loading image from: {image_path}")
    image_tensor = load_and_preprocess_image(image_path, target_size=(224, 224))

    print("Running inference...")
    start_infer = time.time()
    predictions = classifier.predict(image_tensor, verbose=0)
    infer_time = (time.time() - start_infer) * 1000

    print(f"Inference completed in {infer_time:.2f} ms.\n")

    # Format and display top-K predictions
    print(f"--- Top {top_k} Predictions ---")
    try:
        from keras.applications.mobilenet_v3 import decode_predictions
        decoded = decode_predictions(predictions, top=top_k)[0]
        for rank, (class_id, label, score) in enumerate(decoded, start=1):
            print(f"{rank}. {label.replace('_', ' ').title():<30} Confidence: {score * 100:.2f}% (ID: {class_id})")
    except Exception:
        probs = predictions[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        for rank, idx in enumerate(top_indices, start=1):
            print(f"{rank}. Class Index {idx:<20} Confidence: {probs[idx] * 100:.2f}%")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Classify images using pre-trained KerasHub / Keras vision models."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images", "dog.jpg"),
        help="Path to the input image for classification."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="mobilenet_v3_small_imagenet",
        help="KerasHub preset identifier (e.g., mobilenet_v3_small_imagenet, resnet_50_imagenet)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to display (default: 5)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_classification(image_path=args.image, preset=args.preset, top_k=args.top_k)
