#!/usr/bin/env python3
"""
Workshop Environment Cleanup Script.

Removes temporary caches, downloaded model binaries, sample assets,
and optionally virtual environments and Hugging Face/KerasHub local caches.
"""

import argparse
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def remove_path(path: str, description: str):
    if os.path.exists(path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"[REMOVED] {description}: {path}")
        except Exception as e:
            print(f"[ERROR] Could not remove {path}: {e}")
    else:
        print(f"[SKIPPED] {description} not found: {path}")


def clean_temporary_files():
    print("\n--- Cleaning Python and OS temporary files ---")
    for root, dirs, files in os.walk(BASE_DIR):
        for d in list(dirs):
            if d in ["__pycache__", ".ipynb_checkpoints"]:
                full_path = os.path.join(root, d)
                shutil.rmtree(full_path, ignore_errors=True)
                print(f"[CLEANED] Directory: {full_path}")
                dirs.remove(d)
        for f in files:
            if f.endswith((".pyc", ".pyo", ".DS_Store")):
                full_path = os.path.join(root, f)
                try:
                    os.remove(full_path)
                    print(f"[CLEANED] File: {full_path}")
                except Exception:
                    pass


def clean_lab_assets():
    print("\n--- Cleaning downloaded lab assets and model binaries ---")
    assets = [
        (os.path.join(BASE_DIR, "session-01-hf-kerashub-litert", "01-kerashub-image-classification", "sample_images"), "Lab 1 Sample Images"),
        (os.path.join(BASE_DIR, "session-01-hf-kerashub-litert", "02-litert-web-vision", "models", "mobilenet_quant.tflite"), "Lab 2 Model Binary"),
        (os.path.join(BASE_DIR, "session-01-hf-kerashub-litert", "02-litert-web-vision", "models", "imagenet_classes.json"), "Lab 2 Class Labels"),
        (os.path.join(BASE_DIR, "session-01-hf-kerashub-litert", "02-litert-web-vision", "models", "temp_model.tgz"), "Lab 2 Temp Archive")
    ]
    for path, desc in assets:
        remove_path(path, desc)


def clean_deep_caches():
    print("\n--- Cleaning Hugging Face and KerasHub caches ---")
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    keras_cache = os.path.expanduser("~/.keras")
    remove_path(hf_cache, "Hugging Face Cache (~/.cache/huggingface)")
    remove_path(keras_cache, "Keras Cache (~/.keras)")


def clean_virtualenv():
    print("\n--- Cleaning virtual environments ---")
    venv_path = os.path.join(BASE_DIR, ".venv")
    remove_path(venv_path, "Python Virtualenv (.venv)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean workshop temporary files, downloaded weights, and virtual environments."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean everything including lab assets, virtualenv, and deep caches."
    )
    parser.add_argument(
        "--assets-only",
        action="store_true",
        help="Only remove downloaded models and images (preserves virtualenv)."
    )
    parser.add_argument(
        "--include-cache",
        action="store_true",
        help="Also purge global ~/.cache/huggingface and ~/.keras directories."
    )
    parser.add_argument(
        "--include-venv",
        action="store_true",
        help="Also remove local .venv directory."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("Workshop Workspace Cleaner")
    print("=" * 60)

    clean_temporary_files()
    clean_lab_assets()

    if args.all or args.include_cache:
        clean_deep_caches()

    if args.all or args.include_venv:
        clean_virtualenv()

    print("\nCleanup completed successfully.")
