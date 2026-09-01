import os
import ssl
import urllib.request
from PIL import Image, ImageDraw

def download_or_generate_samples():
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
    os.makedirs(target_dir, exist_ok=True)
    
    samples = {
        "dog.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Golden_Retriever_2019.jpg/500px-Golden_Retriever_2019.jpg",
        "cat.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/500px-Cat_November_2010-1a.jpg",
        "car.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg/500px-2019_Toyota_Corolla_Icon_Tech_VVT-i_Hybrid_1.8.jpg",
        "grace_hopper.jpg": "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
    }

    headers = {"User-Agent": "Mozilla/5.0"}
    ssl_context = ssl._create_unverified_context()

    for filename, url in samples.items():
        file_path = os.path.join(target_dir, filename)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10, context=ssl_context) as response, open(file_path, "wb") as out_file:
                out_file.write(response.read())
            print(f"Downloaded sample image: {filename}")
        except Exception as e:
            print(f"Could not download {filename} ({e}). Generating synthetic fallback image.")
            img = Image.new("RGB", (224, 224), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            d.rectangle([(20, 20), (204, 204)], outline="white", fill=(100, 150, 200))
            d.text((30, 100), f"Sample: {filename}", fill="white")
            img.save(file_path)
            print(f"Saved fallback synthetic image: {filename}")

if __name__ == "__main__":
    download_or_generate_samples()
