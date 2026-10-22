from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

IMAGE_DIR = Path("/home/darek123/python_projects/BadAppleCats/Cats256px_gray_3channels/")
OUT_EMB = "embeddings.npy"
OUT_PATHS = "image_paths.txt"

BATCH_SIZE = 64
MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

image_paths = sorted([
    p for p in IMAGE_DIR.rglob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
])

print("images:", len(image_paths))

all_embeddings = []

with torch.no_grad():

    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(
        range(0, len(image_paths), BATCH_SIZE),
        total=total_batches,
        desc="CLIP Embedding",
        unit="batch"
    ):
        batch_paths = image_paths[i:i + BATCH_SIZE]

        images = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print("skip:", p, e)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        # old version. 
        # feats = model.get_image_features(**inputs)
        # feats = feats / feats.norm(dim=-1, keepdim=True)

        # new version.
        outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        feats = outputs.pooler_output
        feats = model.visual_projection(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    
        all_embeddings.append(feats.cpu().numpy())

embeddings = np.concatenate(all_embeddings, axis=0).astype("float32")

np.save(OUT_EMB, embeddings)

with open(OUT_PATHS, "w", encoding="utf-8") as f:
    for p in image_paths:
        f.write(str(p) + "\n")

print("saved:", OUT_EMB, embeddings.shape)
print("saved:", OUT_PATHS)
