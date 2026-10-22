import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

GRID_PATH = "badapple_grid/badapple_grids.npy"
PATHS_TXT = "image_paths.txt"
OUT_DIR = Path("rendered_frames_sample")
OUT_DIR.mkdir(exist_ok=True)

TILE_SIZE = 32
SECONDS = 10
FPS = 10

grids = np.load(GRID_PATH)
grids = grids[:SECONDS * FPS]

with open(PATHS_TXT, "r", encoding="utf-8") as f:
    cat_paths = [line.strip() for line in f]

# 고양이별 평균 밝기 계산
cat_brightness = []

for p in tqdm(cat_paths, desc="Cat brightness"):
    img = Image.open(p).convert("L").resize((TILE_SIZE, TILE_SIZE))
    cat_brightness.append(np.array(img).mean() / 255.0)

cat_brightness = np.array(cat_brightness)

for fi, grid in enumerate(tqdm(grids, desc="Rendering sample")):
    h, w = grid.shape
    canvas = Image.new("RGB", (w * TILE_SIZE, h * TILE_SIZE))

    for y in range(h):
        for x in range(w):
            target = grid[y, x]
            idx = np.argmin(np.abs(cat_brightness - target))

            cat = Image.open(cat_paths[idx]).convert("RGB").resize((TILE_SIZE, TILE_SIZE))
            canvas.paste(cat, (x * TILE_SIZE, y * TILE_SIZE))

    canvas.save(OUT_DIR / f"frame_{fi:05d}.jpg", quality=90)

print("saved frames:", OUT_DIR)
