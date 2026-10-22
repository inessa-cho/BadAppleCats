import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

VIDEO_PATH = "MediaOriginal/bad_apple.mp4"
INDEX_PATH = "cats.index"
PATHS_TXT = "image_paths.txt"

OUT_DIR = Path("rendered_frames_final")
OUT_DIR.mkdir(exist_ok=True)

GRID_W = 48
GRID_H = 36
TILE_SIZE = 32
TARGET_FPS = 10

START_SECONDS = 0
MAX_SECONDS = None   # 전체 렌더링. 테스트면 1, 2, 5 이런 식으로 넣기.

COOLDOWN_FRAMES = 20
TOP_K = 2048
BATCH_SIZE = 128

DARK_THRESHOLD = 0.08
WHITE_THRESHOLD = 0.92

DARK_POOL_SIZE = 75
WHITE_POOL_SIZE = 75

BRIGHTNESS_TOLERANCE = 0.04
CLIP_WEIGHT = 0.15

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

with open(PATHS_TXT, "r", encoding="utf-8") as f:
    cat_paths = [line.strip() for line in f]

cat_count = len(cat_paths)
print("cats:", cat_count)

index = faiss.read_index(INDEX_PATH)

cat_brightness = []
cat_tiles = []

print("loading cat tiles...")
for p in tqdm(cat_paths):
    img = Image.open(p).convert("RGB").resize((TILE_SIZE, TILE_SIZE))
    cat_tiles.append(img.copy())
    cat_brightness.append(np.array(img.convert("L")).mean() / 255.0)

cat_brightness = np.array(cat_brightness, dtype=np.float32)

brightness_order = np.argsort(cat_brightness)
dark_pool = brightness_order[:DARK_POOL_SIZE]
white_pool = brightness_order[-WHITE_POOL_SIZE:]

print("dark pool:", cat_brightness[dark_pool[0]], "~", cat_brightness[dark_pool[-1]])
print("white pool:", cat_brightness[white_pool[0]], "~", cat_brightness[white_pool[-1]])

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

def image_features(images):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        feats = outputs.pooler_output
        feats = model.visual_projection(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu().numpy().astype("float32")

def choose_by_global_brightness(target_brightness, frame_no, last_used, used_this_frame):
    best_idx = None
    best_score = 1e9

    for idx in brightness_order:
        idx = int(idx)

        if idx in used_this_frame:
            continue

        brightness_diff = abs(float(cat_brightness[idx]) - float(target_brightness))
        cooldown_ok = frame_no - last_used[idx] >= COOLDOWN_FRAMES
        cooldown_penalty = 0.0 if cooldown_ok else 0.05

        score = brightness_diff + cooldown_penalty

        if score < best_score:
            best_score = score
            best_idx = idx

    if best_idx is not None:
        return best_idx

    return int(brightness_order[0])

def choose_cat(candidates, sims, target_brightness, frame_no, last_used, used_this_frame):
    if target_brightness <= DARK_THRESHOLD:
        return int(rng.choice(dark_pool))

    if target_brightness >= WHITE_THRESHOLD:
        return int(rng.choice(white_pool))

    best_idx = None
    best_score = -1e9

    for idx, sim in zip(candidates, sims):
        idx = int(idx)

        if idx in used_this_frame:
            continue

        if frame_no - last_used[idx] < COOLDOWN_FRAMES:
            continue

        brightness_diff = abs(float(cat_brightness[idx]) - float(target_brightness))

        if brightness_diff > BRIGHTNESS_TOLERANCE:
            continue

        score = -brightness_diff + CLIP_WEIGHT * float(sim)

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is not None:
        return best_idx

    best_idx = None
    best_score = -1e9

    for idx, sim in zip(candidates, sims):
        idx = int(idx)

        if idx in used_this_frame:
            continue

        brightness_diff = abs(float(cat_brightness[idx]) - float(target_brightness))

        if brightness_diff > BRIGHTNESS_TOLERANCE:
            continue

        score = -brightness_diff + CLIP_WEIGHT * float(sim)

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is not None:
        return best_idx

    return choose_by_global_brightness(
        target_brightness,
        frame_no,
        last_used,
        used_this_frame,
    )

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

src_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, round(src_fps / TARGET_FPS))

start_src_frame = int(START_SECONDS * src_fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_src_frame)

if MAX_SECONDS is None:
    max_render_frames = None
else:
    max_render_frames = int(MAX_SECONDS * TARGET_FPS)

print("source fps:", src_fps)
print("frames:", frame_count)
print("step:", step)
print("start seconds:", START_SECONDS)
print("start source frame:", start_src_frame)
print("max render frames:", max_render_frames)

last_used = np.full(cat_count, -999999, dtype=np.int32)

rendered = 0
frame_no = int(START_SECONDS * TARGET_FPS)

for src_idx in tqdm(range(start_src_frame, frame_count), desc="Rendering FINAL"):
    ret, frame = cap.read()

    if not ret:
        break

    if (src_idx - start_src_frame) % step != 0:
        continue

    if max_render_frames is not None and rendered >= max_render_frames:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(
        frame,
        (GRID_W * TILE_SIZE, GRID_H * TILE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    tile_images = []
    tile_brightness = []

    for y in range(GRID_H):
        for x in range(GRID_W):
            tile = frame[
                y * TILE_SIZE:(y + 1) * TILE_SIZE,
                x * TILE_SIZE:(x + 1) * TILE_SIZE,
            ]

            pil_tile = Image.fromarray(tile).convert("RGB")
            tile_images.append(pil_tile)
            tile_brightness.append(np.array(pil_tile.convert("L")).mean() / 255.0)

    tile_brightness = np.array(tile_brightness, dtype=np.float32)

    selected = []
    used_this_frame = set()

    for start in range(0, len(tile_images), BATCH_SIZE):
        batch_imgs = tile_images[start:start + BATCH_SIZE]
        feats = image_features(batch_imgs)

        sims, ids = index.search(feats, TOP_K)

        for bi in range(len(batch_imgs)):
            global_i = start + bi

            idx = choose_cat(
                ids[bi],
                sims[bi],
                tile_brightness[global_i],
                frame_no,
                last_used,
                used_this_frame,
            )

            selected.append(idx)
            used_this_frame.add(idx)
            last_used[idx] = frame_no

    canvas = Image.new("RGB", (GRID_W * TILE_SIZE, GRID_H * TILE_SIZE))

    i = 0
    for y in range(GRID_H):
        for x in range(GRID_W):
            canvas.paste(cat_tiles[selected[i]], (x * TILE_SIZE, y * TILE_SIZE))
            i += 1

    canvas.save(OUT_DIR / f"frame_{rendered:05d}.jpg", quality=90)

    rendered += 1
    frame_no += 1

cap.release()

print("rendered frames:", rendered)
print("saved:", OUT_DIR)
