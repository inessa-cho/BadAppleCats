import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

VIDEO_PATH = "MediaOriginal/bad_apple.mp4"
OUT_DIR = Path("badapple_grid")
OUT_DIR.mkdir(exist_ok=True)

GRID_W = 48
GRID_H = 36
TARGET_FPS = 10

cap = cv2.VideoCapture(VIDEO_PATH)

src_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / src_fps

print("source fps:", src_fps)
print("frames:", frame_count)
print("duration:", duration)

step = max(1, round(src_fps / TARGET_FPS))
print("frame step:", step)

grids = []
saved_frames = 0

for frame_idx in tqdm(range(frame_count), desc="Grid processing"):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % step != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Bad Apple 원본 비율을 48x36으로 강제 축소
    small = cv2.resize(gray, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)

    # 0.0~1.0 밝기값
    grid = small.astype(np.float32) / 255.0

    grids.append(grid)
    saved_frames += 1

cap.release()

grids = np.stack(grids, axis=0)

np.save(OUT_DIR / "badapple_grids.npy", grids)

print("saved:", OUT_DIR / "badapple_grids.npy")
print("shape:", grids.shape)
