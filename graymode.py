from pathlib import Path
from PIL import Image

input_dir = Path("/home/darek123/python_projects/BadAppleCats/Cats256px_rgb/")
output_dir = Path("/home/darek123/python_projects/BadAppleCats/Cats256px_gray_3channels/")

output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*"):
    try:
        img = Image.open(img_path)

        gray_rgb = img.convert("L").convert("RGB")

        out_path = output_dir / f"{img_path.stem}.jpg"

        gray_rgb.save(out_path, quality=90)

    except Exception as e:
        print(f"Error: {img_path} -> {e}")
