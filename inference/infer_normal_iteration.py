from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_prediction
import os
from datetime import datetime
import cv2

# Configuration
IMAGE_DIR = "test_images"
MODEL_PATH = "model/model_v52.pt"
OUTPUT_DIR = "results/sahi_outputs"

# Dynamic tiling params
SLICE_SCALE = 0.4  # Tile covers 40% of image height/width
MIN_SLICE = 512
MAX_SLICE = 1024


def clamp(val, minval, maxval):
    return max(minval, min(val, maxval))


def infer_with_sahi():
    timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(OUTPUT_DIR, timestamp_folder)
    os.makedirs(output_subdir, exist_ok=True)

    detection_model = UltralyticsDetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=0.4,
        device="cuda",  # or 'cpu'
    )

    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Unable to read {img_path}")
            continue

        result_filename = f"{os.path.splitext(img_name)[0]}_sliced_output.jpg"
        result_path = os.path.join(output_subdir, result_filename)

        print(f"[INFO] Processing {img_name}")

        result = get_prediction(
            img_path,
            detection_model,
        )

        result.export_visuals(
            export_dir=output_subdir,
            file_name=result_filename,
        )
        print(f"[✓] Saved: {result_path}")


if __name__ == "__main__":
    infer_with_sahi()
