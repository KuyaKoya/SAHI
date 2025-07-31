from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
import os
from datetime import datetime
import cv2

# Configuration
IMAGE_DIR = "data/floorplans-roboflow-yolov11/test/images"
MODEL_PATH = "exports/floorplans_yolov11/weights/best.pt"
OUTPUT_DIR = "results/sahi_outputs"

# Dynamic tiling params
SLICE_SCALE = 0.4  # Tile covers 40% of image height/width
MIN_SLICE = 512
MAX_SLICE = 1024


def infer_with_sahi():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        height, width = image.shape[:2]

        # Dynamically calculate slice sizes
        slice_height = max(MIN_SLICE, min(int(height * SLICE_SCALE), MAX_SLICE))
        slice_width = max(MIN_SLICE, min(int(width * SLICE_SCALE), MAX_SLICE))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(
            OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_{timestamp}.jpg"
        )

        print(
            f"[INFO] Processing {img_name} | size=({width}x{height}) → tile=({slice_width}x{slice_height})"
        )

        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25,
            perform_standard_pred=True,
        )

        result.export_visuals(
            export_dir=os.path.dirname(result_path),
            file_name=os.path.basename(result_path),
        )
        print(f"[✓] Saved: {result_path}")


if __name__ == "__main__":
    infer_with_sahi()
