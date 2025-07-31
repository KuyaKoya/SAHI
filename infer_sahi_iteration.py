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

        height, width = image.shape[:2]

        # Dynamically calculate slice sizes
        slice_height = clamp(int(height * SLICE_SCALE), MIN_SLICE, MAX_SLICE)
        slice_width = clamp(int(width * SLICE_SCALE), MIN_SLICE, MAX_SLICE)

        # Dynamically adjust overlap ratios based on slice size
        overlap_height_ratio = clamp(512 / slice_height, 0.10, 0.30)
        overlap_width_ratio = clamp(512 / slice_width, 0.10, 0.30)

        result_filename = f"{os.path.splitext(img_name)[0]}_sliced_output.jpg"
        result_path = os.path.join(output_subdir, result_filename)

        print(
            f"[INFO] Processing {img_name} | size=({width}x{height}) → tile=({slice_width}x{slice_height}), overlap=({overlap_width_ratio:.2f}, {overlap_height_ratio:.2f})"
        )

        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=True,
        )

        result.export_visuals(
            export_dir=output_subdir,
            file_name=result_filename,
        )
        print(f"[✓] Saved: {result_path}")


if __name__ == "__main__":
    infer_with_sahi()
