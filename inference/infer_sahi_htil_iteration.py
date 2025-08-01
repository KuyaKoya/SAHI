from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
import os
from datetime import datetime
import cv2
import json

# Configuration
IMAGE_DIR = "test_images"
MODEL_PATH = "model/model_v52.pt"
OUTPUT_DIR = "results/sahi_outputs"

# Dynamic tiling params
SLICE_SCALE = 0.4
MIN_SLICE = 512
MAX_SLICE = 1024


def clamp(val, minval, maxval):
    return max(minval, min(val, maxval))


def infer_with_feedback_json():
    timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(OUTPUT_DIR, timestamp_folder)
    os.makedirs(output_subdir, exist_ok=True)

    # Create feedback_json subdirectory within the timestamped folder
    feedback_json_dir = os.path.join(output_subdir, "feedback_json")
    os.makedirs(feedback_json_dir, exist_ok=True)

    detection_model = UltralyticsDetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=0.8,
        device="cuda",  # or 'cpu'
    )

    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"[WARN] Unable to read {img_path}")
            continue

        original_height, original_width = original_image.shape[:2]

        # Resize to target size (2048, 1446)
        target_width, target_height = 2048, 1446
        resized_image = cv2.resize(original_image, (target_width, target_height))

        # Create temporary resized image file
        temp_dir = os.path.join(os.path.dirname(img_path), "temp_resized")
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_name = f"temp_resized_{img_name}"
        temp_image_path = os.path.join(temp_dir, temp_image_name)
        cv2.imwrite(temp_image_path, resized_image)

        # Calculate scale factors for coordinate conversion
        scale_x = original_width / target_width
        scale_y = original_height / target_height

        height, width = target_height, target_width
        print(
            f"[INFO] Processing {img_name} | original=({original_width}x{original_height}) → resized=({width}x{height})"
        )
        slice_height = clamp(int(height * SLICE_SCALE), MIN_SLICE, MAX_SLICE)
        slice_width = clamp(int(width * SLICE_SCALE), MIN_SLICE, MAX_SLICE)
        overlap_height_ratio = clamp(512 / slice_height, 0.10, 0.30)
        overlap_width_ratio = clamp(512 / slice_width, 0.10, 0.30)

        print(
            f"    tile=({slice_width}x{slice_height}), overlap=({overlap_width_ratio:.2f}, {overlap_height_ratio:.2f})"
        )

        result = get_sliced_prediction(
            temp_image_path,  # Use resized image
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=True,
        )

        # Export image with boxes
        output_img_path = os.path.join(
            output_subdir, f"{os.path.splitext(img_name)[0]}.jpg"
        )
        result.export_visuals(
            export_dir=output_subdir, file_name=os.path.basename(output_img_path)
        )
        print(f"[✓] Saved: {output_img_path}")

        # Export editable prediction JSON
        editable_json = {
            "image": img_name,
            "size": {
                "width": original_width,
                "height": original_height,
            },  # Use original size
            "detections": [],
        }

        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            # Convert coordinates back to original image size
            original_x1 = bbox[0] * scale_x
            original_y1 = bbox[1] * scale_y
            original_x2 = bbox[2] * scale_x
            original_y2 = bbox[3] * scale_y

            editable_json["detections"].append(
                {
                    "bbox": [
                        original_x1,
                        original_y1,
                        original_x2,
                        original_y2,
                    ],  # x1, y1, x2, y2
                    "label": pred.category.name,
                    "confidence": round(pred.score.value, 4),
                }
            )

        json_filename = os.path.splitext(img_name)[0] + "_pred.json"
        json_path = os.path.join(feedback_json_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(editable_json, f, indent=2)
        print(f"[✓] JSON saved: {json_path}")

        # Cleanup temporary resized image
        try:
            os.remove(temp_image_path)
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    infer_with_feedback_json()
