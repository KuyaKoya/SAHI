# convert_corrections_to_yolo.py
import os
import json
import shutil
from PIL import Image

# Updated to use the new feedback_json location
SAHI_OUTPUT_DIR = "results/sahi_outputs"

LABELS = ["room"]  # assuming single class for now
LABEL_TO_ID = {name: idx for idx, name in enumerate(LABELS)}


def get_latest_feedback_json_dir():
    """Get the latest timestamped feedback_json directory and return both paths"""
    if not os.path.exists(SAHI_OUTPUT_DIR):
        raise Exception(f"SAHI output directory not found: {SAHI_OUTPUT_DIR}")

    # Get all timestamped folders
    subfolders = [
        os.path.join(SAHI_OUTPUT_DIR, d)
        for d in os.listdir(SAHI_OUTPUT_DIR)
        if os.path.isdir(os.path.join(SAHI_OUTPUT_DIR, d))
        and d.replace("_", "").replace("-", "").isdigit()
    ]

    if not subfolders:
        raise Exception("No timestamped folders found in SAHI outputs.")

    # Get the latest folder by modification time
    latest_folder = sorted(subfolders, key=os.path.getmtime)[-1]
    feedback_json_dir = os.path.join(latest_folder, "feedback_json")

    if not os.path.exists(feedback_json_dir):
        raise Exception(f"No feedback_json directory found in: {latest_folder}")

    return feedback_json_dir, latest_folder


def convert_bbox_to_yolo(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return [x_center, y_center, w, h]


count = 0

# Get the latest feedback_json directory and its parent timestamped folder
try:
    feedback_json_dir, latest_output_folder = get_latest_feedback_json_dir()
    print(f"[INFO] Using feedback_json directory: {feedback_json_dir}")

    # Create corrected_dataset directories within the same timestamped folder
    OUTPUT_IMAGE_DIR = os.path.join(latest_output_folder, "corrected_dataset", "images")
    OUTPUT_LABEL_DIR = os.path.join(latest_output_folder, "corrected_dataset", "labels")

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    print(
        f"[INFO] Output will be saved to: {os.path.join(latest_output_folder, 'corrected_dataset')}"
    )

except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)

for fname in os.listdir(feedback_json_dir):
    if not fname.endswith("_pred.json"):  # Updated to match new naming convention
        continue

    with open(os.path.join(feedback_json_dir, fname)) as f:
        data = json.load(f)

    image_name = data["image"]
    image_path = os.path.join("test_images", image_name)
    image_output_path = os.path.join(OUTPUT_IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"[SKIP] Image not found: {image_path}")
        continue

    shutil.copyfile(image_path, image_output_path)

    width = data["size"]["width"]
    height = data["size"]["height"]

    yolo_lines = []
    # Updated to use 'detections' instead of 'final_detections' for the new JSON format
    detections_key = "detections" if "detections" in data else "final_detections"
    for det in data[detections_key]:
        label = det.get("label", "room")  # Default to 'room' if label not present
        class_id = LABEL_TO_ID.get(label, 0)  # default to 'room'

        yolo_box = convert_bbox_to_yolo(det["bbox"], width, height)
        line = f"{class_id} {' '.join(f'{v:.6f}' for v in yolo_box)}"
        yolo_lines.append(line)

    label_output_path = os.path.join(
        OUTPUT_LABEL_DIR, os.path.splitext(image_name)[0] + ".txt"
    )
    with open(label_output_path, "w") as f:
        f.write("\n".join(yolo_lines))

    count += 1

print(f"[✓] Converted {count} correction files to YOLO format.")
