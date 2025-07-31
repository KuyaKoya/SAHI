# infer_sahi.py
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
import os
from datetime import datetime
import cv2

# Configuration
IMAGE_PATH = "test_images/10-page_1.jpg"  # your floorplan image
MODEL_PATH = "model/model_v52.pt"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_PATH = f"results/sahi_outputs/sliced_output_{timestamp}.jpg"


def infer_with_sahi():
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    detection_model = UltralyticsDetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=0.4,
        device="cuda",  # or 'cpu'
    )

    result = get_sliced_prediction(
        IMAGE_PATH,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        perform_standard_pred=True,
    )

    # Save visualized results
    result.export_visuals(
        export_dir=os.path.dirname(RESULT_PATH),
        file_name=os.path.basename(RESULT_PATH),
    )

    print(f"Sliced inference result saved at {RESULT_PATH}")


if __name__ == "__main__":
    infer_with_sahi()
