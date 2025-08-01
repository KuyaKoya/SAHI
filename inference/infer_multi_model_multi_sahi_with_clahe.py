from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
import os
from datetime import datetime
import cv2
import numpy as np
from collections import defaultdict
import json

# Configuration
IMAGE_DIR = "test_images"
MODEL_PATHS = [
    "model/model_v52.pt",
    "model/detectron_cus_model_2.pt",
]
OUTPUT_DIR = "results/sahi_ensemble_outputs"

# Tiling strategies
TILING_STRATEGIES = [
    {
        "name": "medium_tiles_dynamic",
        "slice_scale": 0.4,
        "min_slice": 512,
        "max_slice": 1024,
        "weight": 1.2,
        "resize_target": (2048, 1446),
    },
    {
        "name": "grayscale_clahe_dynamic",
        "slice_scale": 0.35,
        "min_slice": 768,
        "max_slice": 1024,
        "apply_grayscale_clahe": True,
        "weight": 1.5,
        "resize_target": (2048, 1446),
    },
]


def clamp(val, minval, maxval):
    return max(minval, min(val, maxval))


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = (
        max(box1[0], box2[0]),
        max(box1[1], box2[1]),
        min(box1[2], box2[2]),
        min(box1[3], box2[3]),
    )
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def weighted_box_fusion(detections_list, iou_threshold=0.5, skip_box_threshold=0.001):
    if not detections_list:
        return {"boxes": [], "scores": [], "labels": []}
    all_boxes, all_scores, all_labels, all_weights = [], [], [], []
    for det in detections_list:
        for i, box in enumerate(det["boxes"]):
            all_boxes.append(box)
            all_scores.append(det["scores"][i])
            all_labels.append(det["labels"][i])
            all_weights.append(det.get("weight", 1.0))
    all_boxes, all_scores, all_labels, all_weights = map(
        np.array, [all_boxes, all_scores, all_labels, all_weights]
    )
    keep = all_scores >= skip_box_threshold
    all_boxes, all_scores, all_labels, all_weights = (
        all_boxes[keep],
        all_scores[keep],
        all_labels[keep],
        all_weights[keep],
    )
    if len(all_boxes) == 0:
        return {"boxes": [], "scores": [], "labels": []}
    grouped = defaultdict(list)
    for i, label in enumerate(all_labels):
        grouped[label].append(i)
    final_boxes, final_scores, final_labels = [], [], []
    for cls, inds in grouped.items():
        boxes, scores, weights = all_boxes[inds], all_scores[inds], all_weights[inds]
        used = np.zeros(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if not used[j] and calculate_iou(boxes[i], boxes[j]) >= iou_threshold:
                    cluster.append(j)
                    used[j] = True
            c_boxes = boxes[cluster]
            c_scores = scores[cluster]
            c_weights = weights[cluster]
            total_weight = np.sum(c_weights)
            final_boxes.append(
                (c_boxes * c_weights[:, None]).sum(axis=0) / total_weight
            )
            final_scores.append((c_scores * c_weights).sum() / total_weight)
            final_labels.append(cls)
    return {"boxes": final_boxes, "scores": final_scores, "labels": final_labels}


def preprocess_image(image, strategy):
    """Preprocess image according to strategy settings"""
    processed_image = image.copy()

    # Apply resizing if specified
    if "resize_target" in strategy:
        target_width, target_height = strategy["resize_target"]
        processed_image = cv2.resize(processed_image, (target_width, target_height))

    # Apply CLAHE enhancement if specified
    if strategy.get("apply_grayscale_clahe", False):
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return processed_image


def run_strategy(image, model, strategy):
    original_height, original_width = image.shape[:2]

    # Preprocess image (includes resizing if specified)
    processed_image = preprocess_image(image, strategy)
    processed_height, processed_width = processed_image.shape[:2]

    # Calculate dynamic slice sizes based on processed image dimensions
    if "slice_scale" in strategy:
        slice_height = clamp(
            int(processed_height * strategy["slice_scale"]),
            strategy["min_slice"],
            strategy["max_slice"],
        )
        slice_width = clamp(
            int(processed_width * strategy["slice_scale"]),
            strategy["min_slice"],
            strategy["max_slice"],
        )
        # Calculate dynamic overlap ratios
        overlap_height_ratio = clamp(512 / slice_height, 0.10, 0.30)
        overlap_width_ratio = clamp(512 / slice_width, 0.10, 0.30)
    else:
        # Fallback to fixed values if not using dynamic slicing
        slice_height = strategy.get("slice_height", 768)
        slice_width = strategy.get("slice_width", 768)
        overlap_height_ratio = strategy.get("overlap_height_ratio", 0.25)
        overlap_width_ratio = strategy.get("overlap_width_ratio", 0.25)

    # Calculate scale factors for coordinate conversion back to original size
    if "resize_target" in strategy:
        target_width, target_height = strategy["resize_target"]
        scale_x = original_width / target_width
        scale_y = original_height / target_height
    else:
        scale_x = 1.0
        scale_y = 1.0

    print(
        f"    [{strategy['name']}] tile=({slice_width}x{slice_height}), overlap=({overlap_width_ratio:.2f}, {overlap_height_ratio:.2f})"
    )

    result = get_sliced_prediction(
        processed_image,
        model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        perform_standard_pred=True,
    )

    boxes, scores, labels = [], [], []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        # Scale coordinates back to original image dimensions
        original_minx = bbox.minx * scale_x
        original_miny = bbox.miny * scale_y
        original_maxx = bbox.maxx * scale_x
        original_maxy = bbox.maxy * scale_y
        boxes.append([original_minx, original_miny, original_maxx, original_maxy])
        scores.append(pred.score.value)
        labels.append(pred.category.id)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "weight": strategy["weight"],
    }


def draw_results(image, detections, path):
    for box, score, label in zip(
        detections["boxes"], detections["scores"], detections["labels"]
    ):
        x1, y1, x2, y2 = map(int, box)

        # Draw thicker, more visible bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # Prepare text with larger font
        text = f"Room {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 3

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )

        # Draw background rectangle for better text visibility
        text_bg_topleft = (x1, y1 - text_height - 15)
        text_bg_bottomright = (x1 + text_width + 10, y1 - 5)
        cv2.rectangle(image, text_bg_topleft, text_bg_bottomright, (0, 255, 0), -1)

        # Draw text in black for better contrast
        cv2.putText(
            image,
            text,
            (x1 + 5, y1 - 10),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )
    cv2.imwrite(path, image)


def ensemble_inference():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    models = [
        UltralyticsDetectionModel(
            p,
            confidence_threshold=0.85,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        )
        for p in MODEL_PATHS
    ]
    for img_file in os.listdir(IMAGE_DIR):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        path = os.path.join(IMAGE_DIR, img_file)
        image = cv2.imread(path)
        if image is None:
            continue
        print(f"[INFO] Processing {img_file}")
        all_results = []
        for model_idx, model in enumerate(models):
            model_name = os.path.basename(MODEL_PATHS[model_idx])
            print(f"  [MODEL] Running {model_name}")
            for strategy in TILING_STRATEGIES:
                try:
                    print(f"    [STRATEGY] {strategy['name']}")
                    results = run_strategy(image.copy(), model, strategy)
                    all_results.append(results)
                    print(f"    [RESULT] Found {len(results['boxes'])} detections")
                except Exception as e:
                    print(f"    [ERROR] {model_name} + {strategy['name']} failed: {e}")
        fused = weighted_box_fusion(all_results)
        print(
            f"  [FUSION] Combined {len(all_results)} results → {len(fused['boxes'])} final detections"
        )
        out_path = os.path.join(save_dir, img_file)
        draw_results(image, fused, out_path)
        print(f"  [✓] Saved: {out_path}")


if __name__ == "__main__":
    ensemble_inference()
