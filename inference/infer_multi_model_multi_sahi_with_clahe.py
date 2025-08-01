from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
import os
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import json


# Configuration
IMAGE_DIR = "test_images"
YOLO_MODEL_PATH = "model/model_v52.pt"
DETECTRON2_MODEL_PATH = "model/detectron_cus_model_2.pth"
OUTPUT_DIR = "results/sahi_ensemble_outputs"

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


def weighted_box_fusion(detections_list, iou_threshold=0.3, skip_box_threshold=0.001):
    """
    Improved weighted box fusion with lower IoU threshold for better merging of overlapping room detections
    Uses connected components approach for better clustering of overlapping boxes
    """
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

        # Build adjacency matrix for connected components clustering
        n = len(boxes)
        adjacency = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if calculate_iou(boxes[i], boxes[j]) >= iou_threshold:
                    adjacency[i, j] = adjacency[j, i] = True

        # Find connected components
        visited = np.zeros(n, dtype=bool)
        for i in range(n):
            if visited[i]:
                continue

            # BFS to find connected component
            cluster = []
            queue = [i]
            visited[i] = True

            while queue:
                current = queue.pop(0)
                cluster.append(current)
                for j in range(n):
                    if adjacency[current, j] and not visited[j]:
                        visited[j] = True
                        queue.append(j)

            # Merge boxes in cluster
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
    processed_image = image.copy()
    if "resize_target" in strategy:
        target_width, target_height = strategy["resize_target"]
        processed_image = cv2.resize(processed_image, (target_width, target_height))
    if strategy.get("apply_grayscale_clahe", False):
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return processed_image


def run_strategy(image, model, strategy):
    original_height, original_width = image.shape[:2]
    processed_image = preprocess_image(image, strategy)
    height, width = processed_image.shape[:2]

    if "slice_scale" in strategy:
        slice_height = clamp(
            int(height * strategy["slice_scale"]),
            strategy["min_slice"],
            strategy["max_slice"],
        )
        slice_width = clamp(
            int(width * strategy["slice_scale"]),
            strategy["min_slice"],
            strategy["max_slice"],
        )
        overlap_height_ratio = clamp(512 / slice_height, 0.10, 0.30)
        overlap_width_ratio = clamp(512 / slice_width, 0.10, 0.30)
    else:
        slice_height = strategy.get("slice_height", 768)
        slice_width = strategy.get("slice_width", 768)
        overlap_height_ratio = strategy.get("overlap_height_ratio", 0.25)
        overlap_width_ratio = strategy.get("overlap_width_ratio", 0.25)

    if "resize_target" in strategy:
        scale_x = original_width / strategy["resize_target"][0]
        scale_y = original_height / strategy["resize_target"][1]
    else:
        scale_x = scale_y = 1.0

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
        boxes.append(
            [
                bbox.minx * scale_x,
                bbox.miny * scale_y,
                bbox.maxx * scale_x,
                bbox.maxy * scale_y,
            ]
        )
        scores.append(pred.score.value)
        labels.append(pred.category.id)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "weight": strategy["weight"],
    }


def run_detectron2_inference(predictor, image):
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    return {
        "boxes": instances.pred_boxes.tensor.numpy().tolist(),
        "scores": instances.scores.numpy().tolist(),
        "labels": instances.pred_classes.numpy().tolist(),
        "weight": 1.0,
    }


def draw_results(image, detections, path):
    for box, score, label in zip(
        detections["boxes"], detections["scores"], detections["labels"]
    ):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        text = f"Room {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
        cv2.putText(
            image, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3
        )
    cv2.imwrite(path, image)


def load_detectron2_model(model_path, device="cpu"):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    # Configure for your custom model - based on error messages showing (3,1024) and (8,1024) shapes
    # This suggests 2 foreground classes + 1 background = 3 total classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
        2  # 2 foreground classes + background is handled automatically
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85  # Set confidence to 85%
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def ensemble_inference():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    yolo_model = UltralyticsDetectionModel(
        YOLO_MODEL_PATH,
        confidence_threshold=0.85,
        device=device,
    )
    detectron2_predictor = load_detectron2_model(DETECTRON2_MODEL_PATH, device)

    for img_file in os.listdir(IMAGE_DIR):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(IMAGE_DIR, img_file)
        image = cv2.imread(path)
        if image is None:
            continue
        print(f"\n[INFO] Processing {img_file}")
        all_results = []

        # Run YOLO model on tiling strategies
        for strategy in TILING_STRATEGIES:
            try:
                print(f"  [YOLO] Strategy: {strategy['name']}")
                results = run_strategy(image.copy(), yolo_model, strategy)
                all_results.append(results)
                print(f"    Found {len(results['boxes'])} boxes")
            except Exception as e:
                print(f"    [ERROR] YOLO + {strategy['name']}: {e}")

        # Run Detectron2 model on full image
        print(f"  [Detectron2] Full image inference")
        d2_result = run_detectron2_inference(detectron2_predictor, image)
        all_results.append(d2_result)
        print(f"    Found {len(d2_result['boxes'])} boxes")

        # Fuse results with more aggressive merging for overlapping room detections
        fused = weighted_box_fusion(
            all_results, iou_threshold=0.3, skip_box_threshold=0.001
        )
        print(f"  [FUSION] → {len(fused['boxes'])} boxes after fusion")

        # Draw results
        out_path = os.path.join(save_dir, img_file)
        draw_results(image, fused, out_path)
        print(f"  [✓] Saved: {out_path}")


if __name__ == "__main__":
    ensemble_inference()
