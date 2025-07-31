import os
import cv2
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction

# === Configuration ===
IMAGE_DIR = "test_images"
MODEL_PATH = "model/model_v52.pt"
OUTPUT_DIR = "results/sahi_ensemble_outputs"
TARGET_SIZE = (2048, 1446)

# === Tiling Strategies ===
TILING_STRATEGIES = [
    {
        "name": "clahe_lab",
        "clahe_lab": True,
        "slice_height": 768,
        "slice_width": 768,
        "overlap_height_ratio": 0.25,
        "overlap_width_ratio": 0.25,
        "weight": 1.4,
    },
    {
        "name": "grayscale_clahe",
        "grayscale_clahe": True,
        "slice_height": 768,
        "slice_width": 768,
        "overlap_height_ratio": 0.25,
        "overlap_width_ratio": 0.25,
        "weight": 1.5,
    },
]


# === Utility Functions ===


def clamp(val, minval, maxval):
    return max(minval, min(val, maxval))


def apply_clahe_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_clahe_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    return cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(detections_list, iou_threshold=0.5, skip_box_threshold=0.001):
    all_boxes, all_scores, all_labels, all_weights = [], [], [], []
    for det in detections_list:
        for i, box in enumerate(det["boxes"]):
            all_boxes.append(box)
            all_scores.append(det["scores"][i])
            all_labels.append(det["labels"][i])
            all_weights.append(det.get("weight", 1.0))
    if not all_boxes:
        return {"boxes": [], "scores": [], "labels": []}

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_weights = np.array(all_weights)
    keep = all_scores >= skip_box_threshold
    all_boxes, all_scores, all_labels, all_weights = (
        all_boxes[keep],
        all_scores[keep],
        all_labels[keep],
        all_weights[keep],
    )

    final_boxes, final_scores, final_labels = [], [], []
    class_groups = defaultdict(list)
    for i, label in enumerate(all_labels):
        class_groups[label].append(i)

    for class_id, idxs in class_groups.items():
        boxes = all_boxes[idxs]
        scores = all_scores[idxs]
        weights = all_weights[idxs]
        used = np.zeros(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if calculate_iou(boxes[i], boxes[j]) >= iou_threshold:
                    cluster.append(j)
                    used[j] = True
            cluster_boxes = boxes[cluster]
            cluster_scores = scores[cluster]
            cluster_weights = weights[cluster]
            total_weight = cluster_weights.sum()
            weighted_coords = cluster_boxes * cluster_weights[:, None]
            avg_box = weighted_coords.sum(axis=0) / total_weight
            avg_score = (cluster_scores * cluster_weights).sum() / total_weight
            final_boxes.append(avg_box.tolist())
            final_scores.append(avg_score)
            final_labels.append(class_id)
    return {"boxes": final_boxes, "scores": final_scores, "labels": final_labels}


def run_single_strategy(image, detection_model, strategy):
    image = cv2.resize(image, TARGET_SIZE)
    img_path = "_temp.jpg"
    enhanced = image.copy()

    if strategy.get("clahe_lab"):
        enhanced = apply_clahe_lab(enhanced)
    elif strategy.get("grayscale_clahe"):
        enhanced = apply_clahe_grayscale(enhanced)

    cv2.imwrite(img_path, enhanced)

    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=strategy["slice_height"],
        slice_width=strategy["slice_width"],
        overlap_height_ratio=strategy["overlap_height_ratio"],
        overlap_width_ratio=strategy["overlap_width_ratio"],
        perform_standard_pred=True,
    )

    os.remove(img_path)

    boxes, scores, labels = [], [], []
    if hasattr(result, "object_prediction_list"):
        for pred in result.object_prediction_list:
            b = pred.bbox
            boxes.append([b.minx, b.miny, b.maxx, b.maxy])
            scores.append(pred.score.value)
            labels.append(pred.category.id)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "weight": strategy["weight"],
        "result_object": result,
    }


def draw_ensemble_visualization(image, detections, output_path):
    vis = image.copy()
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]
    for i, (box, score, label) in enumerate(
        zip(detections["boxes"], detections["scores"], detections["labels"])
    ):
        x1, y1, x2, y2 = map(int, box)
        color = colors[int(label) % len(colors)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_text = f"cls_{int(label)}: {score:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, y1 - h - 10), (x1 + w + 5, y1), color, -1)
        cv2.putText(
            vis,
            label_text,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    cv2.imwrite(output_path, vis)


# === Main Inference ===


def ensemble_inference():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    ensemble_dir = os.path.join(out_dir, "ensemble_results")
    os.makedirs(ensemble_dir, exist_ok=True)

    detection_model = UltralyticsDetectionModel(
        model_path=MODEL_PATH, confidence_threshold=0.85, device="cuda"
    )

    summary = []

    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(IMAGE_DIR, img_name)
        original = cv2.imread(img_path)
        if original is None:
            print(f"[WARN] Couldn't read {img_path}")
            continue

        print(f"[INFO] Processing {img_name}")
        resized = cv2.resize(original, TARGET_SIZE)

        detections = []
        for strat in TILING_STRATEGIES:
            try:
                result = run_single_strategy(resized, detection_model, strat)
                detections.append(result)
            except Exception as e:
                print(f"[ERROR] {strat['name']} failed: {e}")

        fused = weighted_box_fusion(detections)
        out_path = os.path.join(
            ensemble_dir, f"{os.path.splitext(img_name)[0]}_ensemble.jpg"
        )
        draw_ensemble_visualization(resized, fused, out_path)

        print(f"  [✓] Saved: {out_path}")
        summary.append(
            {
                "image": img_name,
                "strategies": {
                    s["name"]: len(r["boxes"])
                    for s, r in zip(TILING_STRATEGIES, detections)
                },
                "ensemble": len(fused["boxes"]),
            }
        )

    with open(os.path.join(out_dir, "ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] All done! Results saved to: {out_dir}")


if __name__ == "__main__":
    ensemble_inference()
