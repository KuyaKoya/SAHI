from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.cv import read_image
import os
from datetime import datetime
import cv2
import numpy as np
from collections import defaultdict
import json

# Configuration
IMAGE_DIR = "test_images"
MODEL_PATH = "model/model_v52.pt"
OUTPUT_DIR = "results/sahi_ensemble_outputs"

# Multiple tiling strategies
TILING_STRATEGIES = [
    {
        "name": "small_tiles",
        "slice_height": 512,
        "slice_width": 512,
        "overlap_height_ratio": 0.2,
        "overlap_width_ratio": 0.2,
        "weight": 1.0,
    },
    {
        "name": "medium_tiles",
        "slice_height": 768,
        "slice_width": 768,
        "overlap_height_ratio": 0.25,
        "overlap_width_ratio": 0.25,
        "weight": 1.2,
    },
    {
        "name": "large_tiles",
        "slice_height": 1024,
        "slice_width": 1024,
        "overlap_height_ratio": 0.3,
        "overlap_width_ratio": 0.3,
        "weight": 0.8,
    },
    {
        "name": "dynamic_adaptive",
        "slice_scale": 0.4,
        "min_slice": 512,
        "max_slice": 1024,
        "weight": 1.1,
    },
    {
        "name": "resized_small_tiles",
        "resize_target": (2048, 1446),
        "slice_height": 512,
        "slice_width": 512,
        "overlap_height_ratio": 0.2,
        "overlap_width_ratio": 0.2,
        "weight": 1.3,  # Higher weight as resizing often improves detection
    },
    {
        "name": "resized_medium_tiles",
        "resize_target": (2048, 1446),
        "slice_height": 768,
        "slice_width": 768,
        "overlap_height_ratio": 0.25,
        "overlap_width_ratio": 0.25,
        "weight": 1.4,
    },
]


def clamp(val, minval, maxval):
    return max(minval, min(val, maxval))


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def weighted_box_fusion(detections_list, iou_threshold=0.5, skip_box_threshold=0.001):
    """
    Simplified Weighted Box Fusion for combining multiple detection results
    detections_list: List of detection dictionaries with format:
    [{'boxes': [[x1,y1,x2,y2]], 'scores': [conf], 'labels': [cls], 'weights': [w]}]
    """
    if not detections_list:
        return {"boxes": [], "scores": [], "labels": []}

    all_boxes = []
    all_scores = []
    all_labels = []
    all_weights = []

    # Collect all detections with their weights
    for det in detections_list:
        for i, box in enumerate(det["boxes"]):
            all_boxes.append(box)
            all_scores.append(det["scores"][i])
            all_labels.append(det["labels"][i])
            all_weights.append(det.get("weight", 1.0))

    if not all_boxes:
        return {"boxes": [], "scores": [], "labels": []}

    # Convert to numpy arrays
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_weights = np.array(all_weights)

    # Filter out low confidence detections
    keep_mask = all_scores >= skip_box_threshold
    all_boxes = all_boxes[keep_mask]
    all_scores = all_scores[keep_mask]
    all_labels = all_labels[keep_mask]
    all_weights = all_weights[keep_mask]

    if len(all_boxes) == 0:
        return {"boxes": [], "scores": [], "labels": []}

    # Group by class
    class_groups = defaultdict(list)
    for i, label in enumerate(all_labels):
        class_groups[label].append(i)

    final_boxes = []
    final_scores = []
    final_labels = []

    # Process each class separately
    for class_id, indices in class_groups.items():
        class_boxes = all_boxes[indices]
        class_scores = all_scores[indices]
        class_weights = all_weights[indices]

        # Cluster overlapping boxes
        used = np.zeros(len(class_boxes), dtype=bool)

        for i in range(len(class_boxes)):
            if used[i]:
                continue

            cluster_indices = [i]
            used[i] = True

            # Find all boxes that overlap with current box
            for j in range(i + 1, len(class_boxes)):
                if used[j]:
                    continue

                iou = calculate_iou(class_boxes[i], class_boxes[j])
                if iou >= iou_threshold:
                    cluster_indices.append(j)
                    used[j] = True

            # Combine boxes in cluster using weighted average
            if len(cluster_indices) > 0:
                cluster_boxes = class_boxes[cluster_indices]
                cluster_scores = class_scores[cluster_indices]
                cluster_weights = class_weights[cluster_indices]

                # Weighted scores
                weighted_scores = cluster_scores * cluster_weights
                total_weight = np.sum(cluster_weights)
                final_score = (
                    np.sum(weighted_scores) / total_weight if total_weight > 0 else 0
                )

                # Weighted box coordinates
                weighted_coords = cluster_boxes * cluster_weights.reshape(-1, 1)
                final_box = (
                    np.sum(weighted_coords, axis=0) / total_weight
                    if total_weight > 0
                    else cluster_boxes[0]
                )

                final_boxes.append(final_box.tolist())
                final_scores.append(final_score)
                final_labels.append(class_id)

    return {"boxes": final_boxes, "scores": final_scores, "labels": final_labels}


def run_single_strategy(image_path, detection_model, strategy, original_image_shape):
    """Run inference with a single tiling strategy"""
    original_height, original_width = original_image_shape[:2]

    # Handle resizing if specified
    if "resize_target" in strategy:
        target_width, target_height = strategy["resize_target"]

        # Create temporary resized image
        original_image = cv2.imread(image_path)
        resized_image = cv2.resize(original_image, (target_width, target_height))

        # Save temporary resized image
        temp_dir = os.path.join(os.path.dirname(image_path), "temp_resized")
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_name = f"temp_resized_{os.path.basename(image_path)}"
        temp_image_path = os.path.join(temp_dir, temp_image_name)
        cv2.imwrite(temp_image_path, resized_image)

        # Use resized dimensions for tiling calculations
        height, width = target_height, target_width
        inference_image_path = temp_image_path

        # Calculate scale factors for coordinate transformation
        scale_x = original_width / target_width
        scale_y = original_height / target_height

        print(
            f"  └─ {strategy['name']}: resized ({original_width}x{original_height}) → ({target_width}x{target_height})"
        )
    else:
        # Use original image
        height, width = original_height, original_width
        inference_image_path = image_path
        scale_x = 1.0
        scale_y = 1.0

    if "slice_scale" in strategy:
        # Dynamic strategy
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
        # Fixed strategy
        slice_height = strategy["slice_height"]
        slice_width = strategy["slice_width"]
        overlap_height_ratio = strategy["overlap_height_ratio"]
        overlap_width_ratio = strategy["overlap_width_ratio"]

    print(
        f"    tile=({slice_width}x{slice_height}), "
        f"overlap=({overlap_width_ratio:.2f}, {overlap_height_ratio:.2f})"
    )

    result = get_sliced_prediction(
        inference_image_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        perform_standard_pred=True,
    )

    # Extract detections in standardized format
    boxes = []
    scores = []
    labels = []

    if hasattr(result, "object_prediction_list"):
        for pred in result.object_prediction_list:
            bbox = pred.bbox

            # Transform coordinates back to original image scale
            original_minx = bbox.minx * scale_x
            original_miny = bbox.miny * scale_y
            original_maxx = bbox.maxx * scale_x
            original_maxy = bbox.maxy * scale_y

            boxes.append([original_minx, original_miny, original_maxx, original_maxy])
            scores.append(pred.score.value)
            labels.append(pred.category.id)

    # Clean up temporary resized image if created
    if "resize_target" in strategy:
        try:
            os.remove(temp_image_path)
            # Remove temp directory if empty
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass  # Ignore cleanup errors

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "weight": strategy["weight"],
        "result_object": result,  # Keep original result for visualization
    }


def draw_ensemble_visualization(image, detections, output_path):
    """Draw ensemble detection results on image using OpenCV with enhanced readability"""
    vis_image = image.copy()

    # High contrast, vivid colors (extend if more classes are needed)
    colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Sky Blue
        (255, 0, 128),  # Pinkish
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    box_thickness = 3

    for i, (box, score, label) in enumerate(
        zip(detections["boxes"], detections["scores"], detections["labels"])
    ):
        x1, y1, x2, y2 = map(int, box)
        color = colors[int(label) % len(colors)]

        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, box_thickness)

        # Draw label background and text
        label_text = f"class_{int(label)}: {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )
        label_bg_topleft = (x1, y1 - text_height - 12)
        label_bg_bottomright = (x1 + text_width + 4, y1)

        # Draw background rectangle for contrast
        cv2.rectangle(vis_image, label_bg_topleft, label_bg_bottomright, color, -1)

        # Draw label text in white
        cv2.putText(
            vis_image,
            label_text,
            (x1 + 2, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    # Save ensemble visualization
    cv2.imwrite(output_path, vis_image)
    return output_path


def ensemble_inference():
    timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(OUTPUT_DIR, timestamp_folder)
    os.makedirs(output_subdir, exist_ok=True)

    # Create subdirectories for individual strategies
    individual_dirs = {}
    for strategy in TILING_STRATEGIES:
        strategy_dir = os.path.join(output_subdir, f"individual_{strategy['name']}")
        os.makedirs(strategy_dir, exist_ok=True)
        individual_dirs[strategy["name"]] = strategy_dir

    ensemble_dir = os.path.join(output_subdir, "ensemble_results")
    os.makedirs(ensemble_dir, exist_ok=True)

    detection_model = UltralyticsDetectionModel(
        model_path=MODEL_PATH,
        confidence_threshold=0.8,  # Lower threshold for individual strategies
        device="cuda",
    )

    results_summary = []

    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Unable to read {img_path}")
            continue

        print(
            f"\n[INFO] Processing {img_name} | size=({image.shape[1]}x{image.shape[0]})"
        )

        # Run all strategies
        strategy_results = []
        for strategy in TILING_STRATEGIES:
            try:
                result = run_single_strategy(
                    img_path, detection_model, strategy, image.shape
                )
                strategy_results.append(result)

                # Save individual strategy result
                individual_filename = (
                    f"{os.path.splitext(img_name)[0]}_{strategy['name']}.jpg"
                )
                result["result_object"].export_visuals(
                    export_dir=individual_dirs[strategy["name"]],
                    file_name=individual_filename,
                )

            except Exception as e:
                print(f"  [ERROR] Strategy {strategy['name']} failed: {e}")
                continue

        if not strategy_results:
            print(f"  [WARN] No successful strategies for {img_name}")
            continue

        # Combine results using weighted box fusion
        print(f"  [INFO] Combining {len(strategy_results)} strategy results...")
        ensemble_result = weighted_box_fusion(strategy_results, iou_threshold=0.5)

        # Create visualization for ensemble result using OpenCV
        ensemble_filename = f"{os.path.splitext(img_name)[0]}_ensemble.jpg"
        ensemble_path = os.path.join(ensemble_dir, ensemble_filename)

        draw_ensemble_visualization(image, ensemble_result, ensemble_path)
        print(f"  [✓] Ensemble visualization saved: {ensemble_path}")

        # Log results summary
        individual_counts = [len(res["boxes"]) for res in strategy_results]
        ensemble_count = len(ensemble_result["boxes"])

        summary = {
            "image": img_name,
            "individual_detections": {
                strategy["name"]: count
                for strategy, count in zip(TILING_STRATEGIES, individual_counts)
            },
            "ensemble_detections": ensemble_count,
            "improvement": (
                ensemble_count - max(individual_counts) if individual_counts else 0
            ),
        }
        results_summary.append(summary)

        print(f"  [✓] Individual: {individual_counts}, Ensemble: {ensemble_count}")

    # Save summary
    summary_path = os.path.join(output_subdir, "ensemble_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[✓] Ensemble inference complete! Results saved to: {output_subdir}")
    return results_summary


if __name__ == "__main__":
    ensemble_inference()
