import argparse
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


# Configuration
IMAGE_DIR = "test_images"
YOLO_MODEL_PATH = "model/model_v52.pt"
OBB_MODEL_PATH = "model/obb-floorplan-v2.pt"
DETECTRON2_MODEL_PATH = "model/detectron_cus_model_3.pth"
OUTPUT_DIR = "results/sahi_ensemble_outputs"
CONFIDENCE_THRESHOLD = 0.85  # Default confidence threshold for models

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

OBB_STRATEGY = {
    "name": "obb_oriented_detection",
    "slice_scale": 0.3,
    "min_slice": 640,
    "max_slice": 1024,
    "weight": 1.8,
    "resize_target": (2048, 1446),
    "use_obb": True,
}


def clamp(val, minval, maxval):
    return max(minval, min(val, maxval))


def obb_to_regular_bbox(obb_coords):
    """Convert oriented bounding box (8 coordinates) to regular bounding box (4 coordinates)"""
    x_coords = obb_coords[::2]  # x coordinates at indices 0, 2, 4, 6
    y_coords = obb_coords[1::2]  # y coordinates at indices 1, 3, 5, 7

    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    return [min_x, min_y, max_x, max_y]


def calculate_obb_iou(obb1, obb2):
    """Calculate IoU between two oriented bounding boxes"""
    # For now, we'll use the regular bbox approximation for IoU calculation
    # This could be improved with proper polygon intersection calculations
    bbox1 = obb_to_regular_bbox(obb1)
    bbox2 = obb_to_regular_bbox(obb2)
    return calculate_iou(bbox1, bbox2)


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


def calculate_containment_ratio(box1, box2):
    """Calculate how much box1 is contained within box2 (0-1)"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)

    if x2_int <= x1_int or y2_int <= y1_int:
        return 0.0

    intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)

    return intersection_area / box1_area if box1_area > 0 else 0.0


def weighted_box_fusion(
    detections_list,
    iou_threshold=0.3,
    skip_box_threshold=0.001,
    containment_threshold=0.8,
):
    """
    Enhanced weighted box fusion that handles nested/enclosed boxes intelligently
    - If a smaller box is mostly contained in a larger box, choose based on confidence
    - Uses smart merging strategies for overlapping boxes
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

    # Filter by score threshold
    keep = all_scores >= skip_box_threshold
    all_boxes, all_scores, all_labels, all_weights = (
        all_boxes[keep],
        all_scores[keep],
        all_labels[keep],
        all_weights[keep],
    )

    if len(all_boxes) == 0:
        return {"boxes": [], "scores": [], "labels": []}

    # Group by class
    grouped = defaultdict(list)
    for i, label in enumerate(all_labels):
        grouped[label].append(i)

    final_boxes, final_scores, final_labels = [], [], []

    for cls, inds in grouped.items():
        boxes = all_boxes[inds]
        scores = all_scores[inds]
        weights = all_weights[inds]

        n = len(boxes)
        if n == 1:
            final_boxes.append(boxes[0])
            final_scores.append(scores[0])
            final_labels.append(cls)
            continue

        # Enhanced clustering with containment awareness
        used = np.zeros(n, dtype=bool)

        for i in range(n):
            if used[i]:
                continue

            cluster_candidates = [i]
            used[i] = True

            # Find all boxes that should be merged with box i
            for j in range(n):
                if used[j] or i == j:
                    continue

                iou = calculate_iou(boxes[i], boxes[j])
                containment_i_in_j = calculate_containment_ratio(boxes[i], boxes[j])
                containment_j_in_i = calculate_containment_ratio(boxes[j], boxes[i])

                # Decide whether to merge based on multiple criteria
                should_merge = False

                # Case 1: High IoU overlap
                if iou >= iou_threshold:
                    should_merge = True

                # Case 2: One box is mostly contained in another
                elif (
                    containment_i_in_j >= containment_threshold
                    or containment_j_in_i >= containment_threshold
                ):
                    # If one box is contained in another, choose based on confidence and size
                    score_diff = abs(scores[i] - scores[j])

                    # If confidence difference is small (< 0.1), prefer the larger box
                    if score_diff < 0.1:
                        should_merge = True
                    # If confidence difference is significant, merge but weight appropriately
                    elif score_diff < 0.3:
                        should_merge = True

                if should_merge:
                    cluster_candidates.append(j)
                    used[j] = True

            # Merge the cluster
            if len(cluster_candidates) == 1:
                # Single box, no merging needed
                final_boxes.append(boxes[cluster_candidates[0]])
                final_scores.append(scores[cluster_candidates[0]])
                final_labels.append(cls)
            else:
                # Smart merging of multiple boxes
                cluster_boxes = boxes[cluster_candidates]
                cluster_scores = scores[cluster_candidates]
                cluster_weights = weights[cluster_candidates]

                # Enhanced merging strategy
                if len(cluster_candidates) == 2:
                    # For pairs, use smart containment-aware merging
                    box1, box2 = cluster_boxes
                    score1, score2 = cluster_scores
                    weight1, weight2 = cluster_weights

                    containment_1_in_2 = calculate_containment_ratio(box1, box2)
                    containment_2_in_1 = calculate_containment_ratio(box2, box1)

                    # If one box is highly contained in another
                    if containment_1_in_2 >= containment_threshold:
                        # Box 1 is inside box 2
                        if (
                            score2 >= score1 - 0.15
                        ):  # If outer box has reasonable confidence
                            merged_box = box2  # Use the larger box
                            merged_score = max(score1, score2)
                        else:
                            merged_box = (
                                box1  # Use the inner box with higher confidence
                            )
                            merged_score = score1
                    elif containment_2_in_1 >= containment_threshold:
                        # Box 2 is inside box 1
                        if (
                            score1 >= score2 - 0.15
                        ):  # If outer box has reasonable confidence
                            merged_box = box1  # Use the larger box
                            merged_score = max(score1, score2)
                        else:
                            merged_box = (
                                box2  # Use the inner box with higher confidence
                            )
                            merged_score = score2
                    else:
                        # Regular weighted average merging
                        total_weight = np.sum(cluster_weights)
                        merged_box = (cluster_boxes * cluster_weights[:, None]).sum(
                            axis=0
                        ) / total_weight
                        merged_score = (
                            cluster_scores * cluster_weights
                        ).sum() / total_weight
                else:
                    # For clusters with 3+ boxes, use weighted average
                    total_weight = np.sum(cluster_weights)
                    merged_box = (cluster_boxes * cluster_weights[:, None]).sum(
                        axis=0
                    ) / total_weight
                    merged_score = (
                        cluster_scores * cluster_weights
                    ).sum() / total_weight

                final_boxes.append(merged_box)
                final_scores.append(merged_score)
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
    """Run regular YOLO detection strategy on the image"""
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


def run_obb_strategy(image, strategy):
    """Run OBB detection strategy on the image using get_sliced_prediction"""
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

    # Create SAHI-compatible model wrapper for OBB model
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    obb_sahi_model = UltralyticsDetectionModel(
        model_path=OBB_MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,  # Adjusted for OBB model
        device=device,
    )

    # Use get_sliced_prediction just like regular YOLO strategy
    result = get_sliced_prediction(
        processed_image,
        obb_sahi_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        perform_standard_pred=True,
    )

    boxes, scores, labels, obb_coords = [], [], [], []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        # Scale coordinates back to original image size
        scaled_box = [
            bbox.minx * scale_x,
            bbox.miny * scale_y,
            bbox.maxx * scale_x,
            bbox.maxy * scale_y,
        ]

        boxes.append(scaled_box)
        scores.append(pred.score.value)
        labels.append(pred.category.id)

        # Create rectangular OBB coordinates from regular box for visualization
        # Since we're using SAHI which returns regular bboxes, we create dummy OBB coords
        x1, y1, x2, y2 = scaled_box
        dummy_obb = [x1, y1, x2, y1, x2, y2, x1, y2]  # Rectangle as OBB
        obb_coords.append(dummy_obb)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "obb_coords": obb_coords,
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


def draw_results(image, detections, path, model_name="", strategy_name=""):
    """Draw detection results with support for both regular bboxes and OBB coordinates"""
    result_image = image.copy()

    for i, (box, score, label) in enumerate(
        zip(detections["boxes"], detections["scores"], detections["labels"])
    ):
        # Draw regular bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # If OBB coordinates are available, draw the oriented box as well
        if "obb_coords" in detections and i < len(detections["obb_coords"]):
            obb = detections["obb_coords"][i]
            # Convert to points for drawing polygon
            points = np.array([[int(obb[j]), int(obb[j + 1])] for j in range(0, 8, 2)])
            cv2.polylines(result_image, [points], True, (255, 0, 0), 3)  # Blue for OBB

        text = f"Room {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(
            result_image, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1
        )
        cv2.putText(
            result_image,
            text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3,
        )

    # Add model and strategy info to the image if provided
    if model_name and strategy_name:
        info_text = f"{model_name} - {strategy_name}"
        (info_tw, info_th), _ = cv2.getTextSize(
            info_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        cv2.rectangle(
            result_image, (10, 10), (info_tw + 20, info_th + 20), (0, 0, 0), -1
        )
        cv2.putText(
            result_image,
            info_text,
            (15, info_th + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

    cv2.imwrite(path, result_image)


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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run selective model inference with YOLO, OBB, and/or Detectron2 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_selective_models.py --yolo                    # Run only YOLO model
  python infer_selective_models.py --obb                     # Run only OBB model
  python infer_selective_models.py --detectron               # Run only Detectron2 model
  python infer_selective_models.py --yolo --obb              # Run YOLO and OBB models
  python infer_selective_models.py --yolo --obb --detectron  # Run all three models
  python infer_selective_models.py --all                     # Run all three models
        """,
    )

    parser.add_argument(
        "--yolo", action="store_true", help="Run YOLO model with tiling strategies"
    )
    parser.add_argument(
        "--obb", action="store_true", help="Run OBB (Oriented Bounding Box) model"
    )
    parser.add_argument("--detectron", action="store_true", help="Run Detectron2 model")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models (equivalent to --yolo --obb --detectron)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for detections (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--input-dir",
        default=IMAGE_DIR,
        help=f"Input directory containing images (default: {IMAGE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory for results (default: {OUTPUT_DIR})",
    )

    return parser.parse_args()


def ensemble_inference(
    use_yolo=True,
    use_obb=True,
    use_detectron=True,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    input_dir=IMAGE_DIR,
    output_dir=OUTPUT_DIR,
):
    print("\n[INFO] Starting inference with selected models:")
    print(f"  - YOLO: {'✓' if use_yolo else '✗'}")
    print(f"  - OBB: {'✓' if use_obb else '✗'}")
    print(f"  - Detectron2: {'✓' if use_detectron else '✗'}")
    print(f"  - Confidence threshold: {confidence_threshold}")
    print(f"  - Input directory: {input_dir}")
    print(f"  - Output directory: {output_dir}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    print(f"[INFO] Using device: {device}")

    # Initialize models based on selection
    yolo_model = None
    detectron2_predictor = None

    if use_yolo:
        print(f"[INFO] Loading YOLO model from {YOLO_MODEL_PATH}")
        yolo_model = UltralyticsDetectionModel(
            YOLO_MODEL_PATH,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    if use_detectron:
        print(f"[INFO] Loading Detectron2 model from {DETECTRON2_MODEL_PATH}")
        detectron2_predictor = load_detectron2_model(DETECTRON2_MODEL_PATH, device)

    # Process images
    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"[WARNING] No image files found in {input_dir}")
        return

    print(f"[INFO] Found {len(image_files)} images to process\n")

    for img_file in image_files:
        path = os.path.join(input_dir, img_file)
        image = cv2.imread(path)
        if image is None:
            print(f"[WARNING] Could not read image: {img_file}")
            continue

        print(f"[INFO] Processing {img_file}")
        all_results = []

        # Create individual result folder for this image
        img_name = os.path.splitext(img_file)[0]
        img_result_dir = os.path.join(save_dir, img_name)
        os.makedirs(img_result_dir, exist_ok=True)

        # Run YOLO model on tiling strategies
        if use_yolo and yolo_model:
            for strategy in TILING_STRATEGIES:
                try:
                    print(f"  [YOLO] Strategy: {strategy['name']}")
                    results = run_strategy(image.copy(), yolo_model, strategy)
                    all_results.append(results)
                    print(f"    Found {len(results['boxes'])} boxes")

                    # Save individual model prediction
                    individual_output_path = os.path.join(
                        img_result_dir, f"yolo-{strategy['name']}-prediction.jpg"
                    )
                    draw_results(
                        image.copy(),
                        results,
                        individual_output_path,
                        "YOLO",
                        strategy["name"],
                    )
                    print(f"    [✓] Saved YOLO prediction: {individual_output_path}")

                except Exception as e:
                    print(f"    [ERROR] YOLO {strategy['name']}: {e}")

        # Run OBB model
        if use_obb:
            try:
                print(f"  [OBB] Strategy: {OBB_STRATEGY['name']}")
                results = run_obb_strategy(image.copy(), OBB_STRATEGY)
                all_results.append(results)
                print(f"    Found {len(results['boxes'])} boxes")

                # Save individual model prediction
                individual_output_path = os.path.join(
                    img_result_dir, f"obb-{OBB_STRATEGY['name']}-prediction.jpg"
                )
                draw_results(
                    image.copy(),
                    results,
                    individual_output_path,
                    "OBB",
                    OBB_STRATEGY["name"],
                )
                print(f"    [✓] Saved OBB prediction: {individual_output_path}")

            except Exception as e:
                print(f"    [ERROR] OBB {OBB_STRATEGY['name']}: {e}")

        # Run Detectron2 model on full image
        if use_detectron and detectron2_predictor:
            try:
                print("  [Detectron2] Full image inference")
                d2_result = run_detectron2_inference(detectron2_predictor, image)
                all_results.append(d2_result)
                print(f"    Found {len(d2_result['boxes'])} boxes")

                # Save Detectron2 prediction
                d2_output_path = os.path.join(
                    img_result_dir, "detectron2-full_image-prediction.jpg"
                )
                draw_results(
                    image.copy(), d2_result, d2_output_path, "DETECTRON2", "full_image"
                )
                print(f"    [✓] Saved Detectron2 prediction: {d2_output_path}")

            except Exception as e:
                print(f"    [ERROR] Detectron2: {e}")

        # Fuse results if we have multiple models
        if len(all_results) > 1:
            fused = weighted_box_fusion(
                all_results, iou_threshold=0.3, skip_box_threshold=0.001
            )
            print(f"  [FUSION] → {len(fused['boxes'])} boxes after fusion")

            # Save the final fusion result in the individual image folder
            fusion_output_path = os.path.join(
                img_result_dir, "ensemble-fusion-prediction.jpg"
            )
            draw_results(image.copy(), fused, fusion_output_path, "ENSEMBLE", "fusion")
            print(f"  [✓] Saved fusion result: {fusion_output_path}")
        elif len(all_results) == 1:
            # If only one model was used, save it as the final result
            single_model_result = all_results[0]
            single_output_path = os.path.join(
                img_result_dir, "single-model-prediction.jpg"
            )
            draw_results(
                image.copy(), single_model_result, single_output_path, "SINGLE", "model"
            )
            print(f"  [✓] Saved single model result: {single_output_path}")
        else:
            print(f"  [WARNING] No valid predictions generated for {img_file}")

        print()  # Empty line for better readability

    print(f"[INFO] Processing complete! Results saved to: {save_dir}")


def main():
    args = parse_arguments()

    # Check if at least one model is selected
    if args.all:
        use_yolo = use_obb = use_detectron = True
    else:
        use_yolo = args.yolo
        use_obb = args.obb
        use_detectron = args.detectron

    # If no models are selected, show help and exit
    if not (use_yolo or use_obb or use_detectron):
        print("Error: You must specify at least one model to run.")
        print("Use --yolo, --obb, --detectron, or --all")
        print("Run with --help for more information.")
        return

    # Update global confidence threshold
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence

    # Run inference
    ensemble_inference(
        use_yolo=use_yolo,
        use_obb=use_obb,
        use_detectron=use_detectron,
        confidence_threshold=args.confidence,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
