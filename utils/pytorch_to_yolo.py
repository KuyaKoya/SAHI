from ultralytics import YOLO
import torch
import argparse
import os


def convert_yolov8_pth_to_pt(pth_path, yaml_path, output_path):
    # Load model from config
    model = YOLO(yaml_path)  # e.g., yolov8m.yaml

    # Load weights
    state_dict = torch.load(pth_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]  # If it's a wrapped checkpoint

    model.model.load_state_dict(state_dict, strict=False)

    # Save as .pt file
    model.save(output_path)
    print(f"[✓] Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv8 .pth to .pt")
    parser.add_argument("--pth", required=True, help="Path to .pth file")
    parser.add_argument(
        "--yaml", required=True, help="Path to YOLOv8 .yaml model config"
    )
    parser.add_argument("--pt", required=True, help="Output .pt path")
    args = parser.parse_args()

    convert_yolov8_pth_to_pt(args.pth, args.yaml, args.pt)
