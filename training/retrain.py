# retrain_corrected.py
import os
from ultralytics import YOLO


def get_latest_corrected_dataset():
    """Find the latest corrected dataset"""
    sahi_output_dir = "results/sahi_outputs"

    if not os.path.exists(sahi_output_dir):
        raise Exception(f"SAHI output directory not found: {sahi_output_dir}")

    # Get all timestamped folders
    subfolders = [
        os.path.join(sahi_output_dir, d)
        for d in os.listdir(sahi_output_dir)
        if os.path.isdir(os.path.join(sahi_output_dir, d))
        and d.replace("_", "").replace("-", "").isdigit()
    ]

    if not subfolders:
        raise Exception("No timestamped folders found in SAHI outputs.")

    # Find the latest folder with a corrected dataset
    latest_folder = None
    for folder in sorted(subfolders, key=os.path.getmtime, reverse=True):
        corrected_dataset_dir = os.path.join(folder, "corrected_dataset")
        data_yaml_path = os.path.join(corrected_dataset_dir, "data.yaml")

        if os.path.exists(data_yaml_path):
            latest_folder = folder
            break

    if not latest_folder:
        raise Exception("No corrected dataset found in any SAHI output folder.")

    return os.path.join(latest_folder, "corrected_dataset", "data.yaml")


def retrain_on_corrections():
    try:
        data_yaml_path = get_latest_corrected_dataset()
        print(f"[INFO] Using dataset: {data_yaml_path}")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    model = YOLO("model/model_v52.pt")  # start from best checkpoint

    # Change to the corrected_dataset directory so relative paths work
    corrected_dataset_dir = os.path.dirname(data_yaml_path)
    original_cwd = os.getcwd()

    try:
        os.chdir(corrected_dataset_dir)
        print(f"[INFO] Changed directory to: {corrected_dataset_dir}")

        model.train(
            data="data.yaml",  # Use relative path since we're in the correct directory
            epochs=50,
            imgsz=1024,
            batch=8,
            device="cpu",
            project=os.path.join(
                original_cwd, "exports"
            ),  # Use absolute path for exports
            name="fine_tuned_on_corrections",
            exist_ok=True,
        )
    finally:
        # Always return to original directory
        os.chdir(original_cwd)
        print(f"[INFO] Returned to: {original_cwd}")


if __name__ == "__main__":
    retrain_on_corrections()
