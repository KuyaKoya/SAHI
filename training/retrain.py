# retrain_corrected.py
from ultralytics import YOLO


def retrain_on_corrections():
    model = YOLO("model/model_v52.pt")  # start from best checkpoint

    model.train(
        data="corrected_dataset/data.yaml",
        epochs=50,
        imgsz=1024,
        batch=8,
        device="cuda" if YOLO("yolov8m").device.type == "cuda" else "cpu",
        project="exports",
        name="fine_tuned_on_corrections",
        exist_ok=True,
    )


if __name__ == "__main__":
    retrain_on_corrections()
