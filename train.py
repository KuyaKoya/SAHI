# train.py
from ultralytics import YOLO


def train():
    model = YOLO("yolov8m.yaml")  # or yolov8m/l for better performance
    model.train(
        data="data/floorplans-roboflow-yolov11/data.yaml",
        epochs=100,
        imgsz=1024,
        batch=8,
        project="exports",
        name="floorplans_yolov11",
        exist_ok=True,
    )


if __name__ == "__main__":
    train()
