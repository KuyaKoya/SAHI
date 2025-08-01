# train.py
from ultralytics import YOLO


def train():
    model = YOLO("yolov8n.yaml")  # or yolov8m/l for better performance
    model.train(
        data="data/floorplans-roboflow-yolov11/data.yaml",
        epochs=100,
        imgsz=1024,
        batch=8,
        project="exports",
        name="floorplans_yolov11",
        exist_ok=True,
    )


def improvement_train():
    model = YOLO("model/model_v52.pt")  # or the path of your best model
    model.train(
        data="data/floorplans-roboflow-yolov11/data.yaml",
        epochs=100,
        imgsz=1280,
        batch=8,
        workers=8,
        optimizer="Adam",
        patience=60,
        project="exports",
        name="floorplans_yolov11",
        exist_ok=True,
    )


if __name__ == "__main__":
    # train()
    improvement_train()
