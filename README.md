## Individual Inference

To run inference on a single image, use `infer_sahi.py`.

---
## Environment Setup

Recommended Python version: 3.9.6 (use a virtual environment)

Install required packages:

```bash
pip install ultralytics sahi opencv-python-headless pdf2image
```
## Project Flow

1. **Prepare Dataset**
   - Download your YOLOV11 dataset from Roboflow.
   - Place the dataset inside the `data` folder (e.g., `data/floorplans-roboflow-yolov11`).

2. **Training**
   - Run `train.py` to train your model using the dataset in the `data` folder.

3. **Inference**
   - Use `infer_sahi_iteration.py` to run inference on images in `data/floorplans-roboflow-yolov11/test/images`.
   - Results will be saved in the `results/sahi_outputs` folder with timestamped filenames.

4. **Results Folder**
   - If the `results` folder does not exist, it will be created automatically by the scripts.

---
Ensure all dependencies are installed and paths are correctly set before running the scripts.
