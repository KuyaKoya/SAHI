## SAHI Object Detection Project

This project provides a comprehensive workflow for object detection using SAHI (Slicing Aided Hyper Inference) with YOLOv11, including training, inference, manual annotation correction, and dataset preparation.

---
## Environment Setup

Recommended Python version: 3.9.6 (use a virtual environment)

Install required packages:

```bash
pip install ultralytics sahi opencv-python-headless pdf2image pillow tkinter
```

## Project Structure

```
SAHI/
├── data/                           # Training datasets
├── test_images/                    # Images for inference
├── model/                          # Trained model files
├── results/
│   ├── sahi_outputs/              # Timestamped inference results
│   └── sahi_ensemble_outputs/     # Ensemble inference results
├── corrections/json/               # Legacy correction files
├── utils/
│   └── convert_corrections_to_yolo.py  # Convert annotations to YOLO format
├── htil/
│   └── annotate_gui.py            # Manual annotation GUI
├── infer_*.py                     # Various inference scripts
├── train.py                       # Model training
└── README.md
```

## Complete Workflow

### 1. **Prepare Dataset**
   - Download your YOLOv11 dataset from Roboflow
   - Place the dataset inside the `data` folder (e.g., `data/floorplans-roboflow-yolov11`)

### 2. **Training**
   - Run `train.py` to train your model using the dataset in the `data` folder
   - Trained models will be saved in `exports/` folder

### 3. **Inference Options**

#### Single Strategy Inference
- **`infer_sahi.py`** - Basic SAHI inference on single images
- **`infer_sahi_iteration.py`** - Batch inference with dynamic tiling
- **`infer_sahi_htil_iteration.py`** - Enhanced inference with image resizing to (2048, 1446)

#### Multi-Strategy Ensemble Inference
- **`infer_multi_sahi.py`** - Advanced ensemble inference using multiple tiling strategies
- Combines results from different tile sizes and preprocessing methods
- Uses Weighted Box Fusion for improved detection accuracy

### 4. **Results Structure**

All inference results are organized in timestamped folders:

```
results/sahi_outputs/YYYYMMDD_HHMMSS/
├── image1.jpg.png              # Visual results with bounding boxes
├── image2.jpg.png
├── feedback_json/              # Machine-readable predictions
│   ├── image1_pred.json
│   └── image2_pred.json
├── corrections/                # Manual annotations (created by GUI)
│   ├── image1.json
│   └── image2.json
└── corrected_dataset/          # YOLO-format dataset (after conversion)
    ├── data.yaml               # YOLO dataset configuration
    ├── images/                 # Source images
    │   ├── image1.jpg
    │   └── image2.jpg
    └── labels/                 # YOLO format labels
        ├── image1.txt
        └── image2.txt
```

### 5. **Manual Annotation & Correction**

Use the annotation GUI to manually correct or add annotations:

```bash
python htil/annotate_gui.py
```

**Features:**
- Load images from latest inference results
- Draw/edit bounding boxes with mouse
- Save corrections automatically
- Navigate with arrow keys
- Auto-saves when switching images

**Controls:**
- **Left Click + Drag**: Draw new bounding box
- **Right Click**: Delete existing bounding box
- **Left/Right Arrow**: Navigate between images
- **S Key**: Manual save
- **Auto-save**: When switching images

### 6. **Convert to Training Dataset**

Convert manual corrections to YOLO format:

```bash
python utils/convert_corrections_to_yolo.py
```

This automatically:
- Finds the latest inference results
- Converts JSON annotations to YOLO format
- Creates a ready-to-use dataset with `data.yaml`
- Saves everything in the same timestamped folder

### 7. **Retrain with Corrections**

Use the generated dataset for retraining:

```bash
python retrain.py
```

## Advanced Features

### Ensemble Inference
The `infer_multi_sahi.py` script uses multiple strategies:
- Small tiles (512x512)
- Medium tiles (768x768) 
- Large tiles (1024x1024)
- Dynamic adaptive tiling
- Resized image processing
- CLAHE enhancement
- Grayscale CLAHE

### Image Preprocessing
- Automatic resizing to optimal dimensions (2048x1446)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Multiple color space processing

### Smart File Organization
- All outputs from a single run are grouped in timestamped folders
- No file conflicts between different inference runs
- Easy to track which corrections belong to which inference

## Scripts Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `train.py` | Initial model training | Dataset in `data/` | Model in `exports/` |
| `infer_sahi_htil_iteration.py` | Enhanced inference | `test_images/` | Timestamped results |
| `infer_multi_sahi.py` | Ensemble inference | `test_images/` | Multi-strategy results |
| `htil/annotate_gui.py` | Manual annotation | Latest inference | Corrections JSON |
| `utils/convert_corrections_to_yolo.py` | Format conversion | Corrections | YOLO dataset |
| `retrain.py` | Model retraining | Corrected dataset | Updated model |

## Tips

1. **For best results**: Use ensemble inference (`infer_multi_sahi.py`) for challenging images
2. **For speed**: Use single strategy inference (`infer_sahi_htil_iteration.py`)
3. **Manual corrections**: Focus on missed detections and false positives
4. **Iterative improvement**: Retrain → Infer → Correct → Repeat

---
Ensure all dependencies are installed and paths are correctly set before running the scripts.
