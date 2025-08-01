## SAHI Object Detection Project

This project provides a comprehensive workflow for object detection using SAHI (Slicing Aided Hyper Inference) with YOLOv11, including training, inference, manual annotation correction, and dataset preparation.

---
## Environment Setup

**Recommended Python version**: 3.9.6 (use a virtual environment)

### Installation

1. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies**:
   
   **Option A: Using requirements file** (recommended):
   ```bash
   pip install -r requirements-1.txt
   ```
   
   **Option B: Manual installation**:
   ```bash
   # Install PyTorch first (required for Detectron2)
   pip install "torch>=1.8.0" "torchvision>=0.9.0"
   
   # Install other core libraries
   pip install ultralytics sahi opencv-python-headless pdf2image pillow
   
   # Install Detectron2 (for Apple Silicon, use these flags)
   CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
   ```

### Dependencies
- **PyTorch** (≥1.8.0) - Deep learning framework
- **TorchVision** (≥0.9.0) - Computer vision utilities
- **Ultralytics** - YOLOv11 implementation
- **SAHI** - Slicing Aided Hyper Inference
- **Detectron2** - Advanced object detection framework
- **OpenCV** - Computer vision operations
- **Pillow** - Image processing
- **PDF2Image** - PDF to image conversion

**Note**: For Apple Silicon (M1/M2) users, ensure you use the compilation flags shown above when installing Detectron2.

## Project Structure

```
SAHI/
├── data/                           # Training datasets
│   └── floorplans-roboflow-yolov11/   # Example dataset
├── exports/                        # Trained model outputs
│   ├── fine_tuned_on_corrections/     # Retrained models
│   ├── floorplans_yolov11/           # Base model exports
│   └── ...
├── test_images/                    # Images for inference
├── model/                          # Trained model files (.pt, .pth)
├── results/
│   ├── sahi_outputs/              # Single-strategy inference results
│   └── sahi_ensemble_outputs/     # Multi-strategy ensemble results
├── inference/                      # Inference scripts
│   ├── infer_sahi.py              # Basic SAHI inference
│   ├── infer_multi_sahi.py        # Ensemble inference
│   └── ...
├── training/                       # Training scripts
│   ├── train.py                   # Initial training
│   └── retrain.py                 # Retraining with corrections
├── utils/                          # Utility scripts
│   └── convert_corrections_to_yolo.py  # Convert annotations to YOLO format
├── htil/                          # Human-in-the-loop tools
│   └── annotate_gui.py            # Manual annotation GUI
├── pdfs/                          # PDF files for processing
├── requirements-1.txt             # Python dependencies
└── README.md
```

## Complete Workflow

### 1. **Prepare Dataset**
   - Download your YOLOv11 dataset from Roboflow
   - Place the dataset inside the `data` folder (e.g., `data/floorplans-roboflow-yolov11`)

### 2. **Training**
   - Run `training/train.py` to train your model using the dataset in the `data` folder
   - Trained models will be saved in `exports/` folder

### 3. **Inference Options**

#### Single Strategy Inference
- **`inference/infer_sahi.py`** - Basic SAHI inference on single images
- **`inference/infer_sahi_iteration.py`** - Batch inference with dynamic tiling
- **`inference/infer_sahi_htil_iteration.py`** - Enhanced inference with image resizing to (2048, 1446)
- **`inference/infer_clahe_sahi.py`** - SAHI inference with CLAHE preprocessing
- **`inference/infer_normal_iteration.py`** - Standard inference without SAHI

#### Multi-Strategy Ensemble Inference
- **`inference/infer_multi_sahi.py`** - Advanced ensemble inference using multiple tiling strategies
- **`inference/infer_multi_model_multi_sahi_with_clahe.py`** - Multi-model ensemble with CLAHE enhancement
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
python training/retrain.py
```

## Advanced Features

### Multi-Model Ensemble
The project supports multiple model types:
- **YOLOv11 models** (Ultralytics)
- **Detectron2 models** (Facebook Research)
- Custom trained models in both frameworks

### Enhanced Inference Strategies
The `inference/infer_multi_sahi.py` and related scripts use multiple strategies:
- Small tiles (512x512)
- Medium tiles (768x768) 
- Large tiles (1024x1024)
- Dynamic adaptive tiling
- Resized image processing
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Grayscale CLAHE enhancement
- Multi-model ensemble voting

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
| `training/train.py` | Initial model training | Dataset in `data/` | Model in `exports/` |
| `training/retrain.py` | Model retraining | Corrected dataset | Updated model |
| `inference/infer_sahi_htil_iteration.py` | Enhanced SAHI inference | `test_images/` | Timestamped results |
| `inference/infer_multi_sahi.py` | Ensemble inference | `test_images/` | Multi-strategy results |
| `inference/infer_clahe_sahi.py` | CLAHE + SAHI inference | `test_images/` | Enhanced results |
| `inference/infer_multi_model_multi_sahi_with_clahe.py` | Multi-model ensemble | `test_images/` | Combined model results |
| `htil/annotate_gui.py` | Manual annotation | Latest inference | Corrections JSON |
| `utils/convert_corrections_to_yolo.py` | Format conversion | Corrections | YOLO dataset |
| `utils/pytorch_to_yolo.py` | Model conversion | PyTorch model | YOLO format |

## Troubleshooting

### Common Installation Issues

**Detectron2 installation fails on Apple Silicon (M1/M2)**:
```bash
CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

**PyTorch not found during Detectron2 installation**:
- Ensure PyTorch is installed first: `pip install torch torchvision`
- Use `--no-build-isolation` flag when installing Detectron2

**ModuleNotFoundError for torch during setup**:
- Install dependencies in the correct order (see requirements-1.txt)
- PyTorch must be installed before Detectron2

## Tips

1. **For best results**: Use ensemble inference (`inference/infer_multi_sahi.py`) for challenging images
2. **For speed**: Use single strategy inference (`inference/infer_sahi_htil_iteration.py`)
3. **For enhanced contrast**: Use CLAHE preprocessing (`inference/infer_clahe_sahi.py`)
4. **For multiple models**: Use multi-model ensemble (`inference/infer_multi_model_multi_sahi_with_clahe.py`)
5. **Manual corrections**: Focus on missed detections and false positives
6. **Iterative improvement**: Retrain → Infer → Correct → Repeat
7. **PDF processing**: Place PDFs in `pdfs/` folder and use PDF2Image conversion
8. **Virtual environment**: Always use a virtual environment to avoid dependency conflicts

## Model Support

This project supports multiple object detection frameworks:
- **YOLOv11** (Ultralytics) - Fast and efficient
- **Detectron2** (Facebook Research) - High accuracy
- **Custom models** in both frameworks

---
**Note**: Ensure all dependencies are installed correctly using the provided `requirements-1.txt` file. For Apple Silicon users, pay special attention to the Detectron2 installation instructions.
