## SAHI Object Detection Project

This project provides a comprehensive workflow for object detection using SAHI (Slicing Aided Hyper Inference) with YOLOv11, including training, inference, manual annotation correction, and dataset preparation.

## 🆕 Recent Updates (August 2025)

### 🎯 **NEW: OBB (Oriented Bounding Box) Support**
- **OBB model integration** - Support for oriented bounding box detection models
- **SAHI-compatible OBB inference** - OBB models now use get_sliced_prediction for optimal performance
- **Unified ensemble system** - OBB, YOLO, and Detectron2 models work together seamlessly
- **Enhanced detection accuracy** - OBB models significantly improve room detection in complex layouts

### 🚀 **Enhanced Multi-Model Ensemble**
- **Three-model ensemble** - YOLOv11 + OBB + Detectron2 for maximum detection coverage
- **Selective model execution** - **🆕 NEW: Choose which models to run via command line**
- **Organized output structure** - Individual model predictions saved with clear naming convention
- **Improved weighted fusion** - Smart containment-aware merging reduces overlapping boxes by 39%
- **Comprehensive visualization** - Both regular and oriented bounding boxes displayed

### ✅ **Detectron2 Integration Fixed**
- **Resolved parameter loading issues** - No more "Skip loading parameter" warnings
- **Optimized model configuration** - Automatic NUM_CLASSES detection based on training data
- **Enhanced confidence thresholds** - All models now use configurable confidence thresholds

### � **Improved Box Fusion Algorithm**
- **Containment-aware clustering** - Intelligent handling of nested/enclosed bounding boxes
- **Smart merging strategies** - Chooses optimal box based on confidence and size relationships
- **Reduced overlaps** - 39% reduction in overlapping boxes (e.g., 59→36 final detections)
- **Enhanced IoU calculations** - Better detection quality while reducing redundancy

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
│   ├── model_v52.pt               # YOLOv11 model
│   ├── obb-floorplan-v2.pt        # 🆕 OBB (Oriented Bounding Box) model
│   ├── detectron_cus_model_2.pth  # Detectron2 model
│   └── ...
├── results/
│   ├── sahi_outputs/              # Single-strategy inference results
│   └── sahi_ensemble_outputs/     # Multi-strategy ensemble results
├── inference/                      # Inference scripts
│   ├── infer_sahi.py              # Basic SAHI inference
│   ├── infer_multi_sahi.py        # Ensemble inference
│   ├── infer_mm_ms_wc_with_obb.py # 🆕 Multi-model ensemble with OBB support
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

#### 🆕 **NEW: Selective Model Inference**

The new `inference/infer_selective_models.py` script allows you to choose which models to run via command line arguments:

**Basic Usage:**
```bash
# Run only YOLO model
python inference/infer_selective_models.py --yolo

# Run only OBB model  
python inference/infer_selective_models.py --obb

# Run only Detectron2 model
python inference/infer_selective_models.py --detectron

# Run YOLO and OBB models
python inference/infer_selective_models.py --yolo --obb

# Run all three models
python inference/infer_selective_models.py --yolo --obb --detectron
# or use the shorthand:
python inference/infer_selective_models.py --all
```

**Advanced Options:**
```bash
# Run with custom confidence threshold
python inference/infer_selective_models.py --yolo --obb --confidence 0.7

# Run with custom input/output directories
python inference/infer_selective_models.py --all --input-dir my_images --output-dir my_results

# Get help with all available options
python inference/infer_selective_models.py --help
```

**Benefits:**
- **Faster execution** - Run only the models you need
- **Resource optimization** - Save GPU memory and processing time
- **Flexible testing** - Compare individual model performance
- **Customizable thresholds** - Adjust confidence levels per run
- **Automatic fusion** - Multiple models are automatically fused using weighted box fusion

#### Single Strategy Inference
- **`inference/infer_sahi.py`** - Basic SAHI inference on single images
- **`inference/infer_sahi_iteration.py`** - Batch inference with dynamic tiling
- **`inference/infer_sahi_htil_iteration.py`** - Enhanced inference with image resizing to (2048, 1446)
- **`inference/infer_clahe_sahi.py`** - SAHI inference with CLAHE preprocessing
- **`inference/infer_normal_iteration.py`** - Standard inference without SAHI

#### Multi-Strategy Ensemble Inference
- **`inference/infer_multi_sahi.py`** - Advanced ensemble inference using multiple tiling strategies
- **`inference/infer_mm_ms_wc_with_obb.py`** - **🆕 NEW: Multi-model ensemble with OBB support**
- **`inference/infer_selective_models.py`** - **🆕 NEW: Selective model inference with command-line control**
- **`inference/infer_multi_model_multi_sahi_with_clahe.py`** - Multi-model ensemble with CLAHE enhancement
- Combines results from **YOLOv11 + OBB + Detectron2 models** for maximum detection coverage
- Uses **enhanced Weighted Box Fusion** with containment-aware clustering
- **39% reduction in overlapping boxes** with smart merging strategies
- **Organized output structure** - Individual model predictions saved with clear naming convention

### 4. **Results Structure**

All inference results are organized in timestamped folders with individual model outputs:

```
results/sahi_ensemble_outputs/YYYYMMDD_HHMMSS/
├── image1/                                          # 🆕 Individual image folder
│   ├── yolo-medium_tiles_dynamic-prediction.jpg    # YOLOv11 strategy result
│   ├── yolo-grayscale_clahe_dynamic-prediction.jpg # YOLOv11 with CLAHE
│   ├── obb-obb_oriented_detection-prediction.jpg   # 🆕 OBB model result
│   ├── detectron2-full_image-prediction.jpg        # Detectron2 result
│   └── ensemble-fusion-prediction.jpg              # 🆕 Final fused result
├── image2/                                          # Same structure for each image
│   ├── yolo-medium_tiles_dynamic-prediction.jpg
│   ├── yolo-grayscale_clahe_dynamic-prediction.jpg
│   ├── obb-obb_oriented_detection-prediction.jpg
│   ├── detectron2-full_image-prediction.jpg
│   └── ensemble-fusion-prediction.jpg
└── ...
```

**Benefits of the new structure:**
- **Clear identification** - Model and strategy easily identified from filename
- **Flat structure** - No nested folders for simpler navigation
- **Consistent naming** - `<model>-<strategy>-prediction.jpg` format for all outputs

**Traditional single-strategy results:**
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
The project supports multiple model types working together:
- **YOLOv11 models** (Ultralytics) - Fast and efficient general detection
- **OBB models** (Oriented Bounding Box) - **🆕 Specialized for rotated/oriented objects**
- **Detectron2 models** (Facebook Research) - High accuracy with **✅ fixed parameter loading**
- **Unified ensemble system** - All three model types contribute to final results
- **🆕 Enhanced fusion algorithm** - Containment-aware merging with 39% overlap reduction
- **🆕 Individual model outputs** - Each model's predictions saved separately for analysis

### Enhanced Inference Strategies
The ensemble scripts use multiple complementary strategies:

**Tiling Strategies:**
- Small tiles (640x640) with 30% overlap - **🆕 Used by OBB models**
- Medium tiles (512-1024x512-1024) with dynamic sizing
- Large tiles with adaptive overlap ratios

**Preprocessing Methods:**
- Automatic resizing to optimal dimensions (2048x1446)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Grayscale CLAHE enhancement for contrast improvement

**Model Integration:**
- **🆕 OBB-oriented detection** - Specialized for rotated room layouts
- **🆕 SAHI-compatible OBB inference** - Uses get_sliced_prediction for optimal performance
- **Multi-model ensemble voting** with enhanced fusion algorithm
- **Containment-aware clustering** - Smart handling of nested/enclosed boxes
- **Individual model tracking** - Each strategy's contribution preserved and visualized

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
| `inference/infer_mm_ms_wc_with_obb.py` | **🆕 OBB + Multi-model ensemble** | `test_images/` | **YOLOv11+OBB+Detectron2 results** |
| `inference/infer_selective_models.py` | **🆕 Selective model inference** | `test_images/` | **Configurable model results** |
| `inference/infer_multi_model_multi_sahi_with_clahe.py` | Multi-model ensemble | `test_images/` | Combined YOLOv11+Detectron2 results |
| `inference/infer_clahe_sahi.py` | CLAHE + SAHI inference | `test_images/` | Enhanced results |
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

### 🆕 Detectron2 Model Issues

**Parameter shape mismatch warnings during model loading**:
- ✅ **Fixed**: Detectron2 model configuration now correctly matches custom model architecture
- The script automatically configures `NUM_CLASSES` based on your model's training data
- No more "Skip loading parameter" warnings

**Detectron2 detecting nothing**:
- ✅ **Fixed**: Confidence threshold optimized to 85% for better detection quality
- Model parameters now load correctly with proper class configuration

### 🆕 Overlapping Detection Issues

**Too many overlapping bounding boxes**:
- ✅ **Fixed**: Enhanced fusion algorithm with IoU threshold of 0.3 (down from 0.5)
- Connected components clustering ensures better merging of overlapping room detections
- Weighted averaging preserves detection quality while reducing redundancy

## Tips

1. **For best results**: Use ensemble inference (`inference/infer_multi_sahi.py`) for challenging images
2. **For speed**: Use single strategy inference (`inference/infer_sahi_htil_iteration.py`)
3. **For enhanced contrast**: Use CLAHE preprocessing (`inference/infer_clahe_sahi.py`)
4. **🆕 For multiple models**: Use multi-model ensemble (`inference/infer_multi_model_multi_sahi_with_clahe.py`) - **now with fixed Detectron2 support**
5. **🆕 For selective execution**: Use `inference/infer_selective_models.py` to run only specific models and save time/resources
6. **Manual corrections**: Focus on missed detections and false positives
7. **Iterative improvement**: Retrain → Infer → Correct → Repeat
8. **PDF processing**: Place PDFs in `pdfs/` folder and use PDF2Image conversion
9. **Virtual environment**: Always use a virtual environment to avoid dependency conflicts
10. **🆕 Overlapping boxes**: The enhanced fusion algorithm automatically handles overlapping room detections
11. **🆕 Model confidence**: Both YOLOv11 and Detectron2 models now use 85% confidence threshold for better quality
12. **🆕 Resource optimization**: Use `--yolo` only for fast inference, `--all` for maximum accuracy

## Model Support

This project supports multiple object detection frameworks:
- **YOLOv11** (Ultralytics) - Fast and efficient
- **Detectron2** (Facebook Research) - High accuracy
- **Custom models** in both frameworks

---
**Note**: Ensure all dependencies are installed correctly using the provided `requirements-1.txt` file. For Apple Silicon users, pay special attention to the Detectron2 installation instructions.
