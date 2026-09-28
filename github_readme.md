# Real-Time Traffic Object Detection for Autonomous Vehicles (AV)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-000000.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An automated end-to-end computer vision and object detection pipeline engineered using **YOLOv8** to evaluate, fine-tune, and deploy vehicle detection systems across diverse highway traffic environments. 

This repository contains a technical report evaluating single-stage vs. two-stage detectors (**YOLO**, **SSD**, **Faster R-CNN**), automated dataset labeling tools, visual bounding box generators, and a few-shot transfer learning pipeline optimized for safety-critical Autonomous Vehicle (AV) systems.

---

## 📌 Project Overview & Key Features

* **High-Throughput Perception:** Leverages modern anchor-free single-stage detection (YOLOv8) achieving sub-10ms latency for safety-critical collision avoidance.
* **Automated Labeling Pipeline:** Converts raw traffic photos into normalized YOLO-formatted bounding box annotations (`<class> <x_center> <y_center> <width> <height>`) automatically.
* **Visual Bounding Box Renderer:** Generates bounding box overlays, confidence scores, and multi-class labels (`car`, `bus`, `truck`, `motorcycle`).
* **Few-Shot Transfer Learning:** Integrates backbone layer freezing and heavy data augmentations (Mosaic, HSV color jitter, scale jitter, horizontal flipping) to achieve convergence on small custom datasets without overfitting.
* **Edge Deployment Ready:** Includes code templates to export trained checkpoints into hardware-accelerated runtimes (**ONNX**, **TensorRT**).

---

## 📁 Repository Directory Structure

```text
traffic-yolo-av/
├── .venv1/                    # Python virtual environment
├── images/                    # Raw input traffic photos
├── data_split/                # Generated YOLO dataset structure
│   ├── images/                # Train and Validation images
│   │   ├── train/
│   │   └── val/
│   └── labels/                # Normalized coordinate annotations
│       ├── train/
│       └── val/
├── output_visualized/         # Rendered images with visual bounding boxes
├── dataset.yaml               # YOLO dataset class map configuration
├── pipeline.py                # Main end-to-end execution pipeline
├── requirements.txt           # Python environment dependencies
└── README.md                  # Project documentation
```

---

## 📊 Comparative Analysis Matrix

Evaluating detection model paradigms for real-time autonomous driving applications:

| Metric / Parameter | Faster R-CNN | SSD (Single Shot Detector) | YOLO (YOLOv8 / YOLOv11) |
| :--- | :--- | :--- | :--- |
| **Architecture** | Two-Stage (RPN + ROI Pooling) | Single-Stage (Multi-scale maps) | Single-Stage (Unified Grid Regression) |
| **Mean Avg Precision (mAP)** | **High** | **Moderate to High** | **High** (Surpasses Faster R-CNN) |
| **Inference Speed (FPS)** | ~5–15 FPS | ~30–60 FPS | **~60–150+ FPS** |
| **Latency** | > 60 ms | ~15–33 ms | **< 10 ms** |
| **Small/Occluded Objects** | Excellent | Prone to missed detections | High (via PANet & anchor-free head) |
| **AV Real-Time Suitability** | ❌ Unsuitable | ⚠️ Acceptable (Edge devices) | ✅ **Optimal Choice** |

---

## ⚡ Quick Start & Setup

### 1. Prerequisites & Virtual Environment

Ensure Python 3.8+ is installed. Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/your-username/traffic-yolo-av.git
cd traffic-yolo-av

# Create virtual environment
python -m venv .venv1

# Activate environment
# On Windows:
.venv1\Scripts\activate
# On macOS/Linux:
source .venv1/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### Running the End-to-End Pipeline

1. Place your input traffic images inside the `images/` directory (e.g., `img1.jpg`, `img2.jpg`, `img3.jpg`).
2. Run the main execution pipeline:

```bash
python pipeline.py
```

`pipeline.py` executes the following steps automatically:
1. **Directory Setup:** Creates `data_split/` and `output_visualized/` directories.
2. **Auto-Annotation:** Detects vehicles and writes formatted `.txt` labels to `data_split/labels/`.
3. **Bounding Box Drawing:** Renders bounding box outlines with class names and confidence levels saved in `output_visualized/`.
4. **Model Fine-Tuning:** Trains YOLOv8 with backbone layer freezing (`freeze=10`) and heavy data augmentations.
5. **Output Visualization:** Displays processed images via Matplotlib.

---

## ⚙️ Hardware Acceleration & Export

For deployment on embedded platforms (such as NVIDIA Jetson AGX Orin or Tesla Full Self-Driving chips), export the fine-tuned model checkpoint (`best.pt`) to ONNX or TensorRT format:

```python
from ultralytics import YOLO

# Load the trained model checkpoint
model = YOLO('runs/train/traffic_few_shot/weights/best.pt')

# Export to ONNX
model.export(format='onnx', dynamic=True)

# Export to TensorRT Engine (requires NVIDIA GPU with TensorRT installed)
model.export(format='engine', device=0)
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.