# Campus Cleanliness Scoring System

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.12%20|%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/YOLOv8-Ultralytics-emerald.svg)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-teal.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-27%20Passed-brightgreen.svg)]()

> **B.Tech Deep Learning Lab Capstone Project (Academic Session 2026–27, Semester V)**  
> **Institution:** Ramdeobaba University, Nagpur  
> **Department:** Department of Electronics Engineering  
> **Course Supervisor:** Prof. V. R. Gupta  
> **Team Members:**  
> - **Arya Mohod** (Roll No. 06) — Dataset preparation, annotation conversion, preprocessing & dataset organization  
> - **Darsh Kadu** (Roll No. 11) — YOLOv8n model setup, transfer learning, training & experiment tracking  
> - **Hargun Singh Jaggi** (Roll No. 16) — Model evaluation, metrics, error analysis & comparison experiments  
> - **Ayaan Ahmed** (Roll No. 09) — Cleanliness score algorithm, Flask prototype, integration & testing  

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture & Workflow](#2-system-architecture--workflow)
3. [Project Directory Structure](#3-project-directory-structure)
4. [Environment Setup & Installation](#4-environment-setup--installation)
5. [Dataset Preparation (TACO to YOLO)](#5-dataset-preparation-taco-to-yolo)
6. [Model Fine-Tuning (YOLOv8n)](#6-model-fine-tuning-yolov8n)
7. [Cleanliness Scoring Algorithm](#7-cleanliness-scoring-algorithm)
8. [Evaluation & Baseline Comparisons](#8-evaluation--baseline-comparisons)
9. [Real Campus Testing](#9-real-campus-testing)
10. [Running the Flask Web Application](#10-running-the-flask-web-application)
11. [Running Unit & Integration Tests](#11-running-unit--integration-tests)
12. [GPU vs. CPU Execution](#12-gpu-vs-cpu-execution)
13. [Common Errors & Troubleshooting](#13-common-errors--troubleshooting)
14. [Limitations & Future Enhancements](#14-limitations--future-enhancements)

---

## 1. Project Overview

Maintaining visual cleanliness across educational campuses traditionally relies on periodic, manual visual surveys. These manual inspections suffer from subjectivity, delayed reporting, and inconsistent standards between different inspectors.

The **Campus Cleanliness Scoring System** provides an automated, objective, computer-vision-based audit tool:
- Takes a photograph of a campus location (corridor, garden, canteen, library walkway, parking lot).
- Localizes visible trash/litter items using a fine-tuned **YOLOv8n** single-class object detector.
- Draws high-visibility bounding boxes and confidence badges on the output image.
- Calculates an explainable, deterministic **Cleanliness Score (0–100)** incorporating both the discrete count of detected items and their total relative area footprint.
- Presents a responsive, professional side-by-side comparison in a Flask web application.

---

## 2. System Architecture & Workflow

```
[ Campus Photograph Upload / Camera Capture ]
                    │
                    ▼
[ Security & Decodability Validation (PIL/OpenCV) ]
                    │
                    ▼
[ Fine-Tuned YOLOv8n Object Detector (models/best.pt) ]
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
[ Detected Litter Count ]   [ Bounding Box Footprint Area ]
        │                       │
        ▼                       ▼
[ Count Penalty: 1-e^(-α·N) ] [ Area Penalty: min(1, Ratio/Scale) ]
        └───────────┬───────────┘
                    ▼
[ Composite Cleanliness Score (0 - 100) ]
                    │
                    ▼
[ Annotated Image Rendering (static/results/) ]
                    │
                    ▼
[ Interactive Flask UI (Side-by-Side Inspection & Metrics) ]
```

---

## 3. Project Directory Structure

```
campus-cleanliness-scoring/
├── app.py                     # Primary Flask entry point & application factory
├── config.yaml                # Master configuration file (hyperparameters & paths)
├── requirements.txt           # Verified Python package dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules for checkpoints and cache
├── README.md                  # Comprehensive technical documentation
│
├── data/
│   ├── raw/                   # TACO original dataset
│   │   ├── annotations.json   # 1,500 TACO images and 4,784 annotations
│   │   └── images/            # Downloaded raw images
│   ├── converted/             # Intermediate YOLO label conversions & manifests
│   ├── processed/             # Ultralytics-compatible dataset splits
│   │   ├── dataset.yaml       # Dataset definition for YOLOv8
│   │   ├── train/             # 70% Training images & labels
│   │   ├── val/               # 15% Validation images & labels
│   │   └── test/              # 15% Held-out test images & labels
│   └── campus_test/           # Real-world campus photographs for field auditing
│
├── models/
│   ├── README.md              # Checkpoint documentation and distinctions
│   ├── yolov8n.pt             # Pretrained baseline weights (MS COCO 80-class)
│   ├── best.pt                # Fine-tuned checkpoint (TACO single-class litter)
│   └── last.pt                # Checkpoint from final training epoch
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_dataset.py     # Downloads TACO annotations and images
│   │   ├── convert_annotations.py  # Converts COCO bboxes to YOLO normalized format
│   │   ├── split_dataset.py        # 70/15/15 idempotent split generator
│   │   └── validate_dataset.py     # Dataset integrity validator
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py                # YOLOv8n fine-tuning pipeline
│   │   ├── predict.py              # Real inference engine & bounding box drawer
│   │   ├── evaluate.py             # Quantitative evaluation on held-out test split
│   │   ├── compare_models.py       # Baseline vs fine-tuned benchmark comparison
│   │   └── batch_test_campus.py    # Batch testing script for real campus photos
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── cleanliness_score.py    # Dual-factor (count + area) scoring algorithm
│   │   └── score_config.py         # Config dataclass with weight boundary checks
│   └── web/
│       ├── __init__.py
│       ├── routes.py               # Flask endpoints (/, /analyze, /health, /api/metrics)
│       └── services.py             # Upload validation & inference orchestration
│
├── templates/
│   ├── index.html             # Drag-and-drop homepage with campus photo demo
│   └── results.html           # Side-by-side visual inspection & metrics dashboard
│
├── static/
│   ├── css/style.css          # Responsive sustainability-themed stylesheet
│   ├── js/app.js              # Client-side validation, previews & loading states
│   ├── uploads/               # Uploaded images (UUID named)
│   └── results/               # Annotated result images with bounding boxes
│
├── reports/
│   ├── README.md
│   ├── evaluation_metrics_test.json # Measured Precision, Recall, F1, mAP50, mAP50-95
│   ├── model_comparison.json        # Baseline vs Fine-Tuned deltas
│   └── campus_test_results.json     # Real campus photograph evaluation
│
├── tests/
│   ├── test_dataset.py        # Unit tests for coordinate math & format validation
│   ├── test_scoring.py        # Unit tests for scoring bounds, monotonicity & zero-case
│   └── test_web.py            # Integration tests for Flask routes, uploads & errors
│
└── runs/                      # Ultralytics training logs, confusion matrices, loss curves
```

---

## 4. Environment Setup & Installation

### Prerequisites
- Python 3.10, 3.12, or 3.14 (64-bit)
- Windows, macOS, or Linux
- Optional: NVIDIA GPU with CUDA drivers (automatically detected if available)

### Step 1: Clone / Navigate to Project Directory
```bash
cd D:\DL
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Dataset Preparation (TACO to YOLO)

The **TACO (Trash Annotations in Context)** dataset contains 1,500 high-resolution images with 4,784 annotations originally organized into 60 fine-grained litter categories under COCO format.

### Step 5.1: Download TACO Dataset
```bash
# Download annotations and a 100-image sample for fast training:
python src/data/download_dataset.py --max-images 100

# Or download all 1,500 images:
python src/data/download_dataset.py --max-images 0
```

### Step 5.2: Convert Annotations to YOLO Single-Class Format
The conversion pipeline:
1. Resolves images on disk and measures native dimensions.
2. Maps all valid categories to a single unified class: `litter` (class ID 0).
3. Converts bounding boxes $[x_{\min}, y_{\min}, w, h]$ to normalized center coordinates $[x_c, y_c, w_n, h_n]$ clamped to $[0.0, 1.0]$.
4. Clips out-of-boundary boxes and logs skipped degenerate boxes.
5. Preserves negative samples (images without litter) with valid empty `.txt` label files.

Run conversion:
```bash
python src/data/convert_annotations.py
```

### Step 5.3: Split Dataset (70% Train, 15% Val, 15% Test)
Splits samples into reproducible partitions using seed 42, prevents multi-crop leakage, and outputs `data/processed/dataset.yaml`:
```bash
python src/data/split_dataset.py --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

### Step 5.4: Validate Dataset Integrity
Verifies all image files, checks label coordinates, and ensures zero corrupt entries:
```bash
python src/data/validate_dataset.py
```

*Actual Measured Dataset Counts:*
- **Train Split (70%):** 70 images, 204 bounding boxes
- **Validation Split (15%):** 15 images, 54 bounding boxes
- **Test Split (15%):** 15 images, 44 bounding boxes
- **Total:** 100 images, 302 bounding boxes (Validation Status: **PASSED**)

---

## 6. Model Fine-Tuning (YOLOv8n)

Training fine-tunes pretrained **YOLOv8n** weights (`models/yolov8n.pt`) on single-class litter detection.

### Training Configuration
- **Model:** YOLOv8n (nano, ~3.15M parameters)
- **Base Weights:** Pretrained MS COCO
- **Image Size:** 640×640
- **Optimizer:** Adam with initial learning rate $\text{lr}_0 = 0.001$
- **Transfer Strategy:** Backbone freeze (`freeze=10`) to preserve low-level edge features while training the single-class detection head.
- **Patience:** 10 epochs early stopping
- **Device:** Automatic GPU detection with seamless CPU fallback

Run training:
```bash
python src/model/train.py --epochs 10 --batch-size 8
```
The script saves the best checkpoint automatically to `models/best.pt` and training logs/curves to `runs/detect/train_litter/`.

---

## 7. Cleanliness Scoring Algorithm

The Cleanliness Score ($S$) is bounded between **0 and 100**, deterministic, and explainable.

### Mathematical Formulation
$$\text{Area Ratio} = \text{clamp}\left( \frac{\sum_{i=1}^N \text{Area}(\text{bbox}_i)}{\text{Image Area}}, 0.0, 1.0 \right)$$

$$\text{Count Penalty} = 1 - e^{-\alpha \cdot N}$$

$$\text{Area Penalty} = \min\left(1.0, \frac{\text{Area Ratio}}{\text{Area Scale}}\right)$$

$$\text{Score} = \text{round}\Big( 100 \cdot \big[ 1 - (w_{\text{count}} \cdot \text{Count Penalty} + w_{\text{area}} \cdot \text{Area Penalty}) \big] \Big)$$

### Default Hyperparameters
- $\alpha = 0.35$: Controls how fast the score drops with increasing item count ($N$).
- $\text{Area Scale} = 0.15$: Bounding box coverage of 15% saturates the area penalty.
- $w_{\text{count}} = 0.50$ and $w_{\text{area}} = 0.50$: Balanced equal weight ($w_{\text{count}} + w_{\text{area}} = 1.0$).
- Baseline Count-Only Alpha: $\alpha_{\text{base}} = 0.25$.

### Qualitative Classification Table
| Score Range | Cleanliness Category | Operational Action |
|:---:|:---|:---|
| **90 – 100** | Excellent / Clean | Spotless condition; maintain routine schedule. |
| **70 – 89** | Good / Satisfactory | Minor isolated litter; standard daily sweeping recommended. |
| **45 – 69** | Fair / Needs Attention | Noticeable accumulation; prompt sanitation inspection advised. |
| **0 – 44** | Poor / High Litter Density | Urgent priority; immediate custodial dispatch required. |

*Disclaimer:* Bounding boxes approximate projected object footprints and may overlap. Zero detections indicates no visible litter in the camera's field of view, but does not guarantee the absence of micro-debris or occluded items.

---

## 8. Evaluation & Baseline Comparisons

### Held-Out Test Evaluation (15% Split)
Run quantitative evaluation on untouched test images:
```bash
python src/model/evaluate.py --split test
```

*Actual Measured Test Metrics (`reports/evaluation_metrics_test.json`):*
- **Precision:** 0.6333 (63.33%)
- **Recall:** 0.4318 (43.18%)
- **F1-Score:** 0.5135 (51.35%)
- **mAP@0.50:** 0.3638 (36.38%)
- **mAP@0.5:0.95:** 0.2514 (25.14%)

### Baseline Pretrained vs. Fine-Tuned Comparison
Run the comparison benchmark:
```bash
python src/model/compare_models.py
```

*Benchmark Results (`reports/model_comparison.json`):*
- **Pretrained Baseline (`yolov8n.pt`):** mAP@0.5 = 0.0000 on single-class litter evaluation (it detects COCO categories, not specialized campus waste).
- **Fine-Tuned Checkpoint (`models/best.pt`):** mAP@0.5 = 0.3638 (+36.38% improvement).
- **Mean Score Divergence:** 14.67 points between count-only baseline and count-plus-area scoring. Large litter items (e.g. food trays or boxes) properly reduce the score under the count-plus-area method.

---

## 9. Real Campus Testing

Real campus photographs captured around Ramdeobaba University are kept separate in `data/campus_test/`.

### Run Batch Testing
```bash
python src/model/batch_test_campus.py --images-dir data/campus_test
```

*Actual Campus Test Results (`reports/campus_test_results.json`):*
1. `campus_corridor_clean.jpg`: Detections = 0, Cleanliness Score = **100 / 100** (Clean)
2. `campus_canteen_clean.jpg`: Detections = 0, Cleanliness Score = **100 / 100** (Clean)
3. `campus_garden_littered.jpg`: Detections = 1, Cleanliness Score = **81 / 100** (Good)
4. `campus_pathway_bottles.jpg`: Detections = 2, Cleanliness Score = **68 / 100** (Needs Attention)
- **Mean Campus Cleanliness Score:** 87.25 / 100

### Documented Failure Modes
- **Small Litter Items:** Objects $< 16\times 16$ px (cigarette butts, bottle caps) can be missed by YOLOv8n.
- **Occlusion:** Objects partially hidden beneath corridor benches or foliage.
- **Illumination & Shadows:** Harsh midday sunlight causing cast shadows that can be mistaken for wrappers.
- **Background Confusion:** Fallen dry leaves on lawns occasionally trigger false positive detections.

---

## 10. Running the Flask Web Application

From the project directory, after installing the dependencies in Step 3, launch the local web server:
```bash
python app.py
```
The server can also be started by passing the full path to `app.py`. The homepage and health endpoints start without loading the model; image analysis requires the dependencies and model checkpoints listed above.

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Sharing the Site and Community Photos on a Local Network
Visitors can use **Share a Campus Photo** to submit a JPG, PNG, or WEBP image. The app attempts to analyze each new photo automatically. Every post remains in the gallery even if the model is unavailable; select **Analyze Photo** on its gallery card to run or retry analysis later. Once analyzed, the gallery shows the cleanliness score, detected litter count, and annotated detection preview. Original photos are stored in `static/community/`, annotated previews in `static/results/`, and submission/analysis records in `data/community_submissions.sqlite`. These remain on the computer running the app, so keep these files and folders when backing up the project. Uploads are public to anyone who can reach the site. Analysis requires the model dependencies and checkpoint.

To let people on the same Wi-Fi/LAN open the site, start the server with:
```powershell
$env:HOST = "0.0.0.0"
python app.py
```
They should visit `http://<your-computer-LAN-IP>:5000` (find the address with `ipconfig`). Keep the server running and allow Python through Windows Firewall on a trusted private network if prompted. This local setup is not accessible from outside the network; public internet access requires deploying the app and using persistent file/database storage.

### Key UI Features
- **Drag-and-Drop Dropzone:** Intuitive upload zone with real-time preview before submission.
- **Community Photo Gallery:** Visitors can submit public campus photos, which are stored on the server and shown newest first.
- **Sensitivity Slider:** Adjustable confidence threshold (0.10 to 0.80).
- **Quick Demo Gallery:** One-click evaluation of real campus test photographs.
- **Side-by-Side Visualizer:** Displays original photograph alongside YOLOv8n annotated detection with bounding boxes.
- **Detailed Inventory Table:** Localized coordinates $[x_1, y_1, x_2, y_2]$, pixel footprint, and confidence score.

---

## 11. Running Unit & Integration Tests

The test suite covers dataset transformation, label validation, scoring monotonicity, upload security, and web routes:
```bash
python -m pytest tests/ -v
```

*Test Results:*
- `tests/test_dataset.py`: 10 passed (COCO-to-YOLO conversion, clipping, label line validation)
- `tests/test_scoring.py`: 10 passed (Score bounds, zero-case, monotonicity, configuration checks)
- `tests/test_web.py`: 7 passed (HTTP routes, upload rejection, image inference, health checks)
- **Total: 27 passed in ~6 seconds.**

---

## 12. GPU vs. CPU Execution

The code automatically detects whether an NVIDIA GPU with CUDA is available:
- **With CUDA:** Accelerates training from ~2 minutes down to ~20 seconds.
- **With CPU:** Fully supported using standard PyTorch CPU kernels. Inference on a single 640×640 image takes only 45–60 ms on a modern Intel Core i5 processor.

To force CPU or GPU mode:
```bash
python src/model/predict.py data/campus_test/campus_garden_littered.jpg --device cpu
```

---

## 13. Common Errors & Troubleshooting

1. **Permission denied when creating runs directory:**
   - *Fix:* Ultralytics settings are configured in code to store outputs in `D:\DL\runs` instead of user home directory.
2. **File too large (413 Request Entity Too Large):**
   - *Fix:* Maximum upload size is 16 MB. Compress image or downscale resolution.
3. **Unsupported File Format:**
   - *Fix:* Ensure the upload extension is `.jpg`, `.jpeg`, `.png`, or `.webp`.
4. **Missing Checkpoint Error:**
   - *Fix:* Run `python src/model/train.py --epochs 5` to produce `models/best.pt`. If missing, the inference engine automatically falls back to `models/yolov8n.pt`.

---

## 14. Limitations & Future Enhancements

- **Single-Class Scope:** The current model groups all trash into `litter`. Future iterations could classify specific waste streams (Biodegradable, Recyclable Plastic, Paper, Metal) for automated segregation bins.
- **Bounding Box Area Approximation:** Bounding boxes capture rectangular envelopes rather than exact polygonal masks. YOLOv8-seg (instance segmentation) could be explored for pixel-level mask area calculation.
- **CCTV Integration:** Connecting the scoring pipeline to RTSP streams from campus security cameras for real-time automated heatmaps and facilities alerts.
