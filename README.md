# 🪰 Greenfly Detection and Multi-Object Tracking Pipeline (YOLOv26 vs. RF-DETR)

[![YOLOv26 Training Notebook](https://img.shields.io/badge/Kaggle-YOLOv26_Training-blue?logo=kaggle)](https://www.kaggle.com/code/archariox/cse445-assignment1-f)
[![RF-DETR Notebook](https://img.shields.io/badge/Kaggle-RF_DETR_Tracking-blue?logo=kaggle)](https://www.kaggle.com/code/istwestkhan/cse445-assignment1-f-rf-detr)

## 📌 Project Overview
This repository contains a complete computer vision pipeline designed to detect and track greenflies in both static images and drone video footage. The project tackles the challenge of identifying small, clustered insects against complex organic backgrounds. 

The pipeline evaluates two distinct architectural paradigms: the high-speed **YOLOv26 (CNN-based)** and the context-aware **RF-DETR (Transformer-based)**. We compare four standard trackers (**SimpleIoU, Custom SORT, ByteTrack, and BoTSORT**) against a custom-trained Transformer tracking implementation to determine the most robust solution for identity maintenance.

---

## 📊 Part 1: YOLOv26 Training & Performance Analysis
The model was trained on an augmented dataset comprising 3,256 `greenfly` and 3,543 `notgreenfly` instances. Training was conducted over 42 epochs.

### 1. Training Convergence & Metrics
<img src="runs/detect/greenfly_yolo26_train_optimized/results.png" width="800"> 

* **Convergence:** The model converged smoothly, reaching a final `train/box_loss` of 0.941 and `train/cls_loss` of 0.312. 
* **Augmentation Strategy:** A sharp drop in training loss is visible at Epoch 41. This indicates the successful disabling of mosaic augmentation during the final stages, allowing the model to fine-tune on true spatial distributions.
* **Overall Accuracy:** The model achieved a highly impressive **mAP@0.5 of 0.937** across all classes.

### 2. Precision-Recall & F1 Confidence
| Precision-Recall (mAP) | F1-Confidence Curve |
| :---: | :---: |
| <img src="runs/detect/greenfly_yolo26_train_optimized/BoxPR_curve.png" width="400"> | <img src="runs/detect/greenfly_yolo26_train_optimized/BoxF1_curve.png" width="400"> |

* **Greenfly:** 0.911 mAP@0.5
* **Not Greenfly:** 0.963 mAP@0.5
* The curves demonstrate that the model maintains high precision even as recall increases, effectively identifying small insects without flagging background noise.

### 3. Confusion Matrix Analysis
<img src="runs/detect/greenfly_yolo26_train_optimized/confusion_matrix_normalized.png" width="600">

* **True Positives:** The model achieves **92% accuracy** for greenflies and **94% accuracy** for non-greenflies.
* **Background Confusion:** Only 7% of background elements were incorrectly classified as greenflies, a highly optimized result for small-object detection in organic textures.

### 4. Bounding Box Density Heatmap
<img src="runs/detect/greenfly_yolo26_train_optimized/labels.jpg" width="600">

* **Spatial Distribution:** The `x` and `y` heatmaps reveal a strong central clustering, indicating the model is optimized for center-of-frame tracking typical of drone footage.
* **Size Profile:** The width/height heatmaps confirm a high density of small, uniform bounding boxes, perfectly matching the insect profiles.

---

## 🎯 Part 2: Tracker Implementation & Comparison
Maintaining unique IDs during overlaps and erratic movement is critical for accurate entomological data.

### 🎥 Visual Tracking Results
*(Click on any thumbnail to watch the full HD tracking demo on YouTube)*

| SimpleIoU Baseline | ByteTrack (Optimal) | RF-DETR (Transformer) |
| :---: | :---: | :---: |
| [![SimpleIoU](https://img.youtube.com/vi/rxhE6NRB6gA/0.jpg)](https://youtu.be/rxhE6NRB6gA) | [![ByteTrack](https://img.youtube.com/vi/NhJX9miisLs/0.jpg)](https://youtu.be/NhJX9miisLs) | [![RF-DETR](https://img.youtube.com/vi/kZuW9QL2oro/0.jpg)](https://youtu.be/kZuW9QL2oro) |
| *Fast baseline; high fragmentation.* | *Resilient; best speed/accuracy.* | *Maximum stability; Lowest ID switches.* |

| Custom SORT | BoTSORT |
| :---: | :---: |
| [![CustomSORT](https://img.youtube.com/vi/A4ZOOHumD-Y/0.jpg)](https://youtu.be/A4ZOOHumD-Y) | [![BoTSORT](https://img.youtube.com/vi/J6r63uEs5D4/0.jpg)](https://youtu.be/J6r63uEs5D4) |
| *Kalman-stabilized motion prediction.* | *Integrated camera motion compensation.* |

### 📊 Quantitative Performance Comparison

| Model + Tracker | Avg FPS | Total IDs | ID Switches | Performance Tier |
| :--- | :--- | :--- | :--- | :--- |
| **SimpleIoU** | 30.60 | 38 | 17 | Baseline (High Fragmentation) |
| **Custom SORT** | 30.95 | 36 | 16 | Kalman-stabilized Baseline |
| **ByteTrack** | **32.73** | 13 | 6 | **Efficiency Leader** |
| **BoTSORT** | 20.31 | **12** | 6 | High Stability / Computationally Heavy |
| **RF-DETR (Custom)** | 13.58 | 13 | **3** | **Precision Leader (Transformers)** |

**Analytical Deep Dive:**
* **Identity Continuity:** **RF-DETR** is the superior architecture for identity maintenance, achieving a significant reduction in **ID Switches (only 3)**. Its transformer-based global attention allows it to track objects through occlusions that cause CNN-based trackers to trigger new IDs.
* **Inference Speed:** **ByteTrack** remains the champion for real-time deployment, providing **32.73 FPS** (over 2x faster than RF-DETR) while keeping ID switches relatively low.
* **Deployment Context:** For real-time drone monitoring, ByteTrack is preferred. For high-precision research analysis where processing time is secondary to tracking accuracy, RF-DETR is the optimal choice.

### 🗺️ Spatial Tracking Density Analysis
These heatmaps highlight where each tracker maintained the most consistent ID presence throughout the drone footage.

| SimpleIoU Heatmap | ByteTrack Heatmap |
| :---: | :---: |
| ![SimpleIoU](output/density_heatmap_SimpleIoU.png) | ![ByteTrack](output/density_heatmap_ByteTrack.png) |

| Custom SORT Heatmap | BoTSORT Heatmap |
| :---: | :---: |
| ![CustomSORT](output/density_heatmap_CustomSORT.png) | ![BoTSORT](output/density_heatmap_BoTSORT.png) |

---

## 🤖 Part 3: RF-DETR (Transformer) Implementation
The **RF-DETR** model was trained using a COCO-converted version of the dataset. This architecture removes the need for Non-Maximum Suppression (NMS) and excels at global context.

* **Stability metrics:** 13 Total IDs with an estimated 6 Lost Tracks.
* **Transformer Advantage:** Unlike standard IoU-based tracking, the transformer architecture treats tracking as a set-prediction problem, resulting in the lowest ID switch rate (3) in the entire pipeline.

---

## 💻 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/tanvir-dev2000/cse445-yolo-tracking.git](https://github.com/tanvir-dev2000/cse445-yolo-tracking.git)
   cd cse445-yolo-tracking
