# 🪰 Greenfly Detection and Multi-Object Tracking Pipeline (YOLOv26 vs. RF-DETR)

[![YOLOv26 Training Notebook](https://img.shields.io/badge/Kaggle-YOLOv26_Training-blue?logo=kaggle)](https://www.kaggle.com/code/archariox/cse445-assignment1-f)
[![RF-DETR Notebook](https://img.shields.io/badge/Kaggle-RF_DETR_Tracking-blue?logo=kaggle)](https://www.kaggle.com/code/istwestkhan/cse445-assignment1-f-rf-detr)

## 📌 Project Overview
This repository contains a complete computer vision pipeline designed to detect and track greenflies in drone video footage. The project tackles the challenge of identifying small, clustered insects against complex organic backgrounds by evaluating two distinct architectural paradigms: the high-speed **YOLOv26 (CNN-based)** and the context-aware **RF-DETR (Transformer-based)**. 

---

## 📊 Part 1: YOLOv26 Training & Performance Analysis
The YOLOv26 model was trained over 42 epochs on a dataset containing 3,256 greenfly instances.

### 1. Final Training Metrics (Epoch 42)
| Metric | Value |
| :--- | :--- |
| **mAP@50** | 0.9373 |
| **mAP@50-95** | 0.5517 |
| **Precision** | 0.9107 |
| **Recall** | 0.8897 |
| **Total Training Time** | ~1.44 Hours |

### 2. Convergence & Spatial Analysis
<img src="runs/detect/greenfly_yolo26_train_optimized/results.png" width="800"> 

*   **Box & Class Loss:** The model reached a stable plateau with a final `train/box_loss` of 0.941 and `train/cls_loss` of 0.312.
*   **Augmentation Strategy:** A sharp drop in training loss at Epoch 41 signifies the successful disabling of mosaic augmentation, allowing the model to optimize for true spatial distributions.

### 3. Precision-Recall & Confusion Matrix
| Precision-Recall (mAP) | F1-Confidence Curve |
| :---: | :---: |
| <img src="runs/detect/greenfly_yolo26_train_optimized/BoxPR_curve.png" width="400"> | <img src="runs/detect/greenfly_yolo26_train_optimized/BoxF1_curve.png" width="400"> |

<img src="runs/detect/greenfly_yolo26_train_optimized/confusion_matrix_normalized.png" width="600">

*   **Accuracy:** The model achieves **92% accuracy** for greenflies and **94% accuracy** for non-greenflies, with only 7% background confusion.

### 4. Bounding Box Density Heatmap
<img src="runs/detect/greenfly_yolo26_train_optimized/labels.jpg" width="600">

*   **Spatial Distribution:** The heatmaps show strong central clustering, confirming the model is optimized for center-of-frame tracking typical in drone flyovers.

---

## 🤖 Part 2: RF-DETR (Transformer) Implementation
The RF-DETR model was trained as a comparative architecture to leverage global self-attention for tracking stability.

### 1. Model Configuration & Results
*   **Trainable Parameters:** 30.5 Million
*   **Best Regular mAP:** 0.5652
*   **Best EMA mAP:** 0.575
*   **Training Duration:** 15 epochs

Unlike CNNs, this transformer architecture removes the need for Non-Maximum Suppression (NMS), allowing for more precise handling of dense object clusters.

---

## 🎯 Part 3: Tracker Implementation & Comparison
Maintaining identity continuity during erratic movement is the primary challenge in insect tracking.

### 🎥 Visual Tracking Results
*(Click on any thumbnail to watch the full HD tracking demo on YouTube)*

| SimpleIoU Baseline | ByteTrack (Optimal) | RF-DETR (Custom) |
| :---: | :---: | :---: |
| [![SimpleIoU](https://img.youtube.com/vi/rxhE6NRB6gA/0.jpg)](https://youtu.be/rxhE6NRB6gA) | [![ByteTrack](https://img.youtube.com/vi/NhJX9miisLs/0.jpg)](https://youtu.be/NhJX9miisLs) | [![RF-DETR](https://img.youtube.com/vi/kZuW9QL2oro/0.jpg)](https://youtu.be/kZuW9QL2oro) |
| *Fast; struggles with occlusion.* | *Highly resilient ID continuity.* | *Transformer-based set prediction.* |

| Custom SORT | BoTSORT |
| :---: | :---: |
| [![CustomSORT](https://img.youtube.com/vi/A4ZOOHumD-Y/0.jpg)](https://youtu.be/A4ZOOHumD-Y) | [![BoTSORT](https://img.youtube.com/vi/J6r63uEs5D4/0.jpg)](https://youtu.be/J6r63uEs5D4) |
| *Kalman Filter stabilization.* | *Camera motion compensation.* |

### 📊 Quantitative Performance Comparison

| Model + Tracker | Avg FPS | Total IDs | ID Switches | Performance Tier |
| :--- | :--- | :--- | :--- | :--- |
| **SimpleIoU** | 30.60 | 38 | 17 | Baseline (Fragmented) |
| **Custom SORT** | 30.95 | 36 | 16 | Kalman-stabilized |
| **ByteTrack** | **32.73** | 13 | 6 | **Efficiency Champion** |
| **BoTSORT** | 20.31 | **12** | 6 | Maximum Stability / Slow |
| **RF-DETR (Custom)** | 13.58 | 13 | **3** | **Precision Champion** |

---

## 🗺️ Spatial Tracking Density Analysis
These heatmaps highlight where each tracker maintained the most consistent ID presence throughout the drone footage.

| SimpleIoU Heatmap | ByteTrack Heatmap |
| :---: | :---: |
| ![SimpleIoU](output/density_heatmap_SimpleIoU.png) | ![ByteTrack](output/density_heatmap_ByteTrack.png) |

| Custom SORT Heatmap | BoTSORT Heatmap |
| :---: | :---: |
| ![CustomSORT](output/density_heatmap_CustomSORT.png) | ![BoTSORT](output/density_heatmap_BoTSORT.png) |

---

## 🔍 Final Analysis: The "Stability vs. Speed" Trade-off

> **Technical Insight:** 
> Our results indicate a clear architectural split. **RF-DETR** achieved the lowest identity switches in the entire pipeline (**3 switches**), representing a **50% improvement** over ByteTrack. However, this comes at a computational cost, with RF-DETR running at **13.58 FPS** compared to ByteTrack's **32.73 FPS**. 
>
> For real-time drone deployment, **ByteTrack** is the overall winner. For high-fidelity biological research where tracking accuracy is paramount, **RF-DETR** is the superior tool.

---

## 💻 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/tanvir-dev2000/cse445-yolo-tracking.git](https://github.com/tanvir-dev2000/cse445-yolo-tracking.git)
   cd cse445-yolo-tracking
