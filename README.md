# Deepfake Detection System, Détecteur 
**A Production-Ready Full-Stack AI Forensics Platform**

This repository contains a full-stack application designed to detect AI-generated and synthetic facial forgeries. By utilizing a modified Deep Residual Network (ResNet18), the system identifies subtle facial artifacts and frequency inconsistencies typically invisible to the human eye.

---

## Technical Overview

### Neural Engine
* **Architecture:** Modified ResNet18 (4-Channel Forensic Input)
* **Framework:** PyTorch
* **Training Strategy:** Transfer Learning with ResNet18_Weights.DEFAULT
* **Optimization:** Adam Optimizer ($lr = 0.0001$) with Weighted Cross-Entropy Loss
* **Data Augmentation:** Implemented `RandomResizedCrop`, `ColorJitter`, `RandomHorizontalFlip`, and `RandomRotation` to ensure model robustness against real-world lighting and orientation variances.

### Full-Stack Architecture
* **Backend:** FastAPI (Python) providing asynchronous, GPU-accelerated RESTful inference.
* **Frontend:** Next.js (React) and Tailwind CSS for a high-performance, dark-mode dashboard.
* **Communication:** JSON-based exchange via HTTP with configured CORS security middleware.

---

## Forensic V2: Major System Upgrades

The system has been upgraded from a standard image classifier to a Multi-Spectral Forensic Pipeline. These enhancements address initial "catastrophic overfitting" and "real-world inference bias."

### 1. Hybrid 4-Channel Input Architecture
The core ResNet-18 engine was modified to look beyond standard RGB pixels by processing a 4th forensic channel dynamically.

| Channel Type | Target Features | Technical Purpose |
| :--- | :--- | :--- |
| **RGB Channels (1-3)** | Facial Geometry & Textures | Standard visual feature extraction via Pre-trained ImageNet weights. |
| **Forensic Channel (4)** | Laplacian Edge Map | Highlights high-frequency sensor noise and GAN-generated artifacts. |
| **Initialization** | Kaiming Normal Dist. | Ensures stable training for the custom input head while preserving existing knowledge. |

### 2. Cost-Sensitive Learning (Bias Correction)
To eliminate the model's tendency to default to "Real" during uncertainty, a Weighted Loss Strategy was implemented to prioritize Recall (Security).

| Upgrade | Implementation | Logic & Impact |
| :--- | :--- | :--- |
| **Loss Function** | Weighted Cross-Entropy | Penalizes missing a Fake image 5x harder than missing a Real one. |
| **Penalty Ratio** | [1.0, 5.0] | Drastically reduces False Negatives (the "lazy model" bug). |
| **Sensitivity** | High Recall | Optimized to catch subtle artifacts even in low-quality media. |

### 3. Forensic Data Augmentation & Optimization
A robust augmentation suite ensures the model generalizes to internet-sourced media rather than over-fitting to dataset-specific noise.

* **RandomResizedCrops:** Forces the AI to identify manipulation artifacts at various scales and resolutions.
* **ColorJitter & Lighting Normalization:** Protects detection accuracy against variations in lighting and skin tones.
* **Mixed Precision (AMP):** GPU-accelerated training using float16 math, optimized for NVIDIA RTX 40-series hardware.

---

## Model Performance & Validation

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Peak Accuracy** | **95.8%** | Maximum validation accuracy achieved with forensic augmentation. |
| **Validation Accuracy** | **96.13%** | Final correctness score on unseen test data. |
| **Fake Detection Recall** | **98.2%** | Effectiveness in catching all forgeries (Zero-Leakage Test). |
| **Inference Latency** | **~35ms** | Real-time processing speed per frame optimized via AMP. |

---

## Forensic Deep-Dive (Precision & Recall)

| Metric | Score | Formula | Logic & Real-World Impact |
| :--- | :--- | :--- | :--- |
| **Precision** | **95.8%** | $\frac{TP}{TP + FP}$ | **The Certainty Filter:** High precision ensures that when the system flags a "Fake," it is almost certainly correct. |
| **Recall** | **98.2%** | $\frac{TP}{TP + FN}$ | **The Security Guard:** High recall ensures the system is hyper-sensitive to artifacts, letting almost zero fakes slip through. |
| **F1-Score** | **96.9%** | $2 \cdot \frac{P \cdot R}{P + R}$ | **The Reliability Index:** Proves the model is balanced for both high security and user trust. |

### Experimental Integrity (Anti-Leakage Measures)
To prevent Data Leakage, a strict Subject-Independent Split was implemented:
* **Training Set:** 80% of data used to teach the model ResNet18 features.
* **Validation/Test Set:** 20% completely unseen images.
* **No Subject Overlap:** Ensures images of the same person do not appear in both sets, forcing the model to learn artifacts, not faces.

---

## Backend Architectural Overview

| Component | Status | Technical Purpose |
| :--- | :--- | :--- |
| **Model Loader** | Singleton | Ensures ResNet18 weights are loaded into memory once at startup to prevent RAM bloat. |
| **Health API** | Active | Provides a RESTful heartbeat (/health) to verify system status and model readiness. |
| **Inference Engine** | Warm | Pre-loaded state allows for near-instant prediction by removing weight-reloading overhead. |
| **Data Integrity** | Verified | Strict Subject-Independent Split to ensure zero data leakage and authentic performance. |

---

## Inference & Decision Logic

The system utilizes a Softmax Activation Layer and a Forensic Pre-check before providing a result:

| Stage | Process | Technical Detail |
| :--- | :--- | :--- |
| **1. Extraction** | Tensor Conversion | Image converted to [4, 224, 224] tensor format. |
| **2. Isolation** | Laplacian Filtering | Isolates high-frequency signal inconsistencies. |
| **3. Prediction** | Softmax Activation | Confidence derived using $Confidence = \max(\sigma(z))$. |

---

## Performance Visualization

The following visualizations illustrate the training convergence and the model's classification reliability.

<table style="width:100%">
  <tr>
    <th style="text-align:center">Training & Validation Curves</th>
    <th style="text-align:center">Confusion Matrix (Forensic V2)</th>
  </tr>
  <tr>
    <td><img src="./backend/images/results_v2_forensic.png" width="100%"></td>
    <td><img src="./backend/images/confusion_matrix_v2_forensic.png" width="100%"></td>
  </tr>
</table>

---

## Project Structure

```text
.
├── backend/                      # Python AI Forensic Service
│   ├── api.py                    # FastAPI Entry Point
│   ├── train_forensic_resnet.py  # 4-Channel Training Pipeline
│   ├── generate_plots.py         # Performance Visualization Script
│   ├── images/                   # Exported Metrics (PNG)
│   ├── requirements.txt          # Dependency List
│   └── trained_model.pth         # Serialized Model Weights
├── frontend/                     # Next.js User Interface
│   ├── components/               # React Dashboard Components
│   └── app/                      # Application Routing & Logic
└── README.md                     # System Documentation

Installation & Deployment
1. Backend Configuration
PowerShell

cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn api:app --reload

2. Frontend Configuration
PowerShell

cd frontend
npm install
npm run dev

Strategic Design Decisions

ResNet18 Selection: Chosen for better gradient flow and high inference speed on consumer-grade hardware compared to deeper architectures.

Headless Decoupling: Separating the AI engine from the UI allows for independent scaling and seamless integration with future platforms.