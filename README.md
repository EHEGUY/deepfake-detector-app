# Deepfake Detection System, Détecteur 
**A Production-Ready Full-Stack AI Forensics Platform**

This repository contains a full-stack application designed to detect AI-generated and synthetic facial forgeries. By utilizing a Deep Residual Network (ResNet18), the system identifies subtle facial artifacts and frequency inconsistencies typically invisible to the human eye.

---

## Technical Overview

### Neural Engine
* **Architecture:** ResNet18 (Residual Networks)
* **Framework:** PyTorch
* **Training Strategy:** Transfer Learning with ResNet18_Weights.DEFAULT
* **Optimization:** Adam Optimizer ($lr = 0.0001$)
* **Data Augmentation:** Implemented `ColorJitter`, `RandomHorizontalFlip`, and `RandomRotation` to ensure model robustness against real-world lighting and orientation variances.

### Full-Stack Architecture
* **Backend:** FastAPI (Python) providing asynchronous, GPU-accelerated RESTful inference.
* **Frontend:** Next.js (React) and Tailwind CSS for a high-performance, dark-mode dashboard.
* **Communication:** JSON-based exchange via HTTP with configured CORS security middleware.

---

## Model Performance & Validation

The model was evaluated over 5 training epochs, demonstrating high precision in distinguishing authentic human faces from synthetic forgeries.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Peak Accuracy** | **97.78%** | Maximum validation accuracy achieved during training. |
| **Validation Accuracy** | **96.13%** | Final correctness score on unseen test data. |
| **Fake Detection Confidence** | **99.94%** | Average certainty when identifying forgeries. |
| **Inference Latency** | **~45ms** | Real-time processing speed per image frame. |

---

## Forensic Deep-Dive (Precision & Recall)

To satisfy production-grade requirements, we look beyond simple accuracy. These metrics prove the model's reliability in real-world scenarios.

| Metric | Score | Formula | Logic & Real-World Impact |
| :--- | :--- | :--- | :--- |
| **Precision** | **95.8%** | $\frac{TP}{TP + FP}$ | **The "Certainty" Filter:** High precision means the system is extremely cautious. When it flags a face as "Fake," it is almost certainly correct, minimizing "False Accusations" against authentic users. |
| **Recall** | **98.2%** | $\frac{TP}{TP + FN}$ | **The "Security" Guard:** High recall ensures that the system is highly sensitive to artifacts. It "catches" nearly 99% of all forgeries, ensuring that synthetic media rarely "slips through" undetected. |
| **F1-Score** | **96.9%** | $2 \cdot \frac{P \cdot R}{P + R}$ | **The Reliability Index:** Since deepfake detection requires a balance between security (Recall) and user trust (Precision), the F1-Score proves this model is optimized for both. |

### Experimental Integrity (Anti-Leakage Measures)
To prevent **Data Leakage** (where the model "cheats" by memorizing test data), we implemented a strict **Subject-Independent Split**:
* **Training Set:** 80% of data used to teach the model ResNet18 features.
* **Validation Set:** 10% used for hyperparameter tuning.
* **Test Set:** 10% completely "unseen" images used for final evaluation.
* **No Subject Overlap:** We ensured that images of the same person do not appear in both training and testing sets, forcing the model to learn *artifacts*, not *faces*.
---
### Backend Architectural Overview

| Component | Status | Technical Purpose |
| :--- | :--- | :--- |
| **Model Loader** | `Singleton` | Ensures ResNet18 weights are loaded into memory **once** at startup, preventing RAM bloat and memory leaks. |
| **Health API** | `Active` | Provides a RESTful heartbeat (`/health`) to verify system status and model readiness in real-time. |
| **Inference Engine** | `Warm` | Pre-loaded state allows for near-instant prediction (~45ms) by removing the overhead of reloading weights. |
| **Data Integrity** | `Verified` | Implemented a strict **Subject-Independent Split** to ensure zero data leakage and authentic performance. |

### Production Standards
* **Singleton Pattern:** Optimized memory management ensures that multiple requests don't spawn multiple model instances, keeping the server stable.
* **Zero-Leakage Pipeline:** Training and Testing sets were physically separated with no subject overlap, proving the model detects *forgery patterns*, not specific people.
* **Security Hardening:** Utilizes `weights_only=True` during the `torch.load` process to adhere to modern security standards.
* **Container Ready:** The inclusion of a `/health` endpoint makes this project ready for professional deployment via Docker or Kubernetes.


2 **Forensic V2: Major System Upgrades**

The system has been significantly upgraded from a standard image classifier to a Multi-Spectral Forensic Pipeline. These enhancements address initial "catastrophic overfitting" and "real-world inference bias."
1. Hybrid 4-Channel Input Architecture

We modified the core ResNet-18 engine to look beyond standard RGB pixels. The model now processes a 4th forensic channel dynamically.

    RGB Channels (1-3): Process facial geometry and standard textures.

    Forensic Channel (4): A Laplacian Edge Inconsistency Map that highlights high-frequency sensor noise and GAN-generated artifacts.

    Initialization: Utilized Kaiming Normal Distribution for the custom input head to ensure stable training while preserving pre-trained ImageNet knowledge.

 **Cost-Sensitive Learning (Bias Correction)**

To solve the model's tendency to default to "Real" during uncertainty, we implemented a Weighted Loss Strategy. This prioritizes Recall (Security) over simple accuracy.

| **Upgrade** | **Implementation** | **Logic & Impact** |
| :--- | :--- | :--- |
| **Loss Function** | **Weighted Cross-Entropy** | **Penalizes missing a Fake image 5x harder than missing a Real one.** |
| **Penalty Ratio** | **[1.0, 5.0]** | **Drastically reduces False Negatives (the "lazy model" bug).** |
| **Sensitivity** | **High Recall** | **Optimized to catch subtle artifacts even in low-quality media.** |



**Forensic Data Augmentation**

To ensure the model works on internet images (not just the dataset), we added a robust Augmentation Suite.

    RandomResizedCrops: Forces the AI to identify manipulation artifacts at various scales and resolutions.

    ColorJitter & Lighting Normalization: Protects detection accuracy against variations in lighting and skin tones.

    Mixed Precision (AMP): GPU-accelerated training using float16 math, optimized for NVIDIA RTX 40-series hardware.

    | **Feature** | **Baseline (V1)** | **Forensic V2 (Current)** |
| :--- | :--- | :--- |
| **Input Depth** | **3-Channel (RGB)** | **4-Channel (RGB + Laplacian)** |
| **Feature Focus** | **Surface Textures** | **Frequency-Domain Artifacts** |
| **Error Penalty** | **Balanced** | **Weighted (5:1 Bias for Fakes)** |
| **Inference** | **High Bias** | **Zero-Bias Forensic Detection** |
| **Latency** | **~45ms** | **~35ms (Optimized via AMP)** |

### ** Updated Inference Logic**

| **Stage** | **Process** | **Technical Detail** |
| :--- | :--- | :--- |
| **1. Extraction** | **Tensor Conversion** | **Image converted to [4, 224, 224] tensor format.** |
| **2. Isolation** | **Laplacian Filtering** | **Isolates high-frequency signal inconsistencies.** |
| **3. Prediction** | **Softmax Activation** | **Biased toward identifying synthetic patterns.** |

###  Inference & Decision Logic

The system utilizes a **Softmax Activation Layer** to transform raw logit outputs from the ResNet18 head into human-readable probabilities.

- **Calculation:** Confidence is derived using $Confidence = \max(\sigma(z))$, where $\sigma$ is the Softmax function.
- **Thresholding:** While the model provides a binary "Real/Fake" label, the confidence score provides a forensic layer of detail, allowing users to identify "Borderline" cases (e.g., scores between 50% and 70%).
### Performance Visualization


The following visualizations illustrate the training convergence and the model's classification reliability.

<table style="width:100%">
  <tr>
    <th style="text-align:center">Training & Validation Curves</th>
    <th style="text-align:center">Confusion Matrix</th>
  </tr>
  <tr>
    <td><img src="backend/images/results_v2_forensic.png" width="100%"></td>
    <td><img src="backend/images/confusion_matrix_v2_forensic.png" width="100%"></td>
  </tr>
</table>

---

## Project Structure

```text
.
├── backend/                      # Python AI Forensic Service
│   ├── api.py                    # FastAPI Entry Point
│   ├── train.py                  # Model Training Pipeline
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

Access the system locally at: http://localhost:3000
Strategic Design Decisions

    ResNet18 Selection: Chosen for its superior gradient flow and high inference speed on consumer-grade hardware compared to deeper, more computationally expensive architectures.

    Headless Decoupling: Separating the AI engine from the UI allows for independent scaling and seamless integration with future mobile platforms.



Sidd, student

