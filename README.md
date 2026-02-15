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

### Advanced Architectural Features
* **Singleton Model Loader:** Optimized memory management by ensuring the ResNet18 weights are initialized only once.
* **Integrity Check:** Zero-leakage data pipeline with strict physical separation of Training and Validation sets.
* **System Health Monitoring:** Dedicated `/health` endpoint for real-time API status verification.

### Performance Visualization


The following visualizations illustrate the training convergence and the model's classification reliability.

<table style="width:100%">
  <tr>
    <th style="text-align:center">Training & Validation Curves</th>
    <th style="text-align:center">Confusion Matrix</th>
  </tr>
  <tr>
    <td><img src="backend/images/results.png" width="100%"></td>
    <td><img src="backend/images/confusion_matrix.png" width="100%"></td>
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

