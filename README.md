# Bristol Regional Food Network — Advanced AI
### UFCFUR-15-3 | Group Project 2025–26

An AI-powered solution for the Bristol Regional Food Network digital marketplace, automating produce quality classification, intelligent demand prediction, model deployment, and explainable AI.

---

## 📁 Repository Structure

```
BFN-ADV-AI/
├── Task1_Demand_Prediction/       # Intelligent reorder suggestions & demand forecasting
├── Task2_Quality_Classification/  # Fruit & vegetable quality classification (Tasks 2, 3 & 4)
│   ├── api/                       # FastAPI deployment endpoints
│   ├── database/                  # SQLite interaction logging
│   ├── evaluation_results/        # Evaluation charts and metrics
│   ├── model_registry/            # Trained model storage (not tracked by Git)
│   ├── xai_outputs/               # Grad-CAM heatmap outputs
│   ├── preprocess.py              # Data loading and augmentation
│   ├── model.py                   # ResNet50 model architecture
│   ├── train.py                   # Model training script
│   ├── grading.py                 # A/B/C quality grading logic
│   ├── evaluate.py                # Evaluation metrics and charts
│   ├── inventory.py               # Automated inventory updates
│   ├── gradcam.py                 # Grad-CAM implementation
│   └── explain_prediction.py      # XAI explanation pipeline
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🤖 Tasks Overview

### Task 1 — Intelligent Demand Prediction
Analyses customer purchase history to predict frequently ordered items and provide personalised reorder suggestions. Also generates weekly demand forecasts per producer.

- **Algorithm:** Random Forest Classifier
- **Dataset:** Synthetic purchase history (3,315 orders, 30 customers, 13 products)
- **Accuracy:** 87.18% | Precision: 87.96% | F1: 86.26%

### Task 2 — Fruit & Vegetable Quality Classification
Computer vision model that classifies 14 types of produce as Healthy or Rotten and automatically assigns quality grades and inventory actions.

- **Algorithm:** ResNet50 Transfer Learning (PyTorch)
- **Dataset:** [Kaggle — Fruit and Vegetable Disease (Healthy vs Rotten)](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten)
- **Classes:** 28 (14 produce types × Healthy/Rotten)
- **Images:** 29,291
- **Accuracy:** 99.00% | Precision: 99.02% | F1: 99.00%

**Grading thresholds (as per case study):**
| Grade | Colour | Size | Ripeness |
|-------|--------|------|----------|
| A | ≥ 75% | ≥ 80% | ≥ 70% |
| B | ≥ 65% | ≥ 70% | ≥ 60% |
| C | < 65% | < 70% | < 60% |

### Task 3 — Model Deployment (FastAPI)
REST API that allows AI engineers to upload trained models and serves predictions for both quality classification and demand prediction.

**Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API health check |
| POST | `/predict` | Quality classification for an image |
| GET | `/reorder` | Reorder suggestions for a customer |
| GET | `/forecast` | Weekly demand forecast for producers |

### Task 4 — Explainable AI (Grad-CAM)
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to explain model predictions by highlighting which regions of an image most influenced the classification decision.

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (recommended) — CPU also supported but slower

### Step 1 — Clone the repository
```bash
git clone https://github.com/ImmortalSheikh/BFN-ADV-AI.git
cd BFN-ADV-AI
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the dataset
Download from Kaggle and place inside `Task2_Quality_Classification/dataset/`:

🔗 [Fruit and Vegetable Disease Dataset](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten)

The folder structure must be exactly:
```
Task2_Quality_Classification/dataset/Fruit And Vegetable Diseases Dataset/
├── Apple__Healthy/
├── Apple__Rotten/
├── Banana__Healthy/
...
```

---

## 🚀 Running the Project

### Task 1 — Demand Prediction
```bash
cd Task1_Demand_Prediction
python data.py       # Generate dataset
python train.py      # Train model
python evaluate.py   # Evaluate model
python predict.py    # Run predictions demo
```

### Task 2 — Quality Classification
```bash
cd Task2_Quality_Classification
python preprocess.py  # Test data loading
python train.py       # Train model (~1-2 hrs GPU, ~8-10 hrs CPU)
python grading.py     # Test grading on sample image
python evaluate.py    # Full evaluation + charts
python inventory.py   # Batch inspection demo
```

⚠️ After training, copy the model to the registry:
```bash
# Windows
Copy-Item saved_model.pth model_registry\saved_model.pth

# Mac/Linux
cp saved_model.pth model_registry/saved_model.pth
```

### Task 3 — API Deployment
```bash
cd Task2_Quality_Classification

# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
uvicorn api.main:app --reload
```

API runs at: `http://127.0.0.1:8000`
Interactive docs at: `http://127.0.0.1:8000/docs`

### Task 4 — XAI / Grad-CAM
```bash
cd Task2_Quality_Classification

# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python explain_prediction.py
```

---

## 📊 Results

### Task 1 — Demand Prediction
| Metric | Score |
|--------|-------|
| Accuracy | 87.18% |
| Precision | 87.96% |
| Recall | 87.18% |
| F1 Score | 86.26% |
| ROC-AUC | 0.8032 |

### Task 2 — Quality Classification
| Metric | Score |
|--------|-------|
| Accuracy | 99.00% |
| Precision | 99.02% |
| Recall | 99.00% |
| F1 Score | 99.00% |

---

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, torchvision (ResNet50)
- **Machine Learning:** scikit-learn (Random Forest)
- **API:** FastAPI, Uvicorn
- **Database:** SQLite
- **Explainability:** Grad-CAM (custom implementation)
- **Data Processing:** NumPy, Pandas, OpenCV, Pillow
- **Visualisation:** Matplotlib, Seaborn

---

## ⚠️ Important Notes
- The dataset (~5GB) and trained model (`saved_model.pth`, ~226MB) are excluded from this repository via `.gitignore`
- All file paths use relative paths and are cross-platform compatible
- Training with GPU is strongly recommended for Task 2

---

## 👥 Group Members
| Member | Task |
|--------|------|
| Amar | Task 2 — Quality Classification |
| Abdelrahman | Task 1 — Demand Prediction |
| Amr | Task 3 — Model Deployment |
| Sultan | Task 4 — Explainable AI |

---
