# Customer Churn Prediction: Production-Ready Data Science Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Pipeline](https://github.com/actions/starter-workflows/actions/workflows/python-package.yml/badge.svg)](https://github.com)

An end-to-end, modular machine learning system designed to predict customer churn in telecommunications and subscription services. This repository provides data generation, automated preprocessing, feature engineering, multi-model cross-validation, hyperparameter tuning, metric evaluation, and production-ready batch and real-time inference.

---

## 📌 Problem Overview

Customer attrition directly impacts monthly recurring revenue (MRR) and customer acquisition cost (CAC) payback periods. Identifying at-risk customers early enables proactive retention campaigns, targeted discounts, and service interventions.

This project delivers:
1. **Automated Feature Engineering**: Captures contractual commitments, add-on service counts, tenure buckets, and charge ratios.
2. **Multi-Model Benchmark**: Evaluates Logistic Regression, Random Forest, Gradient Boosting, and HistGradientBoosting across Stratified 5-Fold Cross-Validation.
3. **Rigorous Evaluation**: Assesses ROC-AUC, PR-AUC, Sensitivity/Recall, Precision, F1-Score, and Brier Score on a clean holdout test set.
4. **Production Inference Service**: Single-customer risk scoring with actionable retention recommendations, plus high-throughput batch CSV scoring.

---

## 📂 Project Structure

```text
churn-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI pipeline (lint & tests)
├── .gitignore                   # Git ignore patterns
├── README.md                    # Project documentation
├── requirements.txt             # Project dependencies
├── data/
│   ├── generate_data.py         # Synthetic telecom churn dataset generator
│   ├── raw/                     # Raw dataset storage
│   └── processed/               # Train, validation, and test splits
├── models/
│   └── best_churn_model.joblib  # Serialized scikit-learn pipeline
├── reports/
│   ├── figures/                 # Diagnostic plots (ROC, PR, Confusion Matrix, Feature Importance)
│   ├── model_comparison.csv     # Cross-validation comparison table
│   ├── train_metrics.json       # Training metadata and hyperparameters
│   └── test_metrics.json        # Test set evaluation metrics
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data ingestion, schema validation, and splitting
│   ├── preprocessing.py         # Custom feature engineering & ColumnTransformer
│   ├── train.py                 # Multi-model CV, tuning, and pipeline serialization
│   ├── evaluate.py              # Performance evaluation and diagnostic visualizations
│   ├── predict.py               # Single & batch inference engine
│   └── eda.py                   # Exploratory data analysis visualization script
└── tests/
    ├── __init__.py
    ├── test_data.py             # Data loading and preprocessing unit tests
    └── test_pipeline.py         # Pipeline fitting and inference unit tests
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone <repository_url>
cd churn-prediction

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset

Generate 5,000 synthetic customer records with realistic churn dynamics:

```bash
python data/generate_data.py --samples 5000 --output data/raw/customer_churn.csv
```

### 3. Exploratory Data Analysis (EDA)

Generate exploratory distribution and churn correlation charts:

```bash
python src/eda.py
```
Outputs are saved to `reports/figures/`.

### 4. Model Training & Comparison

Execute multi-model cross-validation, hyperparameter optimization, and serialize the best model:

```bash
python src/train.py --data data/raw/customer_churn.csv --output models/best_churn_model.joblib
```

### 5. Evaluation & Diagnostics

Evaluate on the holdout test set, generating metrics and diagnostic curves:

```bash
python src/evaluate.py --model models/best_churn_model.joblib --test data/processed/test.csv
```

Generated plots in `reports/figures/`:
- `roc_curve.png`: Receiver Operating Characteristic curve.
- `precision_recall_curve.png`: Precision-Recall curve.
- `confusion_matrix.png`: Normalized confusion matrix.
- `feature_importance.png`: Top predictive features.

---

## 🔮 Inference & Deployment

### Single-Customer Prediction

Run inference with single customer attributes:

```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor("models/best_churn_model.joblib")

customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 268.5,
}

result = predictor.predict_single(customer)
print(result)
# Output:
# {
#   "churn_prediction": "Yes",
#   "churn_probability": 0.8124,
#   "risk_tier": "High Risk",
#   "retention_action_recommended": True
# }
```

### Batch Scoring via CLI

Score an entire customer portfolio from CSV:

```bash
python src/predict.py --model models/best_churn_model.joblib --input data/processed/test.csv --output data/scored_customers.csv
```

---

## 🧪 Testing

Run automated unit and integration tests:

```bash
pytest -v
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
