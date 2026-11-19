# Customer Churn Prediction: End-to-End Data Science Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5+-orange.svg)](https://scikit-learn.org/)
[![CI Pipeline](https://github.com/actions/starter-workflows/actions/workflows/python-package.yml/badge.svg)](https://github.com)

A production-grade machine learning project predicting customer attrition using the Telco Customer Churn benchmark dataset (7,043 customers). It includes data ingestion, feature engineering, multi-model cross-validation, holdout evaluation, interactive web dashboard, and batch inference.

---

## 📌 Business Objectives & Solution Architecture

- **Primary Goal**: Detect customer attrition risk before contract termination to trigger retention offers.
- **Key Metric**: Optimize **ROC-AUC** and **Recall (Sensitivity)** to minimize missed churners (false negatives).
- **Core Components**:
  1. **Data Acquisition**: Scripted downloader for Telco Customer Churn data.
  2. **Feature Engineering**: Automated domain transformations (tenure bucketing, charge ratios, service bundling).
  3. **Multi-Model Benchmark**: Evaluates Logistic Regression, Random Forest, Gradient Boosting, and HistGradientBoosting across Stratified 5-Fold Cross-Validation.
  4. **Holdout Evaluation**: ROC curves, PR curves, Confusion Matrix, and feature importances.
  5. **Inference & Serving**: Single-instance customer scoring, batch CSV scoring, and a Streamlit web app.

---

## 📂 Project Directory Structure

```text
churn-prediction/
├── .github/workflows/ci.yml       # GitHub Actions CI workflow
├── .gitignore                     # Git ignore rules
├── Dockerfile                     # Container deployment specification
├── Makefile                       # Developer shortcuts (setup, train, eval, test, app)
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
├── pytest.ini                     # Pytest configuration
├── app.py                         # Streamlit interactive web dashboard
├── data/
│   ├── download_data.py           # Automated dataset downloader
│   ├── generate_data.py           # Synthetic dataset generator fallback
│   ├── raw/                       # Raw customer CSV storage
│   └── processed/                 # Stratified train, val, test CSV splits
├── models/
│   └── best_churn_model.joblib    # Serialized scikit-learn pipeline
├── reports/
│   ├── figures/                   # Confusion matrix, ROC curve, PR curve, Feature importance
│   ├── model_comparison.csv       # Cross-validation comparison table
│   ├── model_summary.md           # Model performance summary report
│   ├── train_metrics.json         # Best hyperparameters & validation metrics
│   └── test_metrics.json          # Test set evaluation results
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Data loading, cleaning, and stratified splitting
│   ├── preprocessing.py           # Custom FeatureEngineer and ColumnTransformer
│   ├── train.py                   # Multi-model CV, tuning, and pipeline serialization
│   ├── evaluate.py                # Holdout test set metrics and diagnostic plots
│   ├── predict.py                 # Single and batch prediction engine
│   └── eda.py                     # Exploratory data analysis scripts
└── tests/
    ├── __init__.py
    ├── test_data.py               # Data loading and preprocessing unit tests
    └── test_pipeline.py           # Pipeline fitting and inference unit tests
```

---

## ⚡ Quick Start & Commands

### 1. Environment Setup

```bash
git clone <repository_url>
cd churn-prediction

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run with Makefile

```bash
make data    # Download dataset
make train   # Train and cross-validate models
make eval    # Evaluate holdout test set & generate figures
make test    # Run automated test suite
make app     # Launch Streamlit web app
```

### 3. CLI Prediction Examples

**Single Customer Inference:**
```bash
python src/predict.py
```

**Batch CSV Scoring:**
```bash
python src/predict.py --input data/processed/test.csv --output data/predictions.csv
```

---

## 📊 Benchmark Results (Holdout Test Set)

Evaluated on 1,409 holdout customer accounts:

| Metric | Score |
| :--- | :--- |
| **ROC-AUC** | **0.8415** |
| **Recall (Sensitivity)** | **78.88%** (295 / 374 churners identified) |
| **Precision** | **50.17%** |
| **F1-Score** | **0.6133** |
| **Accuracy** | **73.60%** |
| **Brier Calibration Score** | **0.1677** |

---

## 🐳 Docker Deployment

Build and launch the containerized application:

```bash
docker build -t churn-app .
docker run -p 8501:8501 churn-app
```
Open `http://localhost:8501` in your browser.
