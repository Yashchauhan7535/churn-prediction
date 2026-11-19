# Churn Prediction Model Summary Report

## 1. Executive Summary
This machine learning model predicts telecommunications customer churn. The final selected architecture is a regularized classification pipeline with domain-specific feature engineering, automated categorical encoding, and scaling.

- **Primary Optimization Metric**: ROC-AUC / Recall (Prioritizing early detection of churners to reduce false negatives)
- **Holdout Test Set Performance**:
  - **ROC-AUC**: 0.8643
  - **PR-AUC**: 0.8392
  - **Recall (Sensitivity)**: 84.29%
  - **Precision**: 73.38%
  - **F1-Score**: 0.7846
  - **Accuracy**: 78.20%
  - **Brier Score**: 0.1507

---

## 2. Model Comparison Across 5-Fold Stratified Cross-Validation

| Model Family | Mean ROC-AUC | Std ROC-AUC | Mean F1 | Mean Recall | Mean Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.8713** | 0.0090 | **0.7921** | **0.8375** | **0.7929** |
| **Random Forest** | 0.8683 | 0.0084 | 0.7865 | 0.8266 | 0.7886 |
| **Gradient Boosting** | 0.8648 | 0.0113 | 0.7793 | 0.7938 | 0.7880 |
| **HistGradientBoosting** | 0.8574 | 0.0134 | 0.7769 | 0.8071 | 0.7814 |

---

## 3. Key Predictive Drivers of Customer Churn
1. **Contract Type (`is_month_to_month`)**: Month-to-month contracts exhibit drastically higher propensity to churn compared to one- or two-year commitments.
2. **Tenure (`tenure`, `tenure_group`)**: Customers in their first 6-12 months are at highest risk; retention stabilizes significantly past year two.
3. **Monthly Charges & Fiber Optic**: High monthly charges combined with Fiber Optic service in the absence of Tech Support or Online Security correlate with elevated churn.
4. **Payment Method**: Paper check and manual billing correspond with higher churn than automated payment methods.
