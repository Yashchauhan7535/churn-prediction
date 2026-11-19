# Customer Churn Prediction: Model Evaluation & Business Insights

## 1. Executive Summary
This end-to-end data science project analyzes customer attrition on the official IBM Telco Customer Churn benchmark dataset (7,043 customer accounts). The objective is to identify at-risk customers early to deploy targeted retention interventions, protect monthly recurring revenue (MRR), and reduce customer acquisition payback time.

---

## 2. Test Set Evaluation Metrics (Holdout Partition: 1,409 Customers)

The holdout test evaluation results are:

| Metric | Score | Business Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8415** | Strong separability between churn and retention cohorts across varying classification thresholds. |
| **Recall (Sensitivity)** | **78.88%** | Identifies ~79% of all churning customers (295 out of 374 actual churners caught). |
| **Precision** | **50.17%** | Every 2 outreach calls capture an at-risk customer. |
| **F1-Score** | **0.6133** | Robust balance between catching churners and outreach expenditure efficiency. |
| **Accuracy** | **73.60%** | Overall correct classification rate on holdout test data. |
| **Brier Score** | **0.1677** | Well-calibrated probabilistic output suitable for financial risk weighting. |

---

## 3. Confusion Matrix Breakdown

- **True Negatives (Retained correctly identified)**: 742 customers
- **True Positives (Churners successfully flagged)**: 295 customers
- **False Negatives (Missed churners)**: 79 customers
- **False Positives (Flagged but retained)**: 293 customers

---

## 4. Cross-Validation Model Comparison (5-Fold Stratified CV)

| Model Family | Mean ROC-AUC | Std ROC-AUC | Mean F1 | Mean Recall | Mean Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Tuned)** | **0.8438** | 0.0097 | **0.6235** | **0.7928** | **0.7458** |
| **Random Forest** | 0.8422 | 0.0098 | 0.6325 | 0.7752 | 0.7608 |
| **Gradient Boosting** | 0.8395 | 0.0117 | 0.5719 | 0.5145 | 0.7959 |
| **HistGradientBoosting** | 0.8357 | 0.0130 | 0.6239 | 0.7523 | 0.7594 |

---

## 5. Top Predictive Churn Drivers

1. **Contract Type (`is_month_to_month`)**: Month-to-month contracts are the #1 predictor of churn. Customers on annual contracts churn at less than a third of the rate.
2. **Tenure Duration (`tenure`)**: First 12 months exhibit the highest churn volatility. Retention stabilizes exponentially beyond month 24.
3. **Internet Infrastructure (`has_fiber_optic` vs Add-on Services)**: High churn propensity in Fiber Optic users lacking Tech Support or Online Security add-ons.
4. **Billing & Payment Method**: Paperless billing combined with Electronic Check payment correlates with higher delinquency and cancellation compared to automated bank transfers.
