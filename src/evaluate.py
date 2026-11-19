"""Model evaluation, metrics generation, feature importance, and diagnostic plotting module."""

import os
import sys
import json
import logging
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
)

from src.data_loader import TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str = "models/best_churn_model.joblib",
    test_data_path: str = "data/processed/test.csv",
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """Evaluate trained model on the holdout test set and produce metric artifacts and plots."""
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load artifact & data
    pipeline = joblib.load(model_path)
    df_test = pd.read_csv(test_data_path)

    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    # 2. Predict probabilities & classes
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # 3. Calculate metrics
    roc_auc = float(roc_auc_score(y_test, y_prob))
    avg_precision = float(average_precision_score(y_test, y_prob))
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    brier = float(brier_score_loss(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    cls_report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(avg_precision, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "classification_report": cls_report,
    }
    logger.info("Test Evaluation Summary: %s", metrics)

    # 4. Generate Visualizations
    sns.set_theme(style="whitegrid")

    # (a) Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm_array = np.array(cm)
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
    )
    plt.title("Confusion Matrix (Holdout Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # (b) ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(fig_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # (c) Precision-Recall Curve Plot
    p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(r_curve, p_curve, color="#059669", lw=2.5, label=f"PR Curve (AP = {avg_precision:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = os.path.join(fig_dir, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # (d) Feature Importance Plot (if model supports it)
    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feat_{i}" for i in range(100)]

        importances = None
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])

        if importances is not None and len(importances) == len(feature_names):
            fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi_df["clean_feature"] = fi_df["feature"].str.replace("cat__", "").str.replace("num__", "")
            top_fi = fi_df.sort_values("importance", ascending=False).head(15)

            plt.figure(figsize=(9, 6))
            sns.barplot(data=top_fi, x="importance", y="clean_feature", palette="viridis")
            plt.title("Top 15 Feature Importances", fontsize=13, fontweight="bold", pad=12)
            plt.xlabel("Importance / Absolute Weight")
            plt.ylabel("Engineered Feature")
            plt.tight_layout()
            fi_path = os.path.join(fig_dir, "feature_importance.png")
            plt.savefig(fi_path, dpi=200)
            plt.close()
            metrics["top_features"] = top_fi[["clean_feature", "importance"]].to_dict(orient="records")
    except Exception as e:
        logger.warning("Could not generate feature importance plot: %s", e)

    # 5. Save test metrics JSON
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved evaluation metrics to %s", metrics_path)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained churn prediction model.")
    parser.add_argument("--model", type=str, default="models/best_churn_model.joblib", help="Model path.")
    parser.add_argument("--test", type=str, default="data/processed/test.csv", help="Test CSV path.")
    parser.add_argument("--reports", type=str, default="reports", help="Reports directory.")
    args = parser.parse_args()

    evaluate_model(model_path=args.model, test_data_path=args.test, output_dir=args.reports)


if __name__ == "__main__":
    main()
