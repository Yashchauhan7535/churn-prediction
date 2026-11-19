"""Model training, model comparison, cross-validation, and serialization module."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from src.data_loader import load_raw_data, clean_data, split_data
from src.preprocessing import build_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate classification models with tuned baseline hyperparameters."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            C=1.0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        ),
    }


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
) -> pd.DataFrame:
    """Run stratified cross-validation across all candidate models and rank them."""
    models = get_candidate_models()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
    }

    results = []
    logger.info("Comparing candidate models using %d-fold Stratified CV...", cv_splits)

    for name, model in models.items():
        pipeline = build_full_pipeline(model)
        cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        results.append({
            "model": name,
            "mean_roc_auc": cv_res["test_roc_auc"].mean(),
            "std_roc_auc": cv_res["test_roc_auc"].std(),
            "mean_f1": cv_res["test_f1"].mean(),
            "std_f1": cv_res["test_f1"].std(),
            "mean_precision": cv_res["test_precision"].mean(),
            "mean_recall": cv_res["test_recall"].mean(),
            "mean_accuracy": cv_res["test_accuracy"].mean(),
        })
        logger.info(
            "[%s] CV ROC-AUC: %.4f (+/- %.4f) | F1: %.4f | Recall: %.4f",
            name,
            cv_res["test_roc_auc"].mean(),
            cv_res["test_roc_auc"].std(),
            cv_res["test_f1"].mean(),
            cv_res["test_recall"].mean(),
        )

    df_results = pd.DataFrame(results).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)
    return df_results


def tune_best_model(
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Perform targeted hyperparameter tuning on the top-performing model family."""
    logger.info("Performing hyperparameter tuning for %s...", best_model_name)
    if best_model_name == "GradientBoosting":
        base_model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__learning_rate": [0.03, 0.08, 0.12],
            "classifier__max_depth": [3, 4, 5],
            "classifier__subsample": [0.8, 1.0],
        }
    elif best_model_name == "RandomForest":
        base_model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
        param_grid = {
            "classifier__n_estimators": [100, 180],
            "classifier__max_depth": [8, 12, None],
            "classifier__min_samples_leaf": [2, 4],
        }
    elif best_model_name == "HistGradientBoosting":
        base_model = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
        param_grid = {
            "classifier__learning_rate": [0.03, 0.08, 0.12],
            "classifier__max_iter": [100, 150],
            "classifier__max_depth": [4, 6],
            "classifier__min_samples_leaf": [10, 20],
        }
    else:
        base_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
        }

    pipeline = build_full_pipeline(base_model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    logger.info("Best ROC-AUC score: %.4f with params: %s", grid_search.best_score_, grid_search.best_params_)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_and_export(
    data_path: str = "data/raw/customer_churn.csv",
    output_model_path: str = "models/best_churn_model.joblib",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """Complete end-to-end training and serialization pipeline."""
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    df_raw = load_raw_data(data_path)
    df_clean = clean_data(df_raw)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean, test_size=0.2, val_size=0.1, random_state=42)

    # Save splits for evaluation & reproducibility
    splits_dir = "data/processed"
    os.makedirs(splits_dir, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    # 1. Model comparison
    cv_comparison = compare_models(X_train, y_train, cv_splits=5)
    cv_comparison_path = os.path.join(reports_dir, "model_comparison.csv")
    cv_comparison.to_csv(cv_comparison_path, index=False)
    logger.info("Model comparison results:\n%s", cv_comparison.to_string())

    best_model_name = cv_comparison.iloc[0]["model"]
    logger.info("Top ranking model family: %s", best_model_name)

    # 2. Hyperparameter tuning on the best model
    best_pipeline, best_params, cv_score = tune_best_model(best_model_name, X_train, y_train)

    # 3. Validation set assessment
    y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = best_pipeline.predict(X_val)

    val_metrics = {
        "val_roc_auc": float(roc_auc_score(y_val, y_val_prob)),
        "val_f1": float(f1_score(y_val, y_val_pred)),
        "val_precision": float(precision_score(y_val, y_val_pred)),
        "val_recall": float(recall_score(y_val, y_val_pred)),
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
    }
    logger.info("Validation metrics: %s", val_metrics)

    # 4. Serialize final model pipeline
    joblib.dump(best_pipeline, output_model_path)
    logger.info("Serialized trained pipeline to %s", output_model_path)

    summary = {
        "best_model_name": best_model_name,
        "best_hyperparameters": best_params,
        "best_cv_roc_auc": float(cv_score),
        "validation_metrics": val_metrics,
        "artifact_path": output_model_path,
        "comparison_table": cv_comparison.to_dict(orient="records"),
    }

    metrics_path = os.path.join(reports_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train customer churn prediction models.")
    parser.add_argument("--data", type=str, default="data/raw/customer_churn.csv", help="Path to raw CSV data.")
    parser.add_argument("--output", type=str, default="models/best_churn_model.joblib", help="Target model artifact.")
    parser.add_argument("--reports", type=str, default="reports", help="Reports directory.")
    args = parser.parse_args()

    train_and_export(data_path=args.data, output_model_path=args.output, reports_dir=args.reports)


if __name__ == "__main__":
    main()
