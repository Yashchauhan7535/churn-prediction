"""Feature engineering and preprocessing pipeline module."""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.data_loader import NUMERICAL_COLS, CATEGORICAL_COLS

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering transformer:
    - Adds services_count (total number of active add-on services)
    - Adds is_month_to_month flag
    - Adds has_internet flag
    - Adds charge_difference (ratio of MonthlyCharges to expected monthly amortized charges)
    - Adds tenure_group categorization
    """

    def __init__(self):
        self.services = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Count active internet add-ons
        service_counts = np.zeros(len(X), dtype=int)
        for col in self.services:
            if col in X.columns:
                service_counts += (X[col] == "Yes").astype(int)
        X["services_count"] = service_counts

        # Contract month-to-month indicator
        if "Contract" in X.columns:
            X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)

        # Internet indicator
        if "InternetService" in X.columns:
            X["has_fiber_optic"] = (X["InternetService"] == "Fiber optic").astype(int)
            X["has_internet"] = (X["InternetService"] != "No").astype(int)

        # Automatic payment indicator
        if "PaymentMethod" in X.columns:
            X["is_automatic_payment"] = X["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)

        # Ratio of monthly charges to amortized total charges
        if "MonthlyCharges" in X.columns and "TotalCharges" in X.columns and "tenure" in X.columns:
            amortized = X["TotalCharges"] / np.maximum(X["tenure"], 1)
            X["charge_ratio"] = (X["MonthlyCharges"] / np.maximum(amortized, 1.0)).clip(0.1, 5.0)

        # Tenure binning (0-12, 12-24, 24-48, 48-72+)
        if "tenure" in X.columns:
            X["tenure_group"] = pd.cut(
                X["tenure"],
                bins=[-1, 12, 24, 48, 72, 100],
                labels=["0-12m", "12-24m", "24-48m", "48-72m", "72m+"],
            ).astype(str)

        return X


def build_preprocessor(
    num_cols: List[str] = None,
    cat_cols: List[str] = None,
) -> ColumnTransformer:
    """Build a standard scikit-learn ColumnTransformer."""
    if num_cols is None:
        num_cols = NUMERICAL_COLS + ["services_count", "is_month_to_month", "has_fiber_optic", "has_internet", "is_automatic_payment", "charge_ratio"]
    if cat_cols is None:
        cat_cols = CATEGORICAL_COLS + ["tenure_group"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def build_full_pipeline(classifier, num_cols=None, cat_cols=None) -> Pipeline:
    """Combine FeatureEngineer, ColumnTransformer, and Classifier into a single Pipeline."""
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)),
        ("classifier", classifier),
    ])
