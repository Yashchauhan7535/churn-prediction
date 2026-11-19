"""Data loading, cleaning, and train/test splitting module."""

import os
import logging
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET_COL = "Churn"


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    df = pd.read_csv(data_path)
    logger.info("Loaded raw data with shape %s from %s", df.shape, data_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Coerce TotalCharges to numeric (handles whitespace strings)
    - Handle missing values
    - Convert binary Churn target to 0/1
    """
    df = df.copy()

    # Drop customerID identifier column if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Coerce TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Impute missing TotalCharges with MonthlyCharges * tenure or median
        mask = df["TotalCharges"].isna()
        if mask.any():
            df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"].clip(lower=1)
            logger.info("Imputed %d missing TotalCharges values.", mask.sum())

    # Map target Churn to binary integer
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].replace({"Yes": 1, "No": 0}).astype(int)

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], pd.Series, pd.Series, Optional[pd.Series]]:
    """
    Split cleaned data into train, val, and test partitions with stratification.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # First split into train_val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if val_size > 0:
        val_relative_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
        )
        logger.info(
            "Split data into Train: %s, Val: %s, Test: %s",
            X_train.shape,
            X_val.shape,
            X_test.shape,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        logger.info("Split data into Train: %s, Test: %s", X_train_val.shape, X_test.shape)
        return X_train_val, None, X_test, y_train_val, None, y_test
