"""Tests for data generation and cleaning routines."""

import pytest
import pandas as pd
import numpy as np
from data.generate_data import generate_churn_dataset
from src.data_loader import clean_data, split_data, TARGET_COL


def test_generate_churn_dataset():
    df = generate_churn_dataset(n_samples=100, random_state=42)
    assert len(df) == 100
    assert "Churn" in df.columns
    assert "MonthlyCharges" in df.columns
    assert "tenure" in df.columns
    assert set(df["Churn"].unique()).issubset({"Yes", "No"})
    assert df["MonthlyCharges"].min() >= 18.0


def test_clean_data():
    raw_data = pd.DataFrame({
        "customerID": ["001-A", "002-B"],
        "tenure": [5, 0],
        "MonthlyCharges": [70.5, 20.0],
        "TotalCharges": ["352.5", " "],  # Whitespace missing value
        "Churn": ["Yes", "No"],
    })
    df_clean = clean_data(raw_data)
    assert "customerID" not in df_clean.columns
    assert df_clean["TotalCharges"].dtype == float
    assert df_clean["TotalCharges"].isna().sum() == 0
    assert list(df_clean[TARGET_COL]) == [1, 0]


def test_split_data():
    df = generate_churn_dataset(n_samples=200, random_state=42)
    df_clean = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df_clean, test_size=0.2, val_size=0.1, random_state=42
    )
    assert len(X_train) == 140
    assert len(X_val) == 20
    assert len(X_test) == 40
    assert len(y_train) == 140
    assert len(y_val) == 20
    assert len(y_test) == 40
