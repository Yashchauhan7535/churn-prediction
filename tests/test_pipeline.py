"""Tests for pipeline preprocessing and inference."""

import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data.generate_data import generate_churn_dataset
from src.data_loader import clean_data, TARGET_COL
from src.preprocessing import FeatureEngineer, build_full_pipeline


def test_feature_engineer():
    df = generate_churn_dataset(n_samples=50, random_state=42)
    fe = FeatureEngineer()
    df_transformed = fe.transform(df)

    assert "services_count" in df_transformed.columns
    assert "is_month_to_month" in df_transformed.columns
    assert "has_fiber_optic" in df_transformed.columns
    assert "charge_ratio" in df_transformed.columns
    assert "tenure_group" in df_transformed.columns


def test_pipeline_fit_predict():
    df = generate_churn_dataset(n_samples=100, random_state=42)
    df_clean = clean_data(df)
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    clf = LogisticRegression(max_iter=200)
    pipeline = build_full_pipeline(clf)

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == 100
    assert set(preds).issubset({0, 1})
    assert probs.shape == (100, 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
