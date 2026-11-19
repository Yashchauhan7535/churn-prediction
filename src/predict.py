"""Production inference module for single and batch churn predictions."""

import os
import sys
import json
import logging
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Union
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/best_churn_model.joblib"


class ChurnPredictor:
    """Production predictor wrapping the trained scikit-learn pipeline."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        logger.info("Loaded model pipeline from %s", model_path)

    @staticmethod
    def _categorize_risk(prob: float) -> str:
        if prob >= 0.70:
            return "High Risk"
        elif prob >= 0.40:
            return "Medium Risk"
        else:
            return "Low Risk"

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on a single customer dictionary."""
        df = pd.DataFrame([customer_data])
        prob = float(self.pipeline.predict_proba(df)[:, 1][0])
        pred = int(self.pipeline.predict(df)[0])

        return {
            "churn_prediction": "Yes" if pred == 1 else "No",
            "churn_probability": round(prob, 4),
            "risk_tier": self._categorize_risk(prob),
            "retention_action_recommended": prob >= 0.40,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make batch predictions on a DataFrame."""
        df_out = df.copy()
        probs = self.pipeline.predict_proba(df)[:, 1]
        preds = self.pipeline.predict(df)

        df_out["churn_probability"] = probs.round(4)
        df_out["churn_prediction"] = ["Yes" if p == 1 else "No" for p in preds]
        df_out["risk_tier"] = [self._categorize_risk(p) for p in probs]
        df_out["retention_action"] = probs >= 0.40
        return df_out


def main():
    parser = argparse.ArgumentParser(description="Run inference using trained churn prediction model.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model.")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV for batch prediction.")
    parser.add_argument("--output", type=str, default="data/predictions.csv", help="Path to save predictions.")
    args = parser.parse_args()

    predictor = ChurnPredictor(model_path=args.model)

    if args.input:
        df = pd.read_csv(args.input)
        preds = predictor.predict_batch(df)
        preds.to_csv(args.output, index=False)
        print(f"Batch predictions saved to {args.output}")
        print(preds[["churn_prediction", "churn_probability", "risk_tier"]].head())
    else:
        # Sample customer inference test
        sample_customer = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
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
            "TotalCharges": 179.0,
        }
        res = predictor.predict_single(sample_customer)
        print("Sample Customer Prediction:")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
