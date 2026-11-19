"""Interactive Terminal Demonstration for Customer Churn Prediction."""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import ChurnPredictor


def print_banner():
    print("\n" + "=" * 65)
    print(" 📊 CUSTOMER CHURN PREDICTION — LIVE PRODUCTION DEMO")
    print("=" * 65 + "\n")


def display_customer_result(scenario_title, customer_data, result):
    print(f"--- {scenario_title} ---")
    print(f"• Tenure:           {customer_data['tenure']} months")
    print(f"• Contract:         {customer_data['Contract']}")
    print(f"• Internet:         {customer_data['InternetService']}")
    print(f"• Tech Support:     {customer_data['TechSupport']}")
    print(f"• Payment Method:   {customer_data['PaymentMethod']}")
    print(f"• Monthly Charges:  ${customer_data['MonthlyCharges']}")
    print(f"• Total Charges:    ${customer_data['TotalCharges']}")
    print("-" * 45)

    prob = result["churn_probability"] * 100
    risk = result["risk_tier"]

    if risk == "High Risk":
        badge = "🔴 HIGH CHURN RISK"
    elif risk == "Medium Risk":
        badge = "🟡 MEDIUM CHURN RISK"
    else:
        badge = "🟢 LOW CHURN RISK"

    print(f"Prediction:         {result['churn_prediction']} ({badge})")
    print(f"Churn Probability:  {prob:.1f}%")
    print(f"Action Required:    {'YES - Trigger Retention Campaign' if result['retention_action_recommended'] else 'NO - Routine Service'}")

    if result["retention_action_recommended"]:
        print("\n💡 Prescriptive Action Plan:")
        print("   1. Contract Incentive: Offer 15% discount for 1-year commitment.")
        print("   2. Support Bundling: Provide free Tech Support / Security for 6 months.")
        print("   3. Automated Billing: Credit $10 for enrollment in ACH auto-pay.")
    else:
        print("\n💡 Prescriptive Action Plan:")
        print("   Customer is highly satisfied and stable. Routine engagement.")
    print("=" * 65 + "\n")


def main():
    print_banner()
    model_path = "models/best_churn_model.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Model artifact not found at {model_path}.")
        return

    print("Loading serialized machine learning pipeline...")
    predictor = ChurnPredictor(model_path=model_path)
    print("Model loaded successfully!\n")
    time.sleep(0.5)

    # Scenario 1: High Risk Profile
    cust_high_risk = {
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
    res_high = predictor.predict_single(cust_high_risk)
    display_customer_result("Scenario 1: High-Risk Attrition Profile", cust_high_risk, res_high)
    time.sleep(1)

    # Scenario 2: Low Risk Profile
    cust_low_risk = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 48,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 54.0,
        "TotalCharges": 2592.0,
    }
    res_low = predictor.predict_single(cust_low_risk)
    display_customer_result("Scenario 2: Loyal Long-Term Customer Profile", cust_low_risk, res_low)


if __name__ == "__main__":
    main()
