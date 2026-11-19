"""Interactive Customer Churn Prediction & Retention Dashboard (Streamlit)."""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import ChurnPredictor

st.set_page_config(
    page_title="Customer Churn Prediction Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/best_churn_model.joblib"


@st.cache_resource
def load_predictor():
    if not os.path.exists(MODEL_PATH):
        return None
    return ChurnPredictor(model_path=MODEL_PATH)


def main():
    st.title("📊 Customer Churn Prediction & Retention Intelligence")
    st.markdown(
        "Production machine learning pipeline predicting customer attrition risk, identifying critical risk drivers, and recommending targeted retention interventions."
    )

    predictor = load_predictor()
    if predictor is None:
        st.error(f"Trained model not found at `{MODEL_PATH}`. Please run `python src/train.py` first.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🎯 Single Customer Scoring", "📁 Batch Portfolio Scoring", "📈 Model Performance & Explainability"])

    # --- TAB 1: SINGLE CUSTOMER SCORING ---
    with tab1:
        st.sidebar.header("Customer Profile Parameters")

        # Demographics
        st.sidebar.subheader("1. Demographics")
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        senior = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])

        # Account & Contract
        st.sidebar.subheader("2. Contract & Account")
        tenure = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=6)
        contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.sidebar.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        # Charges
        st.sidebar.subheader("3. Monthly & Total Charges")
        monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0, step=0.5)
        auto_calc_total = st.sidebar.checkbox("Auto-estimate Total Charges", value=True)
        if auto_calc_total:
            total_charges = round(monthly_charges * tenure, 2)
            st.sidebar.info(f"Estimated Total Charges: ${total_charges}")
        else:
            total_charges = st.sidebar.number_input("Total Charges ($)", min_value=18.0, max_value=10000.0, value=450.0)

        # Phone & Internet Services
        st.sidebar.subheader("4. Subscribed Services")
        phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = "No phone service" if phone_service == "No" else st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])

        internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        if internet_service == "No":
            security = backup = device = tech = tv = movies = "No internet service"
        else:
            col_s1, col_s2 = st.sidebar.columns(2)
            with col_s1:
                security = st.selectbox("Online Security", ["No", "Yes"])
                backup = st.selectbox("Online Backup", ["No", "Yes"])
                device = st.selectbox("Device Protection", ["No", "Yes"])
            with col_s2:
                tech = st.selectbox("Tech Support", ["No", "Yes"])
                tv = st.selectbox("Streaming TV", ["No", "Yes"])
                movies = st.selectbox("Streaming Movies", ["No", "Yes"])

        # Construct single customer record
        customer_payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Customer Prediction Result")
            res = predictor.predict_single(customer_payload)
            prob = res["churn_probability"]
            risk_tier = res["risk_tier"]

            # Visual metric
            if risk_tier == "High Risk":
                st.error(f"### ⚠️ High Churn Risk: {prob * 100:.1f}%")
            elif risk_tier == "Medium Risk":
                st.warning(f"### ⚡ Moderate Churn Risk: {prob * 100:.1f}%")
            else:
                st.success(f"### ✅ Low Churn Risk: {prob * 100:.1f}%")

            st.progress(float(prob))

            st.write(f"**Predicted Status:** `{'Will Churn' if res['churn_prediction'] == 'Yes' else 'Will Stay'}`")
            st.write(f"**Risk Category:** `{risk_tier}`")
            st.write(f"**Retention Intervention Required:** `{'Yes' if res['retention_action_recommended'] else 'No'}`")

        with col_right:
            st.subheader("Actionable Retention Strategy")
            if prob >= 0.60:
                st.markdown(
                    """
                    **Recommended Strategy: Immediate Retention Intervention**
                    1. **Contract Upgrade Incentive**: Offer a discount for transitioning from *Month-to-month* to a *1-Year commitment*.
                    2. **Free Tech Support / Security Add-on**: Include bundled tech support or online security for 6 months.
                    3. **Payment Method Incentive**: Provide a \$10 account credit for switching to automated payment (ACH/Credit Card).
                    """
                )
            elif prob >= 0.35:
                st.markdown(
                    """
                    **Recommended Strategy: Proactive Engagement**
                    1. **Customer Satisfaction Check-in**: Automated CSAT outreach regarding connection quality.
                    2. **Loyalty Milestone Reward**: Offer a streaming bundle discount or speed upgrade.
                    """
                )
            else:
                st.markdown(
                    """
                    **Recommended Strategy: Standard Relationship Management**
                    - Customer shows high retention probability. Maintain standard communications and billing.
                    """
                )

    # --- TAB 2: BATCH SCORING ---
    with tab2:
        st.subheader("Portfolio Batch Scoring")
        st.write("Upload a CSV customer file to generate churn probabilities and risk categories at scale.")

        uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} customer records.")

            with st.spinner("Scoring portfolio..."):
                scored_df = predictor.predict_batch(batch_df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Customers", len(scored_df))
            col2.metric("High Risk Customers", int((scored_df["risk_tier"] == "High Risk").sum()))
            col3.metric("Avg Churn Probability", f"{scored_df['churn_probability'].mean() * 100:.1f}%")

            st.dataframe(scored_df[["customerID", "Contract", "tenure", "MonthlyCharges", "churn_probability", "risk_tier"]].head(25) if "customerID" in scored_df.columns else scored_df.head(25))

            csv_data = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored CSV",
                data=csv_data,
                file_name="scored_customers.csv",
                mime="text/csv",
            )

    # --- TAB 3: MODEL PERFORMANCE & EXPLAINABILITY ---
    with tab3:
        st.subheader("Holdout Model Validation & Diagnostics")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if os.path.exists("reports/figures/roc_curve.png"):
                st.image("reports/figures/roc_curve.png", caption="ROC Curve on Holdout Test Set")
            if os.path.exists("reports/figures/confusion_matrix.png"):
                st.image("reports/figures/confusion_matrix.png", caption="Holdout Confusion Matrix")
        with col_m2:
            if os.path.exists("reports/figures/feature_importance.png"):
                st.image("reports/figures/feature_importance.png", caption="Key Predictive Feature Importances")
            if os.path.exists("reports/figures/precision_recall_curve.png"):
                st.image("reports/figures/precision_recall_curve.png", caption="Precision-Recall Curve")


if __name__ == "__main__":
    main()
