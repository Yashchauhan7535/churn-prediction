"""
Synthetic Telecom Customer Churn Dataset Generator.
Produces a realistic dataset following industry churn dynamics (contract types, tenure,
charges, services) with realistic correlations and ground truth churn probabilities.
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_churn_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    # 1. Demographics
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}" for _ in range(n_samples)]
    gender = np.random.choice(["Male", "Female"], size=n_samples)
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    # Dependents correlated with partner
    dependents = []
    for p in partner:
        if p == "Yes":
            dependents.append(np.random.choice(["Yes", "No"], p=[0.55, 0.45]))
        else:
            dependents.append(np.random.choice(["Yes", "No"], p=[0.08, 0.92]))
    dependents = np.array(dependents)

    # 2. Tenure & Contract
    # Contracts: Month-to-month, One year, Two year
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.21, 0.24])
    tenure = []
    for c in contract:
        if c == "Month-to-month":
            t = np.random.exponential(scale=14) + 1
        elif c == "One year":
            t = np.random.normal(loc=35, scale=12)
        else:
            t = np.random.normal(loc=55, scale=12)
        tenure.append(int(np.clip(t, 1, 72)))
    tenure = np.array(tenure)

    # 3. Services
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
    multiple_lines = np.array(multiple_lines)

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22])

    def add_internet_subservice(p_yes):
        res = []
        for iserv in internet_service:
            if iserv == "No":
                res.append("No internet service")
            else:
                res.append(np.random.choice(["Yes", "No"], p=[p_yes, 1.0 - p_yes]))
        return np.array(res)

    online_security = add_internet_subservice(0.35)
    online_backup = add_internet_subservice(0.42)
    device_protection = add_internet_subservice(0.40)
    tech_support = add_internet_subservice(0.32)
    streaming_tv = add_internet_subservice(0.46)
    streaming_movies = add_internet_subservice(0.48)

    # 4. Billing & Payment
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.22, 0.22, 0.22]
    )

    # 5. Charges calculation
    monthly_charges = []
    for i in range(n_samples):
        base = 20.0
        if phone_service[i] == "Yes":
            base += 5.0
            if multiple_lines[i] == "Yes":
                base += 10.0
        if internet_service[i] == "DSL":
            base += 25.0
        elif internet_service[i] == "Fiber optic":
            base += 45.0

        for s in [online_security[i], online_backup[i], device_protection[i], tech_support[i], streaming_tv[i], streaming_movies[i]]:
            if s == "Yes":
                base += 6.5
        # Add slight gaussian noise
        noise = np.random.normal(0, 2.5)
        monthly_charges.append(round(max(18.5, min(120.0, base + noise)), 2))
    monthly_charges = np.array(monthly_charges)

    total_charges = []
    for i in range(n_samples):
        # Total charges = tenure * monthly charges with slight variation
        tot = tenure[i] * monthly_charges[i] * np.random.uniform(0.95, 1.05)
        total_charges.append(round(tot, 2))
    total_charges = np.array(total_charges)

    # 6. Realistic Churn Probability Model (Log-odds)
    # Baseline log-odds
    log_odds = -1.2

    for i in range(n_samples):
        lo = log_odds
        # Month-to-month contracts strongly increase churn
        if contract[i] == "Month-to-month":
            lo += 1.3
        elif contract[i] == "Two year":
            lo -= 1.4
        else:
            lo -= 0.6

        # Tenure effect (newer customers churn more easily)
        if tenure[i] <= 6:
            lo += 1.1
        elif tenure[i] <= 18:
            lo += 0.4
        elif tenure[i] >= 48:
            lo -= 1.0

        # Fiber optic customers without tech support have highest dissatisfaction
        if internet_service[i] == "Fiber optic":
            lo += 0.7
            if tech_support[i] == "No":
                lo += 0.6
            if online_security[i] == "No":
                lo += 0.4

        # Electronic check customers have higher churn propensity
        if payment_method[i] == "Electronic check":
            lo += 0.5
        elif "automatic" in payment_method[i]:
            lo -= 0.4

        # High charges increase risk
        if monthly_charges[i] > 80:
            lo += 0.35

        # Senior citizen slight risk increase
        if senior_citizen[i] == 1:
            lo += 0.25

        # Family support (partner, dependents) stabilizes retention
        if partner[i] == "Yes":
            lo -= 0.25
        if dependents[i] == "Yes":
            lo -= 0.35

        prob = 1.0 / (1.0 + np.exp(-lo))
        churn_decision = "Yes" if np.random.uniform() < prob else "No"
        churn_list = churn_decision if i == 0 else churn_list
        if i == 0:
            churn_arr = [churn_decision]
        else:
            churn_arr.append(churn_decision)

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_arr,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic customer churn dataset.")
    parser.add_argument("--samples", type=int, default=5000, help="Number of customer samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="data/raw/customer_churn.csv", help="Output path for CSV.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = generate_churn_dataset(n_samples=args.samples, random_state=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} samples and saved to {args.output}")
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True).round(3)}")


if __name__ == "__main__":
    main()
