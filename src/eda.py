"""Exploratory Data Analysis script for customer churn dataset."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_eda_figures(
    data_path: str = "data/raw/customer_churn.csv",
    output_dir: str = "reports/figures",
):
    """Generate exploratory data analysis visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Churn by Contract Type
    plt.figure(figsize=(8, 5))
    contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    contract_churn_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
    contract_churn_pct.plot(kind="bar", stacked=True, color=["#3b82f6", "#ef4444"], figsize=(8, 5))
    plt.title("Churn Rate by Contract Type (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Contract Type")
    plt.ylabel("Percentage of Customers")
    plt.legend(title="Churn", labels=["No", "Yes"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_churn_by_contract.png"), dpi=200)
    plt.close()

    # 2. Tenure vs Churn
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x="tenure", hue="Churn", common_norm=False, fill=True, alpha=0.4, palette=["#3b82f6", "#ef4444"])
    plt.title("Tenure Distribution by Churn Status", fontsize=13, fontweight="bold")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_tenure_distribution.png"), dpi=200)
    plt.close()

    # 3. Monthly Charges vs Churn
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=["#3b82f6", "#ef4444"])
    plt.title("Monthly Charges vs Churn Status", fontsize=13, fontweight="bold")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_monthly_charges_boxplot.png"), dpi=200)
    plt.close()

    print(f"Saved EDA visualizations to {output_dir}")


if __name__ == "__main__":
    generate_eda_figures()
