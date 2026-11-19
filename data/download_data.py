"""Dataset acquisition script for IBM Telco Customer Churn data."""

import os
import argparse
import logging
import urllib.request
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
BACKUP_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def download_dataset(output_path: str = "data/raw/Telco-Customer-Churn.csv", force: bool = False) -> str:
    """Download the official IBM Telco Customer Churn benchmark dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and not force:
        logger.info("Dataset already exists at %s. Use --force to re-download.", output_path)
        return output_path

    urls = [PRIMARY_URL, BACKUP_URL]
    success = False

    for url in urls:
        try:
            logger.info("Downloading dataset from: %s ...", url)
            urllib.request.urlretrieve(url, output_path)
            logger.info("Successfully downloaded to %s", output_path)
            success = True
            break
        except Exception as e:
            logger.warning("Failed downloading from %s: %s", url, e)

    if not success:
        raise RuntimeError("Unable to download dataset from any available remote mirrors.")

    # Validation
    df = pd.read_csv(output_path)
    logger.info("Dataset validated: %d rows, %d columns.", df.shape[0], df.shape[1])
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download real Telco Customer Churn dataset.")
    parser.add_argument("--output", type=str, default="data/raw/Telco-Customer-Churn.csv", help="Target output CSV.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if exists.")
    args = parser.parse_args()

    download_dataset(output_path=args.output, force=args.force)


if __name__ == "__main__":
    main()
