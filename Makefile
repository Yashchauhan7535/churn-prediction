.PHONY: help setup data train eval test app clean

help:
	@echo "Available commands:"
	@echo "  make setup   - Install dependencies in virtual environment"
	@echo "  make data    - Download official Telco Customer Churn dataset"
	@echo "  make train   - Train and cross-validate churn prediction models"
	@echo "  make eval    - Evaluate best model on test set and generate plots"
	@echo "  make test    - Run automated test suite with pytest"
	@echo "  make app     - Launch interactive Streamlit dashboard"
	@echo "  make clean   - Remove cache and build files"

setup:
	python3 -m pip install -r requirements.txt

data:
	python data/download_data.py --output data/raw/Telco-Customer-Churn.csv

train:
	python src/train.py --data data/raw/Telco-Customer-Churn.csv --output models/best_churn_model.joblib

eval:
	python src/evaluate.py --model models/best_churn_model.joblib --test data/processed/test.csv

test:
	pytest -v

app:
	streamlit run app.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
