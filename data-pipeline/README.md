# Data Pipeline MLOps – Customer Churn Feature Store

This repository demonstrates an **end-to-end data pipeline** for a customer churn use case, including:

- Data processing & feature engineering
- Data versioning with **DVC**
- Feature storage with **Feast**
- Online feature serving with **Redis**

---

## 1. Repository Installation

Clone the repository:

```bash
git clone https://github.com/dangnha/data-pipeline.git
cd data-pipeline
```

---

## 2. Environment Setup

### Option A: Conda (Recommended)

```bash
conda create -n churn_mlops python=3.10 -y
conda activate churn_mlops
```

### Option B: Virtual Environment (venv)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Data Processing – Create df_processed.csv

Process raw data and create engineered features:

```bash
python scripts/process_raw_data.py
```

This script:

- Loads raw data from `data/raw/customer_churn_dataset-training-master.csv`
- Creates engineered features:
  - `Tenure_Age_Ratio`: Tenure / Age
  - `Spend_per_Usage`: Total Spend / Usage Frequency
  - `Support_Calls_per_Tenure`: Support Calls / Tenure
- Saves processed data to `data/processed/df_processed.csv`

---

## 5. Data Versioning with DVC

### 5.1 Initialize DVC (First Time Only)

```bash
dvc init
```

### 5.2 Configure Remote Storage (S3)

First, create an S3 bucket on AWS Console (e.g., `churn-mlops-dvc`), then configure DVC:

```bash
dvc remote add -d s3remote s3://churn-mlops-dvc/data
dvc remote modify s3remote region ap-southeast-1
```

Commit remote configuration to Git:

```bash
git add .dvc/config
git commit -m "Configure DVC remote storage"
```

### 5.3 Add Data to DVC Tracking

```bash
dvc add data/processed/df_processed.csv
```

### 5.4 Commit Changes to Git

```bash
git add data/processed/df_processed.csv.dvc .gitignore
git commit -m "Add processed data to DVC"
```

### 5.5 Push Data to Remote Storage

```bash
dvc push
```

### 5.6 Pull Data (For Collaborators)

Clone the repository and pull data from S3:

```bash
git clone <repository-url>
cd data-pipeline
dvc pull
```

> **Note:** Ensure AWS credentials are configured (`aws configure`) or set via environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

---

## 6. Start Redis (Online Store)

Run Redis using Docker:

```bash
docker run -d -p 6379:6379 --name redis-feast redis:7
```

Verify Redis is running:

```bash
docker ps
```

---

## 7. Prepare Feast Data

Convert processed data to Feast-compatible Parquet format:

```bash
cd churn_feature_store/churn_features/feature_repo
python prepare_feast_data.py
```

This script:

- Loads `df_processed.csv`
- Adds `event_timestamp` and `created_timestamp` columns
- Maps column names to Feast-compatible format
- Creates additional features: `avg_monthly_spend`, `churn_risk_score`
- Saves as `data/processed_churn_data.parquet`

---

## 8. Feast Feature Store Setup

### 8.1 Apply Feast Repository

```bash
cd churn_feature_store/churn_features/feature_repo
feast apply
```

---

### 8.2 Materialize Features to Online Store

Run incremental materialization using the current timestamp:

**Linux / macOS:**

```bash
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
```

**Windows (PowerShell):**

```powershell
feast materialize-incremental (Get-Date -Format yyyy-MM-ddTHH:mm:ss)
```

**Windows (CMD):**

```cmd
for /f "delims=" %i in ('powershell -Command "Get-Date -Format yyyy-MM-ddTHH:mm:ss"') do feast materialize-incremental %i
```

---

## 9. Online Feature Retrieval

Return to the project root and run the sample retrieval script:

```bash
cd ../../..
python scripts/sample_retrieval.py
```

Expected output:

- Feature values retrieved from Redis
- Retrieves features for `customer_id` range (2-40)
- Features include: `age`, `gender`, `tenure_months`, `subscription_type`, `contract_length`, `usage_frequency`, `support_calls`, `payment_delay_days`, `total_spend`, `last_interaction_days`

---
