#!/usr/bin/env python3
"""
Script to generate test.parquet from raw test CSV data.
Applies the same feature engineering as the training data.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def generate_test_parquet(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Convert raw test CSV to Feast-compatible parquet format.

    Args:
        input_path: Path to raw test CSV file
        output_path: Path to save processed parquet file

    Returns:
        Processed DataFrame
    """
    # Load raw test data
    df = pd.read_csv(input_path)
    print(f"Loaded raw data. Shape: {df.shape}")

    # Add timestamp columns required by Feast
    current_time = datetime.now()

    # Create event timestamp (simulate data from last 90 days)
    np.random.seed(42)  # For reproducibility
    df['event_timestamp'] = current_time - pd.to_timedelta(
        np.random.randint(0, 90 * 24 * 60 * 60, size=len(df)),
        unit='s'
    )

    # Create created timestamp
    df['created_timestamp'] = current_time

    # customer_id as int
    df['customer_id'] = df['CustomerID'].astype(float).astype(int)

    # Compute engineered features
    df['Tenure_Age_Ratio'] = df['Tenure'] / np.maximum(df['Age'], 1)
    df['Spend_per_Usage'] = df['Total Spend'] / np.maximum(df['Usage Frequency'], 1)
    df['Support_Calls_per_Tenure'] = df['Support Calls'] / np.maximum(df['Tenure'], 1)

    # Map column names to Feast-compatible names
    column_mapping = {
        'Age': 'age',
        'Gender': 'gender',
        'Tenure': 'tenure_months',
        'Usage Frequency': 'usage_frequency',
        'Support Calls': 'support_calls',
        'Payment Delay': 'payment_delay_days',
        'Subscription Type': 'subscription_type',
        'Contract Length': 'contract_length',
        'Total Spend': 'total_spend',
        'Last Interaction': 'last_interaction_days',
        'Churn': 'churned',
        'Tenure_Age_Ratio': 'tenure_age_ratio',
        'Spend_per_Usage': 'spend_per_usage',
        'Support_Calls_per_Tenure': 'support_calls_per_tenure',
    }

    df = df.rename(columns=column_mapping)

    # Add avg_monthly_spend
    df['avg_monthly_spend'] = df['total_spend'] / np.maximum(df['tenure_months'], 1)

    # Add churn_risk_score
    df['churn_risk_score'] = (
        df['payment_delay_days'] * 0.3 +
        (df['support_calls'] / np.maximum(df['tenure_months'], 1)) * 0.2 +
        (1 - (df['last_interaction_days'] / 30)) * 0.5
    ).clip(0, 1)

    # Select and order columns for Feast
    feast_columns = [
        'customer_id',
        'event_timestamp',
        'created_timestamp',
        'age',
        'gender',
        'tenure_months',
        'usage_frequency',
        'support_calls',
        'payment_delay_days',
        'subscription_type',
        'contract_length',
        'total_spend',
        'last_interaction_days',
        'tenure_age_ratio',
        'spend_per_usage',
        'support_calls_per_tenure',
        'avg_monthly_spend',
        'churn_risk_score',
        'churned'
    ]

    df_feast = df[feast_columns].copy()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as Parquet
    df_feast.to_parquet(output_path, index=False)

    print(f"Test data prepared. Shape: {df_feast.shape}")
    print(f"Saved to: {output_path}")
    print(f"\nColumns: {df_feast.columns.tolist()}")

    return df_feast


if __name__ == "__main__":
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    input_path = repo_root / "data/raw/customer_churn_dataset-testing-master.csv"
    output_path = repo_root / "churn_feature_store/churn_features/feature_repo/data/test.parquet"

    generate_test_parquet(str(input_path), str(output_path))
