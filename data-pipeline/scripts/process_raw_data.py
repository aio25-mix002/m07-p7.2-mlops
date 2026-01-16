#!/usr/bin/env python3
"""
Process train_period_1.csv to create df_processed.csv with engineered features.

This script creates the following engineered features required by prepare_feast_data.py:
- Tenure_Age_Ratio: Tenure / Age
- Spend_per_Usage: Total Spend / (Usage Frequency + 1)
- Support_Calls_per_Tenure: Support Calls / (Tenure + 1)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def process_raw_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Process raw churn data and create engineered features.

    Args:
        input_path: Path to train_period_1.csv
        output_path: Path to save df_processed.csv

    Returns:
        Processed DataFrame
    """
    # Load raw data
    df = pd.read_csv(input_path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing values")

    # Create engineered features
    # Tenure_Age_Ratio: ratio of tenure to age
    df['Tenure_Age_Ratio'] = df['Tenure'] / np.maximum(df['Age'], 1)

    # Spend_per_Usage: average spend per usage instance
    df['Spend_per_Usage'] = df['Total Spend'] / np.maximum(df['Usage Frequency'], 1)

    # Support_Calls_per_Tenure: support call frequency relative to tenure
    df['Support_Calls_per_Tenure'] = df['Support Calls'] / np.maximum(df['Tenure'], 1)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")
    print(f"Output shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")

    return df


if __name__ == "__main__":
    # Define paths relative to script location
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    input_file = base_dir / "data/raw/customer_churn_dataset-training-master.csv" #train_period_1.csv
    output_file = base_dir / "data/processed/df_processed.csv"

    process_raw_data(str(input_file), str(output_file))
