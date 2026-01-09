"""
Script to add fake timestamps to datasets for Evidently AI drift monitoring.

This script processes data to match production.csv format (keeping original columns) and adds timestamps:
- Training dataset: timestamps from months 1, 2 (reference/baseline data)
- Testing dataset: timestamps from months 3, 4 (current data for comparison)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame to match production.csv format (keep original columns, no encoding).
    
    Args:
        df: Raw DataFrame with original columns
        
    Returns:
        Cleaned DataFrame with original format (like production.csv)
    """
    df = df.copy()
    
    # Rename columns to match production.csv format (handle spaces)
    column_mapping = {
        'Usage Frequency': 'Usage_Frequency',
        'Support Calls': 'Support_Calls',
        'Payment Delay': 'Payment_Delay',
        'Subscription Type': 'Subscription_Type',
        'Contract Length': 'Contract_Length',
        'Total Spend': 'Total_Spend',
        'Last Interaction': 'Last_Interaction'
    }
    df = df.rename(columns=column_mapping)
    
    # Remove CustomerID if exists (not needed for drift monitoring)
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)
    
    # Replace inf with NaN (to be dropped later)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Keep Churn column if it exists (for reference data)
    if 'Churn' in df.columns:
        # Replace inf with NaN
        df['Churn'] = df['Churn'].replace([np.inf, -np.inf], np.nan)
        # Drop rows with NaN in Churn before converting to int
        df = df.dropna(subset=['Churn'])
        df['Churn'] = df['Churn'].astype(int)
    
    # Drop all rows with any NaN values in other columns
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"  Dropped {dropped_rows} rows with NaN values (from {initial_rows} to {len(df)} rows)")
    
    return df


def add_fake_timestamps_by_months(
    input_file: str,
    output_file: str,
    months: list,
    year: int = None,
    timestamp_column: str = "timestamp"
):
    """
    Preprocess data and add fake timestamps distributed across specified months.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        months: List of month numbers (1-12) to distribute timestamps
        year: Year to use (default: current year)
        timestamp_column: Name of timestamp column to add
    """
    print(f"Reading {input_file}...")
    df_raw = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df_raw.shape}")
    
    # Prepare data to match production.csv format (keep original columns)
    print("Preparing data (keeping original format)...")
    df = prepare_dataframe(df_raw)
    
    print(f"Prepared dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Use current year if not specified
    if year is None:
        year = datetime.now().year
    
    # Generate timestamps distributed across specified months
    n_rows = len(df)
    timestamps = []
    
    for i in range(n_rows):
        # Randomly select a month from the list
        month = np.random.choice(months)
        
        # Get number of days in that month
        if month == 12:
            days_in_month = 31
        else:
            next_month = datetime(year, month + 1, 1)
            days_in_month = (next_month - timedelta(days=1)).day
        
        # Random day, hour, minute, second within the month
        day = np.random.randint(1, days_in_month + 1)
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        timestamp = datetime(year, month, day, hour, minute, second)
        timestamps.append(timestamp)
    
    # Sort timestamps to make them more realistic (earlier records first)
    timestamps.sort()
    
    # Add timestamp column to dataframe
    df[timestamp_column] = timestamps
    
    # Add prediction and churn_probability columns to match production.csv format
    # If Churn exists, generate prediction with some accuracy (not 100% match)
    if 'Churn' in df.columns:
        # Generate prediction with ~85% accuracy (realistic model performance)
        # This simulates that the model is not perfect
        np.random.seed(42)  # For reproducibility
        accuracy = 0.85
        
        # Start with Churn as base
        df['prediction'] = df['Churn'].astype(int)
        
        # Introduce errors: flip ~15% of predictions
        n_samples = len(df)
        n_errors = int(n_samples * (1 - accuracy))
        error_indices = np.random.choice(n_samples, size=n_errors, replace=False)
        df.loc[error_indices, 'prediction'] = 1 - df.loc[error_indices, 'prediction']
        
        # Generate churn_probability based on prediction (not Churn)
        # If prediction=1, use high probability (0.7-1.0), else low (0.0-0.3)
        df['churn_probability'] = np.where(
            df['prediction'] == 1,
            np.random.uniform(0.7, 1.0, len(df)),
            np.random.uniform(0.0, 0.3, len(df))
        )
        # Keep Churn column (for reference data and drift monitoring)
    else:
        # If no Churn, set prediction and churn_probability to None/0
        df['prediction'] = 0
        df['churn_probability'] = 0.0
    
    # Reorder columns: features, timestamp, Churn (if exists), prediction, churn_probability
    # Age, Gender, Tenure, Usage_Frequency, Support_Calls, Payment_Delay, 
    # Subscription_Type, Contract_Length, Total_Spend, Last_Interaction, 
    # timestamp, Churn (if exists), prediction, churn_probability
    production_columns = [
        'Age', 'Gender', 'Tenure', 'Usage_Frequency', 'Support_Calls', 
        'Payment_Delay', 'Subscription_Type', 'Contract_Length', 
        'Total_Spend', 'Last_Interaction', timestamp_column
    ]
    
    # Add Churn if it exists
    if 'Churn' in df.columns:
        production_columns.append('Churn')
    
    # Add prediction and churn_probability
    production_columns.extend(['prediction', 'churn_probability'])
    
    # Only keep columns that exist in df
    final_columns = [col for col in production_columns if col in df.columns]
    df = df[final_columns]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Save to output file
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    month_names = {1: "January", 2: "February", 3: "March", 4: "April",
                   5: "May", 6: "June", 7: "July", 8: "August",
                   9: "September", 10: "October", 11: "November", 12: "December"}
    months_str = ", ".join([month_names[m] for m in sorted(months)])
    print(f"✓ Prepared data and added timestamps for {months_str} {year}")
    print(f"✓ Total records: {n_rows}")
    print(f"✓ Saved to: {output_file}\n")


def main():
    """Main function to process both training and testing datasets."""
    
    base_dir = "original_data"
    output_dir = "data_model"
    
    # Training dataset: reference data from months 1, 2
    training_input = os.path.join(base_dir, "customer_churn_dataset-training-master.csv")
    training_output = os.path.join(output_dir, "reference_data.csv")
    
    # Testing dataset: current data from months 3, 4
    testing_input = os.path.join(base_dir, "customer_churn_dataset-testing-master.csv")
    testing_output = os.path.join(output_dir, "current_data.csv")
    
    print("=" * 60)
    print("Adding fake timestamps for Evidently AI drift monitoring")
    print("=" * 60)
    print()
    
    # Process training dataset (reference/baseline) - months 1, 2
    if os.path.exists(training_input):
        print("Processing TRAINING dataset (Reference/Baseline data - Months 1, 2)...")
        add_fake_timestamps_by_months(
            input_file=training_input,
            output_file=training_output,
            months=[1, 2],  # January, February
            timestamp_column="timestamp"
        )
    else:
        print(f"⚠ Warning: {training_input} not found!")
    
    # Process testing dataset (current data) - months 3, 4
    if os.path.exists(testing_input):
        print("Processing TESTING dataset (Current data - Months 3, 4)...")
        add_fake_timestamps_by_months(
            input_file=testing_input,
            output_file=testing_output,
            months=[3, 4],  # March, April
            timestamp_column="timestamp"
        )
    else:
        print(f"⚠ Warning: {testing_input} not found!")
    
    print("=" * 60)
    print("✓ All datasets processed successfully!")
    print("=" * 60)
    print()
    print("Output files:")
    print(f"  - Reference data: {training_output}")
    print(f"  - Current data: {testing_output}")
    print()
    print("You can now use these files with Evidently AI for drift monitoring:")
    print("  reference_df = pd.read_csv('data_model/reference_data.csv')")
    print("  current_df = pd.read_csv('data_model/current_data.csv')")


if __name__ == "__main__":
    main()

