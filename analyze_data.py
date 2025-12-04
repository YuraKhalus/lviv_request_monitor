import pandas as pd
import numpy as np
import os

def analyze_and_optimize():
    """
    Loads, cleans, and optimizes the Lviv appeals dataset according to specific rules.
    """
    input_file = 'data/glm_all_2024_portal.csv'
    output_sample_file = 'data/sample_optimized.csv'

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file, sep=';', low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Data loaded successfully.")
    print("\n--- Initial DataFrame Info ---")
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Initial memory usage: {start_mem:.2f} MB")
    df.info(memory_usage='deep')

    # --- Step 2: Date Conversion ---
    print("\n--- Processing Dates and Target Variable ---")
    date_cols = ['registrationDate', 'executionDate']
    for col in date_cols:
        if col in df.columns:
            # Using format='mixed' is computationally expensive but robust to varied date formats
            df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')
        else:
            print(f"Warning: Date column '{col}' not found in DataFrame.")

    # --- Step 3: Calculate Target Variable ---
    if 'registrationDate' in df.columns and 'executionDate' in df.columns:
        df['days_to_resolve'] = (df['executionDate'] - df['registrationDate']).dt.days
    else:
        print("Error: Cannot create 'days_to_resolve' as date columns are missing.")
        df['days_to_resolve'] = np.nan # Create column to avoid subsequent errors

    # --- Step 4: Filter Data ---
    initial_rows = len(df)
    df.dropna(subset=['days_to_resolve'], inplace=True)
    df = df[df['days_to_resolve'] >= 0]
    print(f"Filtered {initial_rows - len(df)} rows with invalid target values.")


    # --- Step 5: Fix Coordinates ---
    print("\n--- Processing Coordinates ---")
    coord_cols = ['latitude', 'longitude']
    for col in coord_cols:
        if col in df.columns:
            # Ensure the column is a string before replacing, handle non-string data
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        else:
            print(f"Warning: Coordinate column '{col}' not found.")


    # --- Step 6: Optimize Other Columns ---
    print("\n--- Optimizing Remaining Columns ---")
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert object columns with a low ratio of unique values to category
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_total_values > 0 and num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        elif 'year' in col.lower() and df[col].dtype.kind == 'i':
            df[col] = pd.to_numeric(df[col], downcast='integer')


    print("\n--- Final DataFrame Info ---")
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Final memory usage: {end_mem:.2f} MB")
    if start_mem > 0:
        print(f"Optimization reduced memory by {(start_mem - end_mem) / start_mem * 100:.1f}%")
    df.info(memory_usage='deep')

    # --- Step 7: Output Verification ---
    print("\n--- Verification: First 5 rows of key columns ---")
    verify_cols = ['registrationDate', 'executionDate', 'days_to_resolve', 'district']
    existing_verify_cols = [c for c in verify_cols if c in df.columns]
    print(df[existing_verify_cols].head())

    # --- Step 8: Save Sample ---
    print(f"\nSaving 100-row sample to {output_sample_file}...")
    os.makedirs(os.path.dirname(output_sample_file), exist_ok=True)
    df.head(100).to_csv(output_sample_file, index=False)
    print("Sample saved successfully.")


if __name__ == "__main__":
    analyze_and_optimize()
