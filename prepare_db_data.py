import pandas as pd
import numpy as np
import os

def prepare_data_for_db():
    """
    Loads the raw CSV, cleans it, selects specific columns,
    and saves the result for database initialization.
    """
    input_file = 'data/glm_all_2024_portal.csv'
    output_file = 'data/cleaned_appeals.csv'
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file, sep=';', low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Processing data...")
    # Convert dates
    date_cols = ['registrationDate', 'executionDate']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')

    # Calculate target variable
    df['days_to_resolve'] = (df['executionDate'] - df['registrationDate']).dt.days

    # Filter invalid rows
    df.dropna(subset=['days_to_resolve'], inplace=True)
    df = df[df['days_to_resolve'] >= 0]

    # Fix coordinates
    coord_cols = ['latitude', 'longitude']
    for col in coord_cols:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Select final columns
    final_cols = [
        'registrationDate', 'executionDate', 'district', 'category',
        'days_to_resolve', 'latitude', 'longitude'
    ]
    # Ensure all required columns exist, fill missing ones with a default if necessary
    for col in final_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. It will be added with null values.")
            df[col] = np.nan

    df_final = df[final_cols]
    
    # Fill any remaining NaNs in key categorical columns to avoid COPY errors
    df_final['district'].fillna('Unknown', inplace=True)
    df_final['category'].fillna('Unknown', inplace=True)


    print(f"Saving cleaned data to {output_file}...")
    df_final.to_csv(output_file, index=False)
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data_for_db()
