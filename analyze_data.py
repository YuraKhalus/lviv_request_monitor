import pandas as pd
import numpy as np
import os

def optimize_memory(df, date_cols=None):
    """
    Optimize memory usage of a DataFrame by converting dtypes.
    
    :param df: pandas DataFrame
    :param date_cols: list of columns to convert to datetime
    :return: Optimized pandas DataFrame
    """
    print("--- Starting Memory Optimization ---")
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Initial memory usage: {start_mem:.2f} MB")

    if date_cols is None:
        date_cols = []

    for col in df.columns:
        # Convert specified date columns
        if col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            continue # Skip to next col

        # Optimize object columns
        if df[col].dtype == 'object':
            # Check if converting to category is memory efficient
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                try:
                    df[col] = df[col].astype('category')
                except Exception as e:
                    print(f"Could not convert column {col} to category: {e}")
        
        # Optimize numeric columns
        elif df[col].dtype.kind in 'ifc':
            # Downcast integers
            if df[col].dtype.kind == 'i':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            # Downcast floats
            elif df[col].dtype.kind == 'f':
                 df[col] = pd.to_numeric(df[col], downcast='float')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Final memory usage: {end_mem:.2f} MB")
    if start_mem > 0:
        print(f"Optimization reduced memory by {(start_mem - end_mem) / start_mem * 100:.1f}%")
    print("--- Memory Optimization Complete ---")
    return df

def main():
    """
    Main function to load, analyze, and optimize the dataset.
    """
    input_file = 'data/glm_all_2024_portal.csv'
    output_sample_file = 'data/sample_optimized.csv'

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    # --- Step 1: Load the CSV ---
    print(f"Loading data from {input_file}...")
    try:
        # Try loading with semicolon delimiter first
        df = pd.read_csv(input_file, delimiter=';')
        # If it results in a single column, the delimiter was likely wrong
        if df.shape[1] == 1 and ',' in df.columns[0]:
            print("Semicolon delimiter resulted in one column. Retrying with comma.")
            df = pd.read_csv(input_file, delimiter=',')
    except Exception as e:
        print(f"Failed to load with semicolon, trying comma. Error: {e}")
        try:
            df = pd.read_csv(input_file, delimiter=',')
        except Exception as e_comma:
            print(f"Error loading CSV with both ';' and ',' delimiters: {e_comma}")
            return
            
    print("Data loaded successfully.")

    # --- Step 2: Print Initial Memory Usage ---
    print("\n--- Initial DataFrame Info ---")
    df.info(memory_usage='deep')

    # --- Step 3: Print Column Names ---
    print("\n--- Column Names ---")
    print(df.columns.tolist())

    # --- Step 4: Optimize Memory ---
    # Based on Lviv Open Data Portal, these are the likely date columns
    # Using common names found in such datasets.
    date_columns = ['date_start', 'reg_date', 'completion_date', 'modification_date', 'created_at', 'updated_at', 'creation_date']
    # Filter out date columns that don't exist in the dataframe to avoid errors
    existing_date_cols = [col for col in date_columns if col in df.columns]
    
    df_optimized = optimize_memory(df.copy(), date_cols=existing_date_cols)

    # --- Step 5: Print Memory Usage After Optimization ---
    print("\n--- Optimized DataFrame Info ---")
    df_optimized.info(memory_usage='deep')

    # --- Step 6: Save a Sample ---
    print(f"\nSaving 100-row sample to {output_sample_file}...")
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_sample_file), exist_ok=True)
    df_optimized.head(100).to_csv(output_sample_file, index=False)
    print("Sample saved successfully.")

if __name__ == "__main__":
    main()
