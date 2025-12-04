import pandas as pd
from sqlalchemy import create_engine, text

# --- Configuration ---
# This script runs on your host machine, so it connects to localhost.
DATABASE_URL = "postgresql://user:password@localhost:5432/lviv_db"

def inspect_database_values():
    """
    Connects to the database and prints the top 20 most frequent categories
    and all unique district names.
    """
    print(f"Connecting to the database at {DATABASE_URL}...")
    
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            print("Connection successful. Fetching data...")

            # --- Query for Top 20 Categories ---
            top_categories_query = text("""
                SELECT category, COUNT(*) as occurrences
                FROM appeals
                GROUP BY category
                ORDER BY occurrences DESC
                LIMIT 20;
            """)
            top_categories_df = pd.read_sql_query(top_categories_query, connection)

            # --- Query for Unique Districts ---
            unique_districts_query = text("""
                SELECT DISTINCT district FROM appeals;
            """)
            unique_districts_df = pd.read_sql_query(unique_districts_query, connection)

            # --- Display Results ---
            print("\n" + "="*40)
            print("      TOP 20 MOST FREQUENT CATEGORIES")
            print("="*40)
            print(top_categories_df.to_string())
            
            print("\n" + "="*40)
            print("           UNIQUE DISTRICT NAMES")
            print("="*40)
            print(unique_districts_df.to_string())
            print("\n" + "="*40)

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print("Please ensure the following:")
        print("1. The PostgreSQL container ('db') is running.")
        print("2. The container's port 5432 is correctly mapped to your host machine.")
        print("3. The database, user, and password match the credentials.")

if __name__ == "__main__":
    inspect_database_values()
