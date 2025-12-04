import os
import pandas as pd
import psycopg2
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from . import schemas

# Define the path for saving model artifacts
ARTIFACTS_PATH = "model_artifacts.joblib"

class ModelManager:
    def __init__(self):
        self.artifacts = None
        # Load artifacts on initialization if they exist
        if os.path.exists(ARTIFACTS_PATH):
            self.load_artifacts()

    def load_artifacts(self):
        """Loads models, columns, and metrics from a file."""
        try:
            self.artifacts = joblib.load(ARTIFACTS_PATH)
            print("Model artifacts loaded successfully.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            self.artifacts = None

    def train(self):
        """
        Connects to the database, fetches data, trains models,
        and saves them along with metrics and feature columns.
        """
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL environment variable not set.")

        print("Starting model training...")
        try:
            conn = psycopg2.connect(db_url)
            df = pd.read_sql_query("SELECT * FROM appeals", conn)
            conn.close()
        except Exception as e:
            print(f"Database connection failed: {e}")
            return

        print("Data fetched successfully. Preprocessing...")
        df.dropna(subset=['days_to_resolve', 'district', 'category'], inplace=True)
        
        features = ['district', 'category']
        target = 'days_to_resolve'
        
        X = pd.get_dummies(df[features], prefix_sep='_')
        y = df[target]

        # Save the column order and names after one-hot encoding
        feature_columns = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(max_depth=10, n_estimators=20, random_state=42),
            "XGBoost": XGBRegressor(max_depth=5, n_estimators=20, random_state=42)
        }
        
        metrics = {"mae": {}, "rmse": {}}

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics["mae"][name] = mean_absolute_error(y_test, preds)
            metrics["rmse"][name] = np.sqrt(mean_squared_error(y_test, preds))
            print(f"{name} - MAE: {metrics['mae'][name]:.4f}, RMSE: {metrics['rmse'][name]:.4f}")

        # Save all artifacts together
        artifacts_to_save = {
            "models": models,
            "feature_columns": feature_columns,
            "metrics": metrics,
        }
        joblib.dump(artifacts_to_save, ARTIFACTS_PATH)
        print("Model training complete. Artifacts saved.")
        
        # Reload artifacts into memory after training
        self.load_artifacts()

    def predict(self, input_data: schemas.AppealInput) -> schemas.PredictionOutput:
        """
        Generates predictions for a given input using the loaded models.
        """
        if self.artifacts is None:
            raise RuntimeError("Models are not trained or loaded. Please train first via the /train endpoint.")

        # Create a DataFrame from the input
        df_input = pd.DataFrame([input_data.dict(exclude={'registrationDate'})])
        
        # Preprocess the input data (One-Hot Encode)
        X_input = pd.get_dummies(df_input, prefix_sep='_')

        # Align columns with the training data
        # This adds missing columns (with value 0) and removes extra ones.
        X_input_aligned = X_input.reindex(columns=self.artifacts["feature_columns"], fill_value=0)

        predictions = {}
        for name, model in self.artifacts["models"].items():
            pred = model.predict(X_input_aligned)[0]
            predictions[name] = float(pred)
            
        return schemas.PredictionOutput(predictions=predictions)

    def get_metrics(self) -> schemas.MetricsOutput:
        """
        Returns the evaluation metrics from the last training run.
        """
        if self.artifacts is None or "metrics" not in self.artifacts:
            raise RuntimeError("No metrics found. Please train models first.")
        
        return schemas.MetricsOutput(
            mae=self.artifacts["metrics"]["mae"],
            rmse=self.artifacts["metrics"]["rmse"]
        )

