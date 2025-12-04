import os
import pandas as pd
import joblib
import logging
from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from . import schemas

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for saving model artifacts
MODELS_PATH = "models.joblib"
COLUMNS_PATH = "model_columns.joblib"
METRICS_PATH = "metrics.joblib"

class ModelManager:
    def __init__(self):
        self.db_engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///:memory:"))
        self.models = None
        self.model_columns = None
        self.metrics = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads models, columns, and metrics from their respective files."""
        try:
            if os.path.exists(MODELS_PATH):
                self.models = joblib.load(MODELS_PATH)
            if os.path.exists(COLUMNS_PATH):
                self.model_columns = joblib.load(COLUMNS_PATH)
            if os.path.exists(METRICS_PATH):
                self.metrics = joblib.load(METRICS_PATH)
            if self.models and self.model_columns:
                logging.info("Model artifacts loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading artifacts: {e}")
            self.models = self.model_columns = self.metrics = None

    def train(self):
        """Fetches data, trains models, and saves artifacts."""
        logging.info("Starting model training...")
        try:
            with self.db_engine.connect() as conn:
                df = pd.read_sql_query(text("SELECT * FROM appeals"), conn)
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            return

        logging.info("Data fetched successfully. Preprocessing...")
        df.dropna(subset=['days_to_resolve', 'district', 'category'], inplace=True)
        
        X = pd.get_dummies(df[['district', 'category']], prefix_sep='_')
        y = df['days_to_resolve']

        # CRUCIAL: Save the column names and order
        feature_columns = X.columns.tolist()
        joblib.dump(feature_columns, COLUMNS_PATH)
        logging.info(f"Saved {len(feature_columns)} feature columns to {COLUMNS_PATH}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        trained_models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(max_depth=10, n_estimators=20, random_state=42),
            "XGBoost": XGBRegressor(max_depth=5, n_estimators=20, random_state=42)
        }
        calculated_metrics = {"mae": {}, "rmse": {}}

        for name, model in trained_models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            calculated_metrics["mae"][name] = mean_absolute_error(y_test, preds)
            calculated_metrics["rmse"][name] = np.sqrt(mean_squared_error(y_test, preds))

        joblib.dump(trained_models, MODELS_PATH)
        joblib.dump(calculated_metrics, METRICS_PATH)
        logging.info("Model training complete. Artifacts saved.")
        self.load_artifacts()

    def predict(self, input_data: schemas.AppealInput) -> schemas.PredictionOutput:
        """Generates predictions, ensuring feature alignment with robust logging."""
        if not self.models or not self.model_columns:
            raise RuntimeError("Models or columns not loaded. Please train first via the /train endpoint.")

        logging.info(f"Raw prediction input: {input_data.dict()}")
        input_df = pd.DataFrame([input_data.dict(exclude={'registrationDate'})])
        
        input_encoded = pd.get_dummies(input_df, prefix_sep='_')
        logging.info(f"Input columns after one-hot encoding: {input_encoded.columns.tolist()}")

        # CRUCIAL FIX: Align columns with the training data
        input_aligned = input_encoded.reindex(columns=self.model_columns, fill_value=0)
        
        if input_aligned.sum().sum() == 0:
            logging.warning("No matching features found after alignment. Prediction will be based on model bias only.")

        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(input_aligned)[0]
            predictions[name] = float(pred)
            
        logging.info(f"Returning predictions: {predictions}")
        return schemas.PredictionOutput(predictions=predictions)

    def get_metrics(self) -> schemas.MetricsOutput:
        """Returns the evaluation metrics from the last training run."""
        if not self.metrics:
            raise RuntimeError("No metrics found. Please train models first.")
        return schemas.MetricsOutput(mae=self.metrics["mae"], rmse=self.metrics["rmse"])

    def get_actual_case(self, district: str, category: str) -> dict:
        """Queries the database for a single, random historical case using fuzzy matching for category."""
        cat_pattern = f"%{category}%"
        logging.info(f"Fetching actual case for District: '{district}', Category Pattern: '{cat_pattern}'")
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT days_to_resolve FROM appeals 
                    WHERE district = :district AND category ILIKE :cat_pattern
                    ORDER BY random() 
                    LIMIT 1
                """)
                result = conn.execute(query, {"district": district, "cat_pattern": cat_pattern}).fetchone()
                
                if result:
                    logging.info(f"Found actual case with {result[0]} days to resolve.")
                    return {"actual_days": float(result[0])}
                else:
                    logging.warning("No actual case found for the given criteria.")
                    return {"actual_days": None}
        except Exception as e:
            logging.error(f"DB query for actual case failed: {e}")
            return {"actual_days": None}
