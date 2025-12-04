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
PERFORMANCE_PATH = "performance_data.joblib" # New path for performance data

class ModelManager:
    def __init__(self):
        self.db_engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///:memory:"))
        self.models = None
        self.model_columns = None
        self.metrics = None
        self.performance_data = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads all model artifacts from their respective files."""
        try:
            if os.path.exists(MODELS_PATH): self.models = joblib.load(MODELS_PATH)
            if os.path.exists(COLUMNS_PATH): self.model_columns = joblib.load(COLUMNS_PATH)
            if os.path.exists(METRICS_PATH): self.metrics = joblib.load(METRICS_PATH)
            if os.path.exists(PERFORMANCE_PATH): self.performance_data = joblib.load(PERFORMANCE_PATH)
            if self.models and self.model_columns: logging.info("Model artifacts loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading artifacts: {e}")
            # Reset all on failure
            self.models = self.model_columns = self.metrics = self.performance_data = None

    def train(self):
        """Fetches data, trains models with updated hyperparameters, and saves all artifacts."""
        logging.info("Starting model training with updated hyperparameters...")
        try:
            with self.db_engine.connect() as conn:
                df = pd.read_sql_query(text("SELECT * FROM appeals"), conn)
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            return

        df.dropna(subset=['days_to_resolve', 'district', 'category'], inplace=True)
        
        X = pd.get_dummies(df[['district', 'category']], prefix_sep='_')
        y = df['days_to_resolve']

        feature_columns = X.columns.tolist()
        joblib.dump(feature_columns, COLUMNS_PATH)
        logging.info(f"Saved {len(feature_columns)} feature columns.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Updated Hyperparameters to combat underfitting
        trained_models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=50, max_depth=7, learning_rate=0.1, random_state=42)
        }
        
        calculated_metrics = {"mae": {}, "rmse": {}}
        performance_data = {'Actual': y_test[:100].tolist()} # Use first 100 test samples for visualization

        for name, model in trained_models.items():
            logging.info(f"Training {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            calculated_metrics["mae"][name] = mean_absolute_error(y_test, preds)
            calculated_metrics["rmse"][name] = np.sqrt(mean_squared_error(y_test, preds))
            performance_data[f"{name}_Pred"] = preds[:100].tolist()

        joblib.dump(trained_models, MODELS_PATH)
        joblib.dump(calculated_metrics, METRICS_PATH)
        joblib.dump(performance_data, PERFORMANCE_PATH) # Save performance data
        logging.info("Model training complete. All artifacts saved.")
        self.load_artifacts()

    def predict(self, input_data: schemas.AppealInput) -> schemas.PredictionOutput:
        """Generates predictions, ensuring feature alignment."""
        if not self.models or not self.model_columns:
            raise RuntimeError("Models not loaded. Please train first.")
        
        input_df = pd.DataFrame([input_data.dict(exclude={'registrationDate'})])
        input_encoded = pd.get_dummies(input_df, prefix_sep='_')
        input_aligned = input_encoded.reindex(columns=self.model_columns, fill_value=0)

        predictions = {name: max(0.0, float(model.predict(input_aligned)[0])) for name, model in self.models.items()}
        return schemas.PredictionOutput(predictions=predictions)

    def get_performance(self) -> dict:
        """Returns the stored performance data for visualization."""
        if not self.performance_data:
            raise RuntimeError("No performance data found. Please train models first.")
        return self.performance_data

    def get_metrics(self) -> schemas.MetricsOutput:
        """Returns evaluation metrics from the last training run."""
        if not self.metrics:
            raise RuntimeError("No metrics found. Please train models first.")
        return schemas.MetricsOutput(mae=self.metrics["mae"], rmse=self.metrics["rmse"])

    def get_actual_case(self, district: str, category: str) -> dict:
        """Queries the DB for a random historical case using fuzzy matching."""
        cat_pattern = f"%{category}%"
        try:
            with self.db_engine.connect() as conn:
                query = text("SELECT days_to_resolve FROM appeals WHERE district = :district AND category ILIKE :cat_pattern ORDER BY random() LIMIT 1")
                result = conn.execute(query, {"district": district, "cat_pattern": cat_pattern}).fetchone()
                return {"actual_days": float(result[0]) if result else None}
        except Exception as e:
            logging.error(f"DB query for actual case failed: {e}")
            return {"actual_days": None}
