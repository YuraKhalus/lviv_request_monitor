from fastapi import FastAPI, BackgroundTasks, HTTPException
from . import schemas, models
import os

app = FastAPI(
    title="Lviv Request Monitor - Model Service",
    description="API for training and predicting civic appeal resolution times.",
    version="1.0.0"
)

# Initialize the ModelManager. It will load artifacts if they exist.
model_manager = models.ModelManager()

@app.get("/", tags=["General"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "Model service is running."}

@app.post("/train", status_code=202, tags=["Training"])
async def train_models(background_tasks: BackgroundTasks):
    """
    Triggers a model training process in the background.
    """
    # Check if DATABASE_URL is set, otherwise training is impossible
    if not os.getenv("DATABASE_URL"):
        raise HTTPException(
            status_code=500,
            detail="DATABASE_URL environment variable is not set. Cannot start training."
        )
    
    background_tasks.add_task(model_manager.train)
    return {"message": "Model training started in the background. Check logs for progress."}

@app.post("/predict", response_model=schemas.PredictionOutput, tags=["Prediction"])
async def predict_resolution_time(appeal_input: schemas.AppealInput):
    """
    Predicts the resolution time in days based on appeal details.
    """
    try:
        predictions = model_manager.predict(appeal_input)
        return predictions
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/actual", tags=["Prediction"])
async def get_actual_resolution_time(appeal_input: schemas.AppealInput):
    """
    Fetches a random, real historical case from the DB matching the input criteria.
    """
    try:
        actual_case = model_manager.get_actual_case(
            district=appeal_input.district,
            category=appeal_input.category
        )
        return actual_case
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.get("/metrics", response_model=schemas.MetricsOutput, tags=["Metrics"])
async def get_training_metrics():
    """
    Returns the MAE and RMSE for each model from the last training run.
    """
    try:
        metrics = model_manager.get_metrics()
        return metrics
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/performance", tags=["Metrics"])
async def get_performance_data():
    """
    Returns actual vs. predicted data for a sample of the test set.
    """
    try:
        performance_data = model_manager.get_performance()
        return performance_data
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

