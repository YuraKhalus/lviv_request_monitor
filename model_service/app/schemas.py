from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict

class AppealInput(BaseModel):
    """
    Input schema for the prediction endpoint.
    Defaults registrationDate to the current time if not provided.
    """
    district: str
    category: str
    registrationDate: datetime = Field(default_factory=datetime.now)

class PredictionOutput(BaseModel):
    """
    Output schema for the prediction endpoint.
    Returns a dictionary of model predictions.
    """
    predictions: Dict[str, float]

class MetricsOutput(BaseModel):
    """
    Output schema for the metrics endpoint.
    """
    mae: Dict[str, float]
    rmse: Dict[str, float]
