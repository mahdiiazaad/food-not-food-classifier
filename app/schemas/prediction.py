from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=2000,
    )
    
class PredictionResponse(BaseModel):
    prediction_id: int
    label: str
    score: float
    model_version: str