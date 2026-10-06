from pydantic import BaseModel

class HealthResponse(BaseModel):
  status : str


class ModelStatusResponse(BaseModel):
  model_version : str
  model_path : str
  retraining_in_progress : bool