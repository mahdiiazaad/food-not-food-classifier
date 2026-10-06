from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.services.model_manager import ModelManager

from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repository import save_prediction, get_prediction, save_feedback
from app.services.feedback_service import check_retraining_trigger
from app.services.retraining_service import start_retraining
from app.schemas.status import (
    HealthResponse,
    ModelStatusResponse)

from app.services.retraining_service import (
    is_retraining_in_progress)



app = FastAPI()
templates = Jinja2Templates(
    directory="app/templates"
)

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


model_manager = ModelManager()

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest,
            # I need a database session for this endpoint. Get one by calling get_db()
            db: Session = Depends(get_db)):
  
  result = model_manager.predict(request.text)
  
  prediction = result[0][0]
  
  
  saved_prediction= save_prediction(
    db=db,
    text=request.text,
    predicted_label= prediction['label'],
    confidence= prediction["score"],
    model_version=model_manager.model_version
  )
  return {
    "prediction_id": saved_prediction.id,
        "label": saved_prediction.predicted_label,
        "score": saved_prediction.confidence,
        "model_version": saved_prediction.model_version
  }
  
  
@app.post('/feedback', response_model=FeedbackResponse)
def feedback(
  request: FeedbackRequest,
  db: Session = Depends(get_db)
):
  prediction = get_prediction(
    db=db,
    prediction_id=request.prediction_id
  )
  
  if prediction is None:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail='Prediction Not Found!'
    )
    
  feedback_record = save_feedback(
    db=db,
    prediction=prediction,
    feedback=request.feedback
  )
  
  retraining_triggered= False
  retraining_started = False
  
  if request.feedback == 'dislike':
    retraining_triggered = check_retraining_trigger(db=db)
    
    if retraining_triggered:
      retraining_started = start_retraining(model_manager)
    
  print(retraining_triggered)
  print(f"retraining_triggered = {retraining_triggered}")
  print(f"retraining_started = {retraining_started}")
  
  
  return {
    "feedback_id": feedback_record.id,
    "prediction_id": feedback_record.prediction_id,
    "feedback": feedback_record.feedback,
    "corrected_label": feedback_record.corrected_label,
     "retraining_triggered": retraining_triggered,
    "retraining_started": retraining_started,
  }
  
  
@app.get('/health', response_model=HealthResponse)
def health():
  return{
    'status' : 'ok'
  }
  
@app.get('/health/status', response_model=ModelStatusResponse)
def model_status():
  return {
        "model_version": model_manager.model_version,
        "model_path": str(model_manager.model_path),
        "retraining_in_progress": is_retraining_in_progress(),
    }