from enum import Enum

from pydantic import BaseModel


class FeedbackType(str, Enum):
    like = "like"
    dislike = "dislike"


class FeedbackRequest(BaseModel):
    prediction_id: int
    feedback: FeedbackType
    
class FeedbackResponse(BaseModel):
    feedback_id: int
    prediction_id: int
    feedback: FeedbackType
    corrected_label: str
    retraining_triggered: bool
    retraining_started: bool