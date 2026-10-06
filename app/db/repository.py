from sqlalchemy.orm import Session
from sqlalchemy import func, select

from app.db.models import Prediction,Feedback

def save_prediction(
  db: Session,
  text: str,
  predicted_label: str,
  confidence: float,
  model_version: str,
  
) -> Prediction:
  prediction = Prediction(
    text=text,
    predicted_label=predicted_label,
    confidence=confidence,
    model_version=model_version
  )
  
  db.add(prediction)
  db.commit()
  db.refresh(prediction)
  
  return prediction


def save_feedback(
  db: Session,
  prediction: Prediction,
  feedback: str,
  
) -> Feedback:
  if feedback == 'like':
    corrected_label = prediction.predicted_label
  elif feedback == 'dislike':
    if prediction.predicted_label == 'food':
      corrected_label = 'not_food'
    
    else:
      corrected_label = 'food'
  
  else:
      raise ValueError("feedback must be 'like' or 'dislike'")
  
  
  feedback_record = Feedback(
    prediction_id = prediction.id,
    feedback=feedback,
    corrected_label = corrected_label
  )
  
  db.add(feedback_record)
  db.commit()
  db.refresh(feedback_record)
  
  return feedback_record


def get_prediction(
  db:Session,
  prediction_id: int,
  
) -> Prediction | None:
  return db.get(Prediction, prediction_id)


def untrained_dislike(db:Session) -> int:
  statement = (
    select(func.count())
    .select_from(Feedback)
    .where(Feedback.feedback == 'dislike',
           Feedback.use_for_training == False)
  )
  
  return db.scalar(statement=statement) or 0


def get_untrained_dislikes(db:Session):
  statement = (
select(Feedback, Prediction)
# Connect the Feedback table to the Prediction table using prediction_id.
.join(Prediction, 
      Feedback.prediction_id == Prediction.id)
  ).where(
    Feedback.feedback == 'dislike',
    Feedback.use_for_training == False
  )
  
  return db.execute(statement).all()



def get_all_feedback_for_training(db: Session):
    statement = (
        select(Feedback, Prediction)
        .join(
            Prediction,
            Feedback.prediction_id == Prediction.id,
        )
        .where(
            Feedback.feedback == "dislike",
        )
    )

    return db.execute(statement).all()
  

def get_untrained_dislike_ids(db: Session) -> list[int]:
  statement = (
    select(Feedback.id)
    .where(
      Feedback.feedback == 'dislike',
      Feedback.use_for_training == False
    )
  )
  
  return list(db.scalars(statement=statement).all())


def mark_feedback_used_for_training(
    db: Session,
    feedback_ids: list[int],
) -> None:

    if not feedback_ids:
        return

    statement = (
        select(Feedback)
        .where(Feedback.id.in_(feedback_ids))
    )

    feedback_records = db.scalars(statement).all()

    for feedback in feedback_records:
        feedback.use_for_training = True

    db.commit()