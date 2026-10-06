from sqlalchemy.orm import Session
from app.db.repository import get_untrained_dislikes, get_all_feedback_for_training
from ml.dataset import LABEL2ID

def get_feedback_training_data(db:Session) -> list[dict]:
  """ask the dataset: Give me all feedback that is currently available for training"""
  rows = get_untrained_dislikes(db=db)
  
  training_data = []
  
  for feedback, prediction in rows:
    training_data.append(
      {
        'text': prediction.text,
        'label': LABEL2ID[feedback.corrected_label]
      }
    )
  
  return training_data



def get_all_feedback_training_data(
    db: Session,
) -> list[dict]:

    rows = get_all_feedback_for_training(db)

    training_data = []

    for feedback, prediction in rows:
        training_data.append(
            {
                "text": prediction.text,
                "label": LABEL2ID[feedback.corrected_label],
            }
        )

    return training_data

  
if __name__ == '__main__':
  
  from app.db.database import SessionLocal

  db = SessionLocal()

  try:
      training_data = get_feedback_training_data(db)

      for item in training_data:
          print(item)

  finally:
      db.close()