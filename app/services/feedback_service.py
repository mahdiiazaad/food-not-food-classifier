from sqlalchemy.orm import Session
from app.db.repository import get_untrained_dislikes, untrained_dislike

from app.config import RETRAINING_THRESHOLD

def should_retrain(untrained_dislike_count: int) -> bool:
  return untrained_dislike_count >= RETRAINING_THRESHOLD


def check_retraining_trigger(db: Session) -> bool:
  untrained_dislike_count = untrained_dislike(db=db)
  
  return should_retrain(untrained_dislike_count=untrained_dislike_count)    