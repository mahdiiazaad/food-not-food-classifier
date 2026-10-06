from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
  pass

class Prediction(Base):
  __tablename__ = "predictions"
  
  id: Mapped[int] = mapped_column(
    Integer,
    primary_key=True,
    autoincrement=True
  )
  
  text: Mapped[str] = mapped_column(
    Text,
    nullable=False
  )
  
  predicted_label: Mapped[str] = mapped_column(
    String,
    nullable=False
  )
  
  model_version: Mapped[str] = mapped_column(
    String(50),
    nullable=False
  )
  confidence: Mapped[float] = mapped_column(
    Float,
    nullable=False
  )
  created_at: Mapped[datetime] = mapped_column(
    DateTime,
    default=datetime.utcnow,
    nullable=False
  )
  
  
class Feedback(Base):
  __tablename__ = 'feedback'
  
  id: Mapped[int] = mapped_column(
    Integer,
    primary_key=True,
    autoincrement=True
  )
  
  prediction_id: Mapped[int] = mapped_column(
    Integer,
    ForeignKey("predictions.id"),
    nullable=False
  )
  
  feedback: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
  
  created_at: Mapped[datetime] = mapped_column(
    DateTime,
    default=datetime.utcnow,
    nullable=False
  )
  
  corrected_label: Mapped[str] = mapped_column(
        String(50),
        nullable=True,
    )
  
  use_for_training: Mapped[bool] = mapped_column(
    Boolean,
    default=False,
    nullable=False
    
  )