from sqlalchemy import create_engine
from sqlalchemy import text


from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base
from app.config import DATABASE_URL


engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
  bind=engine,
  class_=Session,
  autoflush=False
  )

Base.metadata.create_all(engine)

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":

    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print(result.scalar())