import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models_v"

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

RETRAINING_THRESHOLD = int(
    os.getenv(
        "RETRAINING_THRESHOLD",
        "250",
    )
)

RETRAINING_LOCK_PATH = DATA_DIR / "retraining.lock"
RETRAINING_LOG_PATH = DATA_DIR / "retraining.log"