# Food Not-Food Classifier

An end-to-end machine learning application that classifies text as either `food` or `not_food`, serves predictions through FastAPI, stores predictions and user feedback in PostgreSQL, and automatically retrains the model when enough negative feedback accumulates.

This project was built to go beyond training a classifier. The main goal was to build the surrounding ML application lifecycle: inference, persistence, feedback collection, retraining, evaluation, model versioning, model promotion, and live model reloading.

## Demo

The application provides a browser-based interface served directly by FastAPI.

The main workflow is:

```text
User enters text
      ↓
FastAPI /predict
      ↓
Active Hugging Face model
      ↓
Prediction + confidence + model version
      ↓
Prediction stored in PostgreSQL
      ↓
User provides feedback
      ↓
Negative feedback accumulates
      ↓
Retraining threshold reached
      ↓
Background model training
      ↓
Model evaluation
      ↓
Accept or reject candidate model
      ↓
Promote accepted model
      ↓
Reload ModelManager
      ↓
New predictions use the new active model
```

## What this project demonstrates

The project focuses on the engineering around an ML model, including:

- Hugging Face transformer fine-tuning for text classification
- local CPU inference
- FastAPI model serving
- PostgreSQL persistence
- SQLAlchemy ORM
- Pydantic request and response validation
- user feedback collection
- automatic retraining
- background training processes
- retraining locking to prevent duplicate jobs
- model version generation
- model registry and active-model management
- candidate model evaluation and promotion
- live model reload after promotion
- automated API tests
- a server-rendered Bootstrap frontend
- dependency management with `uv`

## Architecture

```text
                              ┌──────────────────────┐
                              │       Browser        │
                              │   Bootstrap + JS     │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │       FastAPI        │
                              │                      │
                              │  GET /               │
                              │  POST /predict       │
                              │  POST /feedback      │
                              │  GET /health         │
                              │  GET /health/status  │
                              └───────┬───────┬──────┘
                                      │       │
                         ┌────────────┘       └──────────────┐
                         ▼                                   ▼
                ┌──────────────────┐                ┌──────────────────┐
                │   ModelManager   │                │   PostgreSQL     │
                │                  │                │                  │
                │ Active model     │                │ Predictions      │
                │ Inference        │                │ Feedback         │
                │ Model version    │                │ Training state   │
                └────────┬─────────┘                └──────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Model Registry  │
                │                  │
                │ Active version   │
                │ Model paths      │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Retraining       │
                │ Service          │
                │                  │
                │ Threshold check  │
                │ Process launch   │
                │ Locking          │
                │ Completion       │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     ml/train.py  │
                │                  │
                │ Dataset rebuild  │
                │ Fine-tuning      │
                │ Evaluation       │
                │ Version creation │
                │ Promotion        │
                └──────────────────┘
```

## Project structure

```text
food-not-food-classifier/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   │
│   ├── db/
│   │   ├── database.py
│   │   ├── models.py
│   │   └── repository.py
│   │
│   ├── schemas/
│   │   ├── feedback.py
│   │   ├── prediction.py
│   │   └── status.py
│   │
│   ├── services/
│   │   ├── feedback_service.py
│   │   ├── model_manager.py
│   │   ├── retraining_service.py
│   │   └── training_data_service.py
│   │
│   └── templates/
│       └── index.html
│
├── ml/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model_registry.py
│   ├── model_version.py
│   └── train.py
│
├── tests/
│   ├── test_feedback.py
│   ├── test_health.py
│   └── test_prediction.py
│
├── models_v/
│   ├── learn_hf_food_not_food_text_classifier_distilbert/
│   └── registry.json
│
├── data/
│
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

## Machine learning pipeline

### Dataset

The project uses the Hugging Face dataset:

`mrdbourke/learn_hf_food_not_food_image_captions`

The dataset is normalized to the project's binary label mapping:

```text
not_food → 0
food     → 1
```

The original dataset is split reproducibly using seed `42`:

```text
80% training
10% validation
10% test
```

The test set is kept separate from training and validation.

### Model

The classifier is based on:

```text
distilbert/distilbert-base-uncased
```

The model is fine-tuned for binary text classification.

The application supports CPU inference, which makes the project runnable without a dedicated GPU.

### Retraining data

The retraining dataset is built from:

```text
Original training data
        +
Historical negative feedback
```

Negative feedback is converted into a corrected label and added to future retraining data.

The original validation and test sets remain separate so that candidate models can still be evaluated against data that is not used as part of the retraining dataset.

## Feedback-driven retraining

The most important feature of the application is the feedback loop.

When a user submits feedback:

- `like` confirms the current prediction.
- `dislike` records the opposite label as corrected feedback.

Negative feedback is stored in PostgreSQL.

The retraining trigger counts only new negative-feedback records that have not yet been consumed by a retraining cycle.

The configured threshold is:

```text
250 new dislikes
```

Once the threshold is reached, FastAPI starts the retraining process in the background.

The API remains available while training is running.

The currently active model continues serving predictions until the candidate model has completed training and evaluation.

## Retraining lifecycle

The retraining process follows this sequence:

```text
1. Count unprocessed dislikes
2. Reach configured threshold
3. Capture the feedback records for the training cycle
4. Start ml.train as a separate process
5. Train a new model version
6. Evaluate the candidate model
7. Compare candidate performance with the active model
8. Promote candidate if accepted
9. Mark the triggering feedback as used
10. Reload ModelManager
11. Continue serving predictions with the new active model
```

A lock is used to prevent multiple retraining jobs from running simultaneously.

If retraining is already running, another threshold-triggering request does not start a second training process.

## Model versioning

Retrained models are assigned sequential versions:

```text
v001
v002
v003
...
```

The baseline model is kept separate from generated retraining versions.

The model registry keeps track of the active model.

The active version is used by `ModelManager` at application startup.

When a candidate is accepted, the registry is updated and the running application reloads the newly promoted model.

## API

### `GET /`

Returns the application frontend.

The HTML page is rendered by FastAPI using Jinja2.

### `POST /predict`

Classifies submitted text.

Request:

```json
{
  "text": "I had pizza for dinner."
}
```

Response:

```json
{
  "prediction_id": 50,
  "label": "food",
  "score": 0.6998,
  "model_version": "v009"
}
```

The actual model version is returned dynamically. It is not hardcoded in the API.

### `POST /feedback`

Stores user feedback for a prediction.

Request:

```json
{
  "prediction_id": 50,
  "feedback": "dislike"
}
```

Response:

```json
{
  "feedback_id": 1,
  "prediction_id": 50,
  "feedback": "dislike",
  "corrected_label": "not_food",
  "retraining_triggered": false,
  "retraining_started": false
}
```

When enough negative feedback has accumulated, the response can indicate that retraining was triggered and whether the training process started successfully.

### `GET /health`

Basic application health check.

Response:

```json
{
  "status": "ok"
}
```

### `GET /health/status`

Returns the active model version and current retraining state.

Response:

```json
{
  "model_version": "v009",
  "model_path": "models_v\\v009",
  "retraining_in_progress": false
}
```

The frontend uses this endpoint to show whether the model is ready or currently being retrained.

## Frontend

The project includes a server-rendered Bootstrap frontend located at:

```text
app/templates/index.html
```

The frontend provides:

- text input
- character-limit validation
- prediction display
- confidence visualization
- model-version display
- prediction ID
- like/dislike feedback controls
- feedback confirmation
- retraining notifications
- active model status
- automatic model-status polling

The frontend and backend are served from the same FastAPI application.

There is no separate frontend server required.

## Database

PostgreSQL stores application data.

The current schema includes two primary tables:

### Predictions

Stores:

- prediction ID
- input text
- predicted label
- confidence
- model version
- creation time

### Feedback

Stores:

- feedback ID
- prediction ID
- feedback type
- corrected label
- creation time
- whether the feedback has been used by a retraining cycle

SQLAlchemy is used as the ORM layer.

## Configuration

Application configuration is provided through environment variables.

Create a local `.env` file from `.env.example`.

Example:

```env
DATABASE_URL=postgresql+psycopg://postgres:your_password@localhost:5432/food_classifier
RETRAINING_THRESHOLD=250
```

The real `.env` file should never be committed to Git.

An `.env.example` file is included in the repository so that the required configuration is visible without exposing credentials.

## Requirements

The project uses:

- Python 3.13
- `uv`
- PostgreSQL
- a CPU-capable environment for inference and local retraining

Main technologies:

| Area | Technology |
|---|---|
| Language | Python |
| ML | PyTorch |
| NLP | Hugging Face Transformers |
| Dataset | Hugging Face Datasets |
| API | FastAPI |
| Validation | Pydantic |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Frontend | HTML, CSS, Bootstrap, JavaScript |
| Testing | pytest |
| Environment | uv |

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/mahdiiazaad/food-not-food-classifier.git
cd food-not-food-classifier
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Create the environment file

PowerShell:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and provide your PostgreSQL connection string.

### 4. Create the PostgreSQL database

Create a PostgreSQL database named:

```text
food_classifier
```

Make sure PostgreSQL is running before starting the application.

### 5. Start the application

```bash
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Running tests

Run the complete test suite with:

```bash
uv run pytest
```

The tests cover the main API behaviors, including:

- health checks
- prediction handling
- feedback handling
- retraining-trigger behavior
- invalid prediction handling

## Development notes

The application separates responsibilities between the API layer, application services, database layer, and machine-learning layer.

The `app/` package is responsible for the application itself:

```text
FastAPI
database
schemas
model manager
feedback
retraining orchestration
frontend
```

The `ml/` package contains machine-learning concerns:

```text
dataset preparation
training
evaluation
model versioning
model registry
```

This separation makes it possible to change the ML pipeline without embedding training logic directly inside FastAPI routes.

## Design decisions

### Why PostgreSQL?

PostgreSQL was chosen instead of a file-based database because the application needs persistent structured storage for predictions and feedback and because the project is intended to demonstrate a more realistic application architecture.

### Why use a background training process?

Model training should not block API requests.

The FastAPI process launches `ml.train` separately, monitors the training process, and keeps serving the currently active model while training is running.

### Why keep the original training data?

Training only on newly collected feedback would make the candidate model heavily dependent on a small and potentially narrow set of examples.

Instead, retraining combines the original training set with accumulated feedback. This reduces the risk of forgetting the original task while allowing the model to learn from user corrections.

### Why have a model registry?

The registry provides a single source of truth for which model version is active.

This allows the application to:

- load the correct model at startup
- record the model version used for predictions
- promote new models explicitly
- reload the active model after retraining

## Current limitations

This project is intentionally a learning and portfolio project, not a production ML platform.

Current limitations include:

- the original dataset is relatively small
- the evaluation test split is small
- accuracy alone is not sufficient for a complete production evaluation strategy
- retraining currently runs on the local machine
- PostgreSQL is expected to be available
- authentication and authorization are not implemented
- database migrations are not yet part of the project
- the retraining worker is designed for a single local application process
- generated retrained model versions are treated as local runtime artifacts

These are areas for future engineering improvements.

## Future improvements

Potential next steps include:

- precision, recall, and F1 evaluation
- a larger and more representative test dataset
- experiment tracking
- persistent training-run metadata
- database migrations with Alembic
- structured application logging
- authentication
- rate limiting
- production deployment
- containerization with Docker
- cloud deployment
- monitoring and metrics
- richer model analytics in the frontend

## Learning goals

This project was developed as a practical exercise in building an ML-powered application rather than only experimenting in a notebook.

The main learning areas were:

1. Fine-tuning and serving transformer models.
2. Designing an API around an ML model.
3. Persisting model predictions and user feedback.
4. Building a feedback-driven retraining loop.
5. Managing model versions.
6. Evaluating and promoting candidate models.
7. Reloading models without restarting the API.
8. Structuring an ML application into separate layers.
9. Writing automated API tests.
10. Connecting a browser interface to a Python ML backend.

## Author

Mahdi Azad

GitHub: https://github.com/mahdiiazaad

---

Built as a practical end-to-end machine learning engineering project.
