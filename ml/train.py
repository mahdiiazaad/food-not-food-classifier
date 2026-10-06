from pathlib import Path
import pprint

import numpy as np

import evaluate

from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

### new import using the function that we created to have the original data along with the feedback data
from app.db.database import SessionLocal
from app.services.training_data_service import get_all_feedback_training_data
from ml.dataset import (load_original_dataset, 
                        create_feedback_dataset,
                        create_retraining_dataset,
                        ID2LABEL, 
                        LABEL2ID)

from ml.model_registry import (
    get_active_model_path,
    get_active_model_version,
    promote_model,
)

# import the fucntion to get the model versio
from ml.model_version import get_next_model_version

from ml.evaluate import evaluate_model, is_model_better

# getting the current active model version and model path 
current_active_version = get_active_model_version()
current_active_path = get_active_model_path()

print(
    f"[INFO] Current active model: "
    f"{current_active_version}"
)

print(
    f"[INFO] Current active model path: "
    f"{current_active_path}"
)

# Setup variables for model training and saving pipeline
MODEL_NAME = 'distilbert/distilbert-base-uncased'

MODEL_VERSION, model_save_dir = get_next_model_version()

# Create a directory for saving models
print(f"[INFO] Creating model save directory: {model_save_dir}")

model_save_dir.mkdir(parents=True, exist_ok=True)



# 2. Load tokenizer (name of the tokenizer of the model and the Hf model are the same)
print(f'[INFO] tokenizign text for model training with tokenizer{MODEL_NAME}')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# 3. Tokenize dataset
def tokenizer_fn(input):
  return tokenizer(input['text'], truncation=True, padding=True)

# loading the original dataset
dataset = load_original_dataset()

print("[INFO] Loading feedback data from PostgreSQL")
db = SessionLocal()

try:
  feedback_data = get_all_feedback_training_data(db=db)
  
finally:
  db.close()
  
feedback_dataset = create_feedback_dataset(feedback_data)

retraining_train_dataset = create_retraining_dataset(
    original_train_dataset=dataset["train"],
    feedback_dataset=feedback_dataset,
)
# this will be used for training only
tokenized_training_dataset = retraining_train_dataset.map(tokenizer_fn, batched=True, batch_size=1000)

# creating validation and test dataset from the original dataset without teh feedbacks
tokenized_validation = dataset['validation'].map(tokenizer_fn, batched=True, batch_size=1000)
tokenized_test = dataset['test'].map(tokenizer_fn, batched=True, batch_size=1000)

# 4. Define evaluation metrics
accuracy_metrics = evaluate.load('accuracy')

def compute_metrics(predictions_and_labels):
  predictions, labels = predictions_and_labels
  
  if len(predictions.shape) >=2:
    predictions = np.argmax(predictions, axis=1)
  
  return accuracy_metrics.compute(predictions=predictions, references=labels)


# 5. Load pretrained model
print(f'[INFO] loading model -> name: {MODEL_NAME}')

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                           num_labels=len(ID2LABEL),
                                           id2label=ID2LABEL,
                                           label2id=LABEL2ID)
print(f'[INFO] model loading complete')


# 6. Create TrainingArguments
training_arg = TrainingArguments(
  output_dir=model_save_dir,
  
  learning_rate=0.0001,
  
  per_device_train_batch_size=32,
  per_device_eval_batch_size=32,
  
  num_train_epochs= 10,
  
  eval_strategy='epoch',
  save_strategy='epoch',
  logging_strategy='epoch',
  
  save_total_limit=3,
  
  use_cpu=True,
  
  load_best_model_at_end=True,
  metric_for_best_model='accuracy',
  
  
  report_to='none',
  push_to_hub=False
  
)

# 7. Create Trainer
trainer = Trainer(
  model=model,
  args=training_arg,
  train_dataset=tokenized_training_dataset,
  eval_dataset=tokenized_validation,
  processing_class=tokenizer,
  compute_metrics=compute_metrics
)

# 8. Train
print(f'[INFO] training model')
results = trainer.train()
print(f'[INFO] training complete')

# 9. Evaluate
preditions_all = trainer.predict(tokenized_test)
predition_values = preditions_all.predictions
prediction_metrics = preditions_all.metrics
print(f'[INFO] prediction metrics: {prediction_metrics}')

pprint.pprint(prediction_metrics)
# 10. Save model and tokenizer to:
# 9. Save the train model (to a local directory)
print(f'[INFO] model training is complete')
print(f'[INFO] saving model to {model_save_dir}')
trainer.save_model(model_save_dir)
print(f'[INFO] model saved to {model_save_dir}')

print("[INFO] Evaluating current active model")
current_metrics = evaluate_model(
  model_path=current_active_path,
  test_dataset=dataset['test']
)

print(
    f"[INFO] Current model metrics: "
    f"{current_metrics}"
)

new_metrics = evaluate_model(
  model_path=model_save_dir,
  test_dataset = dataset['test']
)

print(
    f"[INFO] New model metrics: "
    f"{new_metrics}"
)

accepted = is_model_better(
  current_metrics=current_metrics,
  new_metrics=new_metrics
)

if accepted:

    print(
        f"[INFO] New model {MODEL_VERSION} "
        f"passed evaluation."
    )

    promote_model(
        version=MODEL_VERSION,
        model_path=model_save_dir,
    )

    print(
        f"[INFO] Model {MODEL_VERSION} "
        f"is now active."
    )

else:

    print(
        f"[INFO] New model {MODEL_VERSION} "
        f"failed evaluation."
    )

    print(
        f"[INFO] Keeping current model "
        f"{current_active_version} active."
    )