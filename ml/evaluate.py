from pathlib import Path

import numpy as np
import evaluate

from transformers import (
  AutoTokenizer,
  AutoModelForSequenceClassification,
  Trainer,
  TrainingArguments
)

from ml.dataset import load_original_dataset


def tokenized_dataset(dataset, tokenizer):
  def tokenize(example):
    return tokenizer(
      example['text'],
      truncation = True,
      padding = True
    )
    
  return dataset.map(
    tokenize,
    batched=True,
    batch_size = 1000
  )
  

def evaluate_model(model_path: Path,
                   test_dataset) -> dict:
  tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    use_fast=True
  )
  
  model = AutoModelForSequenceClassification.from_pretrained(
    model_path
  )
  
  tokenized_test = tokenized_dataset(
    test_dataset,
    tokenizer
  )
  
  accuracy_metric = evaluate.load('accuracy')
  
  def compute_metrics(prediction_and_labels):
    prediction, labels = prediction_and_labels
    
    if len(prediction.shape) >= 2:
      prediction = np.argmax(prediction, axis=1)
    
    return accuracy_metric.compute(
      predictions=prediction,
      references=labels
    )
    
  training_args = TrainingArguments(
    output_dir='tmp/evaluation',
    use_cpu=True,
    report_to='none'
  )
  
  trainer = Trainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
  )
  
  result = trainer.predict(tokenized_test)
  
  return result.metrics


def is_model_better(current_metrics: dict,
                    new_metrics: dict) -> bool:
  return (new_metrics['test_accuracy'] >= current_metrics['test_accuracy'])
  


if __name__ == "__main__":

    dataset = load_original_dataset()

    test_dataset = dataset["test"]

    baseline_path = Path(
        "models_v/learn_hf_food_not_food_text_classifier_distilbert"
    )

    v001_path = Path(
        "models_v/v002"
    )

    baseline_metrics = evaluate_model(
        baseline_path,
        test_dataset,
    )

    v001_metrics = evaluate_model(
        v001_path,
        test_dataset,
    )

    print("\nBaseline:")
    print(baseline_metrics)

    print("\nv001:")
    print(v001_metrics)
    
    accepted = is_model_better(
    baseline_metrics,
    v001_metrics,
    )

    print(f"\nNew model accepted: {accepted}")