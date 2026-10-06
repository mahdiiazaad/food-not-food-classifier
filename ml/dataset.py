from datasets import Dataset, DatasetDict, concatenate_datasets
import datasets


DATASET_NAME = "mrdbourke/learn_hf_food_not_food_image_captions"

LABEL2ID = {
    "not_food": 0,
    "food": 1,
}

ID2LABEL = {
    0: "not_food",
    1: "food",
}


def create_feedback_dataset(
    feedback_data: list[dict],
) -> Dataset:

    return Dataset.from_list(feedback_data)


def create_retraining_dataset(
    original_train_dataset: Dataset,
    feedback_dataset: Dataset,
) -> Dataset:

    return concatenate_datasets(
        [
            original_train_dataset,
            feedback_dataset,
        ]
    )


def map_label_to_ids(example: dict) -> dict:
    return {
        "text": example["text"],
        "label": LABEL2ID[example["label"]],
    }


def load_original_dataset(
    dataset_name: str = DATASET_NAME,
) -> DatasetDict:

    print(
        f"[INFO] downloading dataset from Hugging Face Hub: "
        f"{dataset_name}"
    )

    dataset = datasets.load_dataset(dataset_name)

    dataset = dataset["train"].map(map_label_to_ids)

    train_test = dataset.train_test_split(
        test_size=0.2,
        seed=42,
    )

    validation_test = train_test["test"].train_test_split(
        test_size=0.5,
        seed=42,
    )

    return DatasetDict(
        {
            "train": train_test["train"],
            "validation": validation_test["train"],
            "test": validation_test["test"],
        }
    )
    
    
if __name__ == '__main__':
  from app.db.database import SessionLocal
  from app.services.training_data_service import get_all_feedback_training_data
  from ml.dataset import (
      create_feedback_dataset,
      create_retraining_dataset,
      load_original_dataset,
  )


  db = SessionLocal()

  try:
      original_dataset = load_original_dataset()

      feedback_data = get_all_feedback_training_data(db)

  finally:
      db.close()


  feedback_dataset = create_feedback_dataset(feedback_data)

  retraining_train_dataset = create_retraining_dataset(
      original_train_dataset=original_dataset["train"],
      feedback_dataset=feedback_dataset,
  )


  print("Original train:", len(original_dataset["train"]))
  print("Feedback:", len(feedback_dataset))
  print("Retraining train:", len(retraining_train_dataset))

  print("\nOriginal features:")
  print(original_dataset["train"].features)

  print("\nFeedback features:")
  print(feedback_dataset.features)

  print("\nRetraining features:")
  print(retraining_train_dataset.features)