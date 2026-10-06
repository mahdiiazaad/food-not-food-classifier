from transformers import pipeline
import torch

from ml.model_registry import (
    get_active_model_path,
    get_active_model_version,
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


class ModelManager:

    def __init__(
        self,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    ):
        self.device = device
        self.batch_size = batch_size
        
        self.model_path = None
        self.model_version = None
        self.classifier = None
        
        self.load_model()

    def load_model(self) -> None:
        # getting the active model's path
        model_path = get_active_model_path()
        model_version = get_active_model_version()
        
        print(f"[INFO] Loading active model from: {model_path}")
        
        # create a classification pipeline based on the path
        self.classifier = pipeline(
            task='text-classification',
            model=model_path,
            device=self.device,
            top_k = 1,
            batch_size = self.batch_size
        )
        
        # update the model_path variable
        self.model_path = model_path
        self.model_version = model_version
        
        print(
            f"[INFO] Active model loaded: {model_path}"
        )

        print(
            f"[INFO] Active model version: {model_version}"
        )
        
    def predict(self, text):
        return self.classifier(text)
    
    def reload_model(self) -> None:
        print("[INFO] Reloading active model")

        self.load_model()

if __name__ == "__main__":

    manager = ModelManager()
    
    print(manager.model_path)