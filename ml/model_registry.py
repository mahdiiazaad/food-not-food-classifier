from pathlib import Path
import json

# create a json file to save the active model, model versions and thier relating paths
REGISTRY_PATH = Path("models_v/registry.json")


BASELINE_VERSION = "baseline" # name of the model like baseline, v001, v002 ...
BASELINE_PATH = Path(
    "models_v/learn_hf_food_not_food_text_classifier_distilbert"
)


def create_registry() -> None:

    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if REGISTRY_PATH.exists():
        return

    registry = {
        "active_version": BASELINE_VERSION,
        "models": {
            BASELINE_VERSION: {
                "path": str(BASELINE_PATH),
            }
        },
    }

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            registry,
            file,
            indent=4,
        )


def load_registry() -> dict:
    # see if the file exist or not, if not create it's first version
    create_registry()
    # turn the content of the json file into a python dict
    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_active_model_path() -> Path:
    # registery ----> python[dict]
    registry = load_registry()
    # access the value using the key ---> name of the model that is active
    active_version = registry["active_version"]
    # access all the models using the models key and then use the active model name to get the path
    model_path = registry["models"][active_version]["path"]

    return Path(model_path)

def get_active_model_version() -> str:
    registery = load_registry()
    return registery['active_version']
     

# this function will add the new model name and path to the regitery.json file 
def register_model(
    version: str,
    model_path: Path,
) -> None:

    # load the file as a python dictionary
    registry = load_registry()

    # add the new model name and path to the dictionary
    registry["models"][version] = {
        "path": model_path.as_posix(),
    }

    # open the json file as writing mode and dump teh new content to this file 
    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            registry,
            file,
            indent=4,
        )

# get a model name (version) as an input and set it as the active model
def set_active_model(version: str) -> None:

    registry = load_registry()

    if version not in registry["models"]:
        raise ValueError(
            f"Model version '{version}' is not registered."
        )

    registry["active_version"] = version

    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            registry,
            file,
            indent=4,
        )

def promote_model(version: str,
                  model_path: str) -> None:
    """ register model and make it the active model within the json file"""
    
    register_model(
        version=version,
        model_path=model_path
    )
    
    set_active_model(version=version)
    
        
if __name__ == "__main__":
    
    model_path=Path("models_v") / "v002"
    register_model(
    version='v002',
    model_path=model_path
  )
  
    set_active_model(
    version='v002'
  )
  
    path = get_active_model_path()

    print("Active model:", path)  