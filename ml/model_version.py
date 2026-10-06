from pathlib import Path
import re


MODEL_ROOT_DIR = Path("models_v")


def get_next_model_version(
    model_root_dir: Path = MODEL_ROOT_DIR,
) -> tuple[str, Path]:

    model_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    version_pattern = re.compile(r"^v(\d{3})$")

    existing_versions = []

    for path in model_root_dir.iterdir():

        if not path.is_dir():
            continue

        match = version_pattern.fullmatch(path.name)

        if match:
            existing_versions.append(
                int(match.group(1))
            )

    if existing_versions:
        next_version_number = max(existing_versions) + 1
    else:
        next_version_number = 1

    model_version = f"v{next_version_number:03d}"
    model_path = model_root_dir / model_version

    return model_version, model_path
  
  
if __name__ == "__main__":
  from ml.model_version import get_next_model_version

  version, path = get_next_model_version()

  print("Next version:", version)
  print("Save path:", path)