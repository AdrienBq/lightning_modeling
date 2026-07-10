from pathlib import Path

# NOTE: ROOT_PATH is derived from the *current working directory*, not this
# file's location. It assumes the process runs from a first-level subdirectory
# of the repo (e.g. `notebooks/`), so `parents[0]` resolves to the repo root.
# Running from the repo root itself would make these paths point one level too
# high. The reproduction notebooks live in `notebooks/`, so run them from there.
ROOT_PATH = Path.cwd().parents[0]
DATASET_PATH = ROOT_PATH / "data"
MODELS_PATH = ROOT_PATH / "models"
