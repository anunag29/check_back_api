import os
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
trocr_dir = dir_path.parent.parent  # this file is in src/configs

trocr_repo = "microsoft/trocr-base-handwritten"
model_path = trocr_dir / "models"
