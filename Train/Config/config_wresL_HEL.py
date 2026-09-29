from build_training_dataset import build_training_dataset_hel
from functools import partial
from Transformer_Classes import WeightedResidualHELFeatureExtractor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import os
model_dir="HEL/rL/weighted"
load_data=partial(build_training_dataset_hel,needed_data="unv")

def build_pipeline():
    return Pipeline([
        ("transformer",WeightedResidualHELFeatureExtractor(dt=0.1)),
        ("model",RandomForestClassifier(random_state=42))
    ])

def get_recent_model():
    path=Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    latest_file = max(
    (f for f in path.iterdir() if f.is_file()), 
    key=os.path.getctime,
    default=None
    )
    return latest_file