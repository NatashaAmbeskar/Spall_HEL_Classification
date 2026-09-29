from build_training_dataset import build_training_dataset_gb
from functools import partial
from Transformer_Classes import GBFeatureExtractor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import os
model_dir="GB/FE"
load_data=partial(build_training_dataset_gb,needed_data="unv")

def build_pipeline():
    return Pipeline([
        ("transformer", GBFeatureExtractor()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1))
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






