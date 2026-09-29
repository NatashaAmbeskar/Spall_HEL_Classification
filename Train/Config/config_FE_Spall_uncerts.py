from build_training_dataset import build_training_dataset_spall
from functools import partial
from Transformer_Classes import Spall_GlobalGeometryFeatureExtractor_uncerts
#from test_class import Spall_GlobalGeometryFeatureExtractor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import os
model_dir="Spall/Feature_Eng"
load_data=partial(build_training_dataset_spall,needed_data="unv")

def build_pipeline():
    return Pipeline([
        ("transformer", Spall_GlobalGeometryFeatureExtractor_uncerts()),
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