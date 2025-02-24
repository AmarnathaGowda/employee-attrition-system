# ml_engine/model_registry.py
import joblib
from django.conf import settings
import os

MODEL_PATHS = {
    'logreg': settings.BASE_DIR / 'ml_engine/models/logreg_pipeline.joblib',
    'randomforest': settings.BASE_DIR / 'ml_engine/models/randomforest_pipeline.joblib',
    'xgboost': settings.BASE_DIR / 'ml_engine/models/xgboost_pipeline.joblib'
}

class ModelRegistry:
    def __init__(self):
        self.models = {
            # name: joblib.load(path)
            # for name, path in MODEL_PATHS.items()
        }
        self._loaded = False

    def _load_models(self):
        if not self._loaded:
            for name, path in MODEL_PATHS.items():
                if os.path.exists(path):
                    self.models[name] = joblib.load(path)
                else:
                    raise FileNotFoundError(f"Model file {path} not found. Train models first.")
            self._loaded = True
    
    def get_model(self, name):
        return self.models.get(name)
    
    def predict_proba(self, name, data):
        pipeline = self.get_model(name)
        return pipeline.predict_proba(data)[:, 1]

