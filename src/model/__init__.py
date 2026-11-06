from src.model.train import train_model
from src.model.predict import LitterDetector
from src.model.evaluate import evaluate_model
from src.model.compare_models import compare_models_and_scoring

__all__ = [
    "train_model",
    "LitterDetector",
    "evaluate_model",
    "compare_models_and_scoring",
]

