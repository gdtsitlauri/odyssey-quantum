"""Training exports."""

from odyssey.training.calibration import TemperatureScaler
from odyssey.training.trainer import TrainingResult, predict_torch_model, resolve_device, train_model, train_sklearn_model

__all__ = [
    "TemperatureScaler",
    "TrainingResult",
    "predict_torch_model",
    "resolve_device",
    "train_model",
    "train_sklearn_model",
]

