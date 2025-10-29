"""
Advanced Condition Monitoring and RUL Prediction Package
"""

__version__ = "1.0.0"
__author__ = "mwasifanwar"
__email__ = "your-email@example.com"

from .data.data_loader import CMAPPSDataLoader
from .models.cnn_model import AdvancedCNNModel
from .training.trainer import ModelTrainer
from .evaluation.metrics import EvaluationMetrics

__all__ = [
    "CMAPPSDataLoader",
    "AdvancedCNNModel", 
    "ModelTrainer",
    "EvaluationMetrics",
]