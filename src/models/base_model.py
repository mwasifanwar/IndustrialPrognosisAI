from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all RUL prediction models.
    Defines the interface that all models must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.history = None
        self._build_model()
    
    @abstractmethod
    def _build_model(self) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def compile_model(self) -> None:
        """Compile the model with optimizer, loss, and metrics."""
        pass
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              callbacks: Optional[list] = None) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            callbacks: List of callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call _build_model first.")
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        return self.history
    
    def predict(self, X) -> tf.Tensor:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not built.")
        return self.model.predict(X)
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not built.")
        
        metrics = self.model.evaluate(X, y, verbose=0)
        if not isinstance(metrics, list):
            metrics = [metrics]
        
        metric_names = [m if isinstance(m, str) else f'metric_{i}' 
                       for i, m in enumerate(self.model.metrics_names)]
        results = dict(zip(metric_names, metrics))
        
        logger.info(f"Model evaluation results: {results}")
        return results
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not built.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def summary(self) -> str:
        """Get model summary."""
        if self.model is None:
            return "Model not built."
        
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.copy()