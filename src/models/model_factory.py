from typing import Dict, Any, Type
import logging
from .base_model import BaseModel
from .cnn_model import AdvancedCNNModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating different types of models.
    Centralizes model creation and configuration.
    """
    
    # Registry of available models
    _model_registry = {
        'AdvancedCNN': AdvancedCNNModel,
        # Add other models here as they are implemented
        # 'LSTM': LSTMModel,
        # 'Transformer': TransformerModel,
        # 'Hybrid': HybridModel,
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on type and configuration.
        
        Args:
            model_type: Type of model to create
            config: Model configuration dictionary
            
        Returns:
            Instance of the requested model
        """
        if model_type not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available models: {available_models}"
            )
        
        model_class = cls._model_registry[model_type]
        logger.info(f"Creating {model_type} model")
        return model_class(config)
    
    @classmethod
    def register_model(cls, model_name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type in the factory.
        
        Args:
            model_name: Name to register the model under
            model_class: Model class to register
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must inherit from BaseModel")
        
        cls._model_registry[model_name] = model_class
        logger.info(f"Registered new model: {model_name}")
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._model_registry.keys())
    
    @classmethod
    def get_model_config_template(cls, model_type: str) -> Dict[str, Any]:
        """
        Get a configuration template for a specific model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Configuration template dictionary
        """
        templates = {
            'AdvancedCNN': {
                'model': {
                    'name': 'AdvancedCNN',
                    'type': 'convolutional',
                    'window_length': 25,
                    'feature_num': 13,
                    'architecture': {
                        'conv_type': '1d',  # or '2d'
                        'conv_layers': [
                            {
                                'filters': 64,
                                'kernel_size': 3,
                                'activation': 'relu',
                                'padding': 'same',
                                'pooling': True,
                                'pool_size': 2,
                                'dropout': 0.0
                            },
                            {
                                'filters': 32,
                                'kernel_size': 3,
                                'activation': 'relu',
                                'padding': 'same',
                                'pooling': True,
                                'pool_size': 2,
                                'dropout': 0.0
                            }
                        ],
                        'dense_layers': [
                            {
                                'units': 100,
                                'activation': 'relu',
                                'dropout': 0.3
                            },
                            {
                                'units': 50,
                                'activation': 'relu',
                                'dropout': 0.2
                            },
                            {
                                'units': 1,
                                'activation': 'linear'
                            }
                        ],
                        'batch_norm': True,
                        'l2_regularization': 0.001
                    },
                    'compilation': {
                        'optimizer': 'adam',
                        'loss': 'mse',
                        'metrics': ['mae', 'mse']
                    }
                },
                'training': {
                    'batch_size': 32,
                    'epochs': 100,
                    'learning_rate': 0.001,
                    'early_stopping': {
                        'monitor': 'val_loss',
                        'patience': 15,
                        'restore_best_weights': True
                    },
                    'reduce_lr': {
                        'monitor': 'val_loss',
                        'factor': 0.5,
                        'patience': 10,
                        'min_lr': 0.00001
                    }
                }
            }
        }
        
        if model_type not in templates:
            raise ValueError(f"No template available for model type: {model_type}")
        
        return templates[model_type]