import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, 
    Flatten, Dense, Dropout, BatchNormalization,
    Reshape, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, Any, List
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class AdvancedCNNModel(BaseModel):
    """
    Advanced CNN model for RUL prediction with multiple architectural options.
    Supports both 1D and 2D convolutions.
    """
    
    def _build_model(self) -> None:
        """Build the CNN architecture based on configuration."""
        model_config = self.config['model']['architecture']
        input_shape = self._get_input_shape()
        
        self.model = Sequential()
        
        # Input layer with reshaping if needed
        if model_config.get('conv_type', '1d') == '2d':
            # Reshape for 2D convolutions (channels_last)
            self.model.add(Reshape(input_shape + (1,), input_shape=input_shape))
            conv_layer = Conv2D
            pool_layer = MaxPooling2D
        else:
            # Use 1D convolutions
            conv_layer = Conv1D
            pool_layer = MaxPooling1D
        
        # Convolutional layers
        for i, conv_config in enumerate(model_config['conv_layers']):
            if i == 0:
                # First layer specifies input shape
                self.model.add(conv_layer(
                    filters=conv_config['filters'],
                    kernel_size=conv_config['kernel_size'],
                    activation=conv_config['activation'],
                    padding=conv_config.get('padding', 'same'),
                    kernel_regularizer=l2(model_config.get('l2_regularization', 0.001)),
                    input_shape=input_shape if conv_layer == Conv1D else None
                ))
            else:
                self.model.add(conv_layer(
                    filters=conv_config['filters'],
                    kernel_size=conv_config['kernel_size'],
                    activation=conv_config['activation'],
                    padding=conv_config.get('padding', 'same'),
                    kernel_regularizer=l2(model_config.get('l2_regularization', 0.001))
                ))
            
            # Add batch normalization if configured
            if model_config.get('batch_norm', True):
                self.model.add(BatchNormalization())
            
            # Add pooling layer
            if conv_config.get('pooling', True):
                self.model.add(pool_layer(
                    pool_size=conv_config.get('pool_size', 2),
                    padding='same'
                ))
            
            # Add dropout if configured
            dropout_rate = conv_config.get('dropout', 0.0)
            if dropout_rate > 0:
                self.model.add(Dropout(dropout_rate))
        
        # Flatten before dense layers
        if model_config.get('conv_type', '1d') == '2d':
            self.model.add(Flatten())
        else:
            self.model.add(GlobalAveragePooling1D())
        
        # Dense layers
        for i, dense_config in enumerate(model_config['dense_layers']):
            self.model.add(Dense(
                units=dense_config['units'],
                activation=dense_config['activation'],
                kernel_regularizer=l2(model_config.get('l2_regularization', 0.001))
            ))
            
            # Add batch normalization if configured
            if model_config.get('batch_norm', True) and i < len(model_config['dense_layers']) - 1:
                self.model.add(BatchNormalization())
            
            # Add dropout (except for output layer)
            dropout_rate = dense_config.get('dropout', 0.0)
            if dropout_rate > 0 and i < len(model_config['dense_layers']) - 1:
                self.model.add(Dropout(dropout_rate))
        
        logger.info("Advanced CNN model built successfully")
    
    def _get_input_shape(self) -> tuple:
        """Get input shape based on configuration."""
        window_length = self.config['model']['window_length']
        feature_num = self.config['model']['feature_num']
        conv_type = self.config['model']['architecture'].get('conv_type', '1d')
        
        if conv_type == '2d':
            # For 2D conv, we'll reshape to (window_length, feature_num, 1)
            return (window_length, feature_num)
        else:
            # For 1D conv, input shape is (window_length, feature_num)
            return (window_length, feature_num)
    
    def compile_model(self) -> None:
        """Compile the model with optimizer and loss function."""
        compile_config = self.config['model']['compilation']
        
        optimizer = Adam(
            learning_rate=self.config['training'].get('learning_rate', 0.001),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=compile_config['loss'],
            metrics=compile_config['metrics']
        )
        
        logger.info("Model compiled with optimizer: Adam, "
                   f"loss: {compile_config['loss']}, "
                   f"metrics: {compile_config['metrics']}")
    
    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Get training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping_config = self.config['training']['early_stopping']
        callbacks.append(EarlyStopping(
            monitor=early_stopping_config['monitor'],
            patience=early_stopping_config['patience'],
            restore_best_weights=early_stopping_config['restore_best_weights'],
            verbose=1
        ))
        
        # Learning rate reduction
        reduce_lr_config = self.config['training']['reduce_lr']
        callbacks.append(ReduceLROnPlateau(
            monitor=reduce_lr_config['monitor'],
            factor=reduce_lr_config['factor'],
            patience=reduce_lr_config['patience'],
            min_lr=reduce_lr_config['min_lr'],
            verbose=1
        ))
        
        return callbacks
    
    def create_attention_layer(self, input_tensor):
        """
        Create attention mechanism for the CNN.
        This helps the model focus on important time steps.
        """
        from tensorflow.keras.layers import Dense, Multiply, Activation
        
        # Simple attention mechanism
        attention = Dense(input_tensor.shape[-1], activation='softmax')(input_tensor)
        context_vector = Multiply()([input_tensor, attention])
        return context_vector