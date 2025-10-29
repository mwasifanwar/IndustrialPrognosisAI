import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path
import mlflow
import mlflow.keras
from datetime import datetime

from ..models.base_model import BaseModel
from ..models.model_factory import ModelFactory
from ..data.data_loader import CMAPPSDataLoader
from ..data.preprocessor import DataPreprocessor
from ..evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Advanced model trainer with experiment tracking, cross-validation,
    and comprehensive logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = CMAPPSDataLoader()
        self.preprocessor = DataPreprocessor(config)
        self.model = None
        self.experiment_results = {}
        
    def setup_mlflow(self) -> None:
        """Setup MLflow for experiment tracking."""
        if self.config['experiment']['tracking']:
            mlflow.set_tracking_uri(self.config['experiment']['mlflow_tracking_uri'])
            mlflow.set_experiment(self.config['experiment']['experiment_name'])
            logger.info("MLflow tracking setup completed")
    
    def prepare_data(self) -> Tuple[Dict, Dict]:
        """
        Prepare training and testing data.
        
        Returns:
            Tuple of (train_engines, test_engines)
        """
        logger.info("Preparing data for training...")
        
        # Load training data from multiple datasets
        train_engines = {}
        for dataset_id in range(1, 5):  # Assuming datasets 1-4
            try:
                engines = self.data_loader.get_all_engines(dataset_id, 'train')
                engines_data = self.data_loader.load_multiple_engines(
                    dataset_id, engines[:self.config['data']['train_engines']], 'train'
                )
                train_engines.update(engines_data)
            except Exception as e:
                logger.warning(f"Error loading dataset {dataset_id}: {e}")
                continue
        
        # Load test data
        test_engines = {}
        for dataset_id in range(1, 5):
            try:
                engines = self.data_loader.get_all_engines(dataset_id, 'test')
                engines_data = self.data_loader.load_multiple_engines(
                    dataset_id, engines[:self.config['data']['test_engines']], 'test'
                )
                test_engines.update(engines_data)
            except Exception as e:
                logger.warning(f"Error loading test dataset {dataset_id}: {e}")
                continue
        
        logger.info(f"Loaded {len(train_engines)} training engines and {len(test_engines)} test engines")
        return train_engines, test_engines
    
    def create_sequences(self, engines_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from engine data."""
        window_length = self.config['model']['window_length']
        X, y = self.preprocessor.prepare_multiple_engines_data(engines_data, window_length)
        
        # Reshape for model input
        if self.config['model']['architecture'].get('conv_type', '1d') == '2d':
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        else:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        
        logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, 
                   y_val: Optional[np.ndarray] = None) -> BaseModel:
        """Train the model."""
        logger.info("Starting model training...")
        
        # Create model
        model_type = self.config['model']['name']
        self.model = ModelFactory.create_model(model_type, self.config)
        self.model.compile_model()
        
        # Get callbacks
        callbacks = []
        if hasattr(self.model, 'get_callbacks'):
            callbacks = self.model.get_callbacks()
        
        # Start MLflow run
        if self.config['experiment']['tracking']:
            mlflow.start_run()
            mlflow.log_params(self._flatten_config(self.config))
            mlflow.keras.autolog()
        
        # Train model
        history = self.model.train(
            X_train, y_train, X_val, y_val, callbacks
        )
        
        # Log training results
        self.experiment_results['training_history'] = history.history
        self.experiment_results['best_epoch'] = len(history.history['loss'])
        
        if self.config['experiment']['tracking']:
            # Log metrics
            for metric, values in history.history.items():
                mlflow.log_metric(f"final_{metric}", values[-1])
                mlflow.log_metric(f"best_{metric}", min(values) if 'loss' in metric else max(values))
        
        logger.info("Model training completed")
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        logger.info("Evaluating model on test data...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics_calculator = EvaluationMetrics()
        metrics = metrics_calculator.calculate_all_metrics(y_test, y_pred.flatten())
        
        # Log to MLflow
        if self.config['experiment']['tracking']:
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
        
        self.experiment_results['test_metrics'] = metrics
        self.experiment_results['test_predictions'] = {
            'true': y_test,
            'predicted': y_pred.flatten()
        }
        
        logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete training experiment."""
        logger.info("Starting complete training experiment...")
        
        try:
            # Setup
            self.setup_mlflow()
            
            # Prepare data
            train_engines, test_engines = self.prepare_data()
            
            # Create sequences
            X_train, y_train = self.create_sequences(train_engines)
            X_test, y_test = self.create_sequences(test_engines)
            
            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Train model
            self.train_model(X_train_split, y_train_split, X_val, y_val)
            
            # Evaluate model
            test_metrics = self.evaluate_model(X_test, y_test)
            
            # Save model
            if self.config['experiment']['save_best_model']:
                model_path = f"models/{self.config['model']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.model.save(model_path)
                if self.config['experiment']['tracking']:
                    mlflow.log_artifact(model_path)
            
            # Save preprocessor
            preprocessor_path = f"models/preprocessor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            self.preprocessor.save_scaler(preprocessor_path)
            
            # End MLflow run
            if self.config['experiment']['tracking']:
                mlflow.end_run()
            
            logger.info("Experiment completed successfully")
            return self.experiment_results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            if self.config['experiment']['tracking']:
                mlflow.end_run(status="FAILED")
            raise
    
    def _flatten_config(self, config: Dict, parent_key: str = '') -> Dict:
        """Flatten nested configuration for MLflow logging."""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available."""
        # This would be implemented for tree-based models
        # For neural networks, we might use permutation importance
        logger.info("Feature importance not implemented for CNN models")
        return None