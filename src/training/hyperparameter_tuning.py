import optuna
import numpy as np
from typing import Dict, Any, Callable
import logging
import mlflow
from functools import partial

from ..models.model_factory import ModelFactory
from ..data.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """
    Advanced hyperparameter tuning using Optuna.
    Supports multiple search strategies and early stopping.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.study = None
        self.best_params = None
        
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Validation loss
        """
        # Suggest hyperparameters
        params = self._suggest_parameters(trial)
        
        try:
            # Update config with suggested parameters
            tuned_config = self._update_config(params)
            
            # Create and train model
            model = ModelFactory.create_model(
                self.config['model']['name'], tuned_config
            )
            model.compile_model()
            
            # Train with early stopping
            history = model.train(
                X_train, y_train, X_val, y_val, 
                callbacks=[self._get_early_stopping_callback()]
            )
            
            # Get best validation loss
            best_val_loss = min(history.history['val_loss'])
            
            # Report intermediate values if possible
            for epoch, (train_loss, val_loss) in enumerate(zip(
                history.history['loss'], history.history['val_loss']
            )):
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_val_loss
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        search_space = self.config['hyperparameter_tuning']['search_space']
        params = {}
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                if param_config.get('log', False):
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'], log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                    
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high']
                )
                
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )
        
        return params
    
    def _update_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration with tuned parameters."""
        tuned_config = self.config.copy()
        
        # Map parameter names to config structure
        param_mapping = {
            'learning_rate': ['training', 'learning_rate'],
            'batch_size': ['training', 'batch_size'],
            'hidden_units': ['model', 'architecture', 'dense_layers', 0, 'units'],
            'filters': ['model', 'architecture', 'conv_layers', 0, 'filters'],
            'kernel_size': ['model', 'architecture', 'conv_layers', 0, 'kernel_size'],
            'dropout_rate': ['model', 'architecture', 'dense_layers', 0, 'dropout'],
            'l2_regularization': ['model', 'architecture', 'l2_regularization']
        }
        
        for param_name, value in params.items():
            if param_name in param_mapping:
                config_path = param_mapping[param_name]
                current = tuned_config
                for key in config_path[:-1]:
                    current = current[key]
                current[config_path[-1]] = value
        
        return tuned_config
    
    def _get_early_stopping_callback(self):
        """Get early stopping callback for tuning."""
        import tensorflow as tf
        return tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, min_delta=0.001
        )
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Optimization results
        """
        tuning_config = self.config['hyperparameter_tuning']
        
        # Create study
        study = optuna.create_study(
            direction=tuning_config['direction'],
            pruner=optuna.pruners.HyperbandPruner()
        )
        
        # Create objective function with fixed data
        objective_func = partial(
            self.objective, 
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
        )
        
        # Optimize
        study.optimize(
            objective_func, 
            n_trials=tuning_config['n_trials'],
            timeout=tuning_config['timeout'],
            show_progress_bar=True
        )
        
        self.study = study
        self.best_params = study.best_params
        
        logger.info(f"Hyperparameter optimization completed. Best value: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial,
            'study': study
        }
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("No study available. Run optimize first.")
        
        try:
            import matplotlib.pyplot as plt
            fig = optuna.visualization.plot_optimization_history(self.study)
            return fig
        except ImportError:
            logger.warning("Plotly not available for visualization")
            return None
    
    def plot_parallel_coordinate(self):
        """Plot parallel coordinate plot of optimization results."""
        if self.study is None:
            raise ValueError("No study available. Run optimize first.")
        
        try:
            fig = optuna.visualization.plot_parallel_coordinate(self.study)
            return fig
        except ImportError:
            logger.warning("Plotly not available for visualization")
            return None
    
    def get_optimization_report(self) -> str:
        """Generate a text report of optimization results."""
        if self.study is None:
            return "No optimization study available."
        
        report = []
        report.append("Hyperparameter Optimization Report")
        report.append("=" * 40)
        report.append(f"Best value: {self.study.best_value:.6f}")
        report.append(f"Best trial: #{self.study.best_trial.number}")
        report.append("\nBest parameters:")
        for param, value in self.study.best_params.items():
            report.append(f"  {param}: {value}")
        
        report.append(f"\nTotal trials: {len(self.study.trials)}")
        report.append(f"Completed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        
        return "\n".join(report)