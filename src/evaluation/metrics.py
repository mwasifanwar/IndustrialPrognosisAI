import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for RUL prediction.
    Includes standard regression metrics and domain-specific measures.
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Basic regression metrics
        metrics.update(self._calculate_basic_metrics(y_true, y_pred))
        
        # Error distribution metrics
        metrics.update(self._calculate_error_metrics(y_true, y_pred))
        
        # Domain-specific metrics
        metrics.update(self._calculate_domain_metrics(y_true, y_pred))
        
        # Statistical tests
        metrics.update(self._calculate_statistical_metrics(y_true, y_pred))
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic regression metrics."""
        errors = y_true - y_pred
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': self._mean_absolute_percentage_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'max_error': np.max(np.abs(errors)),
            'explained_variance': self._explained_variance_score(y_true, y_pred)
        }
    
    def _calculate_error_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate error distribution metrics."""
        errors = y_true - y_pred
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'error_skewness': stats.skew(errors),
            'error_kurtosis': stats.kurtosis(errors),
            'error_iqr': np.percentile(errors, 75) - np.percentile(errors, 25)
        }
    
    def _calculate_domain_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate domain-specific metrics for RUL prediction."""
        errors = y_true - y_pred
        
        # Early prediction penalty (penalize late predictions more)
        early_prediction_score = self._calculate_early_prediction_score(y_true, y_pred)
        
        # Conservative prediction score (reward conservative estimates)
        conservative_score = self._calculate_conservative_score(y_true, y_pred)
        
        # Degradation tracking score
        degradation_score = self._calculate_degradation_score(y_true, y_pred)
        
        return {
            'early_prediction_score': early_prediction_score,
            'conservative_score': conservative_score,
            'degradation_tracking_score': degradation_score,
            'prognostic_horizon': self._calculate_prognostic_horizon(y_true, y_pred),
            'alpha_lambda': self._calculate_alpha_lambda(y_true, y_pred)
        }
    
    def _calculate_statistical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate statistical test metrics."""
        errors = y_true - y_pred
        
        # Normality test (p-value)
        _, normality_pvalue = stats.normaltest(errors)
        
        # Correlation tests
        pearson_corr, _ = stats.pearsonr(y_true, y_pred)
        spearman_corr, _ = stats.spearmanr(y_true, y_pred)
        
        return {
            'normality_pvalue': normality_pvalue,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'error_autocorrelation': self._calculate_autocorrelation(errors)
        }
    
    def _mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE with handling for zero values."""
        # Avoid division by zero
        mask = y_true != 0
        if np.sum(mask) == 0:
            return float('inf')
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _explained_variance_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate explained variance score."""
        return 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    def _calculate_early_prediction_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate early prediction score.
        Lower scores for late predictions (under-estimation of RUL).
        """
        errors = y_true - y_pred
        late_predictions = errors[errors < 0]  # Negative errors mean late predictions
        
        if len(late_predictions) == 0:
            return 1.0
        
        penalty = np.exp(-np.abs(late_predictions) / 10).mean()
        return penalty
    
    def _calculate_conservative_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate conservative prediction score.
        Reward predictions that are slightly conservative (over-estimation).
        """
        errors = y_true - y_pred
        conservative_predictions = errors[errors > 0]  # Positive errors mean conservative
        
        if len(conservative_predictions) == 0:
            return 0.0
        
        reward = np.tanh(conservative_predictions / 10).mean()
        return max(0, reward)
    
    def _calculate_degradation_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate degradation tracking score.
        Measures how well the prediction follows the degradation trend.
        """
        if len(y_true) < 2:
            return 0.0
        
        true_trend = np.diff(y_true)
        pred_trend = np.diff(y_pred)
        
        if np.std(true_trend) == 0:
            return 1.0 if np.std(pred_trend) == 0 else 0.0
        
        correlation = np.corrcoef(true_trend, pred_trend)[0, 1]
        return max(0, correlation)  # Return 0 if negative correlation
    
    def _calculate_prognostic_horizon(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    threshold: float = 0.1) -> float:
        """
        Calculate prognostic horizon - how early we can accurately predict failure.
        """
        errors = np.abs(y_true - y_pred) / y_true
        accurate_predictions = errors < threshold
        
        if not np.any(accurate_predictions):
            return 0.0
        
        # Return the earliest point where predictions become accurate
        return np.argmax(accurate_predictions) / len(accurate_predictions)
    
    def _calculate_alpha_lambda(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate alpha-lambda metric for prognostic performance.
        """
        # Simplified implementation - can be enhanced based on specific requirements
        errors = np.abs(y_true - y_pred)
        return 1.0 / (1.0 + np.mean(errors / (y_true + 1e-8)))
    
    def _calculate_autocorrelation(self, errors: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation of errors."""
        if len(errors) <= lag:
            return 0.0
        
        shifted = errors[lag:]
        original = errors[:-lag]
        
        if np.std(original) == 0 or np.std(shifted) == 0:
            return 0.0
        
        return np.corrcoef(original, shifted)[0, 1]
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of all metrics from history."""
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
    
    def compare_models(self, model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """
        Compare multiple models based on their predictions.
        
        Args:
            model_predictions: Dictionary with model names as keys and (y_true, y_pred) as values
            
        Returns:
            DataFrame with metrics comparison
        """
        comparison_data = []
        
        for model_name, (y_true, y_pred) in model_predictions.items():
            metrics = self.calculate_all_metrics(y_true, y_pred)
            metrics['model'] = model_name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        return df.set_index('model')