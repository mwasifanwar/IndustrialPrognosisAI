import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Any, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Advanced model explainability for RUL predictions.
    Provides feature importance, attention visualization, and prediction explanations.
    """
    
    def __init__(self, model, preprocessor, feature_names: List[str]):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explanations = {}
    
    def compute_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                 method: str = 'permutation') -> Dict[str, float]:
        """
        Compute feature importance using various methods.
        
        Args:
            X: Input features
            y: True targets
            method: Method for importance calculation ('permutation', 'gradient', 'shap')
            
        Returns:
            Dictionary of feature importances
        """
        if method == 'permutation':
            return self._permutation_importance(X, y)
        elif method == 'gradient':
            return self._gradient_importance(X)
        elif method == 'shap':
            return self._shap_importance(X)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                              n_repeats: int = 10) -> Dict[str, float]:
        """Compute permutation importance."""
        baseline_score = self._evaluate_model(X, y)
        feature_importance = {}
        
        for feature_idx in tqdm(range(X.shape[-1]), desc="Calculating permutation importance"):
            original_feature = X[:, :, feature_idx].copy()
            scores = []
            
            for _ in range(n_repeats):
                # Shuffle the feature
                X_permuted = X.copy()
                shuffled_feature = original_feature.flatten()
                np.random.shuffle(shuffled_feature)
                X_permuted[:, :, feature_idx] = shuffled_feature.reshape(X.shape[0], X.shape[1])
                
                # Calculate score with shuffled feature
                permuted_score = self._evaluate_model(X_permuted, y)
                scores.append(permuted_score)
            
            # Importance is the decrease in performance
            importance = baseline_score - np.mean(scores)
            feature_name = self.feature_names[feature_idx]
            feature_importance[feature_name] = importance
        
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def _gradient_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Compute importance using gradient-based methods."""
        # Convert to tensorflow tensor
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
        
        # Compute gradients of predictions with respect to inputs
        gradients = tape.gradient(predictions, X_tensor)
        
        # Average absolute gradients across samples and time steps
        importance_scores = np.mean(np.abs(gradients.numpy()), axis=(0, 1))
        
        feature_importance = {}
        for idx, score in enumerate(importance_scores):
            feature_name = self.feature_names[idx]
            feature_importance[feature_name] = score
        
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def _shap_importance(self, X: np.ndarray, n_samples: int = 100) -> Dict[str, float]:
        """Compute SHAP values for feature importance."""
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Using permutation importance instead.")
            return self._permutation_importance(X, np.zeros(len(X)))  # Dummy y
        
        # Sample data for SHAP computation (can be computationally expensive)
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create SHAP explainer
        def predict_fn(x):
            return self.model.predict(x)
        
        explainer = shap.Explainer(predict_fn, X_sample)
        shap_values = explainer(X_sample)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 1))
        
        feature_importance = {}
        for idx, score in enumerate(mean_abs_shap):
            feature_name = self.feature_names[idx]
            feature_importance[feature_name] = score
        
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model and return negative RMSE (for permutation importance)."""
        from ..evaluation.metrics import EvaluationMetrics
        
        y_pred = self.model.predict(X)
        metrics_calculator = EvaluationMetrics()
        metrics = metrics_calculator.calculate_all_metrics(y, y_pred.flatten())
        return -metrics['rmse']  # Negative because lower RMSE is better
    
    def plot_feature_importance(self, importance_scores: Dict[str, float], 
                              top_k: int = 15, interactive: bool = False):
        """Plot feature importance."""
        # Get top k features
        top_features = dict(list(importance_scores.items())[:top_k])
        
        if interactive:
            import plotly.express as px
            
            fig = px.bar(
                x=list(top_features.values()),
                y=list(top_features.keys()),
                orientation='h',
                title=f'Top {top_k} Feature Importance',
                labels={'x': 'Importance', 'y': 'Features'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            features = list(top_features.keys())
            importance = list(top_features.values())
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importance)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Top {top_k} Feature Importance')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            return fig
    
    def analyze_prediction(self, X_single: np.ndarray, y_true: float) -> Dict[str, Any]:
        """
        Analyze a single prediction in detail.
        
        Args:
            X_single: Single input sequence
            y_true: True target value
            
        Returns:
            Dictionary with analysis results
        """
        # Make prediction
        y_pred = self.model.predict(X_single[np.newaxis, ...])[0][0]
        
        # Compute gradients for this prediction
        X_tensor = tf.convert_to_tensor(X_single[np.newaxis, ...], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            prediction = self.model(X_tensor)[0][0]
        
        gradients = tape.gradient(prediction, X_tensor)
        gradient_importance = np.mean(np.abs(gradients.numpy()), axis=1)[0]
        
        # Feature contributions (simplified)
        feature_contributions = {}
        for idx, (grad, feature_name) in enumerate(zip(gradient_importance, self.feature_names)):
            feature_contributions[feature_name] = float(grad)
        
        analysis = {
            'true_value': float(y_true),
            'predicted_value': float(y_pred),
            'error': float(y_true - y_pred),
            'absolute_error': float(np.abs(y_true - y_pred)),
            'feature_contributions': feature_contributions,
            'confidence': self._estimate_confidence(X_single),
            'timestamp_importance': self._compute_timestamp_importance(X_single)
        }
        
        return analysis
    
    def _estimate_confidence(self, X_single: np.ndarray) -> float:
        """Estimate prediction confidence using Monte Carlo dropout."""
        try:
            # Enable dropout at inference time
            predictions = []
            for _ in range(10):
                pred = self.model(X_single[np.newaxis, ...], training=True)
                predictions.append(pred.numpy()[0][0])
            
            confidence = 1.0 / (np.std(predictions) + 1e-8)
            return min(confidence, 1.0)  # Normalize to [0, 1]
        except:
            return 0.5  # Default confidence
    
    def _compute_timestamp_importance(self, X_single: np.ndarray) -> List[float]:
        """Compute importance of each timestamp in the sequence."""
        # Use gradient information across time dimension
        X_tensor = tf.convert_to_tensor(X_single[np.newaxis, ...], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            prediction = self.model(X_tensor)[0][0]
        
        gradients = tape.gradient(prediction, X_tensor)
        timestamp_importance = np.mean(np.abs(gradients.numpy()), axis=2)[0]
        
        return timestamp_importance.tolist()
    
    def generate_explanation_report(self, X: np.ndarray, y: np.ndarray, 
                                  sample_indices: List[int] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation report.
        
        Args:
            X: Input features
            y: True targets
            sample_indices: Indices of samples to analyze in detail
            
        Returns:
            Comprehensive explanation report
        """
        if sample_indices is None:
            sample_indices = list(range(min(5, len(X))))
        
        report = {
            'feature_importance': {},
            'sample_analyses': [],
            'global_insights': {},
            'model_behavior': {}
        }
        
        # Compute feature importance using multiple methods
        logger.info("Computing feature importance...")
        report['feature_importance']['permutation'] = self.compute_feature_importance(X, y, 'permutation')
        report['feature_importance']['gradient'] = self.compute_feature_importance(X, y, 'gradient')
        
        # Analyze specific samples
        logger.info("Analyzing sample predictions...")
        for idx in sample_indices:
            if idx < len(X):
                analysis = self.analyze_prediction(X[idx], y[idx])
                report['sample_analyses'].append({
                    'sample_index': idx,
                    'analysis': analysis
                })
        
        # Generate global insights
        report['global_insights'] = self._generate_global_insights(report['feature_importance'])
        report['model_behavior'] = self._analyze_model_behavior(X, y)
        
        return report
    
    def _generate_global_insights(self, feature_importance: Dict) -> Dict[str, Any]:
        """Generate global insights from feature importance."""
        # Combine importance from different methods
        all_scores = {}
        for method, scores in feature_importance.items():
            for feature, score in scores.items():
                if feature not in all_scores:
                    all_scores[feature] = []
                all_scores[feature].append(score)
        
        # Average scores across methods
        combined_importance = {
            feature: np.mean(scores) for feature, scores in all_scores.items()
        }
        combined_importance = dict(sorted(combined_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
        
        insights = {
            'top_features': list(combined_importance.keys())[:10],
            'feature_categories': self._categorize_features(combined_importance),
            'dominant_features': [f for f, s in combined_importance.items() if s > np.mean(list(combined_importance.values()))],
            'stability_analysis': self._analyze_importance_stability(feature_importance)
        }
        
        return insights
    
    def _categorize_features(self, importance_scores: Dict[str, float]) -> Dict[str, List[str]]:
        """Categorize features based on their names and importance."""
        categories = {
            'sensor_measurements': [],
            'operational_settings': [],
            'temporal_features': [],
            'statistical_features': [],
            'domain_features': []
        }
        
        for feature in importance_scores.keys():
            feature_lower = feature.lower()
            
            if 'sensor' in feature_lower:
                categories['sensor_measurements'].append(feature)
            elif 'opset' in feature_lower or 'setting' in feature_lower:
                categories['operational_settings'].append(feature)
            elif 'cycle' in feature_lower or 'time' in feature_lower:
                categories['temporal_features'].append(feature)
            elif 'mean' in feature_lower or 'std' in feature_lower or 'rolling' in feature_lower:
                categories['statistical_features'].append(feature)
            elif 'health' in feature_lower or 'degradation' in feature_lower:
                categories['domain_features'].append(feature)
            else:
                categories['sensor_measurements'].append(feature)  # Default
        
        return categories
    
    def _analyze_importance_stability(self, feature_importance: Dict) -> Dict[str, Any]:
        """Analyze stability of feature importance across methods."""
        methods = list(feature_importance.keys())
        all_features = set()
        for scores in feature_importance.values():
            all_features.update(scores.keys())
        
        # Calculate rank correlation between methods
        from scipy.stats import spearmanr
        
        if len(methods) >= 2:
            ranks_method1 = [feature_importance[methods[0]].get(f, 0) for f in all_features]
            ranks_method2 = [feature_importance[methods[1]].get(f, 0) for f in all_features]
            correlation, _ = spearmanr(ranks_method1, ranks_method2)
        else:
            correlation = 1.0
        
        stability = {
            'rank_correlation': correlation,
            'consistent_top_features': self._find_consistent_features(feature_importance),
            'method_agreement': 'high' if correlation > 0.7 else 'medium' if correlation > 0.4 else 'low'
        }
        
        return stability
    
    def _find_consistent_features(self, feature_importance: Dict) -> List[str]:
        """Find features that are consistently important across methods."""
        top_features_per_method = []
        for method, scores in feature_importance.items():
            top_k = min(10, len(scores))
            top_features = list(scores.keys())[:top_k]
            top_features_per_method.append(set(top_features))
        
        # Find intersection
        consistent_features = set.intersection(*top_features_per_method)
        return list(consistent_features)
    
    def _analyze_model_behavior(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze overall model behavior."""
        predictions = self.model.predict(X).flatten()
        errors = y - predictions
        
        behavior = {
            'prediction_bias': float(np.mean(errors)),
            'error_distribution': {
                'mean': float(np.mean(np.abs(errors))),
                'std': float(np.std(errors)),
                'skew': float(float('nan') if len(errors) < 3 else pd.Series(errors).skew()),
                'kurtosis': float(float('nan') if len(errors) < 4 else pd.Series(errors).kurtosis())
            },
            'confidence_calibration': self._assess_calibration(predictions, errors),
            'failure_modes': self._identify_failure_modes(X, y, predictions)
        }
        
        return behavior
    
    def _assess_calibration(self, predictions: np.ndarray, errors: np.ndarray) -> Dict[str, Any]:
        """Assess how well calibrated the model's predictions are."""
        # Bin predictions and compute error statistics in each bin
        n_bins = 10
        bins = np.linspace(predictions.min(), predictions.max(), n_bins + 1)
        
        calibration_data = []
        for i in range(n_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i+1])
            if np.sum(mask) > 0:
                bin_errors = errors[mask]
                calibration_data.append({
                    'prediction_range': (float(bins[i]), float(bins[i+1])),
                    'mean_error': float(np.mean(bin_errors)),
                    'std_error': float(np.std(bin_errors)),
                    'n_samples': int(np.sum(mask))
                })
        
        # Calculate calibration score (lower is better)
        calibration_errors = [abs(data['mean_error']) for data in calibration_data]
        calibration_score = np.mean(calibration_errors) if calibration_errors else float('inf')
        
        return {
            'calibration_score': calibration_score,
            'calibration_data': calibration_data,
            'is_well_calibrated': calibration_score < np.std(errors) * 0.5
        }
    
    def _identify_failure_modes(self, X: np.ndarray, y: np.ndarray, 
                              predictions: np.ndarray) -> List[Dict[str, Any]]:
        """Identify common failure modes of the model."""
        errors = y - predictions
        large_errors = np.abs(errors) > np.percentile(np.abs(errors), 90)
        
        failure_modes = []
        
        if np.sum(large_errors) > 0:
            # Analyze patterns in large errors
            X_large_errors = X[large_errors]
            y_large_errors = y[large_errors]
            pred_large_errors = predictions[large_errors]
            
            # Check for systematic over/under prediction
            error_sign = np.sign(errors[large_errors])
            over_prediction_ratio = np.sum(error_sign < 0) / len(error_sign)
            
            failure_modes.append({
                'type': 'large_errors',
                'description': f"Model makes large errors in {np.sum(large_errors)} samples",
                'over_prediction_ratio': float(over_prediction_ratio),
                'avg_error_magnitude': float(np.mean(np.abs(errors[large_errors])))
            })
        
        return failure_modes