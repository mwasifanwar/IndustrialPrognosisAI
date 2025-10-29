import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from sklearn.model_selection import KFold, TimeSeriesSplit
from tqdm import tqdm

from ..models.model_factory import ModelFactory
from ..evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

class CrossValidator:
    """
    Advanced cross-validation for time series data.
    Handles engine-wise splitting and temporal validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_calculator = EvaluationMetrics()
    
    def engine_wise_cv(self, engines_data: Dict, n_splits: int = 5) -> Dict[str, List]:
        """
        Perform engine-wise cross-validation.
        
        Args:
            engines_data: Dictionary of engine data
            n_splits: Number of folds
            
        Returns:
            Dictionary of cross-validation results
        """
        engine_ids = list(engines_data.keys())
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = {
            'fold_scores': [],
            'train_scores': [],
            'val_scores': [],
            'models': [],
            'predictions': []
        }
        
        fold = 1
        for train_idx, val_idx in kf.split(engine_ids):
            logger.info(f"Processing fold {fold}/{n_splits}")
            
            # Split engines
            train_engines = {engine_ids[i]: engines_data[engine_ids[i]] for i in train_idx}
            val_engines = {engine_ids[i]: engines_data[engine_ids[i]] for i in val_idx}
            
            # Train and evaluate model
            fold_results = self._train_evaluate_fold(train_engines, val_engines, fold)
            cv_results['fold_scores'].append(fold_results['val_metrics'])
            cv_results['train_scores'].append(fold_results['train_metrics'])
            cv_results['models'].append(fold_results['model'])
            cv_results['predictions'].append(fold_results['predictions'])
            
            fold += 1
        
        # Calculate overall statistics
        cv_results['mean_scores'] = {
            metric: np.mean([fold[metric] for fold in cv_results['fold_scores']])
            for metric in cv_results['fold_scores'][0].keys()
        }
        cv_results['std_scores'] = {
            metric: np.std([fold[metric] for fold in cv_results['fold_scores']])
            for metric in cv_results['fold_scores'][0].keys()
        }
        
        logger.info(f"Cross-validation completed. Mean RMSE: {cv_results['mean_scores']['rmse']:.4f}")
        return cv_results
    
    def temporal_cv(self, engines_data: Dict, n_splits: int = 5) -> Dict[str, List]:
        """
        Perform temporal cross-validation respecting time order.
        
        Args:
            engines_data: Dictionary of engine data
            n_splits: Number of folds
            
        Returns:
            Dictionary of cross-validation results
        """
        # Combine all engine data and sort by cycle
        all_data = []
        for engine_id, data in engines_data.items():
            data_copy = data.copy()
            data_copy['engine_id'] = engine_id
            all_data.append(data_copy)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values('Cycle').reset_index(drop=True)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            'fold_scores': [],
            'train_sizes': [],
            'test_sizes': [],
            'predictions': []
        }
        
        fold = 1
        for train_idx, test_idx in tscv.split(combined_data):
            logger.info(f"Processing temporal fold {fold}/{n_splits}")
            
            train_data = combined_data.iloc[train_idx]
            test_data = combined_data.iloc[test_idx]
            
            # Separate by engines again
            train_engines = {}
            for engine_id in train_data['engine_id'].unique():
                engine_data = train_data[train_data['engine_id'] == engine_id].drop(columns=['engine_id'])
                train_engines[engine_id] = engine_data
            
            test_engines = {}
            for engine_id in test_data['engine_id'].unique():
                engine_data = test_data[test_data['engine_id'] == engine_id].drop(columns=['engine_id'])
                test_engines[engine_id] = test_data
            
            # Train and evaluate
            fold_results = self._train_evaluate_fold(train_engines, test_engines, fold)
            cv_results['fold_scores'].append(fold_results['val_metrics'])
            cv_results['train_sizes'].append(len(train_data))
            cv_results['test_sizes'].append(len(test_data))
            cv_results['predictions'].append(fold_results['predictions'])
            
            fold += 1
        
        return cv_results
    
    def _train_evaluate_fold(self, train_engines: Dict, val_engines: Dict, 
                           fold: int) -> Dict[str, Any]:
        """Train and evaluate a single fold."""
        from ..data.preprocessor import DataPreprocessor
        
        # Prepare data
        preprocessor = DataPreprocessor(self.config)
        window_length = self.config['model']['window_length']
        
        X_train, y_train = preprocessor.prepare_multiple_engines_data(
            train_engines, window_length
        )
        X_val, y_val = preprocessor.prepare_multiple_engines_data(
            val_engines, window_length
        )
        
        # Reshape data
        if self.config['model']['architecture'].get('conv_type', '1d') == '2d':
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        else:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_val.shape[2])
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
        
        # Create and train model
        model = ModelFactory.create_model(self.config['model']['name'], self.config)
        model.compile_model()
        
        # Train with reduced epochs for CV
        cv_config = self.config.copy()
        cv_config['training']['epochs'] = min(50, self.config['training']['epochs'])
        
        history = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        y_pred = model.predict(X_val)
        val_metrics = self.metrics_calculator.calculate_all_metrics(
            y_val, y_pred.flatten()
        )
        
        # Also evaluate on training for reference
        y_train_pred = model.predict(X_train)
        train_metrics = self.metrics_calculator.calculate_all_metrics(
            y_train, y_train_pred.flatten()
        )
        
        return {
            'model': model,
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'predictions': {
                'true': y_val,
                'predicted': y_pred.flatten(),
                'fold': fold
            },
            'preprocessor': preprocessor
        }
    
    def stratified_engine_cv(self, engines_data: Dict, n_splits: int = 5) -> Dict[str, List]:
        """
        Perform stratified cross-validation based on engine characteristics.
        
        Args:
            engines_data: Dictionary of engine data
            n_splits: Number of folds
            
        Returns:
            Dictionary of cross-validation results
        """
        # Calculate engine characteristics for stratification
        engine_features = []
        for engine_id, data in engines_data.items():
            features = {
                'engine_id': engine_id,
                'max_cycle': len(data),
                'avg_sensor2': data['SensorMeasure2'].mean(),
                'degradation_rate': (data['RUL'].max() - data['RUL'].min()) / len(data)
            }
            engine_features.append(features)
        
        engine_df = pd.DataFrame(engine_features)
        
        # Create strata based on cycle length quartiles
        engine_df['stratum'] = pd.qcut(engine_df['max_cycle'], n_splits, labels=False)
        
        cv_results = {
            'fold_scores': [],
            'stratification_info': [],
            'predictions': []
        }
        
        for stratum in range(n_splits):
            logger.info(f"Processing stratum {stratum + 1}/{n_splits}")
            
            # Split based on stratum
            test_engines = engine_df[engine_df['stratum'] == stratum]['engine_id'].tolist()
            train_engines = engine_df[engine_df['stratum'] != stratum]['engine_id'].tolist()
            
            train_data = {eid: engines_data[eid] for eid in train_engines}
            test_data = {eid: engines_data[eid] for eid in test_engines}
            
            # Train and evaluate
            fold_results = self._train_evaluate_fold(train_data, test_data, stratum + 1)
            cv_results['fold_scores'].append(fold_results['val_metrics'])
            cv_results['stratification_info'].append({
                'stratum': stratum,
                'test_engines': test_engines,
                'test_size': len(test_engines)
            })
            cv_results['predictions'].append(fold_results['predictions'])
        
        return cv_results
    
    def get_cv_summary(self, cv_results: Dict) -> pd.DataFrame:
        """Create a summary DataFrame from cross-validation results."""
        rows = []
        for i, fold_metrics in enumerate(cv_results['fold_scores']):
            row = {'fold': i + 1}
            row.update(fold_metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Add summary row
        summary_row = {'fold': 'mean'}
        for metric in df.columns:
            if metric != 'fold':
                summary_row[metric] = df[metric].mean()
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        
        return df