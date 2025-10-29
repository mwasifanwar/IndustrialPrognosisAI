import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional, Union
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Advanced data preprocessing for condition monitoring data.
    Handles scaling, sequence generation, and data splitting.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = None
        self.feature_names = []
        
    def create_sequences(self, data: np.ndarray, target: np.ndarray, 
                        window_length: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series data.
        
        Args:
            data: Input features (n_samples, n_features)
            target: Target values (n_samples,)
            window_length: Length of the sliding window
            stride: Step size for the sliding window
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []
        
        for i in range(0, len(data) - window_length, stride):
            sequences.append(data[i:i + window_length])
            targets.append(target[i + window_length])
            
        return np.array(sequences), np.array(targets)
    
    def fit_scaler(self, data: Union[pd.DataFrame, np.ndarray], 
                  scaler_type: str = 'minmax') -> None:
        """
        Fit scaler on training data.
        
        Args:
            data: Training data to fit scaler on
            scaler_type: Type of scaler ('minmax', 'standard', 'robust')
        """
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
        if isinstance(data, pd.DataFrame):
            data_values = data.values
            self.feature_names = data.columns.tolist()
        else:
            data_values = data
            self.feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        
        self.scaler.fit(data_values)
        logger.info(f"Fitted {scaler_type} scaler on data with shape {data_values.shape}")
    
    def transform_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        if isinstance(data, pd.DataFrame):
            data_values = data.values
        else:
            data_values = data
            
        transformed = self.scaler.transform(data_values)
        logger.debug(f"Transformed data with shape {data_values.shape}")
        return transformed
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data using fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted.")
        return self.scaler.inverse_transform(data)
    
    def prepare_single_engine_data(self, engine_data: pd.DataFrame, 
                                 window_length: int, target_col: str = 'RUL') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for a single engine with sequence generation.
        
        Args:
            engine_data: DataFrame for a single engine
            window_length: Length of sliding window
            target_col: Name of the target column
            
        Returns:
            Tuple of (sequences, targets)
        """
        # Separate features and target
        features = engine_data.drop(columns=[target_col])
        target = engine_data[target_col]
        
        # Scale features
        if self.scaler is None:
            self.fit_scaler(features, self.config['preprocessing']['scaler'])
        
        features_scaled = self.transform_data(features)
        target_scaled = target.values  # Keep target as is for now
        
        # Create sequences
        sequences, targets = self.create_sequences(
            features_scaled, target_scaled, window_length
        )
        
        logger.info(f"Created {len(sequences)} sequences for engine")
        return sequences, targets
    
    def prepare_multiple_engines_data(self, engines_data: Dict[int, pd.DataFrame],
                                    window_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for multiple engines.
        
        Args:
            engines_data: Dictionary of engine DataFrames
            window_length: Length of sliding window
            
        Returns:
            Tuple of (all_sequences, all_targets)
        """
        all_sequences = []
        all_targets = []
        
        # First pass: fit scaler on all data
        if self.scaler is None:
            all_features = pd.concat([
                data.drop(columns=['RUL']) for data in engines_data.values()
            ])
            self.fit_scaler(all_features, self.config['preprocessing']['scaler'])
        
        # Second pass: create sequences
        for engine_id, engine_data in engines_data.items():
            try:
                sequences, targets = self.prepare_single_engine_data(
                    engine_data, window_length
                )
                all_sequences.append(sequences)
                all_targets.append(targets)
            except Exception as e:
                logger.warning(f"Error processing engine {engine_id}: {e}")
                continue
        
        # Concatenate all sequences
        X = np.concatenate(all_sequences, axis=0)
        y = np.concatenate(all_targets, axis=0)
        
        logger.info(f"Prepared data with {X.shape[0]} sequences")
        return X, y
    
    def train_test_split_engines(self, engines_data: Dict[int, pd.DataFrame],
                               test_size: float = 0.2, random_state: int = 42) -> Tuple[Dict, Dict]:
        """
        Split engines into train and test sets.
        
        Args:
            engines_data: Dictionary of engine DataFrames
            test_size: Proportion of engines for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_engines, test_engines)
        """
        engine_ids = list(engines_data.keys())
        train_ids, test_ids = train_test_split(
            engine_ids, test_size=test_size, random_state=random_state
        )
        
        train_engines = {eid: engines_data[eid] for eid in train_ids}
        test_engines = {eid: engines_data[eid] for eid in test_ids}
        
        logger.info(f"Split {len(engine_ids)} engines into {len(train_ids)} train and {len(test_ids)} test")
        return train_engines, test_engines
    
    def save_scaler(self, filepath: str) -> None:
        """Save fitted scaler to file."""
        if self.scaler is None:
            raise ValueError("No scaler to save.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, filepath)
        logger.info(f"Saved scaler to {filepath}")
    
    def load_scaler(self, filepath: str) -> None:
        """Load scaler from file."""
        self.scaler = joblib.load(filepath)
        logger.info(f"Loaded scaler from {filepath}")