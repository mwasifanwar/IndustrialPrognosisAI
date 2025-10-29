import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import yaml
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CMAPPSDataLoader:
    """
    Advanced data loader for Condition Monitoring and Prognostic Health Management datasets.
    Supports multiple datasets and engines with comprehensive error handling.
    """
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        self.config = self._load_config(config_path)
        self.data_path = Path(self.config['data']['raw_path'])
        self.processed_path = Path(self.config['data']['processed_path'])
        self._validate_paths()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def _validate_paths(self) -> None:
        """Validate that required data paths exist."""
        if not self.data_path.exists():
            logger.warning(f"Raw data path {self.data_path} does not exist. Creating...")
            self.data_path.mkdir(parents=True, exist_ok=True)
        
        if not self.processed_path.exists():
            logger.warning(f"Processed data path {self.processed_path} does not exist. Creating...")
            self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def load_engine_data(self, dataset_id: int, engine_id: int, 
                        data_type: str = 'train') -> pd.DataFrame:
        """
        Load specific engine data from processed datasets.
        
        Args:
            dataset_id: ID of the dataset (1-4)
            engine_id: ID of the specific engine
            data_type: Type of data ('train' or 'test')
            
        Returns:
            DataFrame containing the engine data
        """
        file_pattern = f"Processed_{data_type.capitalize()}_{dataset_id:03d}.csv"
        file_path = self.processed_path / file_pattern
        
        if not file_path.exists():
            available_files = list(self.processed_path.glob("*.csv"))
            raise FileNotFoundError(
                f"Dataset not found: {file_path}. "
                f"Available files: {[f.name for f in available_files]}"
            )
            
        try:
            df = pd.read_csv(file_path)
            engine_data = df[df['ID'] == engine_id].copy()
            
            if engine_data.empty:
                available_engines = df['ID'].unique()
                raise ValueError(
                    f"Engine {engine_id} not found in dataset {dataset_id}. "
                    f"Available engines: {sorted(available_engines)}"
                )
                
            logger.info(f"Loaded data for engine {engine_id} from {file_path.name}")
            return engine_data.drop(columns=['ID'])
            
        except Exception as e:
            logger.error(f"Error loading engine data: {e}")
            raise
    
    def get_all_engines(self, dataset_id: int, data_type: str = 'train') -> List[int]:
        """Get list of all engine IDs in dataset."""
        file_pattern = f"Processed_{data_type.capitalize()}_{dataset_id:03d}.csv"
        file_path = self.processed_path / file_pattern
        
        if not file_path.exists():
            return []
            
        df = pd.read_csv(file_path)
        return sorted(df['ID'].unique())
    
    def load_multiple_engines(self, dataset_id: int, engine_ids: List[int], 
                             data_type: str = 'train') -> Dict[int, pd.DataFrame]:
        """Load data for multiple engines."""
        engine_data = {}
        for engine_id in tqdm(engine_ids, desc="Loading engine data"):
            try:
                data = self.load_engine_data(dataset_id, engine_id, data_type)
                engine_data[engine_id] = data
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Skipping engine {engine_id}: {e}")
                continue
                
        return engine_data
    
    def get_dataset_stats(self, dataset_id: int, data_type: str = 'train') -> Dict:
        """Get statistics for a dataset."""
        engines = self.get_all_engines(dataset_id, data_type)
        stats = {
            'dataset_id': dataset_id,
            'data_type': data_type,
            'total_engines': len(engines),
            'engines': engines
        }
        
        # Calculate additional statistics if engines are available
        if engines:
            cycle_lengths = []
            for engine_id in engines[:10]:  # Sample first 10 for performance
                try:
                    data = self.load_engine_data(dataset_id, engine_id, data_type)
                    cycle_lengths.append(len(data))
                except Exception:
                    continue
            
            if cycle_lengths:
                stats.update({
                    'avg_cycles_per_engine': np.mean(cycle_lengths),
                    'min_cycles': np.min(cycle_lengths),
                    'max_cycles': np.max(cycle_lengths),
                    'std_cycles': np.std(cycle_lengths)
                })
        
        return stats
    
    def validate_data_quality(self, dataset_id: int, data_type: str = 'train') -> Dict:
        """Validate data quality for a dataset."""
        engines = self.get_all_engines(dataset_id, data_type)
        quality_report = {
            'dataset_id': dataset_id,
            'data_type': data_type,
            'total_engines': len(engines),
            'issues': []
        }
        
        for engine_id in tqdm(engines, desc="Validating data quality"):
            try:
                data = self.load_engine_data(dataset_id, engine_id, data_type)
                
                # Check for missing values
                missing_values = data.isnull().sum().sum()
                if missing_values > 0:
                    quality_report['issues'].append({
                        'engine_id': engine_id,
                        'issue': 'missing_values',
                        'count': missing_values
                    })
                
                # Check for constant columns
                constant_cols = data.columns[data.nunique() <= 1]
                if len(constant_cols) > 0:
                    quality_report['issues'].append({
                        'engine_id': engine_id,
                        'issue': 'constant_columns',
                        'columns': constant_cols.tolist()
                    })
                
                # Check for negative RUL
                if 'RUL' in data.columns and (data['RUL'] < 0).any():
                    quality_report['issues'].append({
                        'engine_id': engine_id,
                        'issue': 'negative_rul',
                        'count': (data['RUL'] < 0).sum()
                    })
                    
            except Exception as e:
                quality_report['issues'].append({
                    'engine_id': engine_id,
                    'issue': 'loading_error',
                    'error': str(e)
                })
        
        return quality_report