import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for condition monitoring data.
    Creates statistical, temporal, and domain-specific features.
    """
    
    def __init__(self):
        self.feature_config = {
            'statistical': True,
            'temporal': True,
            'rolling': True,
            'domain': True
        }
    
    def set_feature_config(self, config: Dict) -> None:
        """Set which features to generate."""
        self.feature_config.update(config)
    
    def create_statistical_features(self, data: pd.DataFrame, 
                                  columns: List[str]) -> pd.DataFrame:
        """Create statistical features for specified columns."""
        features = pd.DataFrame(index=data.index)
        
        for col in columns:
            # Basic statistics
            features[f'{col}_mean'] = data[col].mean()
            features[f'{col}_std'] = data[col].std()
            features[f'{col}_var'] = data[col].var()
            features[f'{col}_skew'] = data[col].skew()
            features[f'{col}_kurtosis'] = data[col].kurtosis()
            
            # Percentiles
            for p in [25, 50, 75, 90, 95]:
                features[f'{col}_p{p}'] = np.percentile(data[col], p)
            
            # Range and IQR
            features[f'{col}_range'] = data[col].max() - data[col].min()
            features[f'{col}_iqr'] = np.percentile(data[col], 75) - np.percentile(data[col], 25)
        
        return features
    
    def create_temporal_features(self, data: pd.DataFrame, 
                               value_columns: List[str]) -> pd.DataFrame:
        """Create temporal features (differences, rates of change)."""
        features = pd.DataFrame(index=data.index)
        
        for col in value_columns:
            # Differences
            features[f'{col}_diff_1'] = data[col].diff()
            features[f'{col}_diff_2'] = data[col].diff(2)
            
            # Percentage change
            features[f'{col}_pct_change'] = data[col].pct_change()
            
            # Cumulative features
            features[f'{col}_cumsum'] = data[col].cumsum()
            features[f'{col}_cummean'] = data[col].expanding().mean()
        
        return features.fillna(0)
    
    def create_rolling_features(self, data: pd.DataFrame, 
                              value_columns: List[str],
                              windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Create rolling window features."""
        features = pd.DataFrame(index=data.index)
        
        for col in value_columns:
            for window in windows:
                # Rolling statistics
                features[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window).mean()
                features[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window).std()
                features[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window).min()
                features[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window).max()
                
                # Rolling percentiles
                features[f'{col}_rolling_median_{window}'] = data[col].rolling(window=window).median()
                features[f'{col}_rolling_q25_{window}'] = data[col].rolling(window=window).quantile(0.25)
                features[f'{col}_rolling_q75_{window}'] = data[col].rolling(window=window).quantile(0.75)
        
        return features.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    def create_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for condition monitoring."""
        features = pd.DataFrame(index=data.index)
        
        # Sensor degradation indicators
        if all(col in data.columns for col in ['SensorMeasure2', 'SensorMeasure3']):
            features['sensor_ratio_2_3'] = data['SensorMeasure2'] / data['SensorMeasure3']
            features['sensor_diff_2_3'] = data['SensorMeasure2'] - data['SensorMeasure3']
        
        # Rate of change acceleration
        for col in ['SensorMeasure2', 'SensorMeasure3', 'SensorMeasure4']:
            if col in data.columns:
                first_derivative = data[col].diff()
                features[f'{col}_acceleration'] = first_derivative.diff()
        
        # Operating condition indicators
        if 'Cycle' in data.columns and 'RUL' in data.columns:
            features['degradation_rate'] = (data['RUL'].max() - data['RUL']) / data['Cycle']
            features['remaining_cycles_ratio'] = data['RUL'] / data['Cycle']
        
        return features.fillna(0)
    
    def create_health_indicator(self, data: pd.DataFrame, 
                              sensor_columns: List[str]) -> pd.Series:
        """
        Create a composite health indicator from multiple sensors.
        Based on Mahalanobis distance from healthy state.
        """
        # Use first few cycles as healthy reference
        healthy_reference = data[sensor_columns].iloc[:10]
        healthy_mean = healthy_reference.mean()
        healthy_cov = healthy_reference.cov()
        
        try:
            # Calculate Mahalanobis distance
            health_indicator = []
            for idx, row in data[sensor_columns].iterrows():
                diff = row - healthy_mean
                try:
                    distance = np.sqrt(diff @ np.linalg.inv(healthy_cov) @ diff)
                    health_indicator.append(distance)
                except np.linalg.LinAlgError:
                    # Use Euclidean distance if covariance matrix is singular
                    distance = np.sqrt(np.sum(diff ** 2))
                    health_indicator.append(distance)
        except Exception as e:
            logger.warning(f"Error calculating health indicator: {e}")
            # Fallback: use first principal component
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            health_indicator = pca.fit_transform(data[sensor_columns])[:, 0]
        
        return pd.Series(health_indicator, index=data.index, name='health_indicator')
    
    def engineer_features(self, data: pd.DataFrame, 
                         value_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Main method to engineer all features.
        
        Args:
            data: Input DataFrame
            value_columns: Columns to use for feature engineering
            
        Returns:
            DataFrame with engineered features
        """
        if value_columns is None:
            value_columns = [col for col in data.columns if col not in ['Cycle', 'RUL', 'ID']]
        
        engineered_features = []
        
        if self.feature_config['statistical']:
            logger.info("Creating statistical features...")
            statistical_features = self.create_statistical_features(data, value_columns)
            engineered_features.append(statistical_features)
        
        if self.feature_config['temporal']:
            logger.info("Creating temporal features...")
            temporal_features = self.create_temporal_features(data, value_columns)
            engineered_features.append(temporal_features)
        
        if self.feature_config['rolling']:
            logger.info("Creating rolling features...")
            rolling_features = self.create_rolling_features(data, value_columns)
            engineered_features.append(rolling_features)
        
        if self.feature_config['domain']:
            logger.info("Creating domain features...")
            domain_features = self.create_domain_features(data)
            engineered_features.append(domain_features)
            
            # Health indicator
            health_indicator = self.create_health_indicator(data, value_columns)
            engineered_features.append(pd.DataFrame({'health_indicator': health_indicator}))
        
        # Combine all features
        if engineered_features:
            result = pd.concat([data] + engineered_features, axis=1)
            logger.info(f"Engineered {len(engineed_features)} feature sets. "
                       f"Original features: {len(data.columns)}, "
                       f"Final features: {len(result.columns)}")
            return result
        else:
            return data
    
    def select_important_features(self, data: pd.DataFrame, target: pd.Series,
                                n_features: int = 50) -> List[str]:
        """
        Select most important features using correlation analysis.
        
        Args:
            data: Feature DataFrame
            target: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        correlations = []
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                corr = np.abs(data[col].corr(target))
                correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features = [col for col, corr in correlations[:n_features]]
        
        logger.info(f"Selected top {len(selected_features)} features "
                   f"(min correlation: {correlations[n_features-1][1]:.3f})")
        
        return selected_features