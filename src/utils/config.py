import yaml
from typing import Dict, Any, Optional
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration management class for loading, validating, and accessing config parameters.
    Supports hierarchical configuration with defaults and environment-specific overrides.
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 defaults: Optional[Dict[str, Any]] = None):
        self.config = defaults or {}
        self.config_path = config_path
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        elif config_path:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
        
        self._validate_config()
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                loaded_config = yaml.safe_load(file)
            
            # Deep merge with existing config
            self._deep_merge(self.config, loaded_config)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively merge two dictionaries."""
        for key, value in update.items():
            if (key in base and isinstance(base[key], dict) 
                and isinstance(value, dict)):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_sections = ['data', 'model', 'training']
        
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing configuration section: {section}")
        
        # Validate data paths
        data_config = self.get('data', {})
        if 'raw_path' in data_config:
            raw_path = Path(data_config['raw_path'])
            if not raw_path.exists():
                logger.warning(f"Raw data path does not exist: {raw_path}")
        
        # Validate model parameters
        model_config = self.get('model', {})
        if 'window_length' in model_config:
            window_length = model_config['window_length']
            if not isinstance(window_length, int) or window_length <= 0:
                raise ValueError("window_length must be a positive integer")
        
        logger.info("Configuration validation completed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving config to {filepath}: {e}")
            raise
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        self._deep_merge(self.config, updates)
        logger.info("Configuration updated from dictionary")
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        env_prefix = "CM_RUL_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                # Try to convert to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        value = float(value)
                except (ValueError, AttributeError):
                    pass  # Keep as string
                
                self.set(config_key, value)
        
        logger.info("Configuration updated from environment variables")
    
    def create_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Create experiment-specific configuration."""
        experiment_config = {
            'experiment': {
                'name': experiment_name,
                'timestamp': self._get_timestamp(),
                'config_hash': self._generate_config_hash()
            }
        }
        
        # Include relevant sections
        for section in ['data', 'model', 'training', 'evaluation']:
            if section in self.config:
                experiment_config[section] = self.config[section].copy()
        
        return experiment_config
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for experiment tracking."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _generate_config_hash(self) -> str:
        """Generate hash of configuration for tracking changes."""
        import hashlib
        config_str = yaml.dump(self.config, default_flow_style=False)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like assignment."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()