from typing import Any, Dict, Optional
from pathlib import Path
import yaml

class ConfigAccessor:
    def __init__(self, config: Dict):
        self.config = config
    
    def get_dataset_param(self, param_name: str, default: Any = None) -> Any:
        """Safely access dataset parameters"""
        return self.config.get('dataset', {}).get('params', {}).get(param_name, default)
    
    def get_model_param(self, param_name: str, default: Any = None) -> Any:
        """Safely access model parameters"""
        return self.config.get('model', {}).get('params', {}).get(param_name, default)
    
    def get_metric_param(self, param_name: str, default: Any = None) -> Any:
        """Safely access metric parameters"""
        return self.config.get('metrics', {}).get(param_name, default)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ConfigAccessor':
        """Create ConfigAccessor from yaml file"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config) 