from pathlib import Path
import yaml
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, base_config_path: str = "configs/base_config.yaml"):
        self.base_config = self._load_yaml(base_config_path)
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_experiment_config(self, exp_config_path: str) -> Dict[str, Any]:
        """Load experiment config, overriding base config values"""
        exp_config = self._load_yaml(exp_config_path)
        return self._merge_configs(self.base_config, exp_config)
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two configs, with override taking precedence"""
        merged = base.copy()
        for key, value in override.items():
            if (
                key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged 