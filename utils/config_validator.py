from typing import Dict, Any
import yaml
from pathlib import Path

class ConfigValidationError(Exception):
    pass

class ConfigValidator:
    def __init__(self):
        self.required_structure = {
            'dataset': {
                'name': str,
                'type': str,
                'params': {
                    'batch_size': int,
                    'train_path': str,
                    'test_path': str
                }
            },
            'model': {
                'name': str,
                'type': str,
                'params': {
                    'input_shape': list,
                    'hidden_dims': list,
                    'num_classes': int,
                    'learning_rate': float,
                    'dropout_rate': float
                }
            },
            'metrics': {
                'primary': list,
                'secondary': list
            },
            'tracking': {
                'log_confusion_matrix': bool
            },
            'eval': list
        }

    def validate_type(self, value: Any, expected_type: Any, path: str) -> None:
        """Validate type of a config value"""
        if not isinstance(value, expected_type):
            raise ConfigValidationError(
                f"Config validation error at {path}: "
                f"expected type {expected_type}, got {type(value)}"
            )

    def validate_structure(self, config: Dict, structure: Dict, path: str = "") -> None:
        """Recursively validate config structure"""
        for key, expected in structure.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in config:
                raise ConfigValidationError(
                    f"Missing required key: {current_path}"
                )
            
            if isinstance(expected, dict):
                if not isinstance(config[key], dict):
                    raise ConfigValidationError(
                        f"Expected dict at {current_path}, got {type(config[key])}"
                    )
                self.validate_structure(config[key], expected, current_path)
            else:
                self.validate_type(config[key], expected, current_path)

    def validate_config(self, config: Dict) -> None:
        """Validate entire config"""
        try:
            self.validate_structure(config, self.required_structure)
            
            # Additional validation rules
            if config['dataset']['type'] not in ['classification', 'regression']:
                raise ConfigValidationError(
                    "Dataset type must be 'classification' or 'regression'"
                )
            
            # Validate eval section structure
            for eval_config in config['eval']:
                if not all(k in eval_config for k in ['name', 'version', 'description']):
                    raise ConfigValidationError(
                        "Each eval config must have 'name', 'version', and 'description'"
                    )
            
            # Validate metrics structure
            for metric in config['metrics']['primary'] + config['metrics']['secondary']:
                if not all(k in metric for k in ['name', 'threshold']):
                    raise ConfigValidationError(
                        "Each metric must have 'name' and 'threshold'"
                    )
            
        except Exception as e:
            raise ConfigValidationError(f"Config validation failed: {str(e)}")

def load_and_validate_config(config_path: str) -> Dict:
    """Load and validate config file"""
    validator = ConfigValidator()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate config
    validator.validate_config(config)
    
    return config 