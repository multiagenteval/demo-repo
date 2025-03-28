import pytest
import yaml
from pathlib import Path

REQUIRED_CONFIG_STRUCTURE = {
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
        'version': str,
        'primary': list,
        'secondary': list
    },
    'tracking': {
        'log_confusion_matrix': bool
    },
    'eval': list
}

def test_config_structure():
    """Test that config file has all required fields with correct types"""
    config_path = Path('configs/base_config.yaml')
    assert config_path.exists(), f"Config file not found at {config_path}"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    def validate_structure(config_dict, required_structure, path=""):
        for key, value_type in required_structure.items():
            current_path = f"{path}.{key}" if path else key
            assert key in config_dict, f"Missing required key: {current_path}"
            
            if isinstance(value_type, dict):
                assert isinstance(config_dict[key], dict), \
                    f"Expected dict at {current_path}, got {type(config_dict[key])}"
                validate_structure(config_dict[key], value_type, current_path)
            else:
                assert isinstance(config_dict[key], value_type), \
                    f"Expected {value_type} at {current_path}, got {type(config_dict[key])}"
    
    validate_structure(config, REQUIRED_CONFIG_STRUCTURE)

def test_metrics_version_exists():
    """Specifically test for metrics version since that's causing issues"""
    config_path = Path('configs/base_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    assert 'metrics' in config, "metrics section missing from config"
    assert 'version' in config['metrics'], "version missing from metrics section"
    assert isinstance(config['metrics']['version'], str), "metrics version must be string" 