import pytest
from utils.config_validator import ConfigValidator, ConfigValidationError
import yaml

def test_valid_config():
    validator = ConfigValidator()
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Should not raise any exceptions
    validator.validate_config(config)

def test_missing_required_field():
    validator = ConfigValidator()
    invalid_config = {
        'dataset': {
            'name': 'mnist'
            # missing 'type' field
        }
    }
    
    with pytest.raises(ConfigValidationError) as exc_info:
        validator.validate_config(invalid_config)
    assert "Missing required key" in str(exc_info.value)

def test_invalid_type():
    validator = ConfigValidator()
    invalid_config = {
        'dataset': {
            'name': 'mnist',
            'type': 'classification',
            'params': {
                'batch_size': "32"  # should be int, not string
            }
        }
    }
    
    with pytest.raises(ConfigValidationError) as exc_info:
        validator.validate_config(invalid_config)
    assert "expected type" in str(exc_info.value) 