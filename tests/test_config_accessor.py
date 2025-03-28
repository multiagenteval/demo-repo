import pytest
from utils.config_accessor import ConfigAccessor

def test_config_accessor():
    """Test ConfigAccessor with a sample config"""
    sample_config = {
        'model': {
            'params': {
                'learning_rate': 0.001,
                'hidden_dims': [64, 128]
            }
        },
        'dataset': {
            'params': {
                'batch_size': 32
            }
        }
    }
    
    accessor = ConfigAccessor(sample_config)
    
    # Test model param access
    assert accessor.get_model_param('learning_rate') == 0.001
    assert accessor.get_model_param('hidden_dims') == [64, 128]
    assert accessor.get_model_param('nonexistent', default=42) == 42
    
    # Test dataset param access
    assert accessor.get_dataset_param('batch_size') == 32
    assert accessor.get_dataset_param('nonexistent', default='default') == 'default'

def test_config_accessor_from_yaml():
    """Test ConfigAccessor creation from YAML file"""
    # Create a temporary config file
    import tempfile
    import yaml
    
    config_data = {
        'model': {
            'params': {
                'learning_rate': 0.001
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
        yaml.dump(config_data, f)
        f.flush()
        
        accessor = ConfigAccessor.from_yaml(f.name)
        assert accessor.get_model_param('learning_rate') == 0.001

def test_config_accessor_missing_sections():
    """Test ConfigAccessor with missing sections"""
    empty_config = {}
    accessor = ConfigAccessor(empty_config)
    
    assert accessor.get_model_param('anything') is None
    assert accessor.get_model_param('anything', default=42) == 42 