import pytest
import torch
from train import setup_training
from utils.config_accessor import ConfigAccessor

def test_training_setup():
    """Test training setup with minimal config"""
    # Create minimal test config
    test_config = {
        'model': {
            'params': {
                'input_shape': [1, 28, 28],
                'hidden_dims': [32, 64],
                'num_classes': 10,
                'learning_rate': 0.001
            }
        },
        'dataset': {
            'params': {
                'batch_size': 32
            }
        }
    }
    
    # Write test config to temporary file
    import tempfile
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        
        # Test setup
        setup = setup_training(f.name)
        
        assert isinstance(setup['config'], ConfigAccessor)
        assert isinstance(setup['device'], torch.device)
        assert setup['save_dir'].exists()

def test_training_setup_missing_params():
    """Test training setup with missing parameters"""
    # Create minimal test config with missing params
    test_config = {
        'model': {},
        'dataset': {}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
        yaml.dump(test_config, f)
        f.flush()
        
        # Should still work with defaults
        setup = setup_training(f.name)
        assert setup['config'].get_model_param('learning_rate', 0.001) == 0.001 