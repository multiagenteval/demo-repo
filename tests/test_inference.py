import pytest
from inference import load_model, main
import torch
from pathlib import Path

def test_inference_script(tmp_path):
    """Test the complete inference workflow"""
    # Create minimal config
    config = {
        'metrics': {
            'version': 'v1',
            'primary': [{'name': 'accuracy', 'threshold': 0.7}],
            'secondary': []
        },
        'tracking': {
            'log_confusion_matrix': True
        },
        'model': {
            'params': {
                'input_shape': [1, 28, 28],
                'hidden_dims': [32, 64],
                'num_classes': 10,
                'dropout_rate': 0.1
            }
        }
    }
    
    # Mock model loading
    def mock_load_model(model_path, config):
        return DummyModel()
    
    # Patch the necessary functions
    with pytest.MonkeyPatch() as mp:
        mp.setattr("inference.load_model", mock_load_model)
        mp.setattr("inference.ConfigLoader.load_experiment_config", lambda self, path: config)
        
        # Run inference
        exit_code = main()
        
        assert exit_code == 0, "Inference script failed" 