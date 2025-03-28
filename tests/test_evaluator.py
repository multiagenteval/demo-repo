import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from eval.evaluator import ModelEvaluator
import numpy as np

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eval_mode = False
    
    def eval(self):
        self.eval_mode = True
        return self
    
    def forward(self, x):
        return torch.randn(x.size(0), 10)  # 10 classes

def create_dummy_data():
    """Create dummy data for testing"""
    x = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    return DataLoader(TensorDataset(x, y), batch_size=32)

def test_evaluator_initialization():
    """Test that ModelEvaluator initializes with all required attributes"""
    minimal_config = {
        'metrics': {
            'version': 'v1',
            'primary': [{'name': 'accuracy', 'threshold': 0.7}],
            'secondary': []
        },
        'tracking': {
            'log_confusion_matrix': True
        }
    }
    
    evaluator = ModelEvaluator(minimal_config)
    
    # Test all required attributes are present
    required_attributes = [
        'config', 
        'metrics_version',
        'metrics_dir',
        'plots_dir',
        'required_metrics'  # This would have caught the missing attribute
    ]
    
    for attr in required_attributes:
        assert hasattr(evaluator, attr), f"Missing required attribute: {attr}"
    
    # Test required_metrics contains all necessary metric functions
    required_metric_names = ['accuracy', 'f1', 'precision', 'recall']
    for metric in required_metric_names:
        assert metric in evaluator.required_metrics, f"Missing required metric: {metric}"
        assert callable(evaluator.required_metrics[metric]), f"Metric {metric} is not callable"

def test_evaluator_missing_config():
    """Test that ModelEvaluator raises appropriate errors for missing config"""
    bad_configs = [
        ({}, "metrics missing from config"),
        ({'metrics': {}}, "version missing from metrics"),
        ({'metrics': {'version': 1}}, "version must be string"),
        ({'metrics': {'version': 'v1'}}, "tracking missing from config")
    ]
    
    for config, expected_error in bad_configs:
        with pytest.raises(KeyError, match=expected_error):
            ModelEvaluator(config)

def test_evaluate_method():
    """Test the evaluate method thoroughly"""
    config = {
        'metrics': {
            'version': 'v1',
            'primary': [
                {'name': 'accuracy', 'threshold': 0.7},
                {'name': 'f1', 'threshold': 0.7}
            ],
            'secondary': [
                {'name': 'precision', 'threshold': 0.7},
                {'name': 'recall', 'threshold': 0.7}
            ]
        },
        'tracking': {
            'log_confusion_matrix': True
        }
    }
    
    evaluator = ModelEvaluator(config)
    model = DummyModel()
    data_loader = create_dummy_data()
    device = torch.device('cpu')
    
    # Test basic evaluation
    metrics = evaluator.evaluate(model, data_loader, device)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    
    # Test model is put in eval mode
    assert model.eval_mode
    
    # Test custom name
    metrics = evaluator.evaluate(model, data_loader, device, name='custom')
    assert (evaluator.metrics_dir / 'custom_metrics.json').exists()
    if config['tracking']['log_confusion_matrix']:
        assert (evaluator.plots_dir / 'custom_confusion_matrix.png').exists()

def test_metric_calculations():
    """Test that metrics are calculated correctly"""
    config = {
        'metrics': {
            'version': 'v1',
            'primary': [{'name': 'accuracy', 'threshold': 0.7}],
            'secondary': []
        },
        'tracking': {
            'log_confusion_matrix': False
        }
    }
    
    evaluator = ModelEvaluator(config)
    
    # Test perfect predictions
    targets = np.array([0, 1, 2, 3])
    preds = np.array([0, 1, 2, 3])
    probas = np.eye(4)
    metrics = evaluator._calculate_metrics(targets, preds, probas)
    assert metrics['accuracy'] == 1.0
    
    # Test completely wrong predictions
    preds = np.array([1, 2, 3, 0])
    metrics = evaluator._calculate_metrics(targets, preds, probas)
    assert metrics['accuracy'] == 0.0

def test_error_handling():
    """Test error handling in evaluate method"""
    config = {
        'metrics': {
            'version': 'v1',
            'primary': [{'name': 'accuracy', 'threshold': 0.7}],
            'secondary': []
        },
        'tracking': {
            'log_confusion_matrix': False
        }
    }
    
    evaluator = ModelEvaluator(config)
    model = DummyModel()
    
    with pytest.raises(TypeError):
        evaluator.evaluate(model, None, None)  # Missing data_loader
    
    with pytest.raises(AttributeError):
        evaluator.evaluate(None, create_dummy_data(), torch.device('cpu'))  # Missing model

def test_evaluate_method_signature():
    """Test that evaluate method accepts correct arguments"""
    minimal_config = {
        'metrics': {
            'version': 'v1',
            'primary': [{'name': 'accuracy', 'threshold': 0.7}],
            'secondary': []
        },
        'tracking': {
            'log_confusion_matrix': True
        }
    }
    
    evaluator = ModelEvaluator(minimal_config)
    
    # Test method signature
    import inspect
    sig = inspect.signature(evaluator.evaluate)
    parameters = list(sig.parameters.keys())
    assert parameters == ['self', 'model', 'eval_loaders'], \
        f"Unexpected evaluate method signature: {parameters}"
    
    # Test actual usage pattern
    with pytest.raises(TypeError):
        # This should fail because we're passing wrong arguments
        evaluator.evaluate(None, None, None, name='test')

def test_evaluate_with_single_loader():
    """Test that evaluate works with both dict and single loader"""
    minimal_config = {
        'metrics': {
            'version': 'v1',
            'primary': [{'name': 'accuracy', 'threshold': 0.7}],
            'secondary': []
        },
        'tracking': {
            'log_confusion_matrix': True
        }
    }
    
    evaluator = ModelEvaluator(minimal_config)
    
    # Mock the _evaluate_single method to avoid actual evaluation
    evaluator._evaluate_single = lambda *args, **kwargs: {'accuracy': 0.9}
    
    # Should work with dict of loaders
    eval_loaders = {'test': 'dummy_loader'}
    results = evaluator.evaluate(None, eval_loaders)
    assert isinstance(results, dict)

def test_metrics_consistency():
    """Test that all configured metrics are calculated"""
    config = {
        'metrics': {
            'version': 'v1',
            'primary': [
                {'name': 'accuracy', 'threshold': 0.7},
                {'name': 'f1', 'threshold': 0.65}
            ],
            'secondary': [
                {'name': 'precision', 'threshold': 0.6},
                {'name': 'recall', 'threshold': 0.6}
            ]
        },
        'tracking': {
            'log_confusion_matrix': True
        }
    }
    
    evaluator = ModelEvaluator(config)
    
    # Create dummy data
    targets = np.array([0, 1, 2, 3])
    preds = np.array([0, 1, 2, 3])
    
    # Test each metric individually
    for metric_name in ['accuracy', 'f1', 'precision', 'recall']:
        assert hasattr(evaluator, f'_calculate_{metric_name}'), f"Missing calculation method for {metric_name}"
        
    # Test metric calculation
    metrics = {}
    for metric_name, metric_fn in evaluator.required_metrics.items():
        value = metric_fn(targets, preds)
        assert value is not None, f"Metric {metric_name} returned None"
        metrics[metric_name] = value
    
    # Verify all required metrics are present
    assert set(metrics.keys()) == set(evaluator.required_metrics.keys())

def test_full_evaluation_workflow():
    """Test the complete evaluation workflow"""
    config = {
        'metrics': {
            'version': 'v1',
            'primary': [
                {'name': 'accuracy', 'threshold': 0.7},
                {'name': 'f1', 'threshold': 0.65}
            ],
            'secondary': [
                {'name': 'precision', 'threshold': 0.6},
                {'name': 'recall', 'threshold': 0.6}
            ]
        },
        'tracking': {
            'log_confusion_matrix': True
        }
    }
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Create dummy model and data
    model = DummyModel()
    data_loader = create_dummy_data()
    device = torch.device('cpu')
    
    # Run evaluation
    metrics = evaluator.evaluate(model, data_loader, device, name='test')
    
    # Check all metrics are present
    expected_metrics = {'accuracy', 'f1', 'precision', 'recall'}
    assert set(metrics.keys()) == expected_metrics, \
        f"Missing metrics. Expected {expected_metrics}, got {set(metrics.keys())}"
    
    # Check all metrics have numeric values
    for metric, value in metrics.items():
        assert isinstance(value, (int, float)), \
            f"Metric {metric} has non-numeric value: {value}" 