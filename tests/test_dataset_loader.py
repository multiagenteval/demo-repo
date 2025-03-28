import pytest
from data.dataset_loader import MNISTLoader
import torch

def test_mnist_loader_initialization():
    """Test that MNISTLoader initializes with default and custom parameters"""
    # Test default initialization
    loader = MNISTLoader()
    assert str(loader.data_dir) == 'data/raw'
    assert loader.batch_size == 32

    # Test custom initialization
    loader = MNISTLoader(data_dir='custom/path', batch_size=64)
    assert str(loader.data_dir) == 'custom/path'
    assert loader.batch_size == 64

def test_mnist_loader_data():
    """Test that MNISTLoader returns correct data format"""
    loader = MNISTLoader(batch_size=32)
    train_loader, test_loader = loader.load_data()

    # Check a batch from train loader
    batch_data, batch_labels = next(iter(train_loader))
    assert isinstance(batch_data, torch.Tensor)
    assert isinstance(batch_labels, torch.Tensor)
    assert batch_data.shape[0] == 32  # batch size
    assert batch_data.shape[1] == 1   # channels
    assert batch_data.shape[2] == 28  # height
    assert batch_data.shape[3] == 28  # width
    assert batch_labels.shape[0] == 32 