from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Callable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np

class DatasetVersion(Enum):
    V1 = "v1"  # Original MNIST
    V2 = "v2"  # Balanced MNIST with undersampling

@dataclass
class DatasetInfo:
    name: str
    version: DatasetVersion
    transform: Callable
    sampling_strategy: str
    description: str

class DatasetRegistry:
    def __init__(self):
        self._datasets = {}
        self._register_datasets()
    
    def _register_datasets(self):
        # V1: Original MNIST dataset
        self._datasets[("mnist", DatasetVersion.V1)] = DatasetInfo(
            name="mnist",
            version=DatasetVersion.V1,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            sampling_strategy="original",
            description="Original MNIST dataset with standard normalization"
        )
        
        # V2: Balanced MNIST with undersampling
        self._datasets[("mnist", DatasetVersion.V2)] = DatasetInfo(
            name="mnist",
            version=DatasetVersion.V2,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]),
            sampling_strategy="balanced",
            description="Balanced MNIST dataset with undersampling of majority classes"
        )
    
    def get_dataset_info(self, name: str, version: DatasetVersion) -> DatasetInfo:
        return self._datasets.get((name, version))
    
    def balance_dataset(self, dataset, targets):
        """Balance dataset by undersampling majority classes"""
        class_counts = np.bincount(targets)
        min_count = class_counts.min()
        
        balanced_indices = []
        for class_idx in range(len(class_counts)):
            class_indices = np.where(targets == class_idx)[0]
            selected_indices = np.random.choice(
                class_indices, 
                size=min_count, 
                replace=False
            )
            balanced_indices.extend(selected_indices)
        
        return Subset(dataset, balanced_indices) 