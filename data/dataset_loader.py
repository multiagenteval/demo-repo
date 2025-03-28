from pathlib import Path
import requests
import time
from urllib.error import URLError
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Callable
from .dataset_registry import DatasetRegistry, DatasetVersion
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset

class MNISTLoader:
    def __init__(self, data_dir: str = 'data/raw', batch_size: int = 32, 
                 num_workers: int = 4, pin_memory: bool = True):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        self.train_dataset = None
        self.test_dataset = None

    def _download_with_retry(self, dataset_fn: Callable, max_retries: int = 3):
        """Retry download with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return dataset_fn()
            except (TimeoutError, URLError) as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = 2 ** attempt
                print(f"Download failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    def _manual_download(self):
        """Manually download MNIST files from a different mirror"""
        base_url = "https://ossci-datasets.s3.amazonaws.com/mnist"
        files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]
        
        raw_folder = self.data_dir / 'MNIST/raw'
        raw_folder.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            output_file = raw_folder / file
            if not output_file.exists():
                print(f"Downloading {file}...")
                url = f"{base_url}/{file}"
                response = requests.get(url, stream=True)
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

    def load_data(self):
        """Load MNIST dataset with optimized DataLoader settings"""
        self.train_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        self.test_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        return train_loader, test_loader

    def create_balanced_loader(self, dataset):
        """Create a balanced version of the dataset"""
        targets = dataset.targets
        indices = []
        
        # Get distribution
        unique_classes = torch.unique(targets)
        min_size = min([(targets == c).sum() for c in unique_classes])
        
        # Sample equally from each class
        for c in unique_classes:
            class_indices = (targets == c).nonzero().squeeze()
            selected_indices = class_indices[torch.randperm(len(class_indices))[:min_size]]
            indices.extend(selected_indices.tolist())
        
        # Create balanced loader
        balanced_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(indices)
        )
        return balanced_loader

    def create_adversarial_loader(self, model, loader, epsilon=0.1):
        """Create adversarial examples using FGSM attack"""
        model.eval()
        device = next(model.parameters()).device
        
        adv_data = []
        adv_targets = []
        
        for data, targets in loader:
            # Clone the data to create a leaf variable
            data = data.clone().detach().to(device)
            targets = targets.to(device)
            data.requires_grad = True
            
            # Forward pass
            if device.type == 'mps':
                # Move to CPU for gradient calculation if using MPS
                data = data.cpu()
                targets = targets.cpu()
                model = model.cpu()
            
            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Generate adversarial examples
            data_grad = torch.autograd.grad(loss, data)[0]
            perturbed_data = data + epsilon * data_grad.sign()
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            
            # Move data back to original device if needed
            if device.type == 'mps':
                model = model.to(device)
                perturbed_data = perturbed_data.to(device)
                targets = targets.to(device)
            
            adv_data.append(perturbed_data.cpu().detach())
            adv_targets.append(targets.cpu())
        
        # Create TensorDataset with adversarial examples
        adv_dataset = TensorDataset(
            torch.cat(adv_data),
            torch.cat(adv_targets)
        )
        
        # Create DataLoader
        adv_loader = DataLoader(
            adv_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return adv_loader 