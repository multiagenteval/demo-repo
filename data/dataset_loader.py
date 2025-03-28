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

class MNISTLoader:
    def __init__(self, data_dir: str = 'data/raw', batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])

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
        """Load MNIST dataset"""
        train_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        test_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size
        )
        
        return train_loader, test_loader 