import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class BaselineCNN(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        
        # First conv layer
        self.conv1 = nn.Conv2d(input_shape[0], hidden_dims[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dims[0])
        
        # Second conv layer
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dims[1])
        
        # Calculate size after convolutions
        conv_output_size = (input_shape[1] // 4) * (input_shape[2] // 4) * hidden_dims[1]
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.argmax(logits, dim=1) 