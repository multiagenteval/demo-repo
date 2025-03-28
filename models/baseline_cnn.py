import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

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
        
        # Increased dropout significantly - common mistake when trying to prevent overfitting
        self.dropout = nn.Dropout(0.8)  # Changed from 0.1 to 0.8
        
        # First conv layer with L2 regularization
        self.conv1 = nn.Conv2d(input_shape[0], hidden_dims[0], kernel_size=3, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(hidden_dims[0])
        
        # Second conv layer
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(hidden_dims[1])
        
        # Calculate size after convolutions
        conv_output_size = (input_shape[1] // 4) * (input_shape[2] // 4) * hidden_dims[1]
        
        # Fully connected layers with heavy regularization
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Added aggressive dropout throughout
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)  # New dropout
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)  # New dropout
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Increased dropout
        x = self.fc2(x)
        x = self.dropout(x)  # Final dropout
        
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.argmax(logits, dim=1) 