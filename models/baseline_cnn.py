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
        
        # Increased dropout rate significantly
        self.dropout = nn.Dropout(0.5)  # Changed from 0.1 to 0.5
        
        # Reduced model capacity
        self.conv1 = nn.Conv2d(input_shape[0], hidden_dims[0] // 2, kernel_size=5, padding=2)  # Reduced channels, increased kernel
        self.bn1 = nn.BatchNorm2d(hidden_dims[0] // 2)
        
        # Added more aggressive dropout
        self.conv2 = nn.Conv2d(hidden_dims[0] // 2, hidden_dims[1] // 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(hidden_dims[1] // 2)
        
        # Calculate size after convolutions
        conv_output_size = (input_shape[1] // 4) * (input_shape[2] // 4) * (hidden_dims[1] // 2)
        
        # Smaller fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)  # Reduced from 128 to 64
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Added dropout after every layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, 2)
        
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