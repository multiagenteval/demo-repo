import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class BaselineCNN(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # First conv block with residual connection
        self.conv1 = nn.Conv2d(input_shape[0], hidden_dims[0], kernel_size=3, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(hidden_dims[0])
        
        # Second conv block with residual connection
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(hidden_dims[1])
        
        # Residual projection when channels don't match
        self.res_proj = nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=1) if hidden_dims[0] != hidden_dims[1] else None
        
        # Calculate size after convolutions
        conv_output_size = (input_shape[1] // 4) * (input_shape[2] // 4) * hidden_dims[1]
        
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def fgsm_attack(self, data: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM
        """
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = data + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
        
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None, epsilon: float = 0.0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional adversarial training
        Args:
            x: Input tensor
            target: Optional target tensor for adversarial training
            epsilon: FGSM epsilon parameter (0.0 means no adversarial training)
        """
        # Generate adversarial examples during training if epsilon > 0
        if self.training and epsilon > 0 and target is not None:
            x.requires_grad = True
            # Forward pass for gradient calculation
            outputs = self._forward_impl(x)
            # Calculate loss for gradient
            loss = F.cross_entropy(outputs, target)
            # Zero all existing gradients
            self.zero_grad()
            # Calculate gradients of model in backward pass
            loss.backward()
            # Collect gradients
            data_grad = x.grad.data
            # Generate adversarial example
            x = self.fgsm_attack(x, epsilon, data_grad)
            # Detach for the actual forward pass
            x = x.detach()
        
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the forward pass
        """
        # First conv block
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # Second conv block with residual
        res = x
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Apply residual connection if shapes match
        if self.res_proj is not None:
            res = self.res_proj(res)
        x = x + res
        
        x = F.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self._forward_impl(x)
            return torch.argmax(logits, dim=1) 