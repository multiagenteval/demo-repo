import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DegradedCNN(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float = 0.5  # Different default
    ):
        super().__init__()
        # ... same implementation as current degraded version ... 