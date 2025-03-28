from typing import Callable, Dict, Any
from dataclasses import dataclass

@dataclass
class MetricInfo:
    name: str
    fn: Callable
    requires: list[str]  # required inputs like 'probas', 'labels', etc
    params: Dict[str, Any] = None

class MetricRegistry:
    def __init__(self):
        self._metrics = {}
    
    def register(self, name: str, metric_fn: Callable, requires: list[str], params: Dict[str, Any] = None):
        """Register a new metric"""
        self._metrics[name] = MetricInfo(name, metric_fn, requires, params)
    
    def get_metric(self, name: str) -> MetricInfo:
        """Get a metric by name"""
        return self._metrics.get(name)
    
    def list_metrics(self) -> list[str]:
        """List all registered metrics"""
        return list(self._metrics.keys())

# Global registry instance
METRIC_REGISTRY = MetricRegistry()

# Register common metrics
def register_common_metrics():
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    METRIC_REGISTRY.register(
        "accuracy",
        accuracy_score,
        requires=["y_true", "y_pred"]
    )
    
    METRIC_REGISTRY.register(
        "precision",
        precision_score,
        requires=["y_true", "y_pred"],
        params={"average": "weighted"}
    )
    # ... register other metrics 