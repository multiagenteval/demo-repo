from typing import Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class MetricVersion(Enum):
    V1 = "v1"  # Original metrics
    V2 = "v2"  # New metrics with macro averaging

@dataclass
class MetricInfo:
    name: str
    version: MetricVersion
    fn: Callable
    description: str

class MetricsRegistry:
    def __init__(self):
        self._metrics = {}
        self._register_metrics()
    
    def _register_metrics(self):
        # V1 Metrics (Original)
        self._metrics[("accuracy", MetricVersion.V1)] = MetricInfo(
            name="accuracy",
            version=MetricVersion.V1,
            fn=lambda y_true, y_pred: (y_true == y_pred).mean(),
            description="Simple accuracy (correct / total)"
        )
        
        self._metrics[("f1", MetricVersion.V1)] = MetricInfo(
            name="f1",
            version=MetricVersion.V1,
            fn=lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            description="F1 score with weighted average"
        )
        
        # V2 Metrics (New)
        self._metrics[("accuracy", MetricVersion.V2)] = MetricInfo(
            name="accuracy",
            version=MetricVersion.V2,
            fn=lambda y_true, y_pred: np.mean([
                ((y_true == i) & (y_pred == i)).sum() / (y_true == i).sum()
                for i in np.unique(y_true)
            ]),
            description="Balanced accuracy (mean of per-class accuracies)"
        )
        
        self._metrics[("f1", MetricVersion.V2)] = MetricInfo(
            name="f1",
            version=MetricVersion.V2,
            fn=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            description="F1 score with macro average"
        )
    
    def get_metric(self, name: str, version: MetricVersion) -> MetricInfo:
        return self._metrics.get((name, version))
    
    def list_metrics(self, version: MetricVersion = None) -> Dict[str, str]:
        if version:
            return {
                k[0]: v.description 
                for k, v in self._metrics.items() 
                if k[1] == version
            }
        return {
            f"{k[0]}_{k[1].value}": v.description 
            for k, v in self._metrics.items()
        } 