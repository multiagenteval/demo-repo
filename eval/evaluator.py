import torch
from pathlib import Path
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

class ModelEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_dir = Path('experiments/metrics')
        self.plots_dir = Path('experiments/plots')
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, probas: np.ndarray) -> Dict[str, float]:
        metrics = {
            'accuracy': (y_true == y_pred).mean().item(),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Calculate per-class metrics
        class_metrics = {
            f'class_{i}_accuracy': (
                ((y_true == i) & (y_pred == i)).sum() / (y_true == i).sum()
            ).item()
            for i in range(10)
        }
        metrics.update(class_metrics)
        
        return metrics

    def save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, name: str):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.plots_dir / f'{name}_confusion_matrix.png')
        plt.close()

    def evaluate(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                device: torch.device, name: str = 'eval') -> Dict[str, float]:
        model.eval()
        all_preds = []
        all_targets = []
        all_probas = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                probas = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probas = np.array(all_probas)
        
        metrics = self.calculate_metrics(all_targets, all_preds, all_probas)
        
        # Save results
        with open(self.metrics_dir / f'{name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        if self.config['tracking'].get('log_confusion_matrix', True):
            self.save_confusion_matrix(all_targets, all_preds, name)
        
        return metrics 