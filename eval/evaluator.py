import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
import logging
from .metrics_tracker import MetricsTracker  # Add this import

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class EvaluationError(Exception):
    """Custom exception for evaluation errors"""
    pass

class ModelEvaluator:
    def __init__(self, config):
        logger.info("Initializing evaluator...")
        try:
            # Initialize required metrics first
            logger.debug("Setting up required metrics")
            self.required_metrics = {
                'accuracy': self._calculate_accuracy,
                'f1': self._calculate_f1,
                'precision': self._calculate_precision,
                'recall': self._calculate_recall
            }
            logger.debug(f"Initialized metrics: {list(self.required_metrics.keys())}")
            
            # Validate config
            logger.debug("Validating configuration")
            self._validate_config(config)
            
            # Set instance attributes
            self.config = config
            self.metrics_version = config['metrics']['version']
            self.metrics_dir = Path('experiments/metrics')
            self.plots_dir = Path('experiments/plots')
            
            logger.debug("Setting up directories")
            self._setup_directories()
            
            self.metrics_tracker = MetricsTracker()
            
            logger.info("ModelEvaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ModelEvaluator: {e}")
            raise

    def _setup_directories(self):
        """Create necessary directories"""
        try:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directories: {self.metrics_dir}, {self.plots_dir}")
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise EvaluationError(f"Failed to create directories: {e}")

    def _validate_config(self, config):
        """Validate that config has all required sections and metrics"""
        logger.debug("Starting config validation")
        
        required_sections = ['metrics', 'tracking']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                raise ValueError(f"Missing required section: {section}")
        
        # Validate metrics configuration
        if 'primary' not in config['metrics']:
            logger.error("Missing 'primary' in metrics config")
            raise ValueError("Missing 'primary' in metrics config")
        
        # Validate that all configured metrics are supported
        configured_metrics = [m['name'] for m in config['metrics']['primary']]
        configured_metrics.extend([m['name'] for m in config['metrics'].get('secondary', [])])
        logger.debug(f"Configured metrics: {configured_metrics}")
        
        unsupported = set(configured_metrics) - set(self.required_metrics.keys())
        if unsupported:
            logger.error(f"Unsupported metrics configured: {unsupported}")
            raise ValueError(f"Unsupported metrics configured: {unsupported}")
        
        logger.debug("Config validation successful")

    def _calculate_accuracy(self, targets, preds):
        return (targets == preds).mean()

    def _calculate_f1(self, targets, preds):
        """Calculate F1 with proper zero division handling"""
        try:
            return f1_score(
                targets, 
                preds, 
                average='weighted',
                zero_division=0  # Explicitly handle zero division
            )
        except Exception as e:
            logger.warning(f"Error calculating F1: {e}")
            return 0.0

    def _calculate_precision(self, targets, preds):
        """Calculate precision with proper zero division handling"""
        try:
            return precision_score(
                targets, 
                preds, 
                average='weighted',
                zero_division=0  # Explicitly handle zero division
            )
        except Exception as e:
            logger.warning(f"Error calculating precision: {e}")
            return 0.0

    def _calculate_recall(self, targets, preds):
        """Calculate recall with proper zero division handling"""
        try:
            return recall_score(
                targets, 
                preds, 
                average='weighted',
                zero_division=0  # Explicitly handle zero division
            )
        except Exception as e:
            logger.warning(f"Error calculating recall: {e}")
            return 0.0

    def evaluate(self, model, data_loader, device, name='eval'):
        """Evaluate model and ensure all metrics are calculated"""
        logger.info(f"Starting evaluation: {name}")
        model.eval()
        all_preds = []
        all_targets = []
        
        try:
            logger.debug("Running model inference")
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(data_loader):
                    # Add input validation logging
                    if batch_idx == 0:
                        logger.debug(f"First batch - Input shape: {data.shape}, "
                                   f"Target shape: {target.shape}, "
                                   f"Input range: [{data.min():.3f}, {data.max():.3f}]")
                    
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    preds = torch.argmax(output, dim=1)
                    
                    # Add prediction validation
                    if batch_idx == 0:
                        logger.debug(f"First batch - Output shape: {output.shape}, "
                                   f"Predictions shape: {preds.shape}, "
                                   f"Unique predictions: {torch.unique(preds).cpu().numpy()}")
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

            # Validate collected data
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            logger.debug(f"Final shapes - Predictions: {all_preds.shape}, "
                        f"Targets: {all_targets.shape}")
            logger.debug(f"Unique predictions: {np.unique(all_preds)}")
            logger.debug(f"Unique targets: {np.unique(all_targets)}")

            # Calculate metrics with validation
            metrics = {}
            for metric_name, metric_fn in self.required_metrics.items():
                try:
                    value = metric_fn(all_targets, all_preds)
                    if value is None:
                        logger.warning(f"{metric_name} calculation returned None")
                    metrics[metric_name] = value
                    logger.debug(f"Calculated {metric_name}: {value}")
                except Exception as e:
                    logger.error(f"Failed to calculate {metric_name}: {e}")
                    metrics[metric_name] = None

            return metrics

        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            return {name: None for name in self.required_metrics.keys()}

    def _save_results(self, metrics: Dict, name: str):
        """Save evaluation results"""
        try:
            results_file = self.metrics_dir / f'{name}_metrics.json'
            logger.debug(f"Saving results to {results_file}")
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.debug("Results saved successfully")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise EvaluationError(f"Failed to save results: {e}")

    def _save_confusion_matrix(self, targets, preds, name: str):
        """Save confusion matrix plot"""
        try:
            logger.debug(f"Generating confusion matrix for {name}")
            cm = confusion_matrix(targets, preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            save_path = self.plots_dir / f'{name}_confusion_matrix.png'
            logger.debug(f"Saving confusion matrix to {save_path}")
            plt.savefig(save_path)
            logger.debug("Confusion matrix saved successfully")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix: {e}")
            raise EvaluationError(f"Failed to save confusion matrix: {e}")
        finally:
            plt.close() 