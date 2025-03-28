import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.baseline_cnn import BaselineCNN
from utils.config import ConfigLoader
from data.dataset_loader import MNISTLoader
from eval.evaluator import ModelEvaluator
from pathlib import Path
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from eval.experiment_tracker import ExperimentTracker
import logging

# Set up logging at the start of the file, before any logger usage
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_confusion_matrix(y_true, y_pred, save_path):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')

def calculate_metrics(y_true, y_pred, probas):
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
        for i in range(10)  # 10 classes for MNIST
    }
    metrics.update(class_metrics)
    
    return metrics

def train():
    try:
        # Load config
        logger.info("Loading configuration...")
        config_loader = ConfigLoader()
        config = config_loader.load_experiment_config('configs/base_config.yaml')
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load dataset
        dataset_loader = MNISTLoader(
            data_dir='data/raw',
            batch_size=config['dataset']['params']['batch_size']
        )
        train_loader, test_loader = dataset_loader.load_data()
        
        # Initialize model
        model = BaselineCNN(
            input_shape=config['model']['params']['input_shape'],
            hidden_dims=config['model']['params']['hidden_dims'],
            num_classes=config['model']['params']['num_classes'],
            dropout_rate=config['model']['params'].get('dropout_rate', 0.1)
        ).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['model']['params']['learning_rate']
        )
        
        # Create all necessary directories
        save_dir = Path('models/checkpoints')
        plots_dir = Path('experiments/plots')
        metrics_dir = Path('experiments/metrics')
        
        # Create all directories
        save_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment tracker
        tracker = ExperimentTracker()
        
        # Training loop
        num_epochs = 5
        evaluator = ModelEvaluator(config)
        best_accuracy = 0.0
        
        print("Starting training...")
        
        # Add learning rate warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['model']['params']['learning_rate'],
            epochs=num_epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # Add training sanity checks
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Add data sanity check
                if batch_idx == 0:
                    logger.info(f"Input range: [{data.min():.3f}, {data.max():.3f}]")
                    logger.info(f"Target distribution: {torch.bincount(target)}")
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f'Epoch {epoch+1}/{num_epochs} '
                              f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                              f'({100. * batch_idx / len(train_loader):.0f}%)] '
                              f'Loss: {loss.item():.6f}')
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
            
            # Evaluate
            metrics = evaluator.evaluate(model, test_loader, device, name=f'epoch_{epoch+1}')
            
            # Handle potentially None metrics safely
            accuracy = metrics.get('accuracy')
            if accuracy is not None:
                logger.info(f"\nEpoch {epoch+1} Accuracy: {accuracy:.4f}")
                
                # Save best model if accuracy improved
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), save_dir / 'best_model.pth')
            else:
                logger.warning(f"\nEpoch {epoch+1} Accuracy: Failed to calculate")
        
        # Final summary
        if best_accuracy > 0:
            logger.info(f"\nTraining completed! Best accuracy: {best_accuracy:.4f}")
        else:
            logger.warning("\nTraining completed but no valid accuracy was recorded")
        
        # After training, save experiment results
        final_metrics = evaluator.evaluate(model, test_loader, device, name='final')
        experiment_file = tracker.save_experiment(final_metrics, config)
        
        # Compare with baseline
        comparisons = tracker.compare_with_baseline(final_metrics)
        print("\nComparison with baseline:")
        if isinstance(comparisons, dict) and 'status' in comparisons:
            if comparisons['status'] == 'no_baseline':
                logger.info("No baseline metrics found - this will be the baseline")
            elif comparisons['status'] == 'comparison_error':
                logger.error(f"Error comparing metrics: {comparisons.get('error')}")
        else:
            for metric, comp in comparisons.items():
                if comp.get('status') == 'incomplete_data':
                    logger.warning(f"{metric}: Unable to compare (missing data)")
                    continue
                    
                current = comp.get('current')
                baseline = comp.get('baseline')
                diff_percent = comp.get('diff_percent')
                
                if all(v is not None for v in [current, baseline, diff_percent]):
                    logger.info(
                        f"{metric}:\n"
                        f"  Current: {current:.4f}\n"
                        f"  Baseline: {baseline:.4f}\n"
                        f"  Change: {diff_percent:+.2f}%"
                    )
                else:
                    logger.warning(f"{metric}: Incomplete comparison data")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train() 