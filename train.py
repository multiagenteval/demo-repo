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
        
        # Setup device - optimize for M3
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple M3 GPU (MPS)")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load dataset with optimized settings
        dataset_loader = MNISTLoader(
            data_dir='data/raw',
            batch_size=config['dataset']['params']['batch_size'],
            num_workers=4,  # Optimize data loading
            pin_memory=device.type in ['cuda', 'mps']  # Enable for GPU
        )
        train_loader, test_loader = dataset_loader.load_data()
        
        # Initialize model
        model = BaselineCNN(
            input_shape=config['model']['params']['input_shape'],
            hidden_dims=config['model']['params']['hidden_dims'],
            num_classes=config['model']['params']['num_classes'],
            dropout_rate=config['model']['params'].get('dropout_rate', 0.1)
        ).to(device)
        
        # Use mixed precision for faster training (only for CUDA)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')  # Only enable for CUDA, not MPS
        
        # Training setup with optimized parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(  # Switch to AdamW
            model.parameters(),
            lr=config['model']['params']['learning_rate'],
            weight_decay=0.01
        )
        
        # Create directories
        save_dir = Path('models/checkpoints')
        plots_dir = Path('experiments/plots')
        metrics_dir = Path('experiments/metrics')
        for d in [save_dir, plots_dir, metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracker
        tracker = ExperimentTracker()
        
        # Training loop with optimizations
        num_epochs = 5
        evaluator = ModelEvaluator(config)
        best_accuracy = 0.0
        
        logger.info("Starting training...")
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['model']['params']['learning_rate'],
            epochs=num_epochs,
            steps_per_epoch=len(train_loader)
        )
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Data sanity check (first batch only)
                if batch_idx == 0 and epoch == 0:
                    logger.info(f"Input range: [{data.min():.3f}, {data.max():.3f}]")
                    logger.info(f"Target distribution: {torch.bincount(target)}")
                
                # Mixed precision training
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda'):
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    # Regular training for CPU and MPS
                    output = model(data)
                    loss = criterion(output, target)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                if device.type == 'mps':
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
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
            
            if metrics and 'accuracy' in metrics:
                accuracy = metrics['accuracy']
                logger.info(f"\nEpoch {epoch+1} Accuracy: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), save_dir / 'best_model.pth')
            else:
                logger.warning(f"\nEpoch {epoch+1}: Failed to calculate metrics")
        
        # Final evaluation on all datasets
        eval_loaders = {
            'test': test_loader,
            'test_balanced': dataset_loader.create_balanced_loader(dataset_loader.test_dataset),
            'test_adversarial': dataset_loader.create_adversarial_loader(model, test_loader, epsilon=0.1)
        }
        
        metrics_by_dataset = {}
        for dataset_name, loader in eval_loaders.items():
            logger.info(f"\nEvaluating on {dataset_name} dataset...")
            metrics = evaluator.evaluate(model, loader, device, name=f'{dataset_name}')
            
            # Validate metrics
            if not metrics:
                logger.error(f"No metrics returned for {dataset_name}")
                continue
            
            if metrics['accuracy'] < 0.5:  # Basic sanity check
                logger.warning(f"Unusually low accuracy ({metrics['accuracy']:.4f}) for {dataset_name}")
            
            metrics_by_dataset[dataset_name] = metrics
            logger.info(f"{dataset_name} metrics: {metrics}")
        
        # Verify we have metrics before saving
        if metrics_by_dataset:
            # Save experiment with status
            experiment = tracker.save_experiment(metrics_by_dataset, config, status="pending")
            
            # Verify save
            if not tracker.json_file.exists():
                logger.error("Failed to save JSON metrics file")
            if not tracker.csv_file.exists():
                logger.error("Failed to save CSV metrics file")
        else:
            logger.error("No metrics collected - skipping save")
        
        # Compare with baseline
        comparisons = tracker.compare_with_baseline(metrics_by_dataset)
        if comparisons.get("status") != "no_baseline":
            print("\nComparison with baseline:")
            for dataset_name, dataset_metrics in comparisons.items():
                print(f"\n{dataset_name} Dataset:")
                for metric_name, values in dataset_metrics.items():
                    if isinstance(values, dict) and 'current' in values:  # Check if it's a metric comparison
                        print(f"  {metric_name}:")
                        print(f"    Baseline: {values['baseline']:.4f}")
                        print(f"    Current:  {values['current']:.4f}")
                        print(f"    Change:   {values['diff']:+.4f} ({values['diff_percent']:+.2f}%)")
                        if values.get('status') == 'incomplete_data':
                            print("    Warning: Incomplete data for comparison")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    train() 