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
    # Load config
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
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Evaluate after each epoch
        metrics = evaluator.evaluate(model, test_loader, device, name=f'epoch_{epoch+1}')
        print(f"\nEpoch {epoch+1} Accuracy: {metrics['accuracy']:.4f}")
        
        # Save best model
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            
            # Get predictions for confusion matrix
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    preds = torch.argmax(output, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            # Save confusion matrix
            if config['tracking']['log_confusion_matrix']:
                save_confusion_matrix(
                    np.array(all_targets),
                    np.array(all_preds),
                    plots_dir / f'confusion_matrix_epoch_{epoch+1}.png'
                )
            
            # Save metrics
            with open(metrics_dir / f'metrics_epoch_{epoch+1}.json', 'w') as f:
                json.dump(metrics, f, indent=2)
    
    print(f"\nTraining completed! Best accuracy: {best_accuracy:.4f}")
    
    # After training, save experiment results
    final_metrics = evaluator.evaluate(model, test_loader, device, name='final')
    experiment_file = tracker.save_experiment(final_metrics, config)
    
    # Compare with baseline
    comparisons = tracker.compare_with_baseline(final_metrics)
    print("\nComparison with baseline:")
    for metric, comp in comparisons.items():
        print(f"{metric}:")
        print(f"  Current: {comp['current']:.4f}")
        print(f"  Baseline: {comp['baseline']:.4f}")
        print(f"  Change: {comp['diff_percent']:+.2f}%")

if __name__ == "__main__":
    train() 