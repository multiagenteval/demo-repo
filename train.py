import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.baseline_cnn import BaselineCNN
from utils.config import ConfigLoader
from pathlib import Path
import json

def train_model(config_path: str):
    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_experiment_config(config_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = datasets.MNIST(
        'data/raw', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        'data/raw', train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['params']['batch_size'],
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['dataset']['params']['batch_size']
    )
    
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
    
    # Training loop
    num_epochs = 5  # We'll keep it small for the demo
    metrics_history = []
    
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
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Save model
    save_dir = Path('models/checkpoints')
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / 'baseline_model.pth')
    
    # Evaluate and save metrics
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'model_name': config['model']['name'],
        'dataset': config['dataset']['name']
    }
    
    metrics_dir = Path('experiments/metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_dir / 'baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_model("configs/experiments/exp001_baseline.yaml") 