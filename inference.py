import torch
from pathlib import Path
from models.baseline_cnn import BaselineCNN
from utils.config import ConfigLoader
from data.dataset_loader import MNISTLoader
from eval.evaluator import ModelEvaluator

def load_model(model_path: str, config: dict) -> BaselineCNN:
    model = BaselineCNN(
        input_shape=config['model']['params']['input_shape'],
        hidden_dims=config['model']['params']['hidden_dims'],
        num_classes=config['model']['params']['num_classes'],
        dropout_rate=config['model']['params'].get('dropout_rate', 0.1)
    )
    model.load_state_dict(torch.load(model_path))
    return model

def main():
    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_experiment_config('configs/base_config.yaml')
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model('models/checkpoints/best_model.pth', config)
    model = model.to(device)
    
    # Load test data
    dataset_loader = MNISTLoader(
        data_dir='data/raw',
        batch_size=config['dataset']['params']['batch_size']
    )
    _, test_loader = dataset_loader.load_data()
    
    # Setup evaluator
    evaluator = ModelEvaluator(config)
    
    # Run evaluation
    metrics = evaluator.evaluate(model, test_loader, device, name='inference')
    
    print("\nInference Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main() 