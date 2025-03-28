import torch
import torch.nn.functional as F

def create_adversarial_loader(model, data_loader, epsilon=0.3, device='cuda'):
    """
    Create adversarial examples using Fast Gradient Sign Method (FGSM)
    """
    model.eval()
    adversarial_data = []
    original_targets = []
    
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        # Forward pass
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Create adversarial examples
        data_grad = data.grad.data
        perturbed_data = data + epsilon * data_grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        adversarial_data.append(perturbed_data.detach().cpu())
        original_targets.append(target.cpu())
    
    # Create new dataset
    adversarial_data = torch.cat(adversarial_data)
    original_targets = torch.cat(original_targets)
    
    adversarial_dataset = torch.utils.data.TensorDataset(adversarial_data, original_targets)
    adversarial_loader = torch.utils.data.DataLoader(
        adversarial_dataset,
        batch_size=data_loader.batch_size,
        shuffle=False
    )
    
    return adversarial_loader 