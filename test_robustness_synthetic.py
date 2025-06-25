#!/usr/bin/env python
"""Test robustness using synthetic corruptions (without downloading CIFAR-10-C).

This provides a quick way to evaluate robustness when the full CIFAR-10-C dataset
is not available.
"""
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from models.convnet import SmallConvNet
from optimizers import A2SAM, HATAM
from utils.seed import set_seed
from utils.cifar10_c import create_synthetic_corruptions


def evaluate_model_on_corruption(model, test_loader, corruption_type, severity, device):
    """Evaluate model on synthetic corruption."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            # Apply synthetic corruption
            corrupted_images = create_synthetic_corruptions(images, corruption_type, severity)
            
            corrupted_images = corrupted_images.to(device)
            labels = labels.to(device)
            
            outputs = model(corrupted_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def quick_robustness_test():
    """Run a quick robustness comparison between optimizers."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    
    print("🧪 Quick Robustness Test with Synthetic Corruptions")
    print("=" * 60)
    
    # Load test data
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Use smaller subset for quick testing
    test_dataset = torchvision.datasets.FakeData(
        size=500, image_size=(3, 32, 32), num_classes=10, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)
    
    # Test corruptions
    corruptions = ["gaussian_noise", "brightness", "contrast", "blur"]
    severities = [1, 3, 5]
    
    # Create models (normally you'd load trained models)
    models = {
        "adam": SmallConvNet().to(device),
        "hatam": SmallConvNet().to(device),
    }
    
    # Quick training (just few steps to differentiate models)
    for name, model in models.items():
        print(f"Quick training for {name.upper()}...")
        
        if name == "adam":
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        else:
            optimizer = HATAM(model.parameters(), lr=0.01)
        
        model.train()
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= 3:  # Just 3 batches for differentiation
                break
                
            data, target = data.to(device), target.to(device)
            
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                return loss
            
            # Handle different optimizer interfaces
            if hasattr(optimizer, 'step') and 'closure' in str(optimizer.step.__code__.co_varnames):
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()
    
    # Evaluate robustness
    print(f"\n📊 Robustness Results:")
    print(f"{'Optimizer':<10} {'Corruption':<15} {'Sev 1':<8} {'Sev 3':<8} {'Sev 5':<8} {'Avg':<8}")
    print("-" * 70)
    
    for model_name, model in models.items():
        for corruption in corruptions:
            accuracies = []
            
            for severity in severities:
                acc = evaluate_model_on_corruption(model, test_loader, corruption, severity, device)
                accuracies.append(acc)
            
            avg_acc = sum(accuracies) / len(accuracies)
            
            print(f"{model_name.upper():<10} {corruption:<15} {accuracies[0]:<8.1f} {accuracies[1]:<8.1f} {accuracies[2]:<8.1f} {avg_acc:<8.1f}")
    
    print(f"\n💡 Note: This is a synthetic test with fake data.")
    print(f"   For real robustness evaluation, use --eval-robustness with actual CIFAR-10-C")
    
    # Show how to download real CIFAR-10-C
    print(f"\n📥 To get real CIFAR-10-C (2.9 GB):")
    print(f"   python train.py --eval-robustness --optim hatam --epochs 5")
    print(f"   # or use compare_optimizers.py --eval-robustness")


if __name__ == "__main__":
    quick_robustness_test() 