"""Utilities for CIFAR-10-C robustness evaluation.

CIFAR-10-C is a robustness benchmark with 15 different corruption types
applied to the CIFAR-10 test set at 5 severity levels.

Download from: https://zenodo.org/records/2535967
"""
import os
import numpy as np
import pickle
import torch
import torch.utils.data as data
from typing import List, Dict, Tuple, Optional
import requests
import tarfile
from pathlib import Path


# 15 corruption types used in CIFAR-10-C
CIFAR10_C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

# AlexNet baseline errors for normalization (from original paper)
ALEXNET_CE_BASELINES = {
    'gaussian_noise': 88.6,
    'shot_noise': 89.4,
    'impulse_noise': 92.3,
    'defocus_blur': 82.0,
    'glass_blur': 82.6,
    'motion_blur': 78.6,
    'zoom_blur': 79.8,
    'snow': 86.7,
    'frost': 82.7,
    'fog': 81.9,
    'brightness': 56.5,
    'contrast': 85.3,
    'elastic_transform': 64.6,
    'pixelate': 71.8,
    'jpeg_compression': 60.7
}


def download_cifar10_c(data_root: str = "./data") -> str:
    """Download CIFAR-10-C if not already present.
    
    Returns path to the extracted CIFAR-10-C directory.
    """
    data_root = Path(data_root)
    cifar10_c_dir = data_root / "CIFAR-10-C"
    
    if cifar10_c_dir.exists():
        print(f"CIFAR-10-C already exists at {cifar10_c_dir}")
        return str(cifar10_c_dir)
    
    # Create data directory
    data_root.mkdir(exist_ok=True)
    
    # Download URL (2.9 GB)
    url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
    tar_path = data_root / "CIFAR-10-C.tar"
    
    print(f"Downloading CIFAR-10-C from {url}...")
    print("This is a large file (2.9 GB), download may take a while...")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(tar_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded // 1024 // 1024} MB)", end='', flush=True)
    
    print(f"\nExtracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(data_root)
    
    # Clean up
    os.remove(tar_path)
    print(f"CIFAR-10-C extracted to {cifar10_c_dir}")
    
    return str(cifar10_c_dir)


class CIFAR10CDataset(data.Dataset):
    """CIFAR-10-C dataset loader.
    
    Loads corrupted CIFAR-10 test images for robustness evaluation.
    """
    
    def __init__(self, root: str, corruption: str, severity: int = 5, transform=None):
        """
        Args:
            root: Path to CIFAR-10-C directory
            corruption: Name of corruption (e.g., 'gaussian_noise')
            severity: Corruption severity level (1-5, where 5 is most severe)
            transform: Optional transform to apply to images
        """
        if corruption not in CIFAR10_C_CORRUPTIONS:
            raise ValueError(f"Unknown corruption: {corruption}. Must be one of {CIFAR10_C_CORRUPTIONS}")
        
        if not 1 <= severity <= 5:
            raise ValueError("Severity must be between 1 and 5")
        
        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        
        # Load corrupted images (50k total, 10k per severity level)
        data_file = self.root / f"{corruption}.npy"
        if not data_file.exists():
            raise FileNotFoundError(f"Corruption file not found: {data_file}")
        
        # Load full data and extract severity level
        all_data = np.load(data_file)  # Shape: (50000, 32, 32, 3)
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        self.data = all_data[start_idx:end_idx]
        
        # Load labels (same for all corruption types)
        labels_file = self.root / "labels.npy"
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        self.labels = np.load(labels_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def evaluate_corruption_robustness(model, cifar10_c_root: str, device: str = "cpu",
                                 batch_size: int = 128) -> Dict[str, Dict]:
    """Evaluate model robustness on CIFAR-10-C.
    
    Returns:
        Dictionary with corruption errors and mCE (mean Corruption Error)
    """
    model.eval()
    results = {}
    
    # Standard CIFAR-10 normalization
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    corruption_errors = {}
    
    for corruption in CIFAR10_C_CORRUPTIONS:
        print(f"Evaluating corruption: {corruption}")
        
        # Test all 5 severity levels
        severity_errors = []
        
        for severity in range(1, 6):
            try:
                dataset = CIFAR10CDataset(cifar10_c_root, corruption, severity, transform)
                loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
                
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for images, labels in loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                
                accuracy = 100.0 * correct / total
                error = 100.0 - accuracy
                severity_errors.append(error)
                
                print(f"  Severity {severity}: {error:.2f}% error")
                
            except FileNotFoundError:
                print(f"  Warning: {corruption} severity {severity} not found, skipping")
                severity_errors.append(float('nan'))
        
        # Corruption Error = mean over severity levels 3, 4, 5
        if len(severity_errors) >= 5:
            ce = np.mean(severity_errors[2:5])  # severity 3,4,5 (indices 2,3,4)
            corruption_errors[corruption] = ce
            print(f"  Corruption Error: {ce:.2f}%")
        
    # Calculate mean Corruption Error (mCE) normalized by AlexNet baseline
    ce_normalized = []
    
    for corruption, ce in corruption_errors.items():
        if corruption in ALEXNET_CE_BASELINES and not np.isnan(ce):
            ce_norm = 100 * ce / ALEXNET_CE_BASELINES[corruption]
            ce_normalized.append(ce_norm)
    
    mce = np.mean(ce_normalized) if ce_normalized else float('nan')
    
    results = {
        'corruption_errors': corruption_errors,
        'mCE': mce,
        'individual_ces': {c: 100 * ce / ALEXNET_CE_BASELINES[c] 
                          for c, ce in corruption_errors.items() 
                          if c in ALEXNET_CE_BASELINES and not np.isnan(ce)}
    }
    
    print(f"\nSummary:")
    print(f"Mean Corruption Error (mCE): {mce:.1f}%")
    print(f"(Lower is better, AlexNet baseline = 100%)")
    
    return results


def create_synthetic_corruptions(images: torch.Tensor, corruption_type: str, severity: int = 3) -> torch.Tensor:
    """Create synthetic corruptions for testing when CIFAR-10-C is not available.
    
    This is a simplified version for demo purposes. Real CIFAR-10-C uses more sophisticated corruptions.
    """
    if corruption_type == "gaussian_noise":
        noise = torch.randn_like(images) * (0.05 * severity)
        return torch.clamp(images + noise, 0, 1)
    
    elif corruption_type == "brightness":
        factor = 0.3 + 0.1 * severity
        return torch.clamp(images * factor, 0, 1)
    
    elif corruption_type == "contrast":
        factor = 0.5 + 0.1 * severity  
        mean = images.mean(dim=(2,3), keepdim=True)
        return torch.clamp((images - mean) * factor + mean, 0, 1)
    
    elif corruption_type == "blur":
        # Simple blur using conv2d (not as sophisticated as real corruptions)
        import torch.nn.functional as F
        kernel_size = min(3 + severity, 7)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        kernel = kernel.repeat(images.size(1), 1, 1, 1).to(images.device)
        
        blurred = F.conv2d(images, kernel, padding=kernel_size//2, groups=images.size(1))
        return torch.clamp(blurred, 0, 1)
    
    else:
        # Return original images if corruption not implemented
        return images 