#!/usr/bin/env python
"""Train benchmark script for CIFAR-10 comparing Adam, SAM, A²SAM and HATAM.

Usage example (CPU/generic):
    python train.py --model convnet --optim a2sam --epochs 20 --device cuda:0

By default the script trains on CIFAR-10 train split and evaluates on the test
split.  If the CIFAR-10-C corruption dataset is available in
`<root>/cifar10_c/`, we additionally evaluate robustness over the 15 corruption
categories at severity levels 1–5 and report mean corruption error (mCE).

Important implementation notes:
    * All random seeds are fixed for reproducibility via utils.seed.set_seed.
    * For SAM and A²SAM the training loop must supply a *closure* so that the
      optimiser can perform two forward/backward passes internally.
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import time
from typing import Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as T

from utils.seed import set_seed
from utils.cifar10_c import download_cifar10_c, evaluate_corruption_robustness
from utils.loss_landscape import plot_loss_surface
from models.convnet import SmallConvNet
from models.mlp_mixer import SmallMLPMixer
from optimizers import A2SAM, HATAM
from optimizers.sam_impl import SAM as DavdaSAM

# === WandB (можно отключить CLI флагом) ===
try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # позволит запускать без wandb в requirements

# Улучшаем точность матмуль на современных GPU (PyTorch >=1.12)
torch.set_float32_matmul_precision("medium")

# ----------------------------- datasets helpers ------------------------------


def build_cifar10_dataloaders(root: str, batch_size: int = 128, fake: bool = False) -> Tuple[data.DataLoader, data.DataLoader]:
    """Returns train/test dataloaders.

    If `fake` is True we use a randomly-generated `torchvision.datasets.FakeData`
    with the same shape as CIFAR-10.  Useful for CI / unit-tests where network
    access is limited.
    """
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if fake:
        transform = T.Compose([T.ToTensor(), norm])
        train_ds = torchvision.datasets.FakeData(size=1024, image_size=(3, 32, 32), num_classes=10, transform=transform)
        test_ds = torchvision.datasets.FakeData(size=256, image_size=(3, 32, 32), num_classes=10, transform=transform)
    else:
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            norm,
        ])
        test_tf = T.Compose([
            T.ToTensor(),
            norm,
        ])
        train_ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ----------------------------- model factory ---------------------------------


def create_model(name: str):
    name = name.lower()
    if name == "convnet":
        return SmallConvNet()
    elif name in {"mlp", "mlp-mixer", "mlpmixer", "mixer"}:
        return SmallMLPMixer()
    else:
        raise ValueError(f"Unknown model {name}")


# ----------------------------- optimiser factory -----------------------------


def create_optimizer(name: str, params, lr: float, rho: float, alpha: float = 0.1, k: int = 1, hessian_update_freq: int = 10, gamma: float = 0.1, weight_decay: float = 0.0):
    name = name.lower()
    base_optimizer_cls = torch.optim.SGD # по умолчанию для SAM/A2SAM
    base_optim_name = "sgd"
    optim_name = name

    if '-' in name:
        optim_name, base_optim_name = name.split('-')
        if base_optim_name == "adam":
            base_optimizer_cls = torch.optim.AdamW
        elif base_optim_name == "sgd":
            base_optimizer_cls = torch.optim.SGD
        else:
            raise ValueError(f"Unknown base optimizer: {base_optim_name}")

    if optim_name == "adam":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    elif optim_name == "sam" or optim_name == "asam":
        # Используем локальную реализацию davda54/sam
        is_adaptive = (optim_name == "asam")
        base_kwargs = {"lr": lr}
        if base_optimizer_cls == torch.optim.SGD:
            base_kwargs["momentum"] = 0.9
        return DavdaSAM(params, base_optimizer_cls, rho=rho, adaptive=is_adaptive, **base_kwargs)
    
    elif optim_name == "a2sam":
        base_optimizer_kwargs={"lr": lr}
        if base_optimizer_cls == torch.optim.SGD:
            base_optimizer_kwargs["momentum"] = 0.9
        # ✅ ИСПРАВЛЕНО: Передаем все параметры A²SAM
        return A2SAM(params, base_optimizer_cls=base_optimizer_cls, base_optimizer_kwargs=base_optimizer_kwargs, 
                     rho=rho, alpha=alpha, k=k, hessian_update_freq=hessian_update_freq)
    
    elif optim_name == "hatam":
        # HATAM не поддерживает базовые оптимизаторы
        if base_optim_name != "sgd": # если пользователь ввел hatam-adam
            print("Warning: HATAM does not use a base optimizer concept. Ignoring '-adam'.")
        # ✅ ИСПРАВЛЕНО: Передаем все параметры HATAM
        return HATAM(params, lr=lr, gamma=gamma, weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Unknown optimiser {name}")


# ----------------------------- training utilities ----------------------------


def train_one_epoch(model, loader, optim, device, log_interval: int = 100, epoch: int = 0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # ✅ ДОБАВЛЕНИЕ: Измерение времени оптимизации
    optim_time = 0.0

    # Проверяем, является ли оптимизатор SAM-ом от Davda, чтобы использовать кастомный training loop
    is_davda_sam = isinstance(optim, DavdaSAM)
    
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        if is_davda_sam:
            # --- Кастомный цикл для SAM (davda54) ---
            step_start = time.time()
            
            # Первый шаг
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.first_step(zero_grad=True)
            
            # Второй шаг
            criterion(model(inputs), targets).backward()
            optim.second_step(zero_grad=True)
            
            optim_time += time.time() - step_start

        else:
            # --- Стандартный цикл для остальных оптимизаторов ---
            def closure():
                optim.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss

            loss = closure()
            
            # ✅ ДОБАВЛЕНИЕ: Измеряем время step'а оптимизатора
            step_start = time.time()
            # For optimisers, которые требуют closure, передаём его — для остальных call без аргументов.
            try:
                optim.step(closure)
            except TypeError:
                optim.step()
            optim_time += time.time() - step_start

        # заново получаем outputs без градиентов для метрик
        with torch.no_grad():
            logits = model(inputs)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (i + 1) % log_interval == 0 or (i + 1) == len(loader):
            print(f"Epoch {epoch} [{i+1}/{len(loader)}] loss={running_loss/total:.4f} acc={100.*correct/total:.2f}% optim_time={optim_time:.3f}s", flush=True)
    return running_loss / total, 100.0 * correct / total, optim_time


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
    return loss_sum / total, 100.0 * correct / total


# ----------------------------- main entry ------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--model", type=str, default="convnet", choices=["convnet", "mlp-mixer"])
    p.add_argument("--optim", type=str, default="adam", choices=["adam", "sam", "a2sam", "hatam", "sam-sgd", "sam-adam", "asam-sgd", "asam-adam", "a2sam-sgd", "a2sam-adam"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rho", type=float, default=0.05, help="SAM/ASAM neighborhood size")
    # ✅ ДОБАВЛЕНО: Дополнительные параметры для A²SAM и HATAM
    p.add_argument("--alpha", type=float, default=0.1, help="A²SAM: strength of Hessian contribution in metric M = I + αH_k")
    p.add_argument("--k", type=int, default=1, help="A²SAM: rank of Hessian approximation (number of eigenpairs)")
    p.add_argument("--hessian-update-freq", type=int, default=10, help="A²SAM: steps between Hessian recomputations")
    p.add_argument("--gamma", type=float, default=0.1, help="HATAM: strength of trajectory correction")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=100, help="print training status every N batches")
    p.add_argument("--fake-data", action="store_true", help="use FakeData instead of downloading CIFAR-10 (smoke test)")
    p.add_argument("--eval-robustness", action="store_true", help="evaluate robustness on CIFAR-10-C (requires 2.9GB download)")
    p.add_argument("--track-generalization", action="store_true", help="track detailed generalization metrics throughout training")
    p.add_argument("--plot-landscape", action="store_true", help="generate 3D loss landscape visualization (computationally expensive)")
    p.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default="New_SAM", help="WandB project name")
    args = p.parse_args()

    set_seed(args.seed)

    # ------------------- WandB init -------------------
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb package not installed — add it to requirements or run without --wandb")
        # Загрузка ключа из переменной окружения (безопасно, не хранится в git)
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        # Если ключа нет – дадим wandb самому спросить у пользователя (например, в интерактивной среде Kaggle)
        run_name = f"{args.model}_{args.optim}_seed{args.seed}"
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args), reinit=True)

    train_loader, test_loader = build_cifar10_dataloaders(args.data_root, args.batch_size, fake=args.fake_data)

    model = create_model(args.model).to(args.device)
    optimizer = create_optimizer(args.optim, model.parameters(), lr=args.lr, rho=args.rho, 
                                alpha=args.alpha, k=args.k, hessian_update_freq=args.hessian_update_freq, 
                                gamma=args.gamma, weight_decay=args.weight_decay)

    # Initialize tracking variables
    best_acc = 0.0
    training_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'generalization_gap': [],
        'optim_time': []
    }
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc, optim_time = train_one_epoch(model, train_loader, optimizer, args.device, log_interval=args.log_interval, epoch=epoch)
        test_loss, test_acc = evaluate(model, test_loader, args.device)
        if test_acc > best_acc:
            best_acc = test_acc
        
        # Calculate generalization gap
        generalization_gap = train_acc - test_acc
        
        # Store training history if tracking is enabled
        if args.track_generalization:
            training_history['epoch'].append(epoch)
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['test_loss'].append(test_loss)
            training_history['test_acc'].append(test_acc)
            training_history['generalization_gap'].append(generalization_gap)
            training_history['optim_time'].append(optim_time)
        
        elapsed = time.time() - start

        # ---------------- WandB logging ----------------
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "generalization_gap": generalization_gap,
                "optim_time_s": optim_time,
                "elapsed_s": elapsed,
            })
        
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% | "
              f"gap={generalization_gap:.2f}% | "
              f"optim_time={optim_time:.3f}s ({100*optim_time/elapsed:.1f}% of total) | {elapsed:.1f}s")
    
    print(f"\n=== Final Results ===")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Final generalization gap: {generalization_gap:.2f}%")
    
    mce_value = None  # по умолчанию
    # Evaluate robustness if requested
    if args.eval_robustness and not args.fake_data:
        print(f"\n=== Robustness Evaluation ===")
        try:
            cifar10_c_path = download_cifar10_c(args.data_root)
            robustness_results = evaluate_corruption_robustness(model, cifar10_c_path, args.device)
            print(f"Mean Corruption Error (mCE): {robustness_results['mCE']:.1f}%")
            
            # Show top 5 worst corruptions
            corruption_errors = robustness_results['corruption_errors']
            worst_corruptions = sorted(corruption_errors.items(), key=lambda x: x[1], reverse=True)[:5]
            print("Top 5 worst corruptions:")
            for i, (corruption, error) in enumerate(worst_corruptions, 1):
                print(f"  {i}. {corruption}: {error:.1f}% error")
                
            # ---------------- WandB robustness log ----------------
            mce_value = robustness_results['mCE'] if args.eval_robustness and not args.fake_data and 'robustness_results' in locals() else None
            if wandb_run is not None and mce_value is not None:
                wandb_run.log({"mCE": mce_value})
            
        except Exception as e:
            print(f"Robustness evaluation failed: {e}")
            print("You can still compare models using generalization gap and standard accuracy")
    
    # Save detailed results if tracking was enabled
    if args.track_generalization:
        import json
        results_file = f"training_results_{args.optim}_{args.model}_seed{args.seed}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'args': vars(args),
                'training_history': training_history,
                'best_test_acc': best_acc,
                'final_generalization_gap': generalization_gap,
                'mCE': mce_value,
            }, f, indent=2)
        print(f"Detailed results saved to {results_file}")
    
    print(f"\n=== Generalization Summary ===")
    print(f"Optimizer: {args.optim.upper()}")
    print(f"Generalization gap: {generalization_gap:.2f}% (lower is better)")
    print(f"Test accuracy: {best_acc:.2f}% (higher is better)")
    print(f"Note: Good optimizers should have low generalization gap AND high test accuracy")

    # Generate 3D loss landscape if requested
    if args.plot_landscape and not args.fake_data:
        print(f"\n=== Generating Loss Landscape ===")
        print("Warning: This is computationally expensive and may take several minutes...")
        try:
            import torch.nn as nn
            # Use a subset of test data for faster computation
            subset_size = min(1000, len(test_loader.dataset))
            subset_indices = torch.randperm(len(test_loader.dataset))[:subset_size]
            subset_dataset = torch.utils.data.Subset(test_loader.dataset, subset_indices)
            subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=100, shuffle=False)
            
            loss_fn = nn.CrossEntropyLoss()
            fig = plot_loss_surface(model, loss_fn, subset_loader, steps=15, span=0.5, device=args.device)
            
            # Save figure locally
            landscape_filename = f"loss_landscape_{args.model}_{args.optim}_seed{args.seed}.png"
            fig.savefig(landscape_filename, dpi=150, bbox_inches='tight')
            print(f"Loss landscape saved as {landscape_filename}")
            
            # Log to wandb if available
            if wandb_run is not None:
                wandb_run.log({"loss_landscape": wandb.Image(fig)})
                print("Loss landscape logged to wandb")
            
            import matplotlib.pyplot as plt
            plt.close(fig)  # Free memory
            
        except Exception as e:
            print(f"Failed to generate loss landscape: {e}")

    # ---------------- WandB finish ----------------
    if wandb_run is not None:
        # Записываем summary
        wandb_run.summary.update({
            "best_test_acc": best_acc,
            "final_generalization_gap": generalization_gap,
            "mCE": mce_value,
        })
        wandb_run.finish()


if __name__ == "__main__":
    main() 