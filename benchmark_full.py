#!/usr/bin/env python
"""Comprehensive GPU benchmark over all optimisers, models and scenarios.

Example (single-GPU):
    python benchmark_full.py --device cuda --epochs 30 --wandb

Выполняет:
1. Перебор всех комбинаций (model × optimiser).
2. Запуск train.py c флагами --track-generalization --eval-robustness.
3. Сбор key-metrics из JSON-файлов, агрегирование в таблицу.
4. (опц.) Логирование агрегированных результатов в WandB как Table + plot.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import argparse
from typing import Dict, Any, List

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

import torchvision
from utils.cifar10_c import download_cifar10_c

MODELS = ["convnet", "mlp-mixer"]
OPTIMS = [
    "adam",
    "hatam",
    "sam-sgd",
    "sam-adam",
    "asam-sgd",
    "asam-adam",
    "a2sam-sgd",
    "a2sam-adam",
]

# === Оптимальные гиперпараметры для честного сравнения ===
# Оптимизаторы на базе Adam и SGD требуют разных LR, а SAM/ASAM — разных rho
OPTIM_HPARAMS = {
    "adam":       {"lr": 1e-3},
    "hatam":      {"lr": 3e-4, "gamma": 0.4, "weight_decay": 0.01},  # ✅ ИСПРАВЛЕНО: правильные параметры HATAM
    "sam-sgd":    {"lr": 1e-1, "rho": 0.05}, 
    "sam-adam":   {"lr": 1e-3, "rho": 0.05},
    "asam-sgd":   {"lr": 1e-1, "rho": 2.0},  # ASAM требует больший rho
    "asam-adam":  {"lr": 1e-3, "rho": 2.0},  # 
    "a2sam-sgd":  {"lr": 0.05, "rho": 0.03, "alpha": 0.5, "k": 5, "hessian_update_freq": 30, "power_iter_steps": 4},  # обновлено
    "a2sam-adam": {"lr": 3e-4, "rho": 0.03, "alpha": 0.5, "k": 5, "hessian_update_freq": 30, "power_iter_steps": 4},  # обновлено
}

def prepare_datasets(root: str):
    """Downloads CIFAR10 and CIFAR10-C if they don't exist."""
    print("\n" + "=" * 80)
    print("STEP 1: PREPARING DATASETS")
    print("=" * 80)
    
    # Download CIFAR-10 train and test sets
    print("Checking for CIFAR-10...")
    try:
        torchvision.datasets.CIFAR10(root, train=True, download=True)
        torchvision.datasets.CIFAR10(root, train=False, download=True)
        print("✅ CIFAR-10 is ready.")
    except Exception as e:
        print(f"❌ Failed to download CIFAR-10: {e}")

    # Download CIFAR-10-C for robustness evaluation
    print("\nChecking for CIFAR-10-C...")
    try:
        download_cifar10_c(root)
        print("✅ CIFAR-10-C is ready.")
    except Exception as e:
        print(f"❌ Could not download CIFAR-10-C: {e}")
        print("Robustness evaluation will be skipped if data is missing.")
    print("-" * 80)


def run_single(model: str, optim: str, args) -> Dict[str, Any]:
    """Запускает одну комбинацию и парсит результаты."""
    run_name = f"{model}_{optim}_seed{args.seed}"
    
    # Получаем специфичные гиперпараметры для оптимизатора
    hparams = OPTIM_HPARAMS.get(optim, {})
    lr = hparams.get("lr", args.lr)
    rho = hparams.get("rho", 0.05) # SAM по умолчанию

    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--optim", optim,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(lr),
        "--rho", str(rho),
        "--device", args.device,
        "--seed", str(args.seed),
        "--track-generalization",
        "--eval-robustness", 
        "--plot-landscape",
        "--wandb",
        "--wandb-project", args.wandb_project,
    ]
    
    # ✅ ДОБАВЛЕНО: Передача дополнительных параметров для A²SAM и HATAM
    if "alpha" in hparams:
        cmd.extend(["--alpha", str(hparams["alpha"])])
    if "k" in hparams:
        cmd.extend(["--k", str(hparams["k"])])
    if "hessian_update_freq" in hparams:
        cmd.extend(["--hessian-update-freq", str(hparams["hessian_update_freq"])])
    if "gamma" in hparams:
        cmd.extend(["--gamma", str(hparams["gamma"])])
    if "weight_decay" in hparams:
        cmd.extend(["--weight-decay", str(hparams["weight_decay"])])
    if "power_iter_steps" in hparams:
        cmd.extend(["--power-iter-steps", str(hparams["power_iter_steps"])])
    if args.fake_data:
        cmd.append("--fake-data")

    print("\n" + "#" * 80)
    print(f"🚀 Run {run_name} on {args.device}")
    print("#" * 80)

    start = time.time()
    proc = subprocess.run(cmd, text=True)
    elapsed = time.time() - start

    # Attempt to read JSON results file created by train.py
    json_path = Path(f"training_results_{optim}_{model}_seed{args.seed}.json")
    metrics = {}
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
            metrics = {
                "best_test_acc": data.get("best_test_acc"),
                "final_generalization_gap": data.get("final_generalization_gap"),
                "mCE": data.get("mCE"),
            }
    metrics.update({
        "model": model,
        "optimizer": optim,
        "elapsed_s": elapsed,
        "status": "ok" if proc.returncode == 0 else "fail",
    })
    return metrics


def aggregate_table(rows: List[Dict[str, Any]]):
    """Pretty-print aggregated results in tabular form."""
    header = [
        "Model", "Optimiser", "Acc %", "Gap %", "mCE %", "Time s"
    ]
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    print(f"{header[0]:<10} {header[1]:<12} {header[2]:<7} {header[3]:<6} {header[4]:<7} {header[5]:<7}")
    print("-" * 60)

    def _fmt(value, prec=2):
        if isinstance(value, (int, float)):
            fmt = f"{{:.{prec}f}}"
            return fmt.format(value)
        return "n/a"

    for r in rows:
        acc_str = _fmt(r.get("best_test_acc"), prec=2)
        gap_str = _fmt(r.get("final_generalization_gap"), prec=2)
        mce_str = _fmt(r.get("mCE"), prec=1)
        print(f"{r['model']:<10} {r['optimizer']:<12} {acc_str:<7} {gap_str:<6} {mce_str:<7} {int(r['elapsed_s']):<7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full benchmark combinator")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="New_SAM")
    args = parser.parse_args()

    # Create data root directory if it doesn't exist
    data_root = Path("./data")
    data_root.mkdir(exist_ok=True)
    
    # --- Загружаем все данные ПЕРЕД началом бенчмарка ---
    if not args.fake_data:
        prepare_datasets(str(data_root))

    # WandB aggregated run (optional)
    wandb_tbl_run = None
    if wandb is not None and os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb_tbl_run = wandb.init(project=args.wandb_project, name="benchmark_full", reinit=True)

    print("\n" + "=" * 80)
    print("STEP 2: RUNNING BENCHMARKS")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    for model, optim in itertools.product(MODELS, OPTIMS):
        res = run_single(model, optim, args)
        results.append(res)
        if wandb_tbl_run is not None:
            wandb_tbl_run.log(res)

    aggregate_table(results)

    if wandb_tbl_run is not None:
        # Save as table artifact for later visual inspection
        import pandas as pd  # lightweight dep, already pulled by wandb
        df = pd.DataFrame(results)
        wandb_tbl_run.log({"summary_table": wandb.Table(dataframe=df)})
        wandb_tbl_run.finish() 