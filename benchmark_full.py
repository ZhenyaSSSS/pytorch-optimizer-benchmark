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

MODELS = ["convnet", "mlp-mixer"]
OPTIMS = ["adam", "sam", "a2sam", "hatam"]

# === Оптимальные гиперпараметры для честного сравнения ===
# Оптимизаторы на базе Adam и SGD требуют разных LR
OPTIM_HPARAMS = {
    "adam":    {"lr": 1e-3},
    "hatam":   {"lr": 1e-3},
    "sam":     {"lr": 1e-1, "base_optimizer": "SGD"}, # SGD-based
    "a2sam":   {"lr": 1e-1, "base_optimizer": "SGD"}, # SGD-based
}


def run_single(model: str, optim: str, args) -> Dict[str, Any]:
    """Запускает одну комбинацию и парсит результаты."""
    run_name = f"{model}_{optim}_seed{args.seed}"
    
    # Получаем специфичные гиперпараметры для оптимизатора
    hparams = OPTIM_HPARAMS.get(optim, {})
    lr = hparams.get("lr", args.lr)

    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--optim", optim,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(lr),
        "--device", args.device,
        "--seed", str(args.seed),
        "--track-generalization",
        "--eval-robustness",
        "--wandb",
        "--wandb-project", args.wandb_project,
    ]
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
    print(f"{header[0]:<10} {header[1]:<8} {header[2]:<7} {header[3]:<6} {header[4]:<7} {header[5]:<7}")
    print("-" * 60)
    for r in rows:
        print(f"{r['model']:<10} {r['optimizer']:<8} {r.get('best_test_acc', 'n/a')!s:<7} "
              f"{r.get('final_generalization_gap', 'n/a')!s:<6} {r.get('mCE', 'n/a')!s:<7} {int(r['elapsed_s']):<7}")


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

    # WandB aggregated run (optional)
    wandb_tbl_run = None
    if wandb is not None and os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb_tbl_run = wandb.init(project=args.wandb_project, name="benchmark_full", reinit=True)

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