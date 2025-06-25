#!/usr/bin/env python
"""Compare multiple optimizers with robustness and generalization metrics.

Usage:
    python compare_optimizers.py --optimizers adam hatam a2sam --epochs 10
    python compare_optimizers.py --optimizers adam hatam --eval-robustness
"""
import subprocess
import sys
import argparse
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def run_experiment(optimizer, args):
    """Run training experiment for a single optimizer."""
    print(f"\n{'='*60}")
    print(f"Training with {optimizer.upper()} optimizer")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "train.py",
        "--optim", optimizer,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--model", args.model,
        "--device", args.device,
        "--seed", str(args.seed),
        "--track-generalization"
    ]
    
    if args.eval_robustness:
        cmd.append("--eval-robustness")
    
    if args.fake_data:
        cmd.append("--fake-data")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    training_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Error training {optimizer}: {result.stderr}")
        return None
    
    # Parse results from output
    output_lines = result.stdout.split('\n')
    
    # Extract metrics from the last epoch
    best_acc = None
    generalization_gap = None
    mce = None
    
    for line in output_lines:
        if "Best test accuracy:" in line:
            best_acc = float(line.split(":")[1].strip().replace('%', ''))
        elif "Final generalization gap:" in line:
            generalization_gap = float(line.split(":")[1].strip().replace('%', ''))
        elif "Mean Corruption Error (mCE):" in line:
            mce = float(line.split(":")[1].strip().replace('%', ''))
    
    # Load detailed results if available
    results_file = f"training_results_{optimizer}_{args.model}_seed{args.seed}.json"
    history = None
    if Path(results_file).exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
            history = data['training_history']
    
    return {
        'optimizer': optimizer,
        'best_accuracy': best_acc,
        'generalization_gap': generalization_gap,
        'mce': mce,
        'training_time': training_time,
        'history': history,
        'stdout': result.stdout
    }


def plot_comparison(results, save_path="optimizer_comparison.png"):
    """Create comparison plots."""
    if not results:
        return
    
    # Filter out failed experiments
    results = [r for r in results if r is not None]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Optimizer Comparison on CIFAR-10', fontsize=16, fontweight='bold')
    
    optimizers = [r['optimizer'] for r in results]
    colors = plt.cm.Set3(np.linspace(0, 1, len(optimizers)))
    
    # 1. Accuracy comparison
    accuracies = [r['best_accuracy'] for r in results if r['best_accuracy'] is not None]
    if accuracies:
        axes[0, 0].bar(range(len(accuracies)), accuracies, color=colors[:len(accuracies)])
        axes[0, 0].set_xticks(range(len(accuracies)))
        axes[0, 0].set_xticklabels([r['optimizer'].upper() for r in results if r['best_accuracy'] is not None])
        axes[0, 0].set_ylabel('Test Accuracy (%)')
        axes[0, 0].set_title('Best Test Accuracy (Higher is Better)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Generalization gap comparison
    gaps = [r['generalization_gap'] for r in results if r['generalization_gap'] is not None]
    if gaps:
        axes[0, 1].bar(range(len(gaps)), gaps, color=colors[:len(gaps)])
        axes[0, 1].set_xticks(range(len(gaps)))
        axes[0, 1].set_xticklabels([r['optimizer'].upper() for r in results if r['generalization_gap'] is not None])
        axes[0, 1].set_ylabel('Generalization Gap (%)')
        axes[0, 1].set_title('Generalization Gap (Lower is Better)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training time comparison
    times = [r['training_time'] for r in results]
    axes[1, 0].bar(range(len(times)), times, color=colors)
    axes[1, 0].set_xticks(range(len(times)))
    axes[1, 0].set_xticklabels([r['optimizer'].upper() for r in results])
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Total Training Time (Lower is Better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Robustness comparison (if available)
    mces = [r['mce'] for r in results if r['mce'] is not None]
    if mces:
        axes[1, 1].bar(range(len(mces)), mces, color=colors[:len(mces)])
        axes[1, 1].set_xticks(range(len(mces)))
        axes[1, 1].set_xticklabels([r['optimizer'].upper() for r in results if r['mce'] is not None])
        axes[1, 1].set_ylabel('Mean Corruption Error (%)')
        axes[1, 1].set_title('Robustness (mCE, Lower is Better)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Robustness evaluation\nnot performed', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, style='italic')
        axes[1, 1].set_title('Robustness (mCE)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")


def print_summary(results):
    """Print detailed comparison summary."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE OPTIMIZER COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Filter out failed experiments
    results = [r for r in results if r is not None]
    
    if not results:
        print("No successful experiments to compare.")
        return
    
    # Print table header
    print(f"{'Optimizer':<12} {'Accuracy':<10} {'Gen Gap':<9} {'mCE':<8} {'Time':<8} {'Score':<8}")
    print("-" * 70)
    
    # Calculate composite scores
    for result in results:
        acc = result['best_accuracy'] or 0
        gap = result['generalization_gap'] or 0
        mce = result['mce'] or float('inf')
        time_score = result['training_time']
        
        # Simple composite score (higher is better)
        # Accuracy is good (higher better), gap is bad (lower better), mce is bad (lower better)
        if mce != float('inf'):
            score = acc - gap - (mce - 50)  # normalize mCE around 50%
        else:
            score = acc - gap
        
        result['composite_score'] = score
        
        # Format output
        acc_str = f"{acc:.1f}%" if acc else "N/A"
        gap_str = f"{gap:.1f}%" if gap else "N/A"
        mce_str = f"{mce:.1f}%" if mce != float('inf') else "N/A"
        time_str = f"{time_score:.0f}s"
        score_str = f"{score:.1f}"
        
        print(f"{result['optimizer'].upper():<12} {acc_str:<10} {gap_str:<9} {mce_str:<8} {time_str:<8} {score_str:<8}")
    
    # Find best optimizer
    best_result = max(results, key=lambda x: x['composite_score'])
    print(f"\n🏆 Best overall optimizer: {best_result['optimizer'].upper()}")
    
    # Specific recommendations
    print(f"\n📊 Detailed Analysis:")
    
    # Best accuracy
    best_acc = max(results, key=lambda x: x['best_accuracy'] or 0)
    print(f"• Highest accuracy: {best_acc['optimizer'].upper()} ({best_acc['best_accuracy']:.1f}%)")
    
    # Best generalization
    best_gen = min([r for r in results if r['generalization_gap'] is not None], 
                   key=lambda x: x['generalization_gap'], default=None)
    if best_gen:
        print(f"• Best generalization: {best_gen['optimizer'].upper()} ({best_gen['generalization_gap']:.1f}% gap)")
    
    # Best robustness
    best_robust = min([r for r in results if r['mce'] is not None], 
                      key=lambda x: x['mce'], default=None)
    if best_robust:
        print(f"• Most robust: {best_robust['optimizer'].upper()} ({best_robust['mce']:.1f}% mCE)")
    
    # Fastest training
    fastest = min(results, key=lambda x: x['training_time'])
    print(f"• Fastest training: {fastest['optimizer'].upper()} ({fastest['training_time']:.0f}s)")
    
    print(f"\n💡 Interpretation:")
    print(f"• Generalization gap < 5% is excellent, < 10% is good")
    print(f"• mCE < 60% is excellent robustness, < 80% is good")
    print(f"• Consider your priority: accuracy vs speed vs robustness")


def main():
    parser = argparse.ArgumentParser(description="Compare optimizers with robustness and generalization metrics")
    parser.add_argument("--optimizers", nargs='+', default=['adam', 'hatam', 'a2sam'], 
                       choices=['adam', 'sam', 'a2sam', 'hatam'],
                       help="Optimizers to compare")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model", type=str, default="convnet", choices=["convnet", "mlp-mixer"])
    parser.add_argument("--device", type=str, default="cuda" if __import__('torch').cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-robustness", action="store_true", 
                       help="Evaluate robustness on CIFAR-10-C (requires 2.9GB download)")
    parser.add_argument("--fake-data", action="store_true", 
                       help="Use fake data for quick testing")
    parser.add_argument("--save-plot", type=str, default="optimizer_comparison.png",
                       help="Path to save comparison plot")
    
    args = parser.parse_args()
    
    print("🚀 Starting comprehensive optimizer comparison...")
    print(f"Optimizers: {args.optimizers}")
    print(f"Epochs: {args.epochs}")
    print(f"Robustness eval: {'Yes' if args.eval_robustness else 'No'}")
    
    results = []
    total_start = time.time()
    
    for optimizer in args.optimizers:
        result = run_experiment(optimizer, args)
        results.append(result)
    
    total_time = time.time() - total_start
    
    print(f"\n⏱️  Total comparison time: {total_time:.1f} seconds")
    
    # Generate summary and plots
    print_summary(results)
    plot_comparison(results, args.save_plot)
    
    print(f"\n✅ Comparison complete! Check '{args.save_plot}' for visual comparison.")


if __name__ == "__main__":
    main() 