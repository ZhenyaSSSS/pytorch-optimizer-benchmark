#!/usr/bin/env python
"""Тест исправлений оптимизаторов с генерацией 3D ландшафта."""

import subprocess
import sys

def test_optimizers():
    """Тестирует все варианты оптимизаторов"""
    optimizers = [
        "adam", "sam", "a2sam", "hatam",
        "sam-sgd", "sam-adam", "asam-sgd", "asam-adam", 
        "a2sam-sgd", "a2sam-adam"
    ]
    
    print("=== Тест всех доступных оптимизаторов ===")
    for opt in optimizers:
        print(f"\n🧪 Тестирую оптимизатор: {opt}")
        cmd = [
            sys.executable, "train.py",
            "--model", "convnet",
            "--optim", opt,
            "--epochs", "2",  # Быстрый тест
            "--fake-data",   # Без загрузки данных
            "--plot-landscape",  # Тест 3D ландшафтов
            "--track-generalization"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✅ {opt}: SUCCESS")
            else:
                print(f"❌ {opt}: FAILED")
                print(f"Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏰ {opt}: TIMEOUT")
        except Exception as e:
            print(f"💥 {opt}: EXCEPTION - {e}")

if __name__ == "__main__":
    test_optimizers() 