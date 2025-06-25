import subprocess, sys, pathlib, os

def test_train_smoke():
    script = pathlib.Path(__file__).resolve().parent.parent / 'train.py'
    # run 1 epoch, fake data, small batch; expect exit code 0
    res = subprocess.run([sys.executable, str(script), '--epochs', '1', '--batch-size', '32', '--optim', 'hatam', '--fake-data', '--device', 'cpu'], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert 'Epoch 1/1' in res.stdout 