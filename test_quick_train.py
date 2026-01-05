"""Quick test of training script with 1 epoch"""
import subprocess
import sys

# Run with 1 epoch for testing
result = subprocess.run(
    [sys.executable, "scripts/train_half_dataset.py"],
    capture_output=False,
    text=True
)

sys.exit(result.returncode)
