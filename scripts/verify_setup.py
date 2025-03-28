"""Verify project setup and dependencies."""
import sys
import importlib
from pathlib import Path

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "yaml",
    "sklearn",
    "matplotlib",
    "seaborn",
]

def verify_imports():
    """Verify all required packages can be imported."""
    failed = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            failed.append((package, str(e)))
            print(f"✗ {package}")
    return failed

def verify_directories():
    """Verify required directories exist."""
    required_dirs = [
        "data/raw",
        "models/checkpoints",
        "experiments/metrics",
        "experiments/plots",
    ]
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")

def main():
    print("Verifying imports...")
    failed = verify_imports()
    
    print("\nVerifying directories...")
    verify_directories()
    
    if failed:
        print("\nMissing dependencies:")
        for package, error in failed:
            print(f"  {package}: {error}")
        sys.exit(1)
    
    print("\nSetup verification complete!")

if __name__ == "__main__":
    main() 