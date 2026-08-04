"""
MLA03 Reinforcement Learning Lab - Environment Setup Script
===========================================================
Run this script to install all required dependencies.

Usage:
    python setup.py          # Install all dependencies
    python setup.py --check  # Verify installations
"""

import subprocess
import sys
import importlib

CORE_PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
]

GYMNASIUM_PACKAGES = [
    "gymnasium[classic-control,box2d]",
]

ML_PACKAGES = [
    "tensorflow",
    "torch",
    "stable-baselines3",
]

SPECIALIZED_PACKAGES = [
    "pettingzoo",
    "minigrid",
    "highway-env",
]

OPTIONAL_PACKAGES = [
    "sb3-contrib",  # For TRPO in exp15
]


def install_packages():
    """Install all required packages using pip."""
    all_packages = CORE_PACKAGES + GYMNASIUM_PACKAGES + ML_PACKAGES + SPECIALIZED_PACKAGES + OPTIONAL_PACKAGES

    print("=" * 60)
    print("MLA03 RL Lab - Installing Dependencies")
    print("=" * 60)

    for pkg in all_packages:
        print(f"\nInstalling {pkg}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  [OK] {pkg}")
        except subprocess.CalledProcessError:
            print(f"  [WARN] {pkg} failed - may need manual installation")

    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)


def check_packages():
    """Verify all required packages are importable."""
    print("=" * 60)
    print("MLA03 RL Lab - Checking Dependencies")
    print("=" * 60)

    all_packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "gymnasium": "gymnasium",
        "tensorflow": "tensorflow",
        "torch": "torch",
        "stable_baselines3": "stable-baselines3",
        "pettingzoo": "pettingzoo",
        "minigrid": "minigrid",
        "highway_env": "highway-env",
    }

    optional_packages = {
        "sb3_contrib": "sb3-contrib",
    }

    status = {}
    for module, name in all_packages.items():
        try:
            importlib.import_module(module)
            status[name] = True
            print(f"  [OK] {name}")
        except ImportError:
            status[name] = False
            print(f"  [MISSING] {name}")

    print("\nOptional packages:")
    for module, name in optional_packages.items():
        try:
            importlib.import_module(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name} (needed for exp15 TRPO)")

    print("\n" + "=" * 60)
    missing = [k for k, v in status.items() if not v]
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Run: python setup.py")
    else:
        print("All core packages installed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_packages()
    else:
        install_packages()
        print("\nRunning verification...")
        check_packages()
