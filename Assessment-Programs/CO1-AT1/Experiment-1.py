"""
Experiment 1: Installation of Reinforcement Learning Development Environment
--------------------------------------------------------------------------
Aim: To install and configure Anaconda Navigator, Python, TensorFlow, Keras,
Gymnasium (OpenAI Gym), NumPy, Matplotlib, and other required libraries, and
verify the environment by executing a simple Reinforcement Learning program.
"""

import sys
import importlib


def check_package(display_name, import_name=None):
    import_name = import_name or display_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK]      {display_name:12s} -> version {version}")
        return True
    except ImportError as e:
        print(f"[MISSING] {display_name:12s} -> {e}")
        return False


def main():
    print("=" * 60)
    print("Experiment 1: RL Development Environment Verification")
    print("=" * 60)
    print(f"Python version: {sys.version.split()[0]}\n")

    packages = [
        ("NumPy", "numpy"),
        ("Matplotlib", "matplotlib"),
        ("Gymnasium", "gymnasium"),
        ("TensorFlow", "tensorflow"),
        ("Keras", "keras"),
    ]

    all_ok = True
    for display_name, import_name in packages:
        ok = check_package(display_name, import_name)
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    if not all_ok:
        print("Some packages are missing. Install them first, e.g.:")
        print('  pip install numpy matplotlib "gymnasium[classic-control,toy-text]" tensorflow')
        print("=" * 60)
        return

    print("All required packages are installed. Running a sanity-check program...\n")

    # --- Simple RL sanity check: random agent on CartPole ---
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)
    total_reward = 0
    for _ in range(50):
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    print(f"Ran 50 random steps in CartPole-v1. Total reward collected: {total_reward}")

    # --- TensorFlow / GPU check ---
    import tensorflow as tf

    print(f"\nTensorFlow built with CUDA support: {tf.test.is_built_with_cuda()}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU(s) detected by TensorFlow: {gpus}")
    else:
        print("No GPU detected by TensorFlow (running on CPU). This is fine for all")
        print("experiments in this lab -- the networks used are small.")

    print("\nEnvironment verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
