"""
Experiment 9: TensorFlow and Keras-Based Neural Network for RL
------------------------------------------------------------------
Aim: To build a simple feed-forward neural network using TensorFlow and Keras
for approximating value functions in Reinforcement Learning environments.

Note on hardware: this network is tiny (two 64-unit dense layers), so it
trains almost instantly on CPU. GPU memory-growth is enabled below in case
TensorFlow detects your RTX 3050, but a GPU is not required for this script.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- GPU configuration (safe no-op if no GPU is present) ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected, memory growth enabled: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected -- running on CPU (this network is small, so that's fine).")

np.random.seed(42)
tf.random.set_seed(42)


def build_value_network(input_dim, hidden_units=(64, 64)):
    """Feed-forward network approximating a state-value function V(s)."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units[0], activation="relu"),
        layers.Dense(hidden_units[1], activation="relu"),
        layers.Dense(1, activation="linear"),  # outputs V(s)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def generate_synthetic_data(n_samples=3000, n_features=4):
    """
    Simulate (state, true_value) pairs for a toy control problem whose state
    layout mirrors CartPole (cart position, velocity, pole angle, angular
    velocity), together with a synthetic 'true' value function to approximate.
    """
    X = np.random.uniform(-1, 1, size=(n_samples, n_features))
    y = 10 - 5 * X[:, 2] ** 2 - 2 * X[:, 0] ** 2 - X[:, 1] * X[:, 3]
    return X.astype(np.float32), y.astype(np.float32)


if __name__ == "__main__":
    X, y = generate_synthetic_data(n_samples=3000)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_value_network(input_dim=4)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50, batch_size=32, verbose=1,
    )

    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal test MSE: {test_loss:.4f}")

    sample_preds = model.predict(X_test[:5], verbose=0).flatten()
    print("\nSample predictions vs true values:")
    for pred, true in zip(sample_preds, y_test[:5]):
        print(f"  Predicted V(s) = {pred:6.3f}  |  True V(s) = {true:6.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Value Function Approximation -- Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment9_value_network_training.png", dpi=150)
    print("\nPlot saved as 'experiment9_value_network_training.png'")
    plt.show()
