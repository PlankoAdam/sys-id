from src.core.config import *
from src.core.types import TFType
import matplotlib.pyplot as plt
import numpy as np
import os

def eval_model(model_type: TFType):
    # MODEL_PATH = f"{MODELS_DIR}/tf_predictor_{model_type.name}.pt"
    METRICS_PATH = f"{METRICS_DIR}/tf_predictor_{model_type.name}_metrics.npz"

    metrics = np.load(METRICS_PATH)
    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    training_time = metrics["training_time"]
    # train_accs = metrics["train_accs"]
    # val_accs = metrics["val_accs"]

    epochs = len(train_losses)
    epochs_range = range(1, epochs + 1)

    # -------------------------
    # Plot loss & accuracy
    # -------------------------
    # plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_type.name}\nTraining time: {training_time}s")
    plt.legend()
    plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, train_accs, label="Train Acc")
    # plt.plot(epochs_range, val_accs, label="Val Acc")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy vs Epoch")
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    plt.show()

def eval_all(osc: bool = False):
    rows = 2
    cols = int(np.ceil(len(TFType)/rows))
    
    for i,t in enumerate(TFType):
        metrics_path = f"{OSC_DIR if osc else STEP_DIR}/{METRICS_DIR}/tf_predictor_{t.name}_metrics.npz"

        if not os.path.exists(metrics_path):
            print(f"Metrics file not found for {t.name}, skipping...")
            continue

        metrics = np.load(metrics_path)
        train_losses = metrics["train_losses"]
        val_losses = metrics["val_losses"]
        training_time = metrics["training_time"]

        epochs_range = range(1, len(train_losses) + 1)

        plt.subplot(rows, cols, i+1)
        plt.plot(epochs_range, train_losses, label="Train Loss")
        plt.plot(epochs_range, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{t.name}\nTraining time: {training_time: .1f}s")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()