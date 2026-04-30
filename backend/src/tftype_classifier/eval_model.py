import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.tftype_classifier.classes import TFTypeDataset, TFTypeClassifier
from src.core.types import TFType
from src.core.config import *

# -------------------------
# Config
# -------------------------
BATCH_SIZE = 32

def eval_model(osc: bool = False):
    parent_dir = OSC_DIR if osc else STEP_DIR
    DATASET_DIR = f"{parent_dir}/{VAL_SAVE_DIR}"
    MODEL_PATH = f"{parent_dir}/{MODELS_DIR}/tftype_classifier.pt"
    METRICS_PATH = f"{parent_dir}/{METRICS_DIR}/tftype_classifier_metrics.npz"
    # -------------------------
    # Load dataset (same split!)
    # -------------------------
    dataset = TFTypeDataset(
        dirs = [f"{DATASET_DIR}/{tftype.name}" for tftype in TFType],
        labels=[tftype.value for tftype in TFType]
    )

    val_ratio = 1.0
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # Load model
    # -------------------------
    model = TFTypeClassifier(input_dim=dataset.X.shape[1], num_classes=len(TFType))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Run validation
    # -------------------------
    y_true, y_pred = [], []
    val_loss_sum = 0

    with torch.no_grad():
        for X, y in val_loader:
            logits = model(X)
            loss = criterion(logits, y)
            val_loss_sum += loss.item()

            preds = logits.argmax(dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    val_loss = val_loss_sum / len(val_loader)
    val_acc = np.mean(np.array(y_true) == np.array(y_pred))

    print(f"Validation Loss: {val_loss:.3f}")
    print(f"Validation Accuracy: {val_acc:.3f}")

    # -------------------------
    # Load training metrics
    # -------------------------
    metrics = np.load(METRICS_PATH)
    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    train_accs = metrics["train_accs"]
    val_accs = metrics["val_accs"]
    training_time = metrics["training_time"]

    epochs = len(train_losses)
    epochs_range = range(1, epochs + 1)

    # -------------------------
    # Plot loss & accuracy
    # -------------------------
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch\nTraining time: {training_time: .1f}s")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.ylim(0.0, 1.0)
    plt.plot(epochs_range, train_accs, label="Train Acc")
    plt.plot(epochs_range, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)

    # -------------------------
    # Confusion Matrix
    # -------------------------
    ax = plt.subplot(1, 3, 3)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[tftype.name for tftype in TFType]
    )
    disp.plot(ax=plt.gca(), colorbar=False)
    plt.title("Validation Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    plt.show()
