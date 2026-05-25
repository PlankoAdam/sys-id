import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import numpy as np
from time import time

from src.tf_predictor.classes import *
from src.core.types import TFType
from src.core.config import *
from src.core.utils import *

def train(train_dataset_dir: str, val_dataset_dir: str, model_save_path: str, metrics_save_path: str, epochs: int = 100, batch_size: int = 128, tftype: TFType = None):
    ctx = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using Torch device: {ctx}")
    start_time = time()

    print(f"Training regression model from dataset {train_dataset_dir} ...")
    print(f"No. of epochs: {epochs}")

    train_dataset = TFParamDataset(dir=train_dataset_dir, normalize=False)
    val_dataset = TFParamDataset(dir=val_dataset_dir, normalize=False)

    # val_ratio = 0.2
    # val_size = int(len(train_dataset) * val_ratio)
    # train_size = len(train_dataset) - val_size

    # train_dataset, val_dataset = random_split(
    #     train_dataset,
    #     [train_size, val_size]
    # )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    print("Loaded dataset")
    print("Training data:")
    print(f"\tNo. of samples: \t{len(train_dataset)}")
    print(f"\tNo. of batches: \t{len(train_loader)}")
    print(f"\tBatch size:     \t{batch_size}")

    print("Validation data:")
    print(f"\tNo. of samples: \t{len(val_dataset)}")
    print(f"\tNo. of batches: \t{len(val_loader)}")
    print(f"\tBatch size:     \t{batch_size}")

    model = TFRegressor(input_dim=train_dataset.X_step.shape[1], output_dim=train_dataset.Y.shape[1], tftype=tftype).to(ctx)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = CustomLoss(tftype=tftype).to(ctx) if tftype else nn.L1Loss().to(ctx)
    torch.autograd.set_detect_anomaly(True)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # -----------------------
        # Training
        # -----------------------
        model.train()
        train_total = 0
        train_loss_sum = torch.tensor(0.0, device=ctx)

        for X_step, X_Tmax, Y in train_loader:
            X_step = X_step.to(ctx, non_blocking=True)
            X_Tmax = X_Tmax.to(ctx, non_blocking=True)
            Y = Y.to(ctx, non_blocking=True)
            optimizer.zero_grad()
            preds = model(X_step, X_Tmax)
            loss = criterion(preds, Y, X_Tmax)
            # if torch.isnan(loss):
            #     optimizer.zero_grad()
            #     continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.detach()
            train_total += len(Y)

        train_loss = (train_loss_sum / len(train_loader)).item()

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        val_total = 0
        val_loss_sum = torch.tensor(0.0, device=ctx)

        with torch.no_grad():
            for X_step, X_Tmax, Y in val_loader:
                X_step = X_step.to(ctx, non_blocking=True)
                X_Tmax = X_Tmax.to(ctx, non_blocking=True)
                Y = Y.to(ctx, non_blocking=True)
                
                preds = model(X_step, X_Tmax)
                loss = criterion(preds, Y, X_Tmax)

                val_loss_sum += loss.detach()
                preds = preds.argmax(dim=1)
                val_total += len(Y)

        val_loss = (val_loss_sum / len(val_loader)).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss {train_loss:.3f} | "
            f"Val Loss {val_loss:.3f}"
        )

    end_time = time()

    print("Training finished")
    print(f"Elapsed time: {end_time - start_time}")

    torch.save(model.state_dict(), model_save_path)

    np.savez(
        metrics_save_path,
        train_losses=train_losses,
        val_losses=val_losses,
        training_time=(end_time - start_time)
    )

    print("Saved model and metrics")

def train_type(tftype: TFType, epochs: int = 100, batch_size: int = 128, osc: bool = False):
    parent_dir = OSC_DIR if osc else STEP_DIR
    train(
        train_dataset_dir=f"{parent_dir}/{TRAIN_SAVE_DIR}/{tftype.name}",
        val_dataset_dir=f"{parent_dir}/{VAL_SAVE_DIR}/{tftype.name}",
        model_save_path=f"{parent_dir}/{MODELS_DIR}/tf_predictor_{tftype.name}.pt",
        metrics_save_path=f"{parent_dir}/{METRICS_DIR}/tf_predictor_{tftype.name}_metrics.npz",
        epochs=epochs,
        batch_size=batch_size,
        tftype=tftype
        )

def train_all(epochs: int = 100, batch_size: int = 128, osc: bool = False):
    for tftype in TFType:
        train_type(tftype, epochs, batch_size, osc=osc)