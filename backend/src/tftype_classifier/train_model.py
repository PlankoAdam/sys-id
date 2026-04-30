from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split
from time import time

from src.tftype_classifier.classes import *
from src.core.types import TFType
from src.core.config import *

def train(epochs: int = 50, batch_size: int = 32, osc: bool = False):
    parent_dir = OSC_DIR if osc else STEP_DIR

    ctx = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using Torch device: {ctx}")
    start_time = time()

    print("Started training TFType classifier...")
    print(f"No. of epochs: {epochs}")

    # Create dataset
    train_dataset = TFTypeDataset(
        dirs = [f"{parent_dir}/{TRAIN_SAVE_DIR}/{tftype.name}" for tftype in TFType],
        labels=[tftype.value for tftype in TFType]
    )
    val_dataset = TFTypeDataset(
        dirs = [f"{parent_dir}/{VAL_SAVE_DIR}/{tftype.name}" for tftype in TFType],
        labels=[tftype.value for tftype in TFType]
    )

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

    # Model
    model = TFTypeClassifier(input_dim=train_dataset.X.shape[1], num_classes=len(TFType)).to(ctx)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(ctx)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # -----------------------
        # Training
        # -----------------------
        model.train()

        train_loss_sum = torch.tensor(0.0, device=ctx)
        train_correct  = torch.tensor(0,   device=ctx)
        
        train_total = 0

        for X, Y in train_loader:
            X = X.to(ctx, non_blocking=True)
            Y = Y.to(ctx, non_blocking=True)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, Y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.detach()                     # stays on GPU
            train_correct  += (logits.argmax(1) == Y).sum()     # stays on GPU
            train_total += len(Y)

        train_loss = (train_loss_sum / len(train_loader)).item()
        train_acc  = (train_correct / train_total).item()

        # -----------------------
        # Validation
        # -----------------------
        model.eval()

        val_loss_sum = torch.tensor(0.0, device=ctx)
        val_correct  = torch.tensor(0,   device=ctx)
        
        val_total = 0

        with torch.no_grad():
            for X, Y in val_loader:
                X = X.to(ctx)
                Y = Y.to(ctx)
                logits = model(X)
                loss = criterion(logits, Y)

                val_loss_sum += loss.detach()                     # stays on GPU
                val_correct  += (logits.argmax(1) == Y).sum()     # stays on GPU
                val_total += len(Y)

        val_loss = (val_loss_sum / len(val_loader)).item()
        val_acc = (val_correct / val_total).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss {train_loss:.3f} Acc {train_acc:.3f} | "
            f"Val Loss {val_loss:.3f} Acc {val_acc:.3f}"
        )

    end_time = time()

    print("Training finished")
    print(f"Elapsed time: {end_time - start_time}")

    torch.save(model.state_dict(), f"{parent_dir}/{MODELS_DIR}/tftype_classifier.pt")

    np.savez(
        f"{parent_dir}/{METRICS_DIR}/tftype_classifier_metrics.npz",
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        training_time=(end_time - start_time)
    )

    print("Saved model and metrics")