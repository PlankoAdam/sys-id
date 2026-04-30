import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

class TFTypeDataset(Dataset):
    def __init__(self, dirs, labels):
        X_list = []
        Y_list = []

        for dir, label in zip(dirs, labels):
            X = np.load(f"{dir}/X_step.npy")
            Y = np.full(len(X), label)

            X_list.append(X)
            Y_list.append(Y)

        self.X = np.vstack(X_list)
        self.Y = np.concatenate(Y_list)

        # self.X = self.X / (np.max(np.abs(self.X), axis=1, keepdims=True) + 1e-8)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class TFTypeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)
