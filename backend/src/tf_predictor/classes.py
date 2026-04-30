import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from src.core.types import TFType
from src.core.config import *
from src.core.utils import *

class TFParamDataset(Dataset):
    def __init__(self, dir, normalize=True):
        self.X_step = np.load(f"{dir}/X_step.npy")
        self.X_Tmax = np.load(f"{dir}/X_Tend.npy")
        self.Y = np.load(f"{dir}/Y_params.npy")

        if normalize:
            # Normalize X per sample (shape-based)
            self.X_step = self.X_step / (
                np.max(np.abs(self.X_step), axis=1, keepdims=True) + 1e-8
            )

            # Normalize parameters (important!)
            Y_mean = self.Y.mean(axis=0)
            Y_std = self.Y.std(axis=0) + 1e-8
            self.Y = (self.Y - Y_mean) / Y_std

        self.X_step = torch.tensor(self.X_step, dtype=torch.float32)
        self.X_Tmax = torch.tensor(self.X_Tmax, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_step)

    def __getitem__(self, idx):
        return self.X_step[idx], self.X_Tmax[idx], self.Y[idx]

class TFRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, tftype: TFType):
        super().__init__()
        # Additional feature for the time scale of the step response
        self.tftype = tftype
        # self.output_dim = output_dim
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim + 1, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, output_dim),
        #     nn.Softplus(),
        # )
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # t_end injected late, after feature extraction
        self.head = nn.Sequential(
            nn.Linear(64 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Softplus()
        )

    def forward(self, step_response, time_end):
        features = self.backbone(step_response)
        out = self.head(torch.cat([features, time_end.unsqueeze(1)], dim=1))

        # time_end = time_end.unsqueeze(1)  # Ensure time_end has shape (batch_size, 1)
        # x = torch.cat((step_response, time_end), dim=1)
        # out = self.net(x)
        
        # For ORD2_PER, clamp the damping ratio (3rd parameter) to [0, 1] using tanh
        if self.tftype == TFType.ORD2_PER:
            out[:, 2] = torch.tanh(out[:, 2])
        
        return out

class CustomLoss(nn.Module):
    """
    Custom loss TF regression models. Uses L1Loss multiplied by computed step response difference.
    """
    def __init__(self, tftype: TFType, reduction='mean'):
        super().__init__()
        # use reduction='none' if you need per-element values
        # self.base_loss = nn.L1Loss(reduction='none')
        self.reduction = reduction
        self.tftype = tftype
        self.penalty_mult = 0.1

        self.register_buffer(
            "t",
            torch.linspace(0, 1, TIME_POINTS)
        )

    def forward(self, pred, target, t_end):
        # loss = self.base_loss(pred, target)
        # params_loss = torch.abs(pred - target) / (torch.abs(target).clamp(min=1e-6)) # Relative error for params
        eps = torch.abs(target).mean(dim=0, keepdim=True).clamp(min=1e-3)  # per-param scale from batch
        params_loss = torch.abs(pred - target) / eps

        # custom computation per prediction/target pair
        t = self.t.unsqueeze(0) * t_end.unsqueeze(1)
        sr_pred = step_response_of_type(self.tftype, pred, t)
        sr_target = step_response_of_type(self.tftype, target, t)


        # time_weights = torch.exp(-2 * self.t / self.t[-1])  # (T,) decaying weights
        # time_weights = time_weights / time_weights.sum()     # normalize
        
        # sr_max = torch.abs(sr_target).max(dim=1, keepdim=True).values
        # penalty = torch.sum(torch.abs(sr_pred - sr_target) / sr_max, dim=1)

        sr_loss = torch.mean(torch.abs(sr_pred - sr_target) / (torch.abs(sr_target).clamp(min=1e-6)), dim=1) # Point-wise relative error
        # penalty = torch.tanh(penalty / 100) * self.penalty_mult
        sr_loss = sr_loss * self.penalty_mult
        sr_loss = sr_loss.unsqueeze(1).expand_as(params_loss)
        # print(penalty)
        # input()

        # loss = loss * (1 + penalty)
        loss = params_loss + sr_loss
        
        # Check for NaN and report which sample caused it
        if torch.isnan(loss).any():
            print(f"WARNING: NaN detected in loss computation!")
            print(f"  sr_pred range: [{sr_pred.min():.3e}, {sr_pred.max():.3e}]")
            print(f"  sr_target range: [{sr_target.min():.3e}, {sr_target.max():.3e}]")
            print(f"  pred range: [{pred.min():.3e}, {pred.max():.3e}]")
            print(f"  target range: [{target.min():.3e}, {target.max():.3e}]")
            print(f"  penalty range: [{sr_loss.min():.3e}, {sr_loss.max():.3e}]")

        # apply final reduction
        if self.reduction == 'mean':
            # return loss.mean()
            return loss.mean(dim=0).sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
