import os
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import json
import math
from matplotlib import pyplot as plt
from pathlib import Path
from typing import *
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset


@dataclass
class Config:
    def update(self, param_dict: Dict) -> "Config":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            setattr(self, key, value)
        return self

    def to_json(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(vars(self), f)


def seed_every_thing(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fetch_data(datadir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("fetching data ...")
    train = pd.read_csv(datadir / "train.csv")
    test = pd.read_csv(datadir / "test.csv")
    submission = pd.read_csv(datadir / "sample_submission.csv")
    print("done.")

    return train, test, submission


def plot_metric(filepath, metric="loss") -> None:
    log = pd.read_csv(filepath, index_col=0)
    loss_nms = [c for c in log.columns if metric in c]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for loss_nm in loss_nms:
        ax.plot(log[loss_nm], label=loss_nm)
    ax.legend()
    fig.savefig(Path(filepath).parent / "loss.png")


class VentilatorDataset(Dataset):
    def __init__(
        self, X: np.ndarray, u_outs: np.ndarray, y: Optional[np.ndarray] = None
    ) -> None:
        if y is None:
            y = np.zeros_like(u_outs)

        self.inputs = X
        self.targets = y
        self.u_outs = u_outs

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "target": torch.tensor(self.targets[idx], dtype=torch.float),
        }
        return data


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transormer Model"""

    def __init__(self, n_features: int, dropout: Optional[float] = 0.1, max_len=80):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, n_features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_features, 2).float() * (-math.log(10000.0) / n_features)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :(n_features // 2)]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def compute_metric(
    y_trues: np.ndarray, y_preds: np.ndarray, u_outs: np.ndarray
) -> np.ndarray:
    if np.any([len(series.shape) == 1 for series in [y_trues, y_preds, u_outs]]):
        # reshape (#samples, #time-steps)
        y_trues, y_preds, u_outs = (
            y_trues.reshape(-1, 80),
            y_preds.reshape(-1, 80),
            u_outs.reshape(-1, 80),
        )
    mask = 1 - u_outs
    mae = mask * np.abs(y_trues - y_preds)
    mae = mae.sum(axis=1) / mask.sum(axis=1)

    return mae
