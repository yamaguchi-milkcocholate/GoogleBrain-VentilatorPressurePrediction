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

from tensorflow import keras


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


def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def reduce_tf_gpu_memory(gpu_id: int):
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(
                "{} memory growth: {}".format(
                    device, tf.config.experimental.get_memory_growth(device)
                )
            )
    else:
        print("Not enough GPU hardware devices available")


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


def fetch_custom_data(datadir: Path, n_splits: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("fetching data ...")
    train = pd.read_csv(datadir / f"train_RC_kfold{n_splits}_seed42.csv", index_col=0)
    test = pd.read_csv(datadir / "test.csv")
    test["RC"] = test["R"].astype(str) + "_" + test["C"].astype(str)
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
        pe[:, 1::2] = torch.cos(position * div_term)[:, : (n_features // 2)]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
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


class CustomL1Loss(keras.callbacks.Callback):
    def __init__(self, X_valid, y_valid, u_outs, filepath) -> None:
        super().__init__()
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.u_outs = u_outs
        self.filepath = filepath

    def on_train_begin(self, logs={}):
        self.logs = []
        self.best_score = np.inf

    def on_epoch_end(self, epoch, logs={}):
        y_preds = self.model.predict(self.X_valid).squeeze()

        mae = compute_metric(y_trues=self.y_valid, y_preds=y_preds, u_outs=self.u_outs)
        mae = np.mean(mae)
        logs['val_custom_loss'] = mae
        self.logs.append(mae)
        print(f"customL1: {mae:.5f}")

        if self.best_score > mae:
            self.best_score = mae
            self.model.save_weights(self.filepath)
