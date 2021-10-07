import os
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import json
from matplotlib import pyplot as plt
from pathlib import Path
from typing import *
from dataclasses import dataclass


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
