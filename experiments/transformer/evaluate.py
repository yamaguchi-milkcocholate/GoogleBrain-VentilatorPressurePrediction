from random import seed
import numpy as np
import pandas as pd
import json
import os
import sys
import gc
import shutil
import time
from pathlib import Path
from typing import *
from scipy.stats.stats import mode

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import LayerNorm
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import L1Loss


from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import sys

print(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils import (
    seed_every_thing,
    compute_metric,
    Config,
    plot_metric,
    reduce_mem_usage,
    fetch_custom_data,
    PositionalEncoding,
    VentilatorDataset
)


class TransformerModel(nn.Module):
    """Transformer Model
    Args:
        n_features: num of features(#input dims)
        n_head: num of Multi-head Attentions
        emb_dim: num of embedding dim
        n_hidden: num of feedforward network's dim
        n_layers: num of sub-encoder-layers in the encoder
        dropout: probability of dropout
        fc_layer_units: num of units of each full-connection layers
    """

    def __init__(
        self,
        n_features: int,
        n_encoders: int,
        n_head: int,
        emb_dim: int,
        n_hidden: int,
        n_layers: int,
        dropout: float,
        fc_layer_units: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.liner_embed = nn.Linear(n_features, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim=emb_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_head,
            dim_feedforward=n_hidden,
            dropout=dropout,
            activation="relu",
        )
        encoder_norm = LayerNorm(emb_dim)

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers, norm=encoder_norm
        )

        if fc_layer_units is None:
            fc_layer_units = []

        if len(fc_layer_units) > 0:
            fc_layers = list()
            for i, n_units in enumerate(fc_layer_units):
                if i != 0:
                    fc_layers.append(nn.Linear(fc_layer_units[i - 1], n_units))
                    fc_layers.append(nn.ReLU())
                    fc_layers.append(nn.Dropout(dropout))
                else:
                    fc_layers.append(nn.Linear(emb_dim, n_units))
                    fc_layers.append(nn.ReLU())
                    fc_layers.append(nn.Dropout(dropout))
            self.fc = nn.Sequential(*fc_layers)
            self.output = nn.Linear(fc_layer_units[-1], 1)
        else:
            self.fc = None
            self.output = nn.Linear(emb_dim, 1)
        self._reset_parameters()

    def forward(self, x):
        x = x.permute(1, 0, 2)

        x = self.liner_embed(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        if self.fc is not None:
            x = self.fc(x)
        y = self.output(x)

        y = y.permute(1, 0, 2)
        return y

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


def main(config):
    config = Config().update(config)
    basedir = Path(__file__).resolve().parent
    datadir = basedir / ".." / ".." / "data"
    logdir = basedir / ".." / ".." / "logs" / config.dirname
    cachedir = basedir / ".." / ".." / "cache"

    filepath = cachedir / f"train_lstm-less-cut-addfeatures_debugFalse.csv"
    train_df = reduce_mem_usage(pd.read_csv(filepath))

    kfolds = train_df.iloc[0::config.cut]['kfold'].values

    features = list(train_df.drop(["kfold", "pressure"], axis=1).columns)
    cont_features = [f for f in features if ("RC_" not in f) and ("R_" not in f) and ("C_" not in f) and ("u_out" not in f)]

    RS = RobustScaler()
    train_df[cont_features] = RS.fit_transform(train_df[cont_features])
    train_data = train_df[features].values
    train_data = train_data.reshape(-1, config.cut, train_data.shape[-1])
    targets = train_df[["pressure"]].to_numpy().reshape(-1, config.cut)

    valid_preds = np.empty_like(targets)

    for fold in range(config.n_splits):
        train_idx, test_idx = (kfolds != fold), (kfolds == fold)
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
        savedir = logdir / f"fold{fold}"

        X_valid = train_data[test_idx]
        y_valid = targets[test_idx]
        u_out_valid = train_data[train_idx, :, features.index("u_out")]

        valid_dataset = VentilatorDataset(X=X_valid, u_outs=u_out_valid, y=y_valid)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = TransformerModel(
            n_features=len(features), **config.transformer
        )
        model.load_state_dict(torch.load(str(savedir / "weights_final.pt")))
        model.to(config.device)
        model.eval()

        preds = []
        with torch.no_grad():

            for data in valid_loader:
                pred = model(data["input"].to(config.device)).squeeze(-1)
                preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds, 0)
        valid_preds[test_idx, :] = preds

    pd.DataFrame(valid_preds).to_csv(logdir / "valid_preds.csv")


if __name__ == "__main__":
    cnf_file = sys.argv[1]
    cfg_file_path = Path(__file__).resolve().parent / cnf_file
    with open(cfg_file_path, "rb") as f:
        config = json.load(f)

    main(config=config)