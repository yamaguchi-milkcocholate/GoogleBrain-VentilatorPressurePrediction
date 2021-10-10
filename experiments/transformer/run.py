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
from torch.nn import L1Loss


from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

import sys

print(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils import (
    seed_every_thing,
    fetch_data,
    Config,
    plot_metric,
    compute_metric,
    VentilatorDataset,
    PositionalEncoding,
)


def add_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["one"] = 1
    df["count"] = (df["one"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = np.select(
        [df["breath_id_lag"] == df["breath_id"]], [1], 0
    )
    df["breath_id_lag2same"] = np.select(
        [df["breath_id_lag2"] == df["breath_id"]], [1], 0
    )
    df["u_in_lag"] = df["u_in"].shift(1).fillna(0)
    df["u_in_lag"] = df["u_in_lag"] * df["breath_id_lagsame"]
    df["u_in_lag2"] = df["u_in"].shift(2).fillna(0)
    df["u_in_lag2"] = df["u_in_lag2"] * df["breath_id_lag2same"]
    df["u_out_lag2"] = df["u_out"].shift(2).fillna(0)
    df["u_out_lag2"] = df["u_out_lag2"] * df["breath_id_lag2same"]

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["RC"] = df["R"] + df["C"]
    df = pd.get_dummies(df)
    if is_train:
        df["pressure"] = (1 - df["u_out"]) * df["pressure"]
    return df


class TransformerModel(nn.Module):
    """Transformer Model
    Args:
        n_features: num of features(#input dims)
        n_head: num of Multi-head Attentions
        n_hidden: num of feedforward network's dim
        n_layers: num of sub-encoder-layers in the encoder
        dropout: probability of dropout
        fc_layer_units: num of units of each full-connection layers
    """

    def __init__(
        self,
        n_features: int,
        n_head: int,
        n_hidden: int,
        n_layers: int,
        dropout: float,
        fc_layer_units: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.pos_encoder = PositionalEncoding(n_features=n_features, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=n_features,
            nhead=n_head,
            dim_feedforward=n_hidden,
            dropout=dropout,
            activation="relu",
        )
        encoder_norm = LayerNorm(n_features)
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
                    fc_layers.append(nn.Dropout(dropout))
                else:
                    fc_layers.append(nn.Linear(n_features, n_units))
                    fc_layers.append(nn.Dropout(dropout))
            self.fc = nn.Sequential(*fc_layers)
            self.output = nn.Linear(fc_layer_units[-1], 1)
        else:
            self.fc = None
            self.output = nn.Linear(n_features, 1)
        self._reset_parameters()

    def forward(self, x):
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        if self.fc is not None:
            x = self.fc(x)
        y = self.output(x)
        return y

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


def main(config: Dict[str, Any]):
    config = Config().update(config)
    seed_every_thing(seed=config.seed)

    basedir = Path(__file__).resolve().parent
    datadir = basedir / ".." / ".." / "data"
    logdir = basedir / ".." / ".." / "logs" / config.dirname
    os.makedirs(logdir, exist_ok=True)

    config.to_json(logdir / "config.json")
    train_df, test_df, submission_df = fetch_data(datadir=datadir)

    if config.debug:
        train_df = train_df[: 80 * 100]
        test_df = test_df[: 80 * 100]

    train_df = add_features(df=train_df)
    test_df = add_features(df=test_df, is_train=False)

    features = list(
        train_df.drop(
            [
                "pressure",
                "id",
                "breath_id",
                "one",
                "count",
                "breath_id_lag",
                "breath_id_lag2",
                "breath_id_lagsame",
                "breath_id_lag2same",
                "u_out_lag2",
            ],
            axis=1,
        ).columns
    )

    train_data, test_data = train_df[features], test_df[features]
    RS = RobustScaler()
    train_data = RS.fit_transform(train_data)
    test_data = RS.transform(test_data)

    train_data = train_data.reshape(-1, 80, train_data.shape[-1])
    targets = train_df[["pressure"]].to_numpy().reshape(-1, 80)
    train_u_outs = train_df[["u_out"]].to_numpy().reshape(-1, 80)
    test_data = test_data.reshape(-1, 80, test_data.shape[-1])
    test_u_outs = test_df[["u_out"]].to_numpy().reshape(-1, 80)

    if len(features) % config.transformer["n_head"] == 0:
        emb_dim = len(features)
    else:
        emb_dim = config.transformer["n_head"] * (len(features) // config.transformer["n_head"] + 1)
    _train_data, _test_data = (
        np.zeros((train_data.shape[0], 80, emb_dim)),
        np.zeros((test_data.shape[0], 80, emb_dim)),
    )
    _train_data[:, :, : train_data.shape[-1]] = train_data
    _test_data[:, :, : test_data.shape[-1]] = test_data
    train_data, test_data = _train_data, _test_data
    del _train_data, _test_data
    gc.collect()

    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    valid_preds = np.empty_like(targets)
    test_preds = []
    test_dataset = VentilatorDataset(X=test_data, u_outs=test_u_outs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_data, targets)):
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
        savedir = logdir / f"fold{fold}"
        os.makedirs(savedir, exist_ok=True)

        X_train, X_valid = train_data[train_idx], train_data[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        u_out_train, u_out_valid = train_u_outs[train_idx], train_u_outs[test_idx]

        train_dataset = VentilatorDataset(X=X_train, u_outs=u_out_train, y=y_train)
        valid_dataset = VentilatorDataset(X=X_valid, u_outs=u_out_valid, y=y_valid)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = TransformerModel(
            n_features=emb_dim, **config.transformer
        ).to(config.device)
        optimizer = Adam(model.parameters(), lr=config.lr)
        criterion = L1Loss()

        best_score = np.inf
        log = {nm: [] for nm in ["loss", "val_loss", "score_loss", "epoch"]}

        for epoch in range(config.epochs):
            model.train()
            model.zero_grad()
            start_time = time.time()

            avg_loss = 0
            avg_metric_score = 0
            for data in train_loader:
                pred = model(data["input"].to(config.device)).squeeze(-1)

                loss = criterion(pred, data["target"].to(config.device)).mean()
                loss.backward()
                avg_loss += loss.item() / len(train_loader)

                metric_score = compute_metric(
                    y_trues=data["target"].detach().cpu().numpy(),
                    y_preds=pred.detach().cpu().numpy(),
                    u_outs=data["u_out"].detach().cpu().numpy(),
                ).mean()
                avg_metric_score += metric_score / len(train_loader)

                optimizer.step()

                for param in model.parameters():
                    param.grad = None

            model.eval()
            val_metric_score, avg_val_loss = 0, 0
            preds = []

            with torch.no_grad():
                for data in valid_loader:
                    pred = model(data["input"].to(config.device)).squeeze(-1)

                    loss = criterion(
                        pred.detach(), data["target"].to(config.device)
                    ).mean()
                    avg_val_loss += loss.item() / len(valid_loader)
                    preds.append(pred.detach().cpu().numpy())

            preds = np.concatenate(preds, 0)
            val_metric_score = compute_metric(
                y_trues=valid_dataset.targets, y_preds=preds, u_outs=valid_dataset.u_outs
            ).mean()

            elapsed_time = time.time() - start_time
            print(
                f"Epoch {epoch + 1:03d}/{config.epochs:03d} t={elapsed_time:.0f}s "
                f"loss={avg_loss:.3f} val_loss={avg_val_loss:.3f} score={val_metric_score:.3f}",
            )

            log["epoch"].append(epoch)
            log["loss"].append(avg_loss)
            log["val_loss"].append(avg_val_loss)
            log["score_loss"].append(val_metric_score)

            if best_score > val_metric_score:
                torch.save(model.state_dict, str(savedir / "weights_best.pt"))
                print(f"score improved from {best_score:.3f} to {val_metric_score:.3f}")
                best_score = val_metric_score
        log = pd.DataFrame(log)
        log.to_csv(savedir / "log.csv")
        plot_metric(filepath=savedir / 'log.csv', metric='loss')
        torch.save(model.state_dict(), str(savedir / "weights_final.pt"))

        model.eval()

        preds = []
        with torch.no_grad():
            for data in valid_loader:
                pred = model(data["input"].to(config.device)).squeeze(-1)
                preds.append(pred.detach().cpu().numpy())
        preds = np.concatenate(preds, 0)
        valid_preds[test_idx, :] = preds

        preds = []
        with torch.no_grad():
            for data in test_loader:
                pred = model(data["input"].to(config.device)).squeeze(-1)
                preds.append(pred.detach().cpu().numpy())
        preds = np.concatenate(preds, 0).reshape(-1)
        test_preds.append(preds)

        del (
            X_train,
            X_valid,
            y_train,
            y_valid,
            u_out_train,
            u_out_valid,
            train_dataset,
            valid_dataset,
            model,
        )
        gc.collect()
        with torch.cuda.device(config.device):
            torch.cuda.empty_cache()
    if not config.debug:
        submission_df["pressure"] = sum(test_preds) / 5
        submission_df.to_csv(logdir / "submission.csv", index=False)

    shutil.copyfile(Path(__file__), logdir / 'script.py')


if __name__ == "__main__":
    cnf_file = sys.argv[1]
    cfg_file_path = Path(__file__).resolve().parent / cnf_file
    with open(cfg_file_path, "rb") as f:
        config = json.load(f)

    main(config=config)
