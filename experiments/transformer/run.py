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


def _add_features(df_):
    df = df_.copy()
    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["RC"] = df["R"] + "_" + df["C"]
    df["tmp0"] = 1

    df["corss"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * (1 - df["u_out"])
    df["cross3"] = df["time_step"] * df["u_out"]

    grp_by = df.groupby("breath_id")
    df["time_delta"] = grp_by["time_step"].diff(1).fillna(0.)
    df["time_step_cumsum"] = grp_by["time_step"].cumsum()
    df["u_in_cumsum"] = grp_by["u_in"].cumsum()
    df["count"] = grp_by["tmp0"].cumsum()

    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    # lag
    for n_lag in range(1, 6):
        df[f"u_in_lag_b{n_lag}"] = grp_by["u_in"].shift(n_lag).fillna(0.)
        df[f"u_out_lag_b{n_lag}"] = grp_by["u_out"].shift(n_lag).fillna(0.)
    for n_lag in range(1, 6):
        df[f"u_in_lag_f{n_lag}"] = grp_by["u_in"].shift(-n_lag).fillna(0.)
        df[f"u_out_lag_f{n_lag}"] = grp_by["u_out"].shift(-n_lag).fillna(0.)

    # diff
    for n_diff in range(1, 6):
        df[f"u_in_diff_b{n_diff}"] = grp_by["u_in"].diff(n_diff).fillna(0.)
        df[f"u_out_diff_b{n_diff}"] = grp_by["u_out"].diff(n_diff).fillna(0.)
    for n_diff in range(1, 6):
        df[f"u_in_diff_f{n_diff}"] = grp_by["u_in"].diff(-n_diff).fillna(0.)
        df[f"u_out_diff_f{n_diff}"] = grp_by["u_out"].diff(-n_diff).fillna(0.)

    # window
    cols_list = (
        ["u_in"] + [f"u_in_lag_b{n_lag}" for n_lag in range(1, 6)],  # back
        list(reversed([f"u_in_lag_f{n_lag}" for n_lag in range(1, 6)]))
        + ["u_in"],  # front
        list(reversed([f"u_in_lag_f{n_lag}" for n_lag in range(1, 3)]))
        + ["u_in"]
        + [f"u_in_lag_b{n_lag}" for n_lag in range(1, 6)],  # center
    )
    for cols, prefix in zip(cols_list, ("b", "f", "c")):
        for lam in ["mean", "max", "min", "std"]:
            df[f"u_in_{prefix}window_{lam}"] = getattr(np, lam)(df[cols].values, axis=1)

    weights1 = np.array([(2 / (len(cols_list[0]) + 1)) ** (i + 1) for i in range(len(cols_list[0]))])
    weights1 /= np.sum(weights1)
    weights2 = np.array([(2 / (len(cols_list[-1]) + 1)) ** (i + 1) for i in range(len(cols_list[-1]))])
    weights2 /= np.sum(weights2)
    for cols, weights, prefix in zip(cols_list, (weights1, weights1, weights2), ("b", "f", "c")):
        df[f"u_in_{prefix}window_ewm"] = np.dot(df[cols].values, weights)

    # window x u_in
    for prefix in ("b", "f", "c"):
        for lam in ["mean", "max", "min"]:
            df[f"u_in_{prefix}window_{lam}_diff"] = (
                df["u_in"] - df[f"u_in_{prefix}window_{lam}"]
            )

    df["u_in_diff_sign"] = np.sign(df["u_in_diff_b1"])

    df["tmp1"] = df["time_delta"] * df["u_in"]
    df["tmp2"] = df["time_delta"] * ((1 - df["u_out"]) * df["u_in"])

    grp_by = df.groupby("breath_id")
    df["u_in_diff_change"] = (
        np.sign(grp_by["u_in_diff_sign"].diff(1).fillna(0)) != 0
    ).astype(int)
    df["area"] = grp_by["tmp1"].cumsum()
    df["area_insp"] = grp_by["tmp2"].cumsum()

    df.drop(["tmp0", "tmp1", "tmp2"], axis=1, inplace=True)
    return df


def calc_stats(df_, n_timesteps):
    first_df = df_.loc[0::n_timesteps]
    last_df = df_.loc[n_timesteps - 1::n_timesteps]

    df = pd.DataFrame(
        {"breath_id": first_df["breath_id"].values, "RC": first_df["RC"].values, "R": first_df["R"], "C": first_df["C"]}
    )
    df["area_insp_last"] = last_df["area_insp"].values
    df["total_time"] = last_df["time_step"].values

    grp_by = df_.groupby("breath_id")
    for lam in ["max", "mean", "std"]:
        df[f"u_in_{lam}"] = df["breath_id"].map(
            getattr(grp_by["u_in"], lam)().to_dict()
        )

    for lam in ["max", "mean"]:
        df[f"area_{lam}"] = df["breath_id"].map(
            getattr(grp_by["area"], lam)().to_dict()
        )
        df[f"area_insp_{lam}"] = df["breath_id"].map(
            getattr(grp_by["area_insp"], lam)().to_dict()
        )

    df["vibs"] = df["breath_id"].map(grp_by["u_in_diff_change"].sum().to_dict())
    df = pd.get_dummies(df)

    return df


def add_features(df_, is_debug, cachedir, prefix, n_timesteps):
    filepath = cachedir / f"{prefix}_lstm-less-cut-addfeatures_debug{is_debug}.csv"

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return reduce_mem_usage(df)

    df = df_.copy()
    df = _add_features(df)
    df_stats = calc_stats(df, n_timesteps=n_timesteps)
    df_stats = df_stats.set_index("breath_id")
    cols = df_stats.columns
    for c in cols:
        df[c] = df.breath_id.map(df_stats[c].to_dict())

    df["norm_time_step"] = df["time_step"] / df["total_time"]
    df.drop(["total_time"], axis=1, inplace=True)

    for lam in ["max", "mean"]:
        df[f"u_in_{lam}_diff"] = df["u_in"] - df[f"u_in_{lam}"]
        df[f"area_{lam}_diff"] = df["area"] - df[f"area_{lam}"]
        df[f"area_insp_{lam}_diff"] = df["area_insp"] - df[f"area_insp_{lam}"]

    df.drop(
        ["id", "RC", "R", "C", "breath_id", "u_in_diff_sign", "u_in_diff_change"],
        axis=1,
        inplace=True,
    )
    df = reduce_mem_usage(df)
    df.to_csv(filepath, index=False)

    return df


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

        encoders = []
        for i in range(n_encoders):
            encoder_layer = TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=n_head,
                dim_feedforward=n_hidden,
                dropout=dropout,
                activation="relu",
            )
            encoder_norm = LayerNorm(emb_dim)

            transformer_encoder = TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_layers, norm=encoder_norm
            )
            encoders.append(transformer_encoder)
        self.encoders = nn.Sequential(*encoders)

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
        x = self.encoders(x)
        if self.fc is not None:
            x = self.fc(x)
        y = self.output(x)

        y = y.permute(1, 0, 2)
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
    cachedir = basedir / ".." / ".." / "cache"
    os.makedirs(logdir, exist_ok=True)

    config.to_json(logdir / "config.json")
    train_df, test_df, submission_df = fetch_custom_data(datadir=datadir, n_splits=config.n_splits)
    train_df["count"], test_df["count"] = (np.arange(train_df.shape[0]) % 80).astype(int), (np.arange(test_df.shape[0]) % 80).astype(int)
    train_df = train_df[train_df["count"] < config.cut].reset_index(drop=True)
    test_preds_idx = test_df["count"] < config.cut
    test_df = test_df[test_preds_idx].reset_index(drop=True)
    test_df["pressure"] = 0

    if config.debug:
        train_df = train_df[: config.cut * 1000]
        test_df = test_df[: config.cut * 1000]

    train_df = add_features(train_df, config.debug, cachedir, "train", n_timesteps=config.cut)
    test_df = add_features(test_df, config.debug, cachedir, "test", n_timesteps=config.cut)

    kfolds = train_df.iloc[0::config.cut]['kfold'].values

    features = list(train_df.drop(["kfold", "pressure"], axis=1).columns)
    cont_features = [f for f in features if ("RC_" not in f) and ("R_" not in f) and ("C_" not in f) and ("u_out" not in f)]

    train_data, test_data = train_df[features], test_df[features]
    RS = RobustScaler()
    train_df[cont_features] = RS.fit_transform(train_df[cont_features])
    test_df[cont_features] = RS.transform(test_df[cont_features])
    train_data, test_data = train_df[features].values, test_df[features].values

    train_data = train_data.reshape(-1, config.cut, train_data.shape[-1])
    targets = train_df[["pressure"]].to_numpy().reshape(-1, config.cut)
    test_data = test_data.reshape(-1, config.cut, test_data.shape[-1])


    valid_preds = np.empty_like(targets)
    test_preds = []
    test_dataset = VentilatorDataset(X=test_data, u_outs=test_data[:, :, features.index("u_out")])
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # for fold, (train_idx, test_idx) in enumerate(kf.split(train_data, targets)):
    for fold in range(config.n_splits):
        train_idx, test_idx = (kfolds != fold), (kfolds == fold)
        print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
        savedir = logdir / f"fold{fold}"
        os.makedirs(savedir, exist_ok=True)

        X_train, X_valid = train_data[train_idx], train_data[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]
        u_out_train, u_out_valid = train_data[train_idx, :, features.index("u_out")], train_data[test_idx, :, features.index("u_out")]

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
            n_features=len(features), **config.transformer
        ).to(config.device)
        optimizer = Adam(model.parameters(), lr=config.lr)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, **config.scheduler)
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

            log_str = f"Epoch {epoch + 1:03d}/{config.epochs:03d} t={elapsed_time:.0f}s " + \
                f"loss={avg_loss:.3f} val_loss={avg_val_loss:.3f} score={val_metric_score:.3f} "

            log["epoch"].append(epoch)
            log["loss"].append(avg_loss)
            log["val_loss"].append(avg_val_loss)
            log["score_loss"].append(val_metric_score)

            if best_score > val_metric_score:
                torch.save(model.state_dict, str(savedir / "weights_best.pt"))
                log_str += f"score improved from {best_score:.3f} to {val_metric_score:.3f}"
                best_score = val_metric_score

            print(log_str)

            scheduler.step(val_metric_score)

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

    pd.DataFrame(valid_preds).to_csv(logdir / "valid_preds.csv")

    if not config.debug:
        submission_df.loc[test_preds_idx, "pressure"] = np.median(test_preds, axis=0)
        submission_df.to_csv(logdir / "submission.csv", index=False)

    shutil.copyfile(Path(__file__), logdir / 'script.py')


if __name__ == "__main__":
    cnf_file = sys.argv[1]
    cfg_file_path = Path(__file__).resolve().parent / cnf_file
    with open(cfg_file_path, "rb") as f:
        config = json.load(f)

    main(config=config)
