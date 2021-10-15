from random import seed
import numpy as np
import pandas as pd
import json
import os
import sys
import gc
import shutil
from pprint import pprint
from pathlib import Path
from typing import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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
    reduce_tf_gpu_memory,
)


def _add_features(df_):
    df = df_.copy()
    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["RC"] = df["R"] + "_" + df["C"]

    df["u_in_insp"] = df["u_in"] * (1 - df["u_out"])
    df["u_in_zero"] = (df["u_in"] == 0).astype(int)
    grp_by = df.groupby("breath_id")
    df["time_delta"] = grp_by["time_step"].diff(1).fillna(0.0)
    df["u_in_lag1"] = grp_by["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = grp_by["u_in"].shift(2).fillna(0.0)
    df["spike"] = df["u_in_lag1"] - (0.5 * df["u_in"] + 0.5 * df["u_in_lag2"])
    df["spike_size"] = df["spike"] ** 2
    df["u_in_diff"] = grp_by["u_in"].diff(1).fillna(0.0)
    df["u_in_diff_sign"] = np.sign(df["u_in_diff"])
    df["u_in_zero_diff"] = grp_by["u_in_zero"].diff(1).fillna(0.0)
    df["u_in_zero_start"] = (df["u_in_zero_diff"] == 1).astype(int)

    df["tmp1"] = df["time_delta"] * df["u_in"]
    df["tmp2"] = df["time_delta"] * df["u_in_insp"]

    grp_by = df.groupby("breath_id")
    df["u_in_norm_diff"] = df["u_in_diff"] / (
        df.breath_id.map(grp_by["u_in_diff"].mean().to_dict()) + 1e-8
    )
    df["u_in_diff_change"] = (
        np.sign(grp_by["u_in_diff_sign"].diff(1).fillna(0)) != 0
    ).astype(int)
    df["u_in_insp_diff_change"] = df["u_in_diff_change"] * (1 - df["u_out"])
    df["area"] = grp_by["tmp1"].cumsum()
    df["area_insp"] = grp_by["tmp2"].cumsum()

    df["first"], df["last"] = 0, 0
    df.loc[0::80, "first"] = 1
    df.loc[79::80, "last"] = 1

    df["tmp3"] = df["u_out"].diff(1).fillna(0)
    df["u_out_change"] = (df["tmp3"] == 1).astype(int)

    df.drop(
        ["u_in_zero_diff", "u_in_zero", "tmp1", "tmp2", "tmp3"], axis=1, inplace=True
    )
    return df


def calc_stats(df_):
    first_df = df_.loc[0::80]
    last_df = df_.loc[79::80]
    u_out_change_df = df_[df_.u_out_change == 1]

    df = pd.DataFrame(
        {"breath_id": first_df["breath_id"].values, "RC": first_df["RC"].values}
    )
    df["u_in_first"] = first_df["u_in"].values
    df["u_in_last"] = last_df["u_in"].values
    df["area_last"] = last_df["area"].values
    df["area_insp_last"] = last_df["area_insp"].values
    df["total_time"] = last_df["time_step"].values
    df["u_in_at_u_out_change"] = u_out_change_df["u_in"].values

    df["p_first"] = first_df["pressure"].values
    df["p_last"] = last_df["pressure"].values
    df["p_at_u_out_change"] = u_out_change_df["pressure"].values

    grp_by = df_.groupby("breath_id")
    for lam in ["max", "min", "mean", "std"]:
        df[f"u_in_{lam}"] = df["breath_id"].map(
            getattr(grp_by["u_in"], lam)().to_dict()
        )
        df[f"u_in_insp_{lam}"] = df["breath_id"].map(
            getattr(grp_by["u_in_insp"], lam)().to_dict()
        )
        df[f"spike_size_{lam}"] = df["breath_id"].map(
            getattr(grp_by["spike_size"], lam)().to_dict()
        )

    for lam in ["max", "min"]:
        df[f"p_{lam}"] = df["breath_id"].map(
            getattr(grp_by["pressure"], lam)().to_dict()
        )

    df["vibs"] = df["breath_id"].map(grp_by["u_in_diff_change"].sum().to_dict())
    df["vibs_insp"] = df["breath_id"].map(
        grp_by["u_in_insp_diff_change"].sum().to_dict()
    )

    df["num_zero_start"] = df["breath_id"].map(
        grp_by["u_in_zero_start"].sum().to_dict()
    )
    df["zero_start_time"] = last_df["time_step"].values
    df["zero_start_time"] = df["breath_id"].map(
        df_[df_["u_in_zero_start"] == 1]
        .groupby("breath_id")["time_step"]
        .first()
        .to_dict()
    )
    df["zero_start_time"] = np.where(
        df.zero_start_time.isnull(),
        last_df["time_step"].values,
        df["zero_start_time"].values,
    )

    df["u_in_last_cluster"] = 0
    df.loc[(df.u_in_last > 4.965) & (df.u_in_last < 4.980), "u_in_last_cluster"] = 1
    df.loc[(df.u_in_last > 4.98) & (df.u_in_last < 5), "u_in_last_cluster"] = 2
    df["u_in_last_cluster"] = df["u_in_last_cluster"].astype(str)
    df = pd.get_dummies(df)

    return df


def add_features(df_: pd.DataFrame) -> pd.DataFrame:
    df = df_.copy()

    df = _add_features(df)
    df_stats = calc_stats(df)
    df_stats = df_stats.set_index("breath_id")
    cols = df_stats.drop(
        ["p_first", "p_last", "p_at_u_out_change", "p_max", "p_min"], axis=1
    ).columns
    for c in cols:
        df[c] = df.breath_id.map(df_stats[c].to_dict())

    for lam in ["min", "max", "mean"]:
        for c in ["u_in", "u_in_insp", "spike_size"]:
            df[f"{c}_{lam}_diff"] = df[c] - df[f"{c}_{lam}"]

        df[f"u_in_cross_{lam}"] = df["u_in"] - df[f"u_in_insp_{lam}"]
        df[f"u_in_insp_cross_{lam}"] = df["u_in_insp"] - df[f"u_in_{lam}"]

        df["u_in_first_diff"] = df["u_in"] - df["u_in_first"]
        df["u_in_insp_first_diff"] = df["u_in_insp"] - df["u_in_first"]
        df["u_in_last_diff"] = df["u_in"] - df["u_in_last"]
        df["u_in_insp_last_diff"] = df["u_in_insp"] - df["u_in_last"]

    df["time_to_zero_start"] = df["zero_start_time"] - df["time_step"]

    df.drop(
        [
            "id",
            "breath_id",
            "u_in_diff_sign",
            "u_in_diff_change",
            "u_in_insp_diff_change",
            "first",
            "last",
            "u_out_change",
            "u_in_zero_start",
        ],
        axis=1,
        inplace=True,
    )

    return df


def build_model(config: Config, n_features) -> keras.models.Sequential:
    model = keras.models.Sequential([keras.layers.Input(shape=(80, n_features))])
    for n_unit in config.n_units:
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(n_unit, return_sequences=True, dropout=0)
            )
        )
    model.add(keras.layers.Dense(50, activation="selu"))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.lr), loss="mae")
    return model


def main(config: Dict[str, Any]):
    config = Config().update(config)
    seed_every_thing(seed=config.seed)
    reduce_tf_gpu_memory(gpu_id=config.gpu_id)

    basedir = Path(__file__).resolve().parent
    datadir = basedir / ".." / ".." / "data"
    logdir = basedir / ".." / ".." / "logs" / config.dirname
    os.makedirs(logdir, exist_ok=True)

    config.to_json(logdir / "config.json")
    train_df, test_df, submission_df = fetch_data(datadir=datadir)
    test_df["pressure"] = 0

    if config.debug:
        train_df = train_df[: 80 * 100]
        test_df = test_df[: 80 * 100]

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    features = list(train_df.drop(["pressure"], axis=1).columns)
    pprint(features)
    print(len(features))

    train_data, test_data = train_df[features], test_df[features]
    RS = RobustScaler()
    train_data = RS.fit_transform(train_data)
    test_data = RS.transform(test_data)

    train_data = train_data.reshape(-1, 80, train_data.shape[-1])
    targets = train_df[["pressure"]].to_numpy().reshape(-1, 80)
    test_data = test_data.reshape(-1, 80, test_data.shape[-1])

    with tf.device(f"/GPU:{config.gpu_id}"):
        kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
        valid_preds = np.empty_like(targets)
        test_preds = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(train_data, targets)):
            print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
            savedir = logdir / f"fold{fold}"
            os.makedirs(savedir, exist_ok=True)

            X_train, X_valid = train_data[train_idx], train_data[test_idx]
            y_train, y_valid = targets[train_idx], targets[test_idx]

            model = build_model(config=config, n_features=len(features))

            # es = EarlyStopping(
            #     monitor="val_loss",
            #     patience=config.es_patience,
            #     verbose=1,
            #     mode="min",
            #     restore_best_weights=True,
            # )

            check_point = ModelCheckpoint(
                filepath=savedir / "weights_best.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min",
                save_weights_only=True,
            )

            schedular = ReduceLROnPlateau(
                monitor="val_loss", mode="min", **config.schedular
            )

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=[check_point, schedular]
                # callbacks=[es, check_point, schedular]
            )
            model.save_weights(savedir / "weights_final.h5")

            pd.DataFrame(history.history).to_csv(savedir / "log.csv")
            plot_metric(filepath=savedir / "log.csv", metric="loss")

            valid_preds[test_idx, :] = np.clip(model.predict(X_valid).squeeze(), 0, 100)
            test_preds.append(
                np.clip(
                    model.predict(test_data).squeeze().reshape(-1, 1).squeeze(), 0, 100
                )
            )

            del model, X_train, X_valid, y_train, y_valid
            keras.backend.clear_session()
            gc.collect()

    pd.DataFrame(valid_preds).to_csv(logdir / "valid_preds.csv")

    if not config.debug:
        submission_df["pressure"] = sum(test_preds) / 5
        submission_df.to_csv(logdir / "submission.csv", index=False)

    shutil.copyfile(Path(__file__), logdir / "script.py")


if __name__ == "__main__":
    cnf_file = sys.argv[1]
    cfg_file_path = Path(__file__).resolve().parent / cnf_file
    with open(cfg_file_path, "rb") as f:
        config = json.load(f)

    main(config=config)
