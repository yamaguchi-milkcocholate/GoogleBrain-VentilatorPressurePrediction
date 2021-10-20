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
    reduce_mem_usage,
    fetch_custom_data,
    CustomL1Loss,
    TransformerEncoder
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


def build_model(config: Config, n_features: int) -> keras.models.Sequential:
    n_inputs = config.cut
    transformer_params = config.transformer_params
    bilstm_params = config.bilstm_params
    conv1d_params = config.conv1d_params

    inputs = keras.Input(shape=(n_inputs, n_features))

    # Transformer Part
    # x_trans = inputs
    # x_trans = keras.layers.Dense(transformer_params["encoder_params"]["dim_emb"], activation="relu")(x_trans)
    # for _ in range(transformer_params["num_transformer_blocks"]):
    #     x_trans = TransformerEncoder(**transformer_params["encoder_params"])(x_trans)

    # for dim in transformer_params["mlp_units"]:
    #     x_trans = keras.layers.Dense(dim, activation="relu")(x_trans)
    #     x_trans = keras.layers.Dropout(transformer_params["mlp_dropout"])(x_trans)

    # Bi-LSTM Part
    x_bilstm = inputs
    for n_unit in bilstm_params["units"]:
        x_bilstm = keras.layers.Bidirectional(
            keras.layers.LSTM(
                n_unit,
                return_sequences=True,
            )
        )(x_bilstm)

    for n_unit in bilstm_params["mlp_units"]:
        x_bilstm = keras.layers.Dense(n_unit, activation="selu")(x_bilstm)
        x_bilstm = keras.layers.Dropout(bilstm_params["mlp_dropout"])(x_bilstm)

    # Conv1D Part
    x_conv = inputs
    for (n_filter, k_size, dilation_rate) in zip(conv1d_params["filters"], conv1d_params["kernel_sizes"], conv1d_params["dilation_rates"]):
        x_conv = keras.layers.Conv1D(
            filters=n_filter,
            kernel_size=k_size,
            dilation_rate=dilation_rate,
            strides=1,
            activation="relu",
            padding="valid"
        )(x_conv)
    for n_unit in conv1d_params["mlp_units"]:
        x_conv = keras.layers.Dense(n_unit)(x_conv)
        x_conv = keras.layers.Dropout(conv1d_params["mlp_dropout"])(x_conv)
    x_conv = keras.layers.GlobalMaxPooling1D()(x_conv)
    x_conv = tf.transpose(x_conv[tf.newaxis], perm=[1, 0, 2])
    x_conv = tf.tile(x_conv, tf.constant([1, n_inputs, 1], tf.int32))

    # x_concat = keras.layers.Concatenate(axis=2)([x_trans, x_bilstm, x_conv])
    x_concat = keras.layers.Concatenate(axis=2)([x_bilstm, x_conv])
    outputs = keras.layers.Dense(1)(x_concat)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.lr),
        loss='mae')
    return model


def main(config: Dict[str, Any]):
    config = Config().update(config)
    seed_every_thing(seed=config.seed)
    reduce_tf_gpu_memory(gpu_id=config.gpu_id)

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
    pprint(features)
    print(len(features))

    cont_features = [f for f in features if ("RC_" not in f) and ("R_" not in f) and ("C_" not in f) and ("u_out" not in f)]
    pprint(cont_features)

    RS = RobustScaler()
    train_df[cont_features] = RS.fit_transform(train_df[cont_features])
    test_df[cont_features] = RS.transform(test_df[cont_features])
    train_data, test_data = train_df[features].values, test_df[features].values

    train_data = train_data.reshape(-1, config.cut, train_data.shape[-1])
    targets = train_df[["pressure"]].to_numpy().reshape(-1, config.cut)
    test_data = test_data.reshape(-1, config.cut, test_data.shape[-1])

    with tf.device(f"/GPU:{config.gpu_id}"):
        valid_preds = np.empty_like(targets)
        test_preds = []

        for fold in range(config.n_splits):
            train_idx, test_idx = (kfolds != fold), (kfolds == fold)
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

            customL1 = CustomL1Loss(
                X_valid=X_valid,
                y_valid=y_valid,
                u_outs=X_valid[:, :, features.index("u_out")],
                filepath=savedir / "weights_custom_best.h5"
            )

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
                callbacks=[check_point, schedular, customL1]
            )
            model.save_weights(savedir / "weights_final.h5")

            model.load_weights(savedir / "weights_custom_best.h5")

            pd.DataFrame(history.history).to_csv(savedir / "log.csv")
            plot_metric(filepath=savedir / "log.csv", metric="loss")

            valid_preds[test_idx, :] = model.predict(X_valid).squeeze()
            test_preds.append(model.predict(test_data).squeeze().reshape(-1, 1).squeeze())

            del model, X_train, X_valid, y_train, y_valid
            keras.backend.clear_session()
            gc.collect()

    pd.DataFrame(valid_preds).to_csv(logdir / "valid_preds.csv")

    if not config.debug:
        submission_df.loc[test_preds_idx, "pressure"] = np.median(test_preds, axis=0)
        submission_df.to_csv(logdir / "submission.csv", index=False)

    shutil.copyfile(Path(__file__), logdir / "script.py")


if __name__ == "__main__":
    cnf_file = sys.argv[1]
    cfg_file_path = Path(__file__).resolve().parent / cnf_file
    with open(cfg_file_path, "rb") as f:
        config = json.load(f)

    main(config=config)
