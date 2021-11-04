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
)


def build_model(config: Config, n_features) -> keras.models.Sequential:
    model = keras.models.Sequential([keras.layers.Input(shape=(config.cut, n_features))])
    for n_unit in config.n_units:
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    n_unit,
                    return_sequences=True,
                )
            )
        )
    for n_unit in config.n_dense_units:
        model.add(keras.layers.Dense(n_unit, activation="selu"))
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
    cachedir = basedir / ".." / ".." / "cache"
    os.makedirs(logdir, exist_ok=True)

    config.to_json(logdir / "config.json")
    train_df, test_df, submission_df = fetch_custom_data(datadir=datadir, n_splits=config.n_splits)
    train_df["count"], test_df["count"] = (np.arange(train_df.shape[0]) % 80).astype(int), (
        np.arange(test_df.shape[0]) % 80
    ).astype(int)
    train_df = train_df[train_df["count"] < config.cut].reset_index(drop=True)
    test_preds_idx = test_df["count"] < config.cut
    test_df = test_df[test_preds_idx].reset_index(drop=True)
    test_df["pressure"] = 0

    if config.debug:
        train_df = train_df[: config.cut * 1000]
        test_df = test_df[: config.cut * 1000]

    train_df = reduce_mem_usage(pd.read_csv(cachedir / f"train-reliable-debug{config.debug}.csv"))
    test_df = reduce_mem_usage(pd.read_csv(cachedir / f"test_lstm-less-cut-addfeatures_debug{config.debug}.csv"))

    kfolds = train_df.iloc[0 :: config.cut]["kfold"].values
    reliables = train_df.iloc[0 :: config.cut]["is_reliable"].values

    features = list(train_df.drop(["kfold", "pressure", "is_reliable"], axis=1).columns)
    pprint(features)
    print(len(features))

    cont_features = [
        f
        for f in features
        if ("RC_" not in f) and ("u_out" not in f) and ("R_" not in f) and ("C_" not in f) and ("u_out" not in f)
    ]
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
            verbose_str = f"data size declines {train_idx.sum()} ⇨ "
            train_idx = np.logical_and(train_idx, reliables)
            test_idx_reliable = np.logical_and(test_idx, reliables)
            verbose_str += f"{train_idx.sum()}"
            print(verbose_str)

            savedir = logdir / f"fold{fold}"
            os.makedirs(savedir, exist_ok=True)

            X_train, X_valid, X_valid_reliable = (
                train_data[train_idx],
                train_data[test_idx],
                train_data[test_idx_reliable],
            )
            y_train, y_valid, y_valid_reliable = targets[train_idx], targets[test_idx], targets[test_idx_reliable]

            model = build_model(config=config, n_features=len(features))

            # es = EarlyStopping(
            #     monitor="val_loss",
            #     patience=config.es_patience,
            #     verbose=1,
            #     mode="min",
            #     restore_best_weights=True,
            # )

            customL1 = CustomL1Loss(
                X_valid=X_valid_reliable,
                y_valid=y_valid_reliable,
                u_outs=X_valid_reliable[:, :, features.index("u_out")],
                filepath=savedir / "weights_custom_best.h5",
            )

            check_point = ModelCheckpoint(
                filepath=savedir / "weights_best.h5",
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min",
                save_weights_only=True,
            )

            schedular = ReduceLROnPlateau(mode="min", **config.schedular)

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_valid_reliable, y_valid_reliable),
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=[check_point, schedular, customL1],
            )
            model.save_weights(savedir / "weights_final.h5")

            model.load_weights(savedir / "weights_custom_best.h5")

            pd.DataFrame(history.history).to_csv(savedir / "log.csv")
            plot_metric(filepath=savedir / "log.csv", metric="loss")

            valid_preds[test_idx, :] = model.predict(X_valid).squeeze()
            test_preds.append(model.predict(test_data).squeeze().reshape(-1, 1).squeeze())

            del model, X_train, X_valid, y_train, y_valid, X_valid_reliable, y_valid_reliable
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
