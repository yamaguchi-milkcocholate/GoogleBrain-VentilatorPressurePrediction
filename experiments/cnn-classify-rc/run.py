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
from tensorflow.keras.metrics import categorical_accuracy

from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler

import sys

print(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils import seed_every_thing, Config, plot_metric, reduce_tf_gpu_memory, reduce_mem_usage, fetch_custom_data


def build_model(config: Config, n_features, n_classes) -> keras.models.Sequential:
    model = keras.models.Sequential([keras.layers.Input(shape=(config.cut, n_features))])
    for filters, kernel_size, dilation_rate in zip(
        config.conv1d["filters"], config.conv1d["kernel_sizes"], config.conv1d["dilation_rates"]
    ):
        model.add(
            keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same")
        )
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.lr),
        loss="categorical_crossentropy",
        metrics=[categorical_accuracy],
    )
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
    train_df, test_df, _ = fetch_custom_data(datadir=datadir, n_splits=config.n_splits)
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

    kfolds = train_df.iloc[0 :: config.cut]["kfold"].values

    train_df = reduce_mem_usage(pd.read_csv(cachedir / f"train-classify-rc-debug{config.debug}.csv"))
    test_df = reduce_mem_usage(pd.read_csv(cachedir / f"test-classify-rc-debug{config.debug}.csv"))

    target_cols = [f for f in train_df.columns if "RC_" in f]
    ignore_cols = [f for f in train_df.columns if ("R_" in f) or ("C_" in f)]
    # features = list(train_df.drop(["kfold", "pressure"] + target_cols + ignore_cols, axis=1).columns)
    features = [f for f in train_df.columns if "u_in" in f]
    pprint(features)
    print(len(features))

    cont_features = [f for f in features if ("u_out" not in f)]
    pprint(cont_features)

    RS = RobustScaler()
    train_df[cont_features] = RS.fit_transform(train_df[cont_features])
    test_df[cont_features] = RS.transform(test_df[cont_features])
    train_data, test_data = train_df[features].values, test_df[features].values

    train_data = train_data.reshape(-1, config.cut, train_data.shape[-1])
    targets = train_df.iloc[0 :: config.cut][target_cols].to_numpy()
    test_data = test_data.reshape(-1, config.cut, test_data.shape[-1])

    with tf.device(f"/GPU:{config.gpu_id}"):
        valid_preds = np.empty_like(targets).astype(np.float32)
        test_preds = []

        for fold in range(config.n_splits):
            train_idx, test_idx = (kfolds != fold), (kfolds == fold)
            print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
            savedir = logdir / f"fold{fold}"
            os.makedirs(savedir, exist_ok=True)

            X_train, X_valid = train_data[train_idx], train_data[test_idx]
            y_train, y_valid = targets[train_idx], targets[test_idx]

            model = build_model(config=config, n_features=len(features), n_classes=len(target_cols))

            es = EarlyStopping(
                monitor="val_loss",
                patience=config.es_patience,
                verbose=1,
                mode="min",
                restore_best_weights=True,
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
                validation_data=(X_valid, y_valid),
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=[es, check_point, schedular],
            )
            model.save_weights(savedir / "weights_final.h5")

            model.load_weights(savedir / "weights_best.h5")

            pd.DataFrame(history.history).to_csv(savedir / "log.csv")
            plot_metric(filepath=savedir / "log.csv", metric="loss")

            valid_preds[test_idx, :] = model.predict(X_valid).reshape(-1, len(target_cols))
            test_preds.append(model.predict(test_data).reshape(-1, len(target_cols)))

            del model, X_train, X_valid, y_train, y_valid
            keras.backend.clear_session()
            gc.collect()

    pd.DataFrame(valid_preds).to_csv(logdir / "valid_preds.csv", index=False)
    pd.DataFrame(np.mean(test_preds, axis=0)).to_csv(logdir / "test_preds.csv", index=False)

    shutil.copyfile(Path(__file__), logdir / "script.py")


if __name__ == "__main__":
    cnf_file = sys.argv[1]
    cfg_file_path = Path(__file__).resolve().parent / cnf_file
    with open(cfg_file_path, "rb") as f:
        config = json.load(f)

    main(config=config)
