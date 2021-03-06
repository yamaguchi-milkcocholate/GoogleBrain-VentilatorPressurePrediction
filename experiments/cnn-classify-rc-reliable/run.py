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
    model = keras.models.Sequential([keras.layers.Input(shape=(80, n_features))])
    for n_unit in config.n_units:
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    n_unit,
                    return_sequences=True,
                    dropout=config.dropout
                )
            )
        )
    for n_unit in config.n_dense_units:
        model.add(keras.layers.Dense(n_unit, activation="selu"))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.BatchNormalization())
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
    logdir = basedir / ".." / ".." / "logs" / config.dirname
    cachedir = basedir / ".." / ".." / "cache"
    os.makedirs(logdir, exist_ok=True)

    config.to_json(logdir / "config.json")
    train_df = reduce_mem_usage(pd.read_csv(cachedir / f"train-reliable-debug{config.debug}.csv"))
    test_df = reduce_mem_usage(pd.read_csv(cachedir / f"test_lstm-less-addfeatures_debug{config.debug}.csv"))

    kfolds = train_df.iloc[0::80]["kfold"].values
    reliables = train_df.iloc[0::80]["is_reliable"].values

    target_cols = [f for f in train_df.columns if "RC_" in f]
    ignore_cols = [f for f in train_df.columns if ("R_" in f) or ("C_" in f)]
    features = list(train_df.drop(["kfold", "pressure", "is_reliable"] + target_cols + ignore_cols, axis=1).columns)
    # features = [f for f in train_df.columns if ("u_in" in f) or ("u_out" in f)]
    pprint(features)
    print(len(features))

    cont_features = [f for f in features if ("u_out" not in f)]
    pprint(cont_features)

    RS = RobustScaler()
    train_df[cont_features] = RS.fit_transform(train_df[cont_features])
    test_df[cont_features] = RS.transform(test_df[cont_features])
    train_data, test_data = train_df[features].values, test_df[features].values

    train_data = train_data.reshape(-1, 80, train_data.shape[-1])
    targets = train_df.iloc[0::80][target_cols].to_numpy()
    test_data = test_data.reshape(-1, 80, test_data.shape[-1])

    with tf.device(f"/GPU:{config.gpu_id}"):
        valid_preds = np.empty_like(targets).astype(np.float32)
        test_preds = []

        for fold in range(config.n_splits):
            train_idx, test_idx = (kfolds != fold), (kfolds == fold)
            print("-" * 15, ">", f"Fold {fold+1}", "<", "-" * 15)
            verbose_str = f"data size declines {train_idx.sum()} ??? "
            train_idx = np.logical_and(train_idx, reliables)
            verbose_str += f"{train_idx.sum()}"
            print(verbose_str)

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
