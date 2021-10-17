from random import seed
import numpy as np
import pandas as pd
import json
import os
import sys
import gc
import shutil
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

from src.utils import seed_every_thing, fetch_data, Config, plot_metric


def add_features(df: pd.DataFrame, is_train: bool=True) -> pd.DataFrame:
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


def build_model(config: Config, n_features) -> keras.models.Sequential:
    model = keras.models.Sequential([keras.layers.Input(shape=(80, n_features))])
    for n_unit in config.n_units:
        model.add(
            keras.layers.Bidirectional(keras.layers.LSTM(n_unit, return_sequences=True))
        )
    model.add(keras.layers.Dense(50, activation="selu"))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.lr), loss="mae")
    return model


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
        train_df = train_df[:80 * 100]
        test_df = test_df[:80 * 100]

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
                # "breath_id_lag",
                # "breath_id_lag2",
                # "breath_id_lagsame",
                # "breath_id_lag2same",
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
            model.save_weights(savedir / 'weights_final.h5')

            pd.DataFrame(history.history).to_csv(savedir / 'log.csv')
            plot_metric(filepath=savedir / 'log.csv', metric='loss')

            valid_preds[test_idx, :] = np.clip(model.predict(X_valid).squeeze(), 0, 100)
            test_preds.append(
                np.clip(model.predict(test_data).squeeze().reshape(-1, 1).squeeze(), 0, 100)
            )

            del model, X_train, X_valid, y_train, y_valid
            keras.backend.clear_session()
            gc.collect()

    pd.DataFrame(valid_preds).to_csv(logdir / "valid_preds.csv")

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
