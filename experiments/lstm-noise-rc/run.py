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


class RCNoiseGenerator(keras.utils.Sequence):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise_p: float,
        batch_size: int,
        rc_idxs: List[int],
        r_idxs: List[int],
        c_idxs: List[int],
    ) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.noise_p = noise_p
        self.batch_size = batch_size
        self.rc_idxs = rc_idxs
        self.r_idxs = r_idxs
        self.c_idxs = c_idxs

        self._rc_matrix = np.array([[0, 1, 2], [6, 7, 8], [3, 4, 5]])

    def __getitem__(self, idx: int):
        s_idx, e_idx = idx * self.batch_size, (idx + 1) * self.batch_size
        x, y = self.X[s_idx:e_idx, :, :].copy(), self.y[s_idx:e_idx, :].copy()

        r = x[:, 0, self.r_idxs]
        c = x[:, 0, self.c_idxs]

        r_change_idxs = np.random.rand(x.shape[0]) <= self.noise_p
        c_change_idxs = np.random.rand(x.shape[0]) <= self.noise_p

        inplace_r = np.random.choice(np.arange(3), size=np.sum(r_change_idxs), replace=True)
        inplace_c = np.random.choice(np.arange(3), size=np.sum(c_change_idxs), replace=True)
        inplace_r, inplace_c = np.eye(3)[inplace_r], np.eye(3)[inplace_c]

        r[r_change_idxs, :] = inplace_r
        c[c_change_idxs, :] = inplace_c

        rc = []
        for r_idx, c_idx in zip(np.argmax(r, axis=1), np.argmax(c, axis=1)):
            rc.append(self._find_rc(r_idx, c_idx))
        rc = np.eye(9)[np.array(rc)]

        rc = np.tile(rc, (1, x.shape[1])).reshape(-1, x.shape[1], 9)
        r = np.tile(r, (1, x.shape[1])).reshape(-1, x.shape[1], 3)
        c = np.tile(c, (1, x.shape[1])).reshape(-1, x.shape[1], 3)

        x[:, :, self.rc_idxs] = rc
        x[:, :, self.r_idxs] = r
        x[:, :, self.c_idxs] = c

        return x, y

    def _find_rc(self, r: int, c: int):
        return self._rc_matrix[r][c]

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def on_epoch_end(self):
        idxs = np.arange(self.X.shape[0])
        np.random.shuffle(idxs)

        self.X = self.X[idxs, :, :]
        self.y = self.y[idxs, :]


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
    _, test_df, submission_df = fetch_custom_data(datadir=datadir, n_splits=config.n_splits)
    test_df["count"] = (np.arange(test_df.shape[0]) % 80).astype(int)
    test_preds_idx = test_df["count"] < config.cut
    test_df = test_df[test_preds_idx].reset_index(drop=True)
    test_df["pressure"] = 0

    train_df = reduce_mem_usage(pd.read_csv(cachedir / f"train-10fold-debug{config.debug}.csv"))
    test_df = reduce_mem_usage(pd.read_csv(cachedir / f"test-10fold-debug{config.debug}.csv"))
    train_df = train_df[train_df["count"] <= config.cut].reset_index(drop=True)
    test_df = test_df[test_df["count"] <= config.cut].reset_index(drop=True)

    kfolds = train_df.iloc[0 :: config.cut]["kfold"].values

    features = np.array(list(train_df.drop(["kfold", "pressure"], axis=1).columns))
    cont_features = [
        f for f in features if ("RC_" not in f) and ("R_" not in f) and ("C_" not in f) and ("u_out" not in f)
    ]

    rc_idxs = [
        i
        for i, f in enumerate(features)
        if f
        in ["RC_20_10", "RC_20_20", "RC_20_50", "RC_50_10", "RC_50_20", "RC_50_50", "RC_5_10", "RC_5_20", "RC_5_50"]
    ]
    r_idxs = [i for i, f in enumerate(features) if f in ["R_20", "R_5", "R_50"]]
    c_idxs = [i for i, f in enumerate(features) if f in ["C_10", "C_20", "C_50"]]
    pprint(features[rc_idxs])
    pprint(features[r_idxs])
    pprint(features[c_idxs])

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

            RC_noise_generator = RCNoiseGenerator(
                X=X_train,
                y=y_train,
                noise_p=config.noise_p,
                batch_size=config.batch_size,
                rc_idxs=rc_idxs,
                r_idxs=r_idxs,
                c_idxs=c_idxs,
            )

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
                u_outs=X_valid[:, :, features.tolist().index("u_out")],
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

            history = model.fit_generator(
                generator=RC_noise_generator,
                validation_data=(X_valid, y_valid),
                epochs=config.epochs,
                callbacks=[check_point, schedular, customL1],
                workers=config.workers,
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
