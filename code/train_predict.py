"""
GridLock 2.0 - Traffic Demand Prediction
=========================================
Reproducible training + prediction pipeline.

Run:
    python train_predict.py

Outputs (written next to this script):
    submission.csv          - final predictions for the test set
    cv_report.txt           - cross-validation score summary

The pipeline is fully deterministic: all random seeds are fixed, so every
run produces a byte-identical submission.csv.
"""
from __future__ import annotations

import hashlib
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
SEED = 42
N_FOLDS = 5
N_SEEDS = 2               # average several LightGBM seeds for a stable result
TARGET_MEAN_SMOOTH = 20.0  # smoothing strength for location mean-encoding

np.random.seed(SEED)


# ----------------------------------------------------------------------------
# Geohash decoding (base-32) -> approximate latitude / longitude
# ----------------------------------------------------------------------------
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE = {c: i for i, c in enumerate(_BASE32)}


def decode_geohash(gh: str):
    """Decode a geohash string into (lat, lon) center coordinates."""
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    is_lon = True
    for ch in gh:
        idx = _DECODE.get(ch, 0)
        for bit in range(4, -1, -1):
            mask = 1 << bit
            if is_lon:
                mid = (lon_lo + lon_hi) / 2
                if idx & mask:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if idx & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            is_lon = not is_lon
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def parse_minutes(ts: str) -> int:
    h, m = ts.split(":")
    return int(h) * 60 + int(m)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Time of day in minutes and cyclical encodings (traffic is periodic)
    df["tmin"] = df["timestamp"].map(parse_minutes)
    ang = 2 * np.pi * df["tmin"] / 1440.0
    df["t_sin"] = np.sin(ang)
    df["t_cos"] = np.cos(ang)
    df["hour"] = (df["tmin"] // 60).astype(int)

    # Spatial coordinates from geohash
    coords = df["geohash"].map(decode_geohash)
    df["lat"] = coords.map(lambda c: c[0])
    df["lon"] = coords.map(lambda c: c[1])

    # Categorical -> integer codes (LightGBM native categorical handling)
    for col, mapping in [
        ("RoadType", {"Residential": 0, "Street": 1, "Highway": 2}),
        ("LargeVehicles", {"Not Allowed": 0, "Allowed": 1}),
        ("Landmarks", {"No": 0, "Yes": 1}),
        ("Weather", {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}),
    ]:
        df[col + "_code"] = df[col].map(mapping).fillna(-1).astype(int)

    df["lanes"] = df["NumberofLanes"].astype(int)
    df["temp"] = df["Temperature"].astype(float)
    df["temp_missing"] = df["Temperature"].isna().astype(int)
    return df


BASE_FEATURES = [
    "tmin", "t_sin", "t_cos", "hour",
    "lat", "lon",
    "RoadType_code", "LargeVehicles_code", "Landmarks_code", "Weather_code",
    "lanes", "temp", "temp_missing",
]
CATEGORICAL = ["RoadType_code", "LargeVehicles_code", "Landmarks_code", "Weather_code"]


def smoothed_location_mean(train_keys, train_target, lookup_keys, global_mean, smooth):
    """Smoothed target mean per location key, computed only on the given train rows."""
    agg = pd.DataFrame({"k": train_keys, "y": train_target}).groupby("k")["y"].agg(["sum", "count"])
    enc = (agg["sum"] + global_mean * smooth) / (agg["count"] + smooth)
    return lookup_keys.map(enc).fillna(global_mean).to_numpy()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def lgb_params(seed):
    return dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        max_depth=-1,
        seed=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train = add_base_features(train)
    test = add_base_features(test)

    global_mean = float(train["demand"].mean())
    y = train["demand"].to_numpy()

    # Location key = geohash (the dominant signal, corr ~0.83 with demand)
    train_geo = train["geohash"]
    test_geo = test["geohash"]

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    feature_cols = BASE_FEATURES + ["geo_mean"]

    for fold, (tr_idx, va_idx) in enumerate(folds.split(train)):
        # Out-of-fold location mean encoding (no leakage)
        geo_mean_tr = smoothed_location_mean(
            train_geo.iloc[tr_idx], y[tr_idx], train_geo.iloc[tr_idx],
            global_mean, TARGET_MEAN_SMOOTH)
        geo_mean_va = smoothed_location_mean(
            train_geo.iloc[tr_idx], y[tr_idx], train_geo.iloc[va_idx],
            global_mean, TARGET_MEAN_SMOOTH)
        geo_mean_te = smoothed_location_mean(
            train_geo.iloc[tr_idx], y[tr_idx], test_geo,
            global_mean, TARGET_MEAN_SMOOTH)

        X_tr = train.iloc[tr_idx][BASE_FEATURES].copy()
        X_tr["geo_mean"] = geo_mean_tr
        X_va = train.iloc[va_idx][BASE_FEATURES].copy()
        X_va["geo_mean"] = geo_mean_va
        X_te = test[BASE_FEATURES].copy()
        X_te["geo_mean"] = geo_mean_te

        fold_va = np.zeros(len(va_idx))
        for s in range(N_SEEDS):
            model = lgb.LGBMRegressor(n_estimators=1500, **lgb_params(SEED + 100 * s))
            model.fit(
                X_tr, y[tr_idx],
                eval_set=[(X_va, y[va_idx])],
                eval_metric="rmse",
                categorical_feature=CATEGORICAL,
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            )
            fold_va += model.predict(X_va, num_iteration=model.best_iteration_) / N_SEEDS
            test_pred += model.predict(X_te, num_iteration=model.best_iteration_) / (N_SEEDS * N_FOLDS)

        oof[va_idx] = fold_va
        rmse = np.sqrt(np.mean((y[va_idx] - fold_va) ** 2))
        print(f"fold {fold}: rmse={rmse:.6f}")

    # Clip to the valid demand range [0, 1]
    oof = np.clip(oof, 0.0, 1.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    oof_rmse = np.sqrt(np.mean((y - oof) ** 2))
    ss_res = np.sum((y - oof) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    oof_r2 = 1 - ss_res / ss_tot
    print(f"\nOOF RMSE = {oof_rmse:.6f}")
    print(f"OOF R2   = {oof_r2:.6f}")

    sub = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": test_pred})
    out_path = os.path.join(HERE, "submission.csv")
    sub.to_csv(out_path, index=False)

    with open(os.path.join(HERE, "cv_report.txt"), "w") as f:
        f.write(f"OOF RMSE = {oof_rmse:.6f}\n")
        f.write(f"OOF R2   = {oof_r2:.6f}\n")
        f.write(f"test mean = {test_pred.mean():.6f}\n")
        f.write(f"test std  = {test_pred.std():.6f}\n")
