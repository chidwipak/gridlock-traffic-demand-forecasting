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
