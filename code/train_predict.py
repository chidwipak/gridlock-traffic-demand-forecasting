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
