# GridLock 2.0 — Submission Package

This folder contains everything needed to reproduce our leaderboard submission.

## Contents

```
submission_package/
├── APPROACH.md            <- plain-English explanation of our method
├── README.md              <- this file
├── submission.csv         <- final predictions (copy of code/submission.csv)
└── code/
    ├── train_predict.py   <- the full training + prediction pipeline
    ├── requirements.txt   <- Python dependencies
    ├── data/
    │   ├── train.csv      <- provided training data
    │   └── test.csv       <- provided test data
    ├── submission.csv     <- generated predictions (produced by the script)
    └── cv_report.txt      <- cross-validation score (produced by the script)
```

## Result

Cross-validation (5-fold, out-of-fold):

- **OOF R² = 0.9464**
- OOF RMSE = 0.0329

The generated `submission.csv` has md5 `5d34532d66929d218be1d3f8b57fcd8c` and is
reproduced byte-for-byte on every run (verified across two independent runs).
