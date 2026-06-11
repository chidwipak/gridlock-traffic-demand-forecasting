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
