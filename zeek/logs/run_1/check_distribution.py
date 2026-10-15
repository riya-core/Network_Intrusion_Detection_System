"""
check_distribution.py
=====================
Compares the scaled features your live Zeek pipeline produced with the training data.

Training features were standardised, so on the training set every scaled feature has mean ~0
and std ~1. If a live feature has a very different mean/std, or many values far outside the
training range, that feature is being computed differently from training (or your network
traffic is simply very different). Models can behave badly on drifted features.

    python live_detector.py --conn-log conn.log --model xgb --dump-features features_live.npy
    python check_distribution.py --artifacts processed --live features_live.npy
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="processed")
    ap.add_argument("--live", default="features_live.npy")
    ap.add_argument("--top", type=int, default=15, help="rows to print")
    ap.add_argument("--csv", default="distribution_report.csv")
    args = ap.parse_args()

    art = Path(args.artifacts)
    names = json.loads((art / "meta.json").read_text())["feature_names"]
    X_train = np.load(art / "unsw_nb15_processed.npz")["X_train"]
    X_live = np.load(args.live)
    if X_live.shape[1] != X_train.shape[1]:
        raise SystemExit(f"feature count mismatch: live {X_live.shape[1]} vs training {X_train.shape[1]}")

    t_mean, t_std = X_train.mean(0), X_train.std(0)
    l_mean, l_std = X_live.mean(0), X_live.std(0)
    shift = (l_mean - t_mean) / np.maximum(t_std, 1e-6)           # mean shift in training std units
    lo, hi = np.percentile(X_train, [0.5, 99.5], axis=0)
    outside = ((X_live < lo) | (X_live > hi)).mean(0) * 100        # % of live rows outside training range

    rep = pd.DataFrame({
        "feature": names, "train_mean": t_mean, "train_std": t_std,
        "live_mean": l_mean, "live_std": l_std,
        "mean_shift_in_std": shift, "pct_outside_train_range": outside,
    })
    rep["flag"] = np.where((rep.mean_shift_in_std.abs() > 1) | (rep.pct_outside_train_range > 10), "CHECK", "")
    rep = rep.reindex(rep.mean_shift_in_std.abs().sort_values(ascending=False).index)
    rep.to_csv(args.csv, index=False)

    print(f"live rows: {len(X_live):,}   training rows: {len(X_train):,}\n")
    print(rep.head(args.top).round(3).to_string(index=False))
    print(f"\n{(rep.flag == 'CHECK').sum()} features flagged. Full report: {args.csv}")
    print("Flagged features are the first place to look if live predictions seem wrong.")


if __name__ == "__main__":
    main()
