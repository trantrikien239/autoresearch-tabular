"""
run_experiment.py — Experiment orchestrator. FIXED — agent cannot edit.

Loads data, applies feature pipeline, trains model across all expanding
windows, computes per-window score and summary stats, appends to results.tsv.
"""

import fcntl
import json
import pickle
import sys
import time
import traceback

import numpy as np

from prepare import (
    load_data, load_feature_names, get_expanding_windows,
    NON_FEATURE_COLS, TARGET_COL, MAX_FEATURES, METRIC_NAME, compute_metric,
)


def _get_feature_cols(df_columns: list[str]) -> list[str]:
    """Extract feature column names (everything not in NON_FEATURE_COLS)."""
    return [c for c in df_columns if c not in NON_FEATURE_COLS]


def _next_exp_id() -> int:
    """Read results.tsv and return the next experiment ID."""
    try:
        with open("results.tsv", "r") as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return 1
            last_line = lines[-1].strip()
            if last_line:
                return int(last_line.split("\t")[0]) + 1
            return 1
    except FileNotFoundError:
        return 1


def main():
    start_time = time.time()

    # Acquire exclusive lock to prevent concurrent experiments
    lock_fp = open(".experiment.lock", "w")
    try:
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: Another experiment is already running. Wait for it to finish.")
        sys.exit(1)

    description = sys.argv[1] if len(sys.argv) > 1 else "no description"

    # Import agent-editable modules (fresh import each run)
    import importlib
    import features
    import model
    importlib.reload(features)
    importlib.reload(model)

    # Load data
    print("Loading data...")
    df = load_data()
    base_features = load_feature_names()
    windows = get_expanding_windows(df)
    print(f"Loaded {len(df)} rows, {len(base_features)} base features, {len(windows)} windows")

    # Run across all expanding windows
    results = []
    prediction_cache = []

    for i, win in enumerate(windows):
        cut = win["cut_month"]
        train_df = win["train_df"]
        test_df = win["test_df"]

        # Filter training data (agent can remove noisy rows)
        train_filtered = features.filter_train(train_df)

        # Feature engineering (stateless, applied independently)
        train_feat = features.build_features(train_filtered, base_features)
        test_feat = features.build_features(test_df, base_features)

        # Determine feature columns dynamically
        feat_cols = _get_feature_cols(train_feat.columns)
        test_feat_cols = _get_feature_cols(test_feat.columns)

        # Verify consistency
        if set(feat_cols) != set(test_feat_cols):
            train_only = set(feat_cols) - set(test_feat_cols)
            test_only = set(test_feat_cols) - set(feat_cols)
            if train_only:
                print(f"  WARNING [{cut}]: {len(train_only)} cols only in train, dropping")
                feat_cols = [c for c in feat_cols if c not in train_only]
            if test_only:
                print(f"  WARNING [{cut}]: {len(test_only)} cols only in test, dropping")

        # Enforce feature cap
        if len(feat_cols) > MAX_FEATURES:
            print(f"ERROR: {len(feat_cols)} features exceeds MAX_FEATURES={MAX_FEATURES}. "
                  f"Add feature selection in features.py to stay within the limit.")
            sys.exit(1)

        # Convert to numpy at the model boundary
        X_train = train_feat.select(feat_cols).to_pandas()
        y_train = train_feat[TARGET_COL].to_numpy().astype(float)
        X_test = test_feat.select(feat_cols).to_pandas()
        y_test = test_feat[TARGET_COL].to_numpy().astype(float)

        # Check for degenerate test set
        if len(np.unique(y_test)) < 2:
            print(f"  SKIP [{cut}]: test set has single class")
            continue

        # Train and predict
        y_pred = model.train_and_predict(X_train, y_train, X_test, feat_cols)

        # Compute metric
        score = compute_metric(y_test, y_pred)

        results.append({
            "cut_month": cut,
            "score": score,
            "train_n": len(train_filtered),
            "test_n": len(test_df),
        })

        # Cache predictions for error analysis
        prediction_cache.append({
            "cut_month": cut,
            "y_true": y_test,
            "y_pred": y_pred,
            "test_df": test_df,
        })

        print(f"  [{cut}] {METRIC_NAME}={score:.4f} (train={len(train_filtered)}, test={len(test_df)})")

    if not results:
        print("ERROR: no valid windows produced results")
        sys.exit(1)

    # Summary statistics
    scores = [r["score"] for r in results]
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    n_features = len(feat_cols)
    total_time = time.time() - start_time

    # Print summary (agent reads this via grep)
    print(f"\n{'='*60}")
    print(f"mean_score: {mean_score:.6f}")
    print(f"std_score: {std_score:.6f}")
    print(f"n_features: {n_features}")
    print(f"n_windows: {len(results)}")
    print(f"metric: {METRIC_NAME}")
    print(f"total_time: {total_time:.1f}s")
    print(f"{'='*60}")

    # Save prediction cache for error analysis
    with open("last_predictions.pkl", "wb") as f:
        pickle.dump(prediction_cache, f)

    # Append to results.tsv
    exp_id = _next_exp_id()
    per_window = {r["cut_month"]: round(r["score"], 6) for r in results}

    header = "exp_id\tmean_score\tstd_score\tn_features\tmetric\tper_window_scores\tdescription\n"
    row = f"{exp_id}\t{mean_score:.6f}\t{std_score:.6f}\t{n_features}\t{METRIC_NAME}\t{json.dumps(per_window)}\t{description}\n"

    try:
        with open("results.tsv", "r") as f:
            pass
    except FileNotFoundError:
        with open("results.tsv", "w") as f:
            f.write(header)

    with open("results.tsv", "a") as f:
        f.write(row)

    print(f"\nResults appended to results.tsv (exp_id={exp_id})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
