"""
analysis.py — Analysis primitives and free-form analysis. AGENT EDITS THIS FILE.

Built-in primitives for data exploration + error analysis.
All findings are appended to findings.md.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

FINDINGS_PATH = Path("findings.md")
PREDICTIONS_PATH = Path("last_predictions.pkl")


# ─── Output helper ───────────────────────────────────────────────────────────

def _append_finding(title: str, content: str) -> None:
    """Append a finding to findings.md with a markdown header and timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(FINDINGS_PATH, "a") as f:
        f.write(f"\n## {title} ({timestamp})\n\n{content}\n")


# ─── Primitives ──────────────────────────────────────────────────────────────

def univariate_auc(
    df: pl.DataFrame,
    features: list[str],
    target_col: str = None,
) -> pl.DataFrame:
    """
    Compute ROC AUC of each feature vs target (univariate).

    Returns polars DataFrame with columns: feature, auc, non_null_count
    sorted by abs(auc - 0.5) descending (most discriminative first).
    """
    from prepare import TARGET_COL
    if target_col is None:
        target_col = TARGET_COL

    y = df[target_col].to_numpy().astype(float)
    results = []

    for feat in features:
        col = df[feat].to_numpy().astype(float)
        mask = ~np.isnan(col) & ~np.isnan(y)
        if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
            continue
        try:
            auc = roc_auc_score(y[mask], col[mask])
            results.append({"feature": feat, "auc": round(auc, 6), "non_null_count": int(mask.sum())})
        except Exception:
            continue

    result_df = pl.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort((pl.col("auc") - 0.5).abs(), descending=True)
    return result_df


def correlation_pairs(
    df: pl.DataFrame,
    features: list[str],
    threshold: float = 0.95,
) -> pl.DataFrame:
    """
    Find highly correlated feature pairs (|corr| > threshold).

    Returns polars DataFrame with columns: feat_a, feat_b, correlation
    sorted by abs(correlation) descending.
    """
    available = [f for f in features if f in df.columns]
    if len(available) > 200:
        variances = []
        for f in available:
            v = df[f].drop_nulls().to_numpy().astype(float)
            variances.append((f, np.nanvar(v) if len(v) > 0 else 0))
        variances.sort(key=lambda x: x[1], reverse=True)
        available = [v[0] for v in variances[:200]]

    mat = df.select(available).to_numpy().astype(float)
    n = mat.shape[1]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            mask = ~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j])
            if mask.sum() < 50:
                continue
            corr = np.corrcoef(mat[mask, i], mat[mask, j])[0, 1]
            if abs(corr) > threshold:
                pairs.append({
                    "feat_a": available[i],
                    "feat_b": available[j],
                    "correlation": round(corr, 4),
                })

    result_df = pl.DataFrame(pairs)
    if len(result_df) > 0:
        result_df = result_df.sort(pl.col("correlation").abs(), descending=True)
    return result_df


def null_rates(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """
    Compute null percentage per feature.

    Returns polars DataFrame with columns: feature, null_pct
    sorted by null_pct descending.
    """
    available = [f for f in features if f in df.columns]
    n_rows = len(df)
    results = []
    for f in available:
        null_count = df[f].null_count()
        results.append({
            "feature": f,
            "null_pct": round(null_count / n_rows * 100, 2),
        })

    result_df = pl.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort("null_pct", descending=True)
    return result_df


def feature_importance_from_model() -> pl.DataFrame:
    """
    Extract feature importance from the last trained model.

    Requires that run_experiment.py was run first (model._last_model is set).
    Returns polars DataFrame with columns: feature, importance_gain, importance_split.
    Works with LightGBM by default — adapt if using a different model.
    """
    try:
        from model import _last_model, _last_feature_names
    except ImportError:
        return pl.DataFrame()

    if _last_model is None:
        print("WARNING: No model cached. Run an experiment first.")
        return pl.DataFrame()

    gain = _last_model.booster_.feature_importance(importance_type="gain")
    split = _last_model.booster_.feature_importance(importance_type="split")

    result_df = pl.DataFrame({
        "feature": _last_feature_names,
        "importance_gain": gain.tolist(),
        "importance_split": split.tolist(),
    })
    return result_df.sort("importance_gain", descending=True)


def error_analysis(threshold: float = 0.5) -> dict:
    """
    Analyze prediction errors from the last experiment run.

    Loads last_predictions.pkl and for each window identifies:
    - False negatives: positive (y=1) but predicted low (< threshold)
    - False positives: negative (y=0) but predicted high (>= threshold)

    Returns dict with summary statistics per window.
    """
    if not PREDICTIONS_PATH.exists():
        print("WARNING: No predictions cached. Run an experiment first.")
        return {}

    with open(PREDICTIONS_PATH, "rb") as f:
        cache = pickle.load(f)

    results = {}
    for entry in cache:
        cut = entry["cut_month"]
        y_true = entry["y_true"]
        y_pred = entry["y_pred"]

        fn_mask = (y_true == 1) & (y_pred < threshold)
        fp_mask = (y_true == 0) & (y_pred >= threshold)
        tp_mask = (y_true == 1) & (y_pred >= threshold)
        tn_mask = (y_true == 0) & (y_pred < threshold)

        n_total = len(y_true)
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())

        results[cut] = {
            "total": n_total,
            "positives": n_pos,
            "negatives": n_neg,
            "fn": int(fn_mask.sum()),
            "fp": int(fp_mask.sum()),
            "tp": int(tp_mask.sum()),
            "tn": int(tn_mask.sum()),
            "fn_rate": round(fn_mask.sum() / max(n_pos, 1), 4),
            "fp_rate": round(fp_mask.sum() / max(n_neg, 1), 4),
            "fn_mean_pred": round(float(y_pred[fn_mask].mean()), 4) if fn_mask.any() else None,
            "fp_mean_pred": round(float(y_pred[fp_mask].mean()), 4) if fp_mask.any() else None,
        }

    return results


def error_feature_patterns(top_n: int = 20) -> str:
    """
    Compare feature distributions between correctly and incorrectly predicted samples.

    Uses last_predictions.pkl. For the largest test window, computes mean feature
    values for false negatives vs true positives, and false positives vs true negatives.
    Returns a formatted string summary.
    """
    if not PREDICTIONS_PATH.exists():
        print("WARNING: No predictions cached. Run an experiment first.")
        return ""

    with open(PREDICTIONS_PATH, "rb") as f:
        cache = pickle.load(f)

    if not cache:
        return ""

    largest = max(cache, key=lambda e: len(e["y_true"]))
    y_true = largest["y_true"]
    y_pred = largest["y_pred"]
    test_df = largest["test_df"]
    cut = largest["cut_month"]

    from prepare import NON_FEATURE_COLS
    feat_cols = [c for c in test_df.columns if c not in NON_FEATURE_COLS]

    median_pred = float(np.median(y_pred))

    fn_mask = (y_true == 1) & (y_pred < median_pred)
    tp_mask = (y_true == 1) & (y_pred >= median_pred)

    lines = [f"Error feature patterns for window {cut} (n={len(y_true)})\n"]
    lines.append(f"FN: {fn_mask.sum()} samples, TP: {tp_mask.sum()} samples\n")
    lines.append(f"Median prediction threshold: {median_pred:.4f}\n\n")

    if fn_mask.sum() > 0 and tp_mask.sum() > 0:
        lines.append("Top features where FN differs most from TP (by mean):\n")
        diffs = []
        for feat in feat_cols[:200]:
            fn_vals = test_df[feat].to_numpy().astype(float)[fn_mask]
            tp_vals = test_df[feat].to_numpy().astype(float)[tp_mask]
            fn_mean = np.nanmean(fn_vals)
            tp_mean = np.nanmean(tp_vals)
            if np.isnan(fn_mean) or np.isnan(tp_mean):
                continue
            diff = abs(fn_mean - tp_mean)
            scale = max(abs(tp_mean), 1e-8)
            diffs.append((feat, fn_mean, tp_mean, diff / scale))

        diffs.sort(key=lambda x: x[3], reverse=True)
        for feat, fn_mean, tp_mean, rel_diff in diffs[:top_n]:
            lines.append(f"  {feat}: FN_mean={fn_mean:.4f}, TP_mean={tp_mean:.4f}, rel_diff={rel_diff:.2f}\n")

    return "".join(lines)


def error_by_segment(segment_col: str = None) -> str:
    """
    Compute score and positive rate by segment from cached predictions.

    Returns formatted string with per-segment metrics.
    """
    from prepare import TIME_COL, METRIC_NAME, compute_metric
    if segment_col is None:
        segment_col = TIME_COL

    if not PREDICTIONS_PATH.exists():
        print("WARNING: No predictions cached. Run an experiment first.")
        return ""

    with open(PREDICTIONS_PATH, "rb") as f:
        cache = pickle.load(f)

    all_y_true = []
    all_y_pred = []
    all_segments = []

    for entry in cache:
        y_true = entry["y_true"]
        y_pred = entry["y_pred"]
        test_df = entry["test_df"]

        if segment_col not in test_df.columns:
            return f"Column '{segment_col}' not found in test data"

        seg_vals = test_df[segment_col].to_list()
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_segments.extend(seg_vals)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_segments = np.array(all_segments)

    lines = [f"Performance by {segment_col}:\n\n"]
    for seg in sorted(set(all_segments)):
        mask = all_segments == seg
        n = mask.sum()
        if n < 30 or len(np.unique(all_y_true[mask])) < 2:
            continue
        score = compute_metric(all_y_true[mask], all_y_pred[mask])
        pos_rate = all_y_true[mask].mean()
        lines.append(f"  {seg}: {METRIC_NAME}={score:.4f}, positive_rate={pos_rate:.4f}, n={n}\n")

    return "".join(lines)


# ─── Main entry point ────────────────────────────────────────────────────────

def run_analysis(
    df: pl.DataFrame,
    base_features: list[str],
    windows: list[dict],
) -> None:
    """
    Main analysis entry point. Agent modifies this function body.

    Available primitives:
    - univariate_auc(df, features) — ROC AUC of each feature vs target
    - correlation_pairs(df, features, threshold) — highly correlated pairs
    - null_rates(df, features) — null % per feature
    - feature_importance_from_model() — model feature importance (requires prior run)
    - error_analysis(threshold) — FP/FN breakdown per window
    - error_feature_patterns(top_n) — feature diffs between FN vs TP
    - error_by_segment(segment_col) — score by segment
    """
    from prepare import NON_FEATURE_COLS, TARGET_COL

    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"Analyzing {len(feat_cols)} features...")

    # Agent: write your analysis code here.
    # Example starter analysis:
    #
    #   # Univariate AUC
    #   auc_df = univariate_auc(df, feat_cols)
    #   top20 = auc_df.head(20)
    #   _append_finding("Univariate AUC", str(top20))
    #
    #   # Null rates
    #   nulls = null_rates(df, feat_cols)
    #   high_null = nulls.filter(pl.col("null_pct") > 50)
    #   _append_finding("High Null Features", str(high_null))
    #
    #   # Correlations
    #   corr = correlation_pairs(df, feat_cols, threshold=0.95)
    #   _append_finding("Highly Correlated Pairs", str(corr))

    print("No analysis configured yet. Edit analysis.py run_analysis() to add your analysis.")
