"""
prepare.py — Data loading and expanding window generation. FIXED — agent cannot edit.

Loads tabular data from parquet using polars, filters to labeled rows,
and generates expanding time-window splits.

Configure the constants below for your dataset.
"""

import pickle
from datetime import date
from dateutil.relativedelta import relativedelta

import polars as pl

# ─── Constants (CONFIGURE THESE FOR YOUR DATASET) ─────────────────────────────

TARGET_COL = "target"                # Binary target column (0/1)
ID_COL = "id"                        # Row identifier column
TIME_COL = "period"                  # Time period column (YYYY-MM format)
MIN_TRAIN_PERIOD = "2024-01"         # Earliest period to include in training
FIRST_CUT_PERIOD = "2025-08"         # First expanding window cut point
LAST_LABELED_PERIOD = "2026-02"      # Last period with non-null target labels

# Agent constraints — tune these for your dataset size and compute budget
MAX_FEATURES = 300                   # Max features the agent can use per experiment
N_ESTIMATORS = 1000                   # Fixed number of boosting rounds

# Evaluation metric — the agent optimizes this
METRIC_NAME = "auc"                  # Display name used in output and results.tsv

def compute_metric(y_true, y_pred):
    """Compute the evaluation metric. Override this for a different metric."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)

NON_FEATURE_COLS = {
    # Columns that are NOT features — add your ID, time, and target columns here
    "id", "period", "target",
}


# ─── Month arithmetic helpers ────────────────────────────────────────────────

def _month_add(ym: str, n: int) -> str:
    """Add n months to a 'YYYY-MM' string, return 'YYYY-MM'."""
    y, m = int(ym[:4]), int(ym[5:7])
    d = date(y, m, 1) + relativedelta(months=n)
    return d.strftime("%Y-%m")


def _generate_test_months(cut_month: str, max_months: int = 6) -> list[str]:
    """Generate up to max_months test months after cut_month, capped at LAST_LABELED_PERIOD."""
    months = []
    for i in range(1, max_months + 1):
        m = _month_add(cut_month, i)
        if m > LAST_LABELED_PERIOD:
            break
        months.append(m)
    return months


# ─── Data loading ────────────────────────────────────────────────────────────

def load_data(data_dir: str = "data/original") -> pl.DataFrame:
    """
    Load tabular data with non-null target labels.

    Returns a polars DataFrame with all columns (features + meta + target).
    Rows with null target are dropped.

    Customize this function for your data format.
    """
    df = pl.read_parquet(f"{data_dir}/**/*.parquet")

    df = df.filter(
        (pl.col(TIME_COL) >= MIN_TRAIN_PERIOD)
        & pl.col(TARGET_COL).is_not_null()
    )

    return df


def load_feature_names(path: str = "data/original/feature_names.pkl") -> list[str]:
    """Load the canonical feature list from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Expanding windows ──────────────────────────────────────────────────────

def get_expanding_windows(df: pl.DataFrame) -> list[dict]:
    """
    Generate expanding time-window splits.

    For each cut period x from FIRST_CUT_PERIOD onward:
      - train: period <= x
      - test:  period in [x+1 .. min(x+6, LAST_LABELED_PERIOD)]

    Returns list of dicts with keys: cut_month, train_df, test_df
    """
    # Generate cut months from FIRST_CUT_PERIOD up to one month before LAST_LABELED_PERIOD
    cut_months = []
    m = FIRST_CUT_PERIOD
    while m <= _month_add(LAST_LABELED_PERIOD, -1):
        cut_months.append(m)
        m = _month_add(m, 1)

    windows = []
    for cut in cut_months:
        test_months = _generate_test_months(cut)
        if not test_months:
            continue

        train_df = df.filter(pl.col(TIME_COL) <= cut)
        test_df = df.filter(pl.col(TIME_COL).is_in(test_months))

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        windows.append({
            "cut_month": cut,
            "train_df": train_df,
            "test_df": test_df,
        })

    return windows
