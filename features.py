"""
features.py — Feature engineering and data curation. AGENT EDITS THIS FILE.

Two entry points:
  - filter_train(df) -> df : remove noisy rows from training data
  - build_features(df, base_features) -> df : feature interactions and selection
"""

import polars as pl


def filter_train(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter training data to remove noisy or unrepresentative rows.

    Examples of what the agent might add here:
    - Remove rows from periods with unusual target rates
    - Remove outlier rows based on error analysis patterns
    - Focus on recent, high-quality training data

    By default, uses all available training data.
    """
    return df


def _engineer_features(df: pl.DataFrame, base_features: list[str]) -> pl.DataFrame:
    """
    Generate engineered features. No selection — that happens in build_features().

    Examples of what the agent might add here:
    - Ratio features (e.g., short-term / long-term lookback windows)
    - Interaction features (e.g., feature_a * feature_b)
    - Domain-specific transformations (e.g., log, bins, differences)

    Rules:
    - Stateless per-row transforms only (no aggregations across rows)
    - Called independently on train and test — must be deterministic
    - Use _safe_ratio() for division to handle zeros/nulls
    """
    return df


def _safe_ratio(df: pl.DataFrame, num: str, den: str, name: str) -> pl.DataFrame:
    """Safely compute num/den, returning null when denominator is 0 or missing."""
    if num in df.columns and den in df.columns:
        df = df.with_columns(
            (pl.col(num) / pl.col(den).replace(0, None)).alias(name)
        )
    return df


def build_features(df: pl.DataFrame, base_features: list[str]) -> pl.DataFrame:
    """
    Main feature pipeline: engineer features, then optionally select a subset.

    The agent modifies this to add feature selection logic (e.g., keeping only
    a curated list of features to stay within MAX_FEATURES).
    """
    from prepare import NON_FEATURE_COLS

    # Step 1: Engineer features
    df = _engineer_features(df, base_features)

    # Step 2: Feature selection (optional — add when feature count exceeds MAX_FEATURES)
    # Example:
    #   from selected_features import SELECTED_FEATURES
    #   keep_cols = set(SELECTED_FEATURES) | NON_FEATURE_COLS
    #   drop_cols = [c for c in df.columns if c not in keep_cols]
    #   if drop_cols:
    #       df = df.drop(drop_cols)

    return df
