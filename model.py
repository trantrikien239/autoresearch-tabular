"""
model.py — Model configuration. AGENT EDITS THIS FILE.

Single entry point: train_and_predict(X_train, y_train, X_test, feature_names)
The agent tunes hyperparameters here. N_ESTIMATORS is set in prepare.py.

Default: LightGBM — chosen for fast training to maximize experiment throughput.
The agent (or user) can swap in any model that implements fit() + predict_proba().
"""

import numpy as np
import lightgbm as lgb

from prepare import N_ESTIMATORS

# Module-level storage for analysis access
_last_model = None
_last_feature_names = None


def train_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """
    Train LightGBM and return predicted probabilities on X_test.

    Args:
        X_train: Training features, shape (n_train, n_features)
        y_train: Training labels, shape (n_train,), binary 0/1
        X_test: Test features, shape (n_test, n_features)
        feature_names: List of feature names for LightGBM

    Returns:
        np.ndarray of predicted probabilities, shape (n_test,)
    """
    global _last_model, _last_feature_names

    params = {
        "objective": "binary",
        "metric": "auc",  # LightGBM's internal eval; the official metric is in prepare.py
        "n_estimators": N_ESTIMATORS,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    _last_model = model
    _last_feature_names = feature_names

    return model.predict_proba(X_test)[:, 1]
