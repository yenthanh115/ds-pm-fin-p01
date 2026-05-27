"""Model training, evaluation, and scoring module for LendSafe pipeline."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


# --- Module-level constants ---

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


@dataclass
class TrainedModel:
    """Container for a trained model and its cross-validation results.

    Attributes:
        name: Human-readable model name.
        model: Fitted sklearn estimator.
        cv_scores: Array of AUC-ROC scores from K-fold cross-validation.
        cv_mean: Mean of cv_scores.
        feature_names: List of feature names used during training.
    """

    name: str
    model: Any
    cv_scores: np.ndarray
    cv_mean: float
    feature_names: list[str]


def stratified_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform an 80/20 stratified train/test split with a fixed random seed.

    Preserves the class distribution of the target variable in both splits.

    Args:
        X: Feature matrix.
        y: Target series (binary).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


def cross_validate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Perform 5-fold stratified cross-validation returning AUC-ROC scores.

    Uses StratifiedKFold to preserve class distribution in each fold.

    Args:
        model: A fitted or unfitted sklearn estimator.
        X_train: Training feature matrix.
        y_train: Training target series.

    Returns:
        Array of AUC-ROC scores, one per fold.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    return scores


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> TrainedModel:
    """Train a logistic regression model with class_weight='balanced'.

    Addresses class imbalance by adjusting weights inversely proportional
    to class frequencies.

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.

    Returns:
        TrainedModel wrapping the fitted logistic regression.
    """
    model = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    cv_scores = cross_validate_model(model, X_train, y_train)

    return TrainedModel(
        name="Logistic Regression",
        model=model,
        cv_scores=cv_scores,
        cv_mean=float(cv_scores.mean()),
        feature_names=list(X_train.columns),
    )


def train_tree_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> TrainedModel:
    """Train a Random Forest model with class_weight='balanced_subsample'.

    Addresses class imbalance by adjusting weights per bootstrap sample.

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.

    Returns:
        TrainedModel wrapping the fitted Random Forest.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    cv_scores = cross_validate_model(model, X_train, y_train)

    return TrainedModel(
        name="Random Forest",
        model=model,
        cv_scores=cv_scores,
        cv_mean=float(cv_scores.mean()),
        feature_names=list(X_train.columns),
    )


def select_best_model(models: list[TrainedModel]) -> TrainedModel:
    """Select the model with the highest mean cross-validation AUC-ROC.

    Args:
        models: List of TrainedModel instances to compare.

    Returns:
        The TrainedModel with the highest cv_mean.

    Raises:
        ValueError: If models list is empty.
    """
    if not models:
        raise ValueError("Cannot select best model from an empty list.")
    return max(models, key=lambda m: m.cv_mean)
