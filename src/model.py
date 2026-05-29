"""Model training, evaluation, and scoring module for LendSafe pipeline."""

import json
import os
from dataclasses import dataclass
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.features import engineer_features
from src.preprocess import strip_formatting, parse_employment_length


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


# --- Evaluation and Reporting ---


@dataclass
class EvaluationReport:
    """Container for model evaluation metrics and feature importances.

    Attributes:
        accuracy: Classification accuracy on the test set.
        precision: Precision for the positive class (default).
        recall: Recall for the positive class (default).
        f1: F1 score for the positive class (default).
        auc_roc: Area Under the ROC Curve.
        feature_importances: Mapping of feature name to importance score.
        threshold: Operating threshold selected for classification.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    feature_importances: dict[str, float]
    threshold: float


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> float:
    """Find the classification threshold that minimizes expected cost.

    Searches candidate thresholds to minimize:
        cost_fp * FP + cost_fn * FN

    The cost ratio reflects the business reality that approving a defaulter
    (false negative) is costlier than rejecting a good applicant (false positive).

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        cost_fp: Cost of a false positive (default 1.0).
        cost_fn: Cost of a false negative (default 5.0).

    Returns:
        Optimal threshold as a float in [0.0, 1.0].
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Candidate thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        cost = cost_fp * fp + cost_fn * fn

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    return best_threshold


def compute_feature_importance(
    model: TrainedModel, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute feature importance scores for a trained model.

    Uses logistic regression coefficients for LR models and permutation
    importance for tree-based models.

    Args:
        model: A TrainedModel instance.
        X_test: Test feature matrix.
        y_test: Test target series.

    Returns:
        Dictionary mapping feature names to importance scores.
    """
    estimator = model.model

    if isinstance(estimator, LogisticRegression):
        # Use absolute values of coefficients as importance
        coefficients = np.abs(estimator.coef_[0])
        importances = dict(zip(model.feature_names, coefficients.tolist()))
    else:
        # Tree-based models: use permutation importance
        result = permutation_importance(
            estimator,
            X_test,
            y_test,
            n_repeats=10,
            random_state=RANDOM_SEED,
            scoring="roc_auc",
        )
        importances = dict(
            zip(model.feature_names, result.importances_mean.tolist())
        )

    # Sort by importance descending
    importances = dict(
        sorted(importances.items(), key=lambda item: item[1], reverse=True)
    )
    return importances


def evaluate_model(
    model: TrainedModel, X_test: pd.DataFrame, y_test: pd.Series
) -> EvaluationReport:
    """Compute all evaluation metrics on the held-out test set.

    Computes accuracy, precision, recall, F1, AUC-ROC, feature importances,
    and the optimal operating threshold.

    Args:
        model: A TrainedModel instance with a fitted estimator.
        X_test: Test feature matrix.
        y_test: Test target series.

    Returns:
        EvaluationReport with all computed metrics.
    """
    y_proba = model.model.predict_proba(X_test)[:, 1]

    # Find optimal threshold based on business cost
    threshold = find_optimal_threshold(y_test, y_proba)

    # Predictions at optimal threshold
    y_pred = (y_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0.0)
    recall = recall_score(y_test, y_pred, zero_division=0.0)
    f1 = f1_score(y_test, y_pred, zero_division=0.0)
    auc_roc = roc_auc_score(y_test, y_proba)

    feature_importances = compute_feature_importance(model, X_test, y_test)

    return EvaluationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        feature_importances=feature_importances,
        threshold=threshold,
    )


def plot_roc_curve(
    y_test: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    output_path: str,
) -> None:
    """Generate and save an ROC curve plot.

    Args:
        y_test: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        model_name: Name of the model (used in plot title).
        output_path: File path to save the plot image.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(
    y_test: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    output_path: str,
) -> None:
    """Generate and save a precision-recall curve plot.

    Args:
        y_test: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        model_name: Name of the model (used in plot title).
        output_path: File path to save the plot image.
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_findings_report(
    reports: list[EvaluationReport], output_path: str
) -> None:
    """Write a findings report summarizing model evaluation results.

    Generates a markdown report containing top predictive features,
    performance summary for each model, and recommended threshold.

    Args:
        reports: List of EvaluationReport instances (one per model).
        output_path: File path to write the markdown report.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines: list[str] = []
    lines.append("# LendSafe Model Evaluation Findings\n")
    lines.append("")

    # --- Performance Summary ---
    lines.append("## Performance Summary\n")
    lines.append("")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Threshold |")
    lines.append("|-------|----------|-----------|--------|----|---------|-----------|\n")

    for i, report in enumerate(reports):
        model_label = f"Model {i + 1}"
        lines.append(
            f"| {model_label} | {report.accuracy:.4f} | {report.precision:.4f} | "
            f"{report.recall:.4f} | {report.f1:.4f} | {report.auc_roc:.4f} | "
            f"{report.threshold:.4f} |"
        )

    lines.append("")
    lines.append("")

    # --- Recommended Threshold ---
    best_report = max(reports, key=lambda r: r.auc_roc)
    lines.append("## Recommended Threshold\n")
    lines.append("")
    lines.append(
        f"The recommended operating threshold is **{best_report.threshold:.4f}**, "
        f"selected to minimize expected cost with a false-negative-to-false-positive "
        f"cost ratio of 5:1."
    )
    lines.append("")
    lines.append("")

    # --- Top Predictive Features ---
    lines.append("## Top Predictive Features\n")
    lines.append("")
    lines.append(
        "The following features have the highest importance scores "
        "from the best-performing model:"
    )
    lines.append("")

    top_features = list(best_report.feature_importances.items())[:10]
    lines.append("| Rank | Feature | Importance |")
    lines.append("|------|---------|------------|")
    for rank, (feature, importance) in enumerate(top_features, start=1):
        lines.append(f"| {rank} | {feature} | {importance:.4f} |")

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def score_applicant(
    features: dict | pd.Series,
    model_path: str = "models/best_model.joblib",
    feature_names_path: str = "models/feature_names.json",
) -> float:
    """Score a single applicant.

    Loads the persisted model and expected feature names, validates that all
    required features are present in the input, applies the same preprocessing
    and feature engineering transformations used during training, and returns
    the probability of default.

    Args:
        features: Dictionary or pandas Series of raw applicant features.
        model_path: Path to persisted best model (joblib format).
        feature_names_path: Path to JSON file listing expected feature names.

    Returns:
        Float in [0.0, 1.0] representing the probability of default.

    Raises:
        ValueError: If required features are missing from the input.
                    The error message lists all missing feature names.
    """
    # Load persisted model
    model = joblib.load(model_path)

    # Load expected feature names
    with open(feature_names_path, "r", encoding="utf-8") as f:
        expected_feature_names = json.load(f)

    # Convert input to a dict if it's a pandas Series
    if isinstance(features, pd.Series):
        features = features.to_dict()

    # Validate all required features are present (fail fast before transformation)
    input_keys = set(features.keys())
    required_keys = set(expected_feature_names)
    missing = sorted(required_keys - input_keys)

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    # Build a single-row DataFrame from the input features
    df = pd.DataFrame([features])

    # Apply preprocessing transformations (same as training)
    # Strip formatting from numeric columns
    numeric_cols = [
        "loan_amount", "interest_rate", "monthly_payment",
        "annual_income", "dti", "revolving_balance",
        "revolving_credit_limit",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = strip_formatting(df[col])

    # Parse employment length if present
    if "employment_length" in df.columns:
        df["employment_length"] = parse_employment_length(df["employment_length"])

    # Apply feature engineering (same as training)
    df = engineer_features(df)

    # Align columns to what the model expects (handle missing one-hot columns)
    model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(df.columns)
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Ensure column order matches model expectations
    df = df[model_features]

    # Return probability of default (positive class)
    proba = model.predict_proba(df)[:, 1]
    return float(proba[0])
