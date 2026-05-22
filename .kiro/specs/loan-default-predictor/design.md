# Design Document

## Overview

This document describes the technical architecture for LendSafe — a consumer loan default prediction system. The system ingests raw loan data, preprocesses it, engineers predictive features, trains classification models, evaluates performance, and exposes a scoring function that returns a probability of default for new applicants.

The design follows a hybrid organisation: Jupyter notebooks for exploratory data analysis and reusable Python modules in `src/` for the deterministic pipeline logic.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LendSafe Pipeline                         │
├─────────────┬──────────────┬──────────────┬─────────────────────┤
│  Data Load  │ Preprocessor │ Feature Eng. │  Model Train/Score  │
│  (loader)   │ (preprocess) │  (features)  │     (model)         │
└──────┬──────┴──────┬───────┴──────┬───────┴──────────┬──────────┘
       │             │              │                   │
       ▼             ▼              ▼                   ▼
   raw CSV →   processed.csv →  feature matrix →  trained model
                                                       │
                                                       ▼
                                              score_applicant()
                                                  → P(default)
```

### Data Flow

1. **Load** — Read raw CSV into pandas DataFrame, validate columns
2. **Preprocess** — Clean formatting, parse strings, encode target, drop ambiguous rows, output `processed.csv`
3. **Feature Engineering** — Compute derived ratios, bin categoricals, one-hot encode, log-transform skewed columns
4. **Model Training** — Stratified split, train logistic regression + tree-based model, cross-validate, select best
5. **Evaluation** — Compute metrics, generate plots, select threshold, produce findings report
6. **Scoring** — Accept new applicant features, apply same transformations, return P(default)

## Components and Interfaces

### 1. Data Loader (`src/preprocess.py::load_dataset`)

**Responsibility:** Load raw CSV files into pandas DataFrames with column validation.

```python
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset, preserving all columns.
    
    Args:
        file_path: Path to the CSV file.
    
    Returns:
        DataFrame with all columns from the source file.
    
    Raises:
        FileNotFoundError: If file_path does not exist.
        IOError: If file cannot be read (permissions, corruption).
    """
```

**Design decisions:**
- Uses `pd.read_csv` with `low_memory=False` for the Lending Club dataset to avoid mixed-type inference
- For the Lending Club dataset (>100MB), uses `chunksize` parameter to process in batches and concatenate
- Error messages include the file path and the underlying OS error for debuggability

### 2. Preprocessor (`src/preprocess.py`)

**Responsibility:** Transform raw data into a clean, analysis-ready format.

```python
COLUMN_SCHEMA = {
    "german": {"mapping": {...}},  # German Credit → common schema
    "lending_club": {"mapping": {...}},  # Lending Club → common schema
}

AMBIGUOUS_STATUSES = {"Current", "In Grace Period", "Late (16-30 days)",
                      "Late (31-120 days)", "Does not meet the credit policy. Status:Charged Off",
                      "Does not meet the credit policy. Status:Fully Paid"}

def strip_formatting(series: pd.Series) -> pd.Series:
    """Remove $, %, commas from string series and cast to float."""

def parse_employment_length(series: pd.Series) -> pd.Series:
    """Convert employment length strings to integer years.
    
    Mapping:
        '< 1 year' → 0
        '1 year' → 1
        '2 years' → 2
        ...
        '10+ years' → 10
        NaN → NaN (preserved for downstream imputation)
    """

def encode_loan_status(series: pd.Series) -> pd.Series:
    """Encode 'Fully Paid' → 0, 'Charged Off' → 1, drop ambiguous."""

def preprocess(df: pd.DataFrame, dataset_type: str = "lending_club") -> pd.DataFrame:
    """Full preprocessing pipeline.
    
    Steps:
        1. Map columns to common schema (if needed)
        2. Strip formatting from numeric columns
        3. Parse employment length
        4. Filter ambiguous loan statuses
        5. Encode target variable
        6. Save processed.csv
    """
```

**Design decisions:**
- Column schema mapping is explicit and declarative — a dictionary maps source column names to common names
- Ambiguous statuses are defined as a constant set for easy extension
- Employment length parsing uses regex: `r'(\d+)'` to extract the numeric portion
- The preprocessor is stateless — it does not fit on training data, so it can be applied identically at scoring time

### 3. Feature Engine (`src/features.py`)

**Responsibility:** Compute derived features from cleaned data.

```python
SKEWNESS_THRESHOLD = 1.0

EMPLOYMENT_BINS = [0, 1, 3, 5, 10, float('inf')]
EMPLOYMENT_LABELS = ['<1yr', '1-3yr', '3-5yr', '5-10yr', '10+yr']

def compute_payment_to_income(df: pd.DataFrame) -> pd.Series:
    """monthly_payment / monthly_income. Returns 0.0 when income is 0."""

def compute_credit_utilisation(df: pd.DataFrame) -> pd.Series:
    """revolving_balance / revolving_credit_limit. Returns 0.0 when limit is 0."""

def compute_open_account_ratio(df: pd.DataFrame) -> pd.Series:
    """open_accounts / total_accounts. Returns 0.0 when total is 0."""

def bin_employment_length(series: pd.Series) -> pd.Series:
    """Bin integer employment length into categorical bands."""

def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply one-hot encoding, dropping the first category to avoid multicollinearity."""

def log_transform_skewed(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply log1p transform to columns with skewness > SKEWNESS_THRESHOLD."""

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline. Returns feature matrix ready for modelling."""
```

**Design decisions:**
- Division-by-zero is handled explicitly by returning 0.0 (not NaN) — this avoids downstream issues with models that cannot handle NaN
- `log1p` (log(1+x)) is used instead of `log` to handle zero values safely
- One-hot encoding drops the first category (`drop_first=True`) to prevent perfect multicollinearity in logistic regression
- Skewness threshold is a module-level constant for easy tuning
- The feature engine stores no state — all transformations are deterministic given the input DataFrame

### 4. Model Trainer (`src/model.py`)

**Responsibility:** Train, evaluate, and persist classification models.

```python
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

@dataclass
class TrainedModel:
    name: str
    model: Any  # fitted sklearn estimator
    cv_scores: np.ndarray  # 5-fold AUC-ROC scores
    cv_mean: float
    feature_names: list[str]

def stratified_split(X: pd.DataFrame, y: pd.Series) -> tuple:
    """80/20 stratified train/test split with fixed seed."""

def train_logistic_regression(X_train, y_train) -> TrainedModel:
    """Train logistic regression with class_weight='balanced'."""

def train_tree_model(X_train, y_train) -> TrainedModel:
    """Train Random Forest with class_weight='balanced_subsample'."""

def cross_validate_model(model, X_train, y_train) -> np.ndarray:
    """5-fold stratified CV returning AUC-ROC scores per fold."""

def select_best_model(models: list[TrainedModel]) -> TrainedModel:
    """Select model with highest mean CV AUC-ROC."""

def find_optimal_threshold(y_true, y_proba, cost_fp: float = 1.0, cost_fn: float = 5.0) -> float:
    """Find threshold minimizing expected cost: cost_fp * FP + cost_fn * FN."""
```

**Design decisions:**
- Fixed random seed (42) ensures reproducibility across runs
- Class imbalance is addressed via `class_weight='balanced'` in both models — this is simpler and more memory-efficient than SMOTE for large datasets
- The cost ratio (FN 5× more expensive than FP) reflects the business reality that approving a defaulter is costlier than rejecting a good applicant
- Models are wrapped in a `TrainedModel` dataclass for uniform handling
- Model persistence uses `joblib.dump` for efficient serialisation of numpy arrays

### 5. Evaluator (`src/model.py`)

**Responsibility:** Compute metrics, generate plots, produce findings report.

```python
@dataclass
class EvaluationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    feature_importances: dict[str, float]
    threshold: float

def evaluate_model(model: TrainedModel, X_test, y_test) -> EvaluationReport:
    """Compute all metrics on held-out test set."""

def plot_roc_curve(y_test, y_proba, model_name: str, output_path: str) -> None:
    """Generate and save ROC curve plot."""

def plot_precision_recall_curve(y_test, y_proba, model_name: str, output_path: str) -> None:
    """Generate and save precision-recall curve plot."""

def compute_feature_importance(model: TrainedModel, X_test, y_test) -> dict[str, float]:
    """Coefficients for logistic regression, permutation importance for tree models."""

def generate_findings_report(reports: list[EvaluationReport], output_path: str) -> None:
    """Write reports/findings.md with top features, metrics summary, threshold."""
```

### 6. Scorer (`src/model.py::score_applicant`)

**Responsibility:** Accept raw applicant features and return P(default).

```python
def score_applicant(
    features: dict | pd.Series,
    model_path: str = "models/best_model.joblib",
    feature_names_path: str = "models/feature_names.json"
) -> float:
    """
    Score a single applicant.
    
    Args:
        features: Dictionary or Series of raw applicant features.
        model_path: Path to persisted best model.
        feature_names_path: Path to expected feature names list.
    
    Returns:
        Float in [0.0, 1.0] representing probability of default.
    
    Raises:
        ValueError: If required features are missing. Error message lists
                    all missing feature names.
    """
```

**Design decisions:**
- The scorer applies the same preprocessing and feature engineering as training — this is achieved by calling the same functions from `src/preprocess.py` and `src/features.py`
- Feature names expected by the model are persisted alongside the model to enable validation at scoring time
- Input validation happens before any transformation — fail fast with a clear error
- Returns `model.predict_proba(X)[:, 1]` — the probability of the positive class (default)

## Data Models

### Common Schema (after column mapping)

| Column | Type | Description |
|--------|------|-------------|
| loan_amount | float | Funded loan amount |
| term | int | Loan term in months |
| interest_rate | float | Annual interest rate |
| monthly_payment | float | Monthly installment |
| grade | str | Lending Club grade (A–G) |
| employment_length | int | Years of employment |
| home_ownership | str | RENT, OWN, MORTGAGE |
| annual_income | float | Self-reported annual income |
| purpose | str | Loan purpose category |
| dti | float | Debt-to-income ratio |
| open_accounts | int | Number of open credit lines |
| total_accounts | int | Total number of credit lines |
| revolving_balance | float | Total revolving balance |
| revolving_credit_limit | float | Total revolving credit limit |
| loan_status | int | Target: 0 = Fully Paid, 1 = Charged Off |

### Derived Features (after feature engineering)

| Feature | Formula | Edge Case |
|---------|---------|-----------|
| payment_to_income | monthly_payment / (annual_income / 12) | 0.0 if income = 0 |
| credit_utilisation | revolving_balance / revolving_credit_limit | 0.0 if limit = 0 |
| open_account_ratio | open_accounts / total_accounts | 0.0 if total = 0 |
| employment_bin | pd.cut(employment_length, bins) | NaN preserved |

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| File not found | `FileNotFoundError` with file path in message |
| File unreadable | `IOError` with path and OS error description |
| Missing scoring features | `ValueError` listing all missing feature names |
| Division by zero in ratios | Return 0.0 (not NaN) |
| Unknown dataset type | `ValueError` listing supported types |
| Memory pressure (large dataset) | Process in chunks via `pd.read_csv(chunksize=...)` |

## Directory Structure

```
ds-pm-fin-p01/
├── data/
│   ├── raw/                    # Original CSV files (gitignored)
│   └── processed.csv           # Output of preprocessing
├── models/
│   ├── best_model.joblib       # Persisted best model
│   └── feature_names.json      # Expected feature names
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── reports/
│   └── findings.md
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── features.py
│   └── model.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_features.py
│   └── test_model.py
├── requirements.txt
├── goals.md
└── README.md
```

## Dependencies

```
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
joblib==1.3.2
imbalanced-learn==0.11.0
jupyter==1.0.0
pytest==7.4.3
hypothesis==6.92.1
```

## Testing Strategy

The project uses a dual testing approach:

- **Property-based tests** (via Hypothesis): Verify universal properties across randomly generated inputs — used for pure transformation functions in preprocessing, feature engineering, and scoring
- **Unit tests** (via pytest): Verify specific examples, integration points, and smoke checks — used for model training, evaluation, and file I/O
- **Integration tests**: End-to-end pipeline execution on sample data to verify components work together

Test files live in `tests/` and mirror the `src/` module structure:
- `tests/test_preprocess.py` — property tests for formatting, parsing, encoding, filtering
- `tests/test_features.py` — property tests for ratio computations, encoding, transforms
- `tests/test_model.py` — property tests for splitting, scoring; unit tests for training and evaluation

Property tests run a minimum of 100 iterations per property to ensure coverage of edge cases.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Column preservation during loading

*For any* valid CSV file with N columns, loading it via the data loader SHALL produce a DataFrame with exactly the same N column names as the source file.

**Validates: Requirements 1.1, 1.2**

### Property 2: Descriptive error on invalid file path

*For any* file path that does not exist or is unreadable, the data loader SHALL raise an error whose message contains the file path string.

**Validates: Requirements 1.3**

### Property 3: Numeric formatting strip round-trip

*For any* numeric value V and any combination of formatting characters ($, %, commas), formatting V with those characters and then applying `strip_formatting` SHALL produce a value equal to V (within floating-point tolerance).

**Validates: Requirements 2.1**

### Property 4: Employment length parsing

*For any* integer year value Y in [0, 10], the employment length string representation (e.g., "Y years", "< 1 year", "10+ years") parsed by the preprocessor SHALL produce the integer Y.

**Validates: Requirements 2.2**

### Property 5: Ambiguous status filtering

*For any* DataFrame containing rows with ambiguous loan statuses, after preprocessing the resulting DataFrame SHALL contain zero rows with any ambiguous status value.

**Validates: Requirements 2.3**

### Property 6: Loan status encoding correctness

*For any* DataFrame containing rows with "Fully Paid" or "Charged Off" statuses, after encoding ALL "Fully Paid" rows SHALL have target value 0 and ALL "Charged Off" rows SHALL have target value 1.

**Validates: Requirements 2.4, 2.5**

### Property 7: Payment-to-income computation

*For any* observation with monthly_payment P and annual_income I, the computed payment_to_income SHALL equal P / (I / 12) when I > 0, and SHALL equal 0.0 when I = 0.

**Validates: Requirements 4.1, 4.8**

### Property 8: Credit utilisation computation

*For any* observation with revolving_balance B and revolving_credit_limit L, the computed credit_utilisation SHALL equal B / L when L > 0, and SHALL equal 0.0 when L = 0.

**Validates: Requirements 4.2, 4.7**

### Property 9: Open account ratio computation

*For any* observation with open_accounts O and total_accounts T, the computed open_account_ratio SHALL equal O / T when T > 0, and SHALL equal 0.0 when T = 0.

**Validates: Requirements 4.3**

### Property 10: One-hot encoding binary invariant

*For any* DataFrame with categorical columns, after one-hot encoding each row SHALL have exactly one 1 among the indicator columns derived from each original categorical column (when using drop_first=False) or at most one 1 (when using drop_first=True), and all indicator values SHALL be either 0 or 1.

**Validates: Requirements 4.5**

### Property 11: Log transform reduces skewness

*For any* numeric column with skewness greater than 1.0, applying the log transformation SHALL produce a column with skewness strictly less than the original skewness.

**Validates: Requirements 4.6**

### Property 12: Stratified split preserves class distribution

*For any* dataset with a binary target, after an 80/20 stratified split the class ratio in the training set and the class ratio in the test set SHALL each be within 2 percentage points of the class ratio in the original dataset.

**Validates: Requirements 5.1**

### Property 13: Cross-validation produces exactly K folds

*For any* dataset and model, 5-fold stratified cross-validation SHALL return exactly 5 AUC-ROC scores, each in the range [0.0, 1.0].

**Validates: Requirements 5.4**

### Property 14: Reproducibility with fixed seed

*For any* dataset, running the full training pipeline twice with the same random seed SHALL produce identical train/test splits and identical model predictions.

**Validates: Requirements 5.6**

### Property 15: Optimal threshold minimizes expected cost

*For any* set of ground-truth labels and predicted probabilities, the selected threshold SHALL produce an expected cost (cost_fp × FP + cost_fn × FN) that is less than or equal to the expected cost at any other threshold in the candidate set.

**Validates: Requirements 6.4**

### Property 16: Scorer output range

*For any* valid applicant feature input (dictionary or pandas Series containing all required features), the scorer SHALL return a float value V where 0.0 ≤ V ≤ 1.0.

**Validates: Requirements 7.1, 7.3**

### Property 17: Scorer transformation consistency

*For any* raw applicant features, applying the preprocessing and feature engineering functions directly SHALL produce the same feature vector as the internal transformations applied by the scorer.

**Validates: Requirements 7.2**

### Property 18: Scorer missing features error

*For any* input that is missing at least one required feature, the scorer SHALL raise a ValueError whose message contains every missing feature name.

**Validates: Requirements 7.4**

### Property 19: Column schema mapping completeness

*For any* DataFrame with column names from either the German Credit or Lending Club schema, applying the column mapping SHALL produce a DataFrame whose columns are a subset of the common schema columns.

**Validates: Requirements 8.3**

### Property 20: Class imbalance ratio accuracy

*For any* dataset with known counts of Default (D) and Non_Default (N) observations, the reported class imbalance ratio SHALL equal D / N (within floating-point tolerance).

**Validates: Requirements 3.4**

### Property 21: Pinned dependency versions

*For any* line in requirements.txt that declares a dependency, the line SHALL contain an exact version pin using the `==` specifier.

**Validates: Requirements 9.5**
