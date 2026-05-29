# Implementation Plan: LendSafe Loan Default Predictor

## Overview

This plan implements the LendSafe consumer loan default prediction pipeline in Python. The implementation follows the hybrid code organisation: reusable modules in `src/` for the deterministic pipeline (preprocessing, feature engineering, model training, scoring) and Jupyter notebooks in `notebooks/` for exploratory data analysis. Tasks are ordered to build incrementally — project structure first, then data loading, preprocessing, feature engineering, modelling, evaluation, scoring, and finally integration wiring.

## Tasks

- [x] 1. Set up project structure and dependencies
  - [x] 1.1 Create directory structure and initialise Python package
    - Create `src/`, `tests/`, `data/raw/`, `models/`, `notebooks/`, `reports/` directories
    - Create `src/__init__.py` and `tests/__init__.py`
    - Create `requirements.txt` with all pinned dependencies from the design (pandas==2.1.4, numpy==1.26.2, scikit-learn==1.3.2, matplotlib==3.8.2, seaborn==0.13.0, joblib==1.3.2, imbalanced-learn==0.11.0, jupyter==1.0.0, pytest==7.4.3, hypothesis==6.92.1)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 2. Implement data loading and preprocessing
  - [x] 2.1 Implement `load_dataset` in `src/preprocess.py`
    - Create `src/preprocess.py` with `load_dataset(file_path: str) -> pd.DataFrame`
    - Use `pd.read_csv` with `low_memory=False`
    - For large files (>100MB), use `chunksize` parameter to process in batches
    - Raise `FileNotFoundError` with file path in message if file missing
    - Raise `IOError` with path and OS error if file unreadable
    - _Requirements: 1.1, 1.2, 1.3, 8.2_

  - [x]* 2.2 Write property tests for data loading
    - **Property 1: Column preservation during loading**
    - **Property 2: Descriptive error on invalid file path**
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [x] 2.3 Implement preprocessing functions in `src/preprocess.py`
    - Implement `COLUMN_SCHEMA` dictionary for German Credit and Lending Club column mappings
    - Implement `AMBIGUOUS_STATUSES` constant set
    - Implement `strip_formatting(series)` to remove $, %, commas and cast to float
    - Implement `parse_employment_length(series)` using regex to extract integer years
    - Implement `encode_loan_status(series)` to encode Fully Paid → 0, Charged Off → 1
    - Implement `preprocess(df, dataset_type)` orchestrating the full pipeline and saving `data/processed.csv`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 8.3_

  - [x]* 2.4 Write property tests for preprocessing
    - **Property 3: Numeric formatting strip round-trip**
    - **Property 4: Employment length parsing**
    - **Property 5: Ambiguous status filtering**
    - **Property 6: Loan status encoding correctness**
    - **Property 19: Column schema mapping completeness**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 8.3**

- [x] 3. Checkpoint - Ensure data loading and preprocessing tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement feature engineering
  - [x] 4.1 Implement feature computation functions in `src/features.py`
    - Create `src/features.py` with module-level constants (`SKEWNESS_THRESHOLD`, `EMPLOYMENT_BINS`, `EMPLOYMENT_LABELS`)
    - Implement `compute_payment_to_income(df)` returning 0.0 when income is 0
    - Implement `compute_credit_utilisation(df)` returning 0.0 when limit is 0
    - Implement `compute_open_account_ratio(df)` returning 0.0 when total_accounts is 0
    - Implement `bin_employment_length(series)` using `pd.cut` with defined bins
    - Implement `one_hot_encode(df, columns)` with `drop_first=True`
    - Implement `log_transform_skewed(df, columns)` using `np.log1p` for columns with skewness > threshold
    - Implement `engineer_features(df)` orchestrating the full feature pipeline
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

  - [x]* 4.2 Write property tests for feature engineering
    - **Property 7: Payment-to-income computation**
    - **Property 8: Credit utilisation computation**
    - **Property 9: Open account ratio computation**
    - **Property 10: One-hot encoding binary invariant**
    - **Property 11: Log transform reduces skewness**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.5, 4.6, 4.7, 4.8**

- [ ] 5. Implement model training and evaluation
  - [x] 5.1 Implement model training functions in `src/model.py`
    - Create `src/model.py` with constants (`RANDOM_SEED=42`, `TEST_SIZE=0.2`, `CV_FOLDS=5`)
    - Define `TrainedModel` dataclass (name, model, cv_scores, cv_mean, feature_names)
    - Implement `stratified_split(X, y)` using `train_test_split` with stratify and fixed seed
    - Implement `train_logistic_regression(X_train, y_train)` with `class_weight='balanced'`
    - Implement `train_tree_model(X_train, y_train)` using Random Forest with `class_weight='balanced_subsample'`
    - Implement `cross_validate_model(model, X_train, y_train)` with 5-fold stratified CV returning AUC-ROC
    - Implement `select_best_model(models)` selecting highest mean CV AUC-ROC
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x]* 5.2 Write property tests for model training
    - **Property 12: Stratified split preserves class distribution**
    - **Property 13: Cross-validation produces exactly K folds**
    - **Property 14: Reproducibility with fixed seed**
    - **Validates: Requirements 5.1, 5.4, 5.6**

  - [x] 5.3 Implement model evaluation and reporting in `src/model.py`
    - Define `EvaluationReport` dataclass (accuracy, precision, recall, f1, auc_roc, feature_importances, threshold)
    - Implement `evaluate_model(model, X_test, y_test)` computing all metrics
    - Implement `plot_roc_curve(y_test, y_proba, model_name, output_path)`
    - Implement `plot_precision_recall_curve(y_test, y_proba, model_name, output_path)`
    - Implement `compute_feature_importance(model, X_test, y_test)` — coefficients for LR, permutation importance for tree
    - Implement `find_optimal_threshold(y_true, y_proba, cost_fp=1.0, cost_fn=5.0)` minimizing expected cost
    - Implement `generate_findings_report(reports, output_path)` writing `reports/findings.md`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 5.4 Write property test for threshold optimisation
    - **Property 15: Optimal threshold minimizes expected cost**
    - **Validates: Requirements 6.4**

- [ ] 6. Checkpoint - Ensure model training and evaluation tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement scoring function
  - [ ] 7.1 Implement `score_applicant` in `src/model.py`
    - Implement `score_applicant(features, model_path, feature_names_path)` function
    - Load persisted model from `models/best_model.joblib`
    - Load expected feature names from `models/feature_names.json`
    - Validate all required features are present — raise `ValueError` listing missing names if not
    - Apply same preprocessing and feature engineering transformations as training
    - Return `model.predict_proba(X)[:, 1]` as float in [0.0, 1.0]
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 7.2 Write property tests for scoring function
    - **Property 16: Scorer output range**
    - **Property 17: Scorer transformation consistency**
    - **Property 18: Scorer missing features error**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

- [ ] 8. Implement exploratory data analysis notebooks
  - [ ] 8.1 Create EDA notebook `notebooks/01_eda.ipynb`
    - Import from `src.preprocess` to load and clean data
    - Produce univariate distribution plots for annual_income, loan_amount, and dti
    - Produce default rate breakdowns by grade, purpose, and home_ownership
    - Produce correlation heatmap for numeric features
    - Compute and report class imbalance ratio (Default / Non_Default)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 9.4_

  - [ ]* 8.2 Write property test for class imbalance ratio
    - **Property 20: Class imbalance ratio accuracy**
    - **Validates: Requirements 3.4**

  - [ ] 8.3 Create feature engineering notebook `notebooks/02_feature_engineering.ipynb`
    - Import from `src.features` to compute derived features
    - Visualise derived feature distributions
    - Validate feature engineering outputs
    - _Requirements: 9.4_

  - [ ] 8.4 Create modelling notebook `notebooks/03_modelling.ipynb`
    - Import from `src.model` to train and evaluate models
    - Run full training pipeline and display evaluation results
    - Compare logistic regression vs tree-based model performance
    - _Requirements: 9.4_

- [ ] 9. Integration and pipeline wiring
  - [ ] 9.1 Wire end-to-end pipeline execution
    - Create a pipeline runner (script or notebook cell) that executes: load → preprocess → engineer features → train → evaluate → persist model
    - Persist best model to `models/best_model.joblib`
    - Persist feature names to `models/feature_names.json`
    - Verify pipeline completes on German Credit dataset within 60 seconds
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ]* 9.2 Write integration tests for end-to-end pipeline
    - Test full pipeline on small sample data
    - Verify processed.csv is created
    - Verify model file is persisted
    - Verify scorer returns valid probability
    - _Requirements: 8.1_

- [ ] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The pipeline uses Python with pandas, scikit-learn, and Hypothesis for property-based testing
- All modules in `src/` are stateless — transformations are deterministic given inputs

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4", "4.1"] },
    { "id": 4, "tasks": ["4.2", "5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3"] },
    { "id": 6, "tasks": ["5.4", "7.1"] },
    { "id": 7, "tasks": ["7.2", "8.1", "8.3", "8.4"] },
    { "id": 8, "tasks": ["8.2", "9.1"] },
    { "id": 9, "tasks": ["9.2"] }
  ]
}
```
