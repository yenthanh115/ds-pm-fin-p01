# Requirements Document

## Introduction

LendSafe is a consumer loan default prediction system that predicts whether a loan applicant will default on their loan. The system delivers a probability-of-default score and a report explaining which features contribute most to the prediction. The pipeline supports two datasets — the German Credit dataset (1,000 rows) for fast iteration and the Lending Club dataset (~2M rows) for production-scale validation. The system uses a hybrid code organisation: Jupyter notebooks for EDA and exploration, and reusable Python modules in `src/` for the core pipeline (preprocessing, feature engineering, model training, scoring).

## Glossary

- **Pipeline**: The end-to-end sequence of data loading, preprocessing, feature engineering, model training, evaluation, and scoring
- **Preprocessor**: The Python module (`src/preprocess.py`) responsible for cleaning raw data and producing analysis-ready datasets
- **Feature_Engine**: The Python module (`src/features.py`) responsible for computing derived features from cleaned data
- **Model_Trainer**: The Python module (`src/model.py`) responsible for training, evaluating, and persisting machine learning models
- **Scorer**: The `score_applicant(features)` function that accepts applicant features and returns a probability of default
- **Default**: A loan outcome where the borrower fails to repay (loan_status = Charged Off, encoded as 1)
- **Non_Default**: A loan outcome where the borrower fully repays (loan_status = Fully Paid, encoded as 0)
- **German_Credit_Dataset**: A publicly available dataset of 1,000 loan records used for fast iteration
- **Lending_Club_Dataset**: A publicly available dataset of approximately 2 million accepted loan records (2007–2018)
- **AUC_ROC**: Area Under the Receiver Operating Characteristic curve, a threshold-independent classification metric
- **SMOTE**: Synthetic Minority Over-sampling Technique, a method for addressing class imbalance
- **Stratified_Split**: A train/test partitioning method that preserves the class distribution of the target variable

## Requirements

### Requirement 1: Data Acquisition and Loading

**User Story:** As a data scientist, I want to load and parse raw loan datasets, so that I have structured data ready for cleaning.

#### Acceptance Criteria

1. WHEN the German_Credit_Dataset file path is provided, THE Pipeline SHALL load the dataset into a pandas DataFrame with all columns preserved.
2. WHEN the Lending_Club_Dataset file path is provided, THE Pipeline SHALL load the dataset into a pandas DataFrame with all columns preserved.
3. IF a dataset file is missing or unreadable, THEN THE Pipeline SHALL raise a descriptive error indicating the file path and the nature of the failure.

### Requirement 2: Data Preprocessing

**User Story:** As a data scientist, I want raw data cleaned and transformed into a consistent format, so that downstream feature engineering and modelling receive valid inputs.

#### Acceptance Criteria

1. WHEN raw data is loaded, THE Preprocessor SHALL strip formatting characters (e.g., `%`, `$`, commas) from numeric columns and cast them to numeric types.
2. WHEN the employment length column contains string values (e.g., "10+ years", "< 1 year"), THE Preprocessor SHALL parse the values into integer years.
3. WHEN the loan_status column contains ambiguous statuses (e.g., "Current", "In Grace Period"), THE Preprocessor SHALL drop those rows from the dataset.
4. WHEN the loan_status column contains "Fully Paid", THE Preprocessor SHALL encode the value as 0.
5. WHEN the loan_status column contains "Charged Off", THE Preprocessor SHALL encode the value as 1.
6. THE Preprocessor SHALL output a cleaned CSV file named `processed.csv` in the `data/` directory.

### Requirement 3: Exploratory Data Analysis

**User Story:** As a data scientist, I want to visualise distributions and relationships in the data, so that I can identify patterns, outliers, and class imbalance before modelling.

#### Acceptance Criteria

1. THE Pipeline SHALL produce univariate distribution plots for annual income, loan amount, and debt-to-income ratio.
2. THE Pipeline SHALL produce default rate breakdowns segmented by loan grade, loan purpose, and home ownership status.
3. THE Pipeline SHALL produce a correlation heatmap to identify multicollinearity among numeric features.
4. THE Pipeline SHALL quantify and report the class imbalance ratio between Default and Non_Default observations.

### Requirement 4: Feature Engineering

**User Story:** As a data scientist, I want derived features that capture financial risk signals, so that the model has informative predictors beyond raw columns.

#### Acceptance Criteria

1. THE Feature_Engine SHALL compute `payment_to_income` as monthly payment divided by monthly income for each observation.
2. THE Feature_Engine SHALL compute `credit_utilisation` as revolving balance divided by revolving credit limit for each observation.
3. THE Feature_Engine SHALL compute `open_account_ratio` as the number of open accounts divided by total accounts for each observation.
4. WHEN the employment length column is present, THE Feature_Engine SHALL bin employment length into discrete categories.
5. WHEN categorical columns are present, THE Feature_Engine SHALL apply one-hot encoding to produce binary indicator columns.
6. WHEN annual income or revolving balance columns exhibit skewness greater than 1.0, THE Feature_Engine SHALL apply a log transformation to reduce skewness.
7. IF revolving credit limit is zero for an observation, THEN THE Feature_Engine SHALL assign a credit_utilisation value of 0.0 for that observation to avoid division by zero.
8. IF monthly income is zero for an observation, THEN THE Feature_Engine SHALL assign a payment_to_income value of 0.0 for that observation to avoid division by zero.

### Requirement 5: Model Training

**User Story:** As a data scientist, I want to train a baseline logistic regression and a tree-based comparison model, so that I can evaluate which approach better predicts loan default.

#### Acceptance Criteria

1. THE Model_Trainer SHALL split the dataset into 80% training and 20% test sets using a Stratified_Split on the target variable.
2. THE Model_Trainer SHALL train a logistic regression model using scikit-learn on the training set.
3. THE Model_Trainer SHALL train one tree-based model (Random Forest or XGBoost) on the training set.
4. THE Model_Trainer SHALL perform 5-fold stratified cross-validation on the training set for each model.
5. WHEN class imbalance is present, THE Model_Trainer SHALL apply at least one imbalance mitigation technique from: class weights, SMOTE, or threshold tuning.
6. THE Model_Trainer SHALL use a fixed random seed for reproducibility of splits and model training.

### Requirement 6: Model Evaluation and Reporting

**User Story:** As a data scientist, I want comprehensive evaluation metrics and visualisations, so that I can select the best model and communicate results to stakeholders.

#### Acceptance Criteria

1. THE Model_Trainer SHALL compute accuracy, precision, recall, F1 score, and AUC_ROC on the held-out test set for each trained model.
2. THE Model_Trainer SHALL produce an ROC curve plot for each trained model.
3. THE Model_Trainer SHALL produce a precision-recall curve plot for each trained model.
4. THE Model_Trainer SHALL select an operating threshold based on business cost assumptions that balance false positives and false negatives.
5. THE Model_Trainer SHALL compute feature importance using logistic regression coefficients for the baseline model.
6. THE Model_Trainer SHALL compute feature importance using permutation importance for the tree-based model.
7. THE Pipeline SHALL generate a one-page findings report in `reports/findings.md` containing: top predictive features, performance summary, and recommended threshold.

### Requirement 7: Scoring Function

**User Story:** As a consumer of the model, I want a single function that accepts applicant features and returns a default probability, so that the model can be integrated into downstream systems.

#### Acceptance Criteria

1. THE Scorer SHALL accept a dictionary or pandas Series of applicant features as input.
2. THE Scorer SHALL apply the same preprocessing and feature engineering transformations used during training to the input features.
3. THE Scorer SHALL return a float value between 0.0 and 1.0 representing the probability of default.
4. IF required input features are missing from the input, THEN THE Scorer SHALL raise a descriptive error listing the missing feature names.
5. THE Scorer SHALL use the trained model with the highest cross-validation AUC_ROC for generating predictions.

### Requirement 8: Pipeline Scalability

**User Story:** As a data scientist, I want the pipeline to work on both the small German Credit dataset and the large Lending Club dataset, so that I can iterate quickly and then validate at scale.

#### Acceptance Criteria

1. THE Pipeline SHALL execute end-to-end (preprocessing through scoring) on the German_Credit_Dataset within 60 seconds on a standard development machine.
2. THE Pipeline SHALL execute end-to-end (preprocessing through scoring) on the Lending_Club_Dataset without exceeding available system memory by processing data in chunks where necessary.
3. WHEN the Lending_Club_Dataset is used, THE Preprocessor SHALL handle column name differences between the two datasets by mapping to a common schema.

### Requirement 9: Code Organisation

**User Story:** As a data scientist, I want a clear separation between exploratory notebooks and reusable pipeline code, so that the project remains maintainable and reproducible.

#### Acceptance Criteria

1. THE Pipeline SHALL expose preprocessing logic as importable functions in `src/preprocess.py`.
2. THE Pipeline SHALL expose feature engineering logic as importable functions in `src/features.py`.
3. THE Pipeline SHALL expose model training and scoring logic as importable functions in `src/model.py`.
4. THE Pipeline SHALL provide Jupyter notebooks in `notebooks/` for EDA and exploration that import from the `src/` modules.
5. THE Pipeline SHALL declare all Python dependencies with pinned versions in `requirements.txt`.
