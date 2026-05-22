# LendSafe — Consumer Loan Default Predictor

Predict whether a loan applicant will default, delivering a probability-of-default score and a report explaining which features matter most.

## Goals

| # | Learning Objective | Status |
|---|---|---|
| 1 | Python + pandas for data wrangling | ⬜ |
| 2 | Exploratory data analysis (EDA) | ⬜ |
| 3 | Logistic regression (first predictive model) | ⬜ |
| 4 | Feature engineering (income ratios, employment length, credit utilisation) | ⬜ |
| 5 | Train/test split, cross-validation | ⬜ |
| 6 | Evaluation metrics: accuracy, precision, recall, AUC-ROC | ⬜ |
| 7 | Handling imbalanced classes | ⬜ |

## Dataset

**Primary:** Lending Club "accepted loans" (2007–2018, ~2M rows)  
**Alternative:** German Credit dataset (1,000 rows — useful for fast iteration)

Target variable: `loan_status` → binary (Fully Paid = 0, Charged Off = 1).

## Project Plan

### Phase 1 — Data Acquisition & Wrangling

- Download and load raw dataset.
- Strip formatting from numeric columns (e.g. `%` in interest rate).
- Parse employment length ("10+ years" → 10).
- Define binary target; drop ambiguous statuses (current, in grace period).
- Output: clean `processed.csv` ready for analysis.

### Phase 2 — Exploratory Data Analysis

- Univariate distributions: income, loan amount, DTI.
- Default rate segmented by grade, purpose, home ownership.
- Correlation heatmap to identify multicollinearity.
- Quantify class imbalance.

### Phase 3 — Feature Engineering

- `payment_to_income` = monthly payment / monthly income
- `credit_utilisation` = revolving balance / revolving credit limit
- `open_account_ratio` = open accounts / total accounts
- Bin employment length; one-hot encode categoricals.
- Log-transform skewed features (annual income, revolving balance).

### Phase 4 — Modelling

- Baseline: logistic regression (`sklearn`).
- 80/20 stratified train/test split.
- 5-fold stratified cross-validation.
- Address imbalance: class weights, SMOTE, threshold tuning.

### Phase 5 — Evaluation & Reporting

- Metrics: accuracy, precision, recall, F1, AUC-ROC.
- ROC curve and precision-recall curve.
- Select operating threshold based on business cost assumptions.
- Feature importance via coefficients + permutation importance.

### Phase 6 — Deliverable

- `score_applicant(features)` function returning P(default).
- One-page findings report: top predictive features, performance summary, recommended threshold.

## Repo Structure

```
ds-pm-fin-p01/
├── data/              # raw + processed CSVs (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── model.py
├── reports/
│   └── findings.md
├── requirements.txt
├── goals.md
└── README.md
```

## Getting Started

```bash
# Create environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter lab notebooks/
```

## License

For educational use only. Dataset subject to its own licensing terms.
