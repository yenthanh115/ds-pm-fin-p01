"""Feature engineering module for LendSafe pipeline."""

import numpy as np
import pandas as pd


# --- Module-level constants ---

SKEWNESS_THRESHOLD = 1.0

EMPLOYMENT_BINS = [0, 1, 3, 5, 10, float("inf")]
EMPLOYMENT_LABELS = ["<1yr", "1-3yr", "3-5yr", "5-10yr", "10+yr"]


def compute_payment_to_income(df: pd.DataFrame) -> pd.Series:
    """Compute payment-to-income ratio for each observation.

    Formula: monthly_payment / (annual_income / 12)
    Returns 0.0 when annual_income is 0 to avoid division by zero.

    Args:
        df: DataFrame with 'monthly_payment' and 'annual_income' columns.

    Returns:
        Series with payment-to-income ratio values.
    """
    monthly_income = df["annual_income"] / 12.0
    result = np.where(
        monthly_income == 0,
        0.0,
        df["monthly_payment"] / monthly_income,
    )
    return pd.Series(result, index=df.index, name="payment_to_income")


def compute_credit_utilisation(df: pd.DataFrame) -> pd.Series:
    """Compute credit utilisation ratio for each observation.

    Formula: revolving_balance / revolving_credit_limit
    Returns 0.0 when revolving_credit_limit is 0 to avoid division by zero.

    Args:
        df: DataFrame with 'revolving_balance' and 'revolving_credit_limit' columns.

    Returns:
        Series with credit utilisation ratio values.
    """
    result = np.where(
        df["revolving_credit_limit"] == 0,
        0.0,
        df["revolving_balance"] / df["revolving_credit_limit"],
    )
    return pd.Series(result, index=df.index, name="credit_utilisation")


def compute_open_account_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute open account ratio for each observation.

    Formula: open_accounts / total_accounts
    Returns 0.0 when total_accounts is 0 to avoid division by zero.

    Args:
        df: DataFrame with 'open_accounts' and 'total_accounts' columns.

    Returns:
        Series with open account ratio values.
    """
    result = np.where(
        df["total_accounts"] == 0,
        0.0,
        df["open_accounts"] / df["total_accounts"],
    )
    return pd.Series(result, index=df.index, name="open_account_ratio")


def bin_employment_length(series: pd.Series) -> pd.Series:
    """Bin integer employment length into categorical bands.

    Bins: [0,1), [1,3), [3,5), [5,10), [10, inf)
    Labels: '<1yr', '1-3yr', '3-5yr', '5-10yr', '10+yr'

    Args:
        series: A pandas Series of integer employment length values.

    Returns:
        A categorical Series with employment length bins.
    """
    return pd.cut(
        series,
        bins=EMPLOYMENT_BINS,
        labels=EMPLOYMENT_LABELS,
        right=False,
    )


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply one-hot encoding to specified categorical columns.

    Uses drop_first=True to avoid multicollinearity in linear models.

    Args:
        df: DataFrame containing the columns to encode.
        columns: List of column names to one-hot encode.

    Returns:
        DataFrame with original categorical columns replaced by binary indicators.
    """
    return pd.get_dummies(df, columns=columns, drop_first=True)


def log_transform_skewed(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply log1p transform to columns with skewness above threshold.

    Only transforms columns whose skewness exceeds SKEWNESS_THRESHOLD.
    Uses np.log1p (log(1+x)) to safely handle zero values.

    Args:
        df: DataFrame containing numeric columns.
        columns: List of column names to check and potentially transform.

    Returns:
        DataFrame with skewed columns log-transformed.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            skewness = df[col].skew()
            if skewness > SKEWNESS_THRESHOLD:
                df[col] = np.log1p(df[col])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Orchestrates all feature transformations:
        1. Compute payment_to_income ratio
        2. Compute credit_utilisation ratio
        3. Compute open_account_ratio
        4. Bin employment_length into categories
        5. Log-transform skewed numeric columns
        6. One-hot encode categorical columns

    Args:
        df: Cleaned DataFrame from preprocessing (common schema).

    Returns:
        Feature matrix ready for modelling (all numeric, no NaN from division).
    """
    df = df.copy()

    # Step 1-3: Compute derived ratio features
    df["payment_to_income"] = compute_payment_to_income(df)
    df["credit_utilisation"] = compute_credit_utilisation(df)
    df["open_account_ratio"] = compute_open_account_ratio(df)

    # Step 4: Bin employment length
    if "employment_length" in df.columns:
        df["employment_bin"] = bin_employment_length(df["employment_length"])
        df = df.drop(columns=["employment_length"])

    # Step 5: Log-transform skewed numeric columns
    numeric_cols_to_check = [
        "annual_income",
        "revolving_balance",
        "loan_amount",
        "monthly_payment",
    ]
    cols_present = [c for c in numeric_cols_to_check if c in df.columns]
    df = log_transform_skewed(df, cols_present)

    # Step 6: One-hot encode categorical columns
    categorical_cols = ["grade", "home_ownership", "purpose"]
    if "employment_bin" in df.columns:
        categorical_cols.append("employment_bin")
    cats_present = [c for c in categorical_cols if c in df.columns]
    if cats_present:
        df = one_hot_encode(df, cats_present)

    return df
