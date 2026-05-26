"""Data loading and preprocessing module for LendSafe pipeline."""

import os
import re

import pandas as pd


# File size threshold (100MB) for switching to chunked reading
_LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100 MB
_CHUNK_SIZE = 50_000  # rows per chunk for large files


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
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    try:
        file_size = os.path.getsize(file_path)

        if file_size > _LARGE_FILE_THRESHOLD:
            # Process large files in chunks to avoid memory pressure
            chunks = pd.read_csv(
                file_path, low_memory=False, chunksize=_CHUNK_SIZE
            )
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path, low_memory=False)

    except FileNotFoundError:
        raise
    except Exception as e:
        raise IOError(
            f"Unable to read file '{file_path}': {e}"
        ) from e

    return df


# --- Column Schema Mapping ---

COLUMN_SCHEMA = {
    "german": {
        "mapping": {
            "credit_amount": "loan_amount",
            "duration": "term",
            "installment_rate": "interest_rate",
            "installment_commitment": "monthly_payment",
            "class": "grade",
            "employment": "employment_length",
            "housing": "home_ownership",
            "income": "annual_income",
            "purpose": "purpose",
            "existing_credits": "dti",
            "num_dependents": "open_accounts",
            "number_of_credits": "total_accounts",
            "savings_status": "revolving_balance",
            "checking_status": "revolving_credit_limit",
            "credit_risk": "loan_status",
        }
    },
    "lending_club": {
        "mapping": {
            "loan_amnt": "loan_amount",
            "term": "term",
            "int_rate": "interest_rate",
            "installment": "monthly_payment",
            "grade": "grade",
            "emp_length": "employment_length",
            "home_ownership": "home_ownership",
            "annual_inc": "annual_income",
            "purpose": "purpose",
            "dti": "dti",
            "open_acc": "open_accounts",
            "total_acc": "total_accounts",
            "revol_bal": "revolving_balance",
            "revol_util": "revolving_credit_limit",
            "loan_status": "loan_status",
        }
    },
}

# Loan statuses that are neither clearly "Fully Paid" nor "Charged Off"
AMBIGUOUS_STATUSES = {
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
    "Does not meet the credit policy. Status:Fully Paid",
}


def strip_formatting(series: pd.Series) -> pd.Series:
    """Remove $, %, commas from string series and cast to float.

    Args:
        series: A pandas Series potentially containing formatted numeric strings.

    Returns:
        A pandas Series with formatting characters removed and values cast to float.
    """
    cleaned = series.astype(str).str.replace("$", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def parse_employment_length(series: pd.Series) -> pd.Series:
    """Convert employment length strings to integer years.

    Mapping:
        '< 1 year' → 0
        '1 year' → 1
        '2 years' → 2
        ...
        '10+ years' → 10
        NaN → NaN (preserved for downstream imputation)

    Args:
        series: A pandas Series containing employment length strings.

    Returns:
        A pandas Series with integer year values (or NaN).
    """
    def _parse_value(val):
        if pd.isna(val):
            return float("nan")
        val_str = str(val).strip()
        if val_str.lower().startswith("< 1"):
            return 0
        match = re.search(r"(\d+)", val_str)
        if match:
            return int(match.group(1))
        return float("nan")

    return series.apply(_parse_value).astype(float)


def encode_loan_status(series: pd.Series) -> pd.Series:
    """Encode loan status to binary target variable.

    'Fully Paid' → 0
    'Charged Off' → 1

    Args:
        series: A pandas Series containing loan status strings.

    Returns:
        A pandas Series with encoded integer values (0 or 1).
    """
    mapping = {"Fully Paid": 0, "Charged Off": 1}
    return series.map(mapping)


def preprocess(df: pd.DataFrame, dataset_type: str = "lending_club") -> pd.DataFrame:
    """Full preprocessing pipeline.

    Steps:
        1. Map columns to common schema (if needed)
        2. Strip formatting from numeric columns
        3. Parse employment length
        4. Filter ambiguous loan statuses
        5. Encode target variable
        6. Save processed.csv

    Args:
        df: Raw DataFrame loaded from CSV.
        dataset_type: One of 'german' or 'lending_club'.

    Returns:
        Cleaned DataFrame with common schema columns.

    Raises:
        ValueError: If dataset_type is not supported.
    """
    if dataset_type not in COLUMN_SCHEMA:
        raise ValueError(
            f"Unsupported dataset_type '{dataset_type}'. "
            f"Supported types: {list(COLUMN_SCHEMA.keys())}"
        )

    # Step 1: Map columns to common schema
    mapping = COLUMN_SCHEMA[dataset_type]["mapping"]
    # Only rename columns that exist in the DataFrame
    rename_map = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Keep only common schema columns that are present
    common_columns = list(mapping.values())
    available_columns = [col for col in common_columns if col in df.columns]
    df = df[available_columns].copy()

    # Step 2: Strip formatting from numeric columns
    numeric_cols = ["loan_amount", "interest_rate", "monthly_payment",
                    "annual_income", "dti", "revolving_balance",
                    "revolving_credit_limit"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = strip_formatting(df[col])

    # Step 3: Parse employment length
    if "employment_length" in df.columns:
        df["employment_length"] = parse_employment_length(df["employment_length"])

    # Step 4: Filter ambiguous loan statuses
    if "loan_status" in df.columns:
        df = df[~df["loan_status"].isin(AMBIGUOUS_STATUSES)].copy()

    # Step 5: Encode target variable
    if "loan_status" in df.columns:
        df["loan_status"] = encode_loan_status(df["loan_status"])
        # Drop rows where loan_status couldn't be mapped (neither Fully Paid nor Charged Off)
        df = df.dropna(subset=["loan_status"]).copy()
        df["loan_status"] = df["loan_status"].astype(int)

    # Step 6: Save processed.csv
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed.csv")
    df.to_csv(output_path, index=False)

    return df
