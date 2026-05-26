"""Property-based tests for src/features.py - feature engineering functionality."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.features import (
    compute_payment_to_income,
    compute_credit_utilisation,
    compute_open_account_ratio,
    one_hot_encode,
    log_transform_skewed,
    SKEWNESS_THRESHOLD,
)


# --- Strategies ---

# Strategy for non-negative financial values (payments, incomes, balances)
_positive_float = st.floats(min_value=0.01, max_value=1e8, allow_nan=False, allow_infinity=False)
_non_negative_float = st.floats(min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False)
_non_negative_int = st.integers(min_value=0, max_value=1000)
_positive_int = st.integers(min_value=1, max_value=1000)


# --- Property 7: Payment-to-income computation ---


class TestPaymentToIncomeProperties:
    """Property-based tests for compute_payment_to_income."""

    @given(
        payment=_positive_float,
        income=_positive_float,
    )
    @settings(max_examples=100)
    def test_payment_to_income_positive_income(self, payment, income):
        """Property 7: Payment-to-income computation (positive income case).

        For any observation with monthly_payment P and annual_income I,
        the computed payment_to_income SHALL equal P / (I / 12) when I > 0.

        **Validates: Requirements 4.1, 4.8**
        """
        df = pd.DataFrame({
            "monthly_payment": [payment],
            "annual_income": [income],
        })

        result = compute_payment_to_income(df)

        expected = payment / (income / 12.0)
        assert np.isclose(result.iloc[0], expected, rtol=1e-7), (
            f"Expected {expected}, got {result.iloc[0]} for payment={payment}, income={income}"
        )

    @given(payment=_non_negative_float)
    @settings(max_examples=100)
    def test_payment_to_income_zero_income(self, payment):
        """Property 7: Payment-to-income computation (zero income case).

        For any observation with monthly_payment P and annual_income I,
        the computed payment_to_income SHALL equal 0.0 when I = 0.

        **Validates: Requirements 4.1, 4.8**
        """
        df = pd.DataFrame({
            "monthly_payment": [payment],
            "annual_income": [0.0],
        })

        result = compute_payment_to_income(df)

        assert result.iloc[0] == 0.0, (
            f"Expected 0.0 when income is 0, got {result.iloc[0]}"
        )


# --- Property 8: Credit utilisation computation ---


class TestCreditUtilisationProperties:
    """Property-based tests for compute_credit_utilisation."""

    @given(
        balance=_non_negative_float,
        limit=_positive_float,
    )
    @settings(max_examples=100)
    def test_credit_utilisation_positive_limit(self, balance, limit):
        """Property 8: Credit utilisation computation (positive limit case).

        For any observation with revolving_balance B and revolving_credit_limit L,
        the computed credit_utilisation SHALL equal B / L when L > 0.

        **Validates: Requirements 4.2, 4.7**
        """
        df = pd.DataFrame({
            "revolving_balance": [balance],
            "revolving_credit_limit": [limit],
        })

        result = compute_credit_utilisation(df)

        expected = balance / limit
        assert np.isclose(result.iloc[0], expected, rtol=1e-7), (
            f"Expected {expected}, got {result.iloc[0]} for balance={balance}, limit={limit}"
        )

    @given(balance=_non_negative_float)
    @settings(max_examples=100)
    def test_credit_utilisation_zero_limit(self, balance):
        """Property 8: Credit utilisation computation (zero limit case).

        For any observation with revolving_balance B and revolving_credit_limit L,
        the computed credit_utilisation SHALL equal 0.0 when L = 0.

        **Validates: Requirements 4.2, 4.7**
        """
        df = pd.DataFrame({
            "revolving_balance": [balance],
            "revolving_credit_limit": [0.0],
        })

        result = compute_credit_utilisation(df)

        assert result.iloc[0] == 0.0, (
            f"Expected 0.0 when limit is 0, got {result.iloc[0]}"
        )


# --- Property 9: Open account ratio computation ---


class TestOpenAccountRatioProperties:
    """Property-based tests for compute_open_account_ratio."""

    @given(
        open_accounts=_non_negative_int,
        total_accounts=_positive_int,
    )
    @settings(max_examples=100)
    def test_open_account_ratio_positive_total(self, open_accounts, total_accounts):
        """Property 9: Open account ratio computation (positive total case).

        For any observation with open_accounts O and total_accounts T,
        the computed open_account_ratio SHALL equal O / T when T > 0.

        **Validates: Requirements 4.3**
        """
        # Ensure open_accounts <= total_accounts for realistic data
        assume(open_accounts <= total_accounts)

        df = pd.DataFrame({
            "open_accounts": [open_accounts],
            "total_accounts": [total_accounts],
        })

        result = compute_open_account_ratio(df)

        expected = open_accounts / total_accounts
        assert np.isclose(result.iloc[0], expected, rtol=1e-7), (
            f"Expected {expected}, got {result.iloc[0]} for open={open_accounts}, total={total_accounts}"
        )

    @given(open_accounts=_non_negative_int)
    @settings(max_examples=100)
    def test_open_account_ratio_zero_total(self, open_accounts):
        """Property 9: Open account ratio computation (zero total case).

        For any observation with open_accounts O and total_accounts T,
        the computed open_account_ratio SHALL equal 0.0 when T = 0.

        **Validates: Requirements 4.3**
        """
        df = pd.DataFrame({
            "open_accounts": [open_accounts],
            "total_accounts": [0],
        })

        result = compute_open_account_ratio(df)

        assert result.iloc[0] == 0.0, (
            f"Expected 0.0 when total_accounts is 0, got {result.iloc[0]}"
        )


# --- Property 10: One-hot encoding binary invariant ---


# Strategy for generating categorical DataFrames
_category_values = st.sampled_from(["A", "B", "C", "D", "E"])


class TestOneHotEncodingProperties:
    """Property-based tests for one_hot_encode."""

    @given(
        n_rows=st.integers(min_value=2, max_value=50),
        categories=st.lists(
            _category_values, min_size=2, max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_one_hot_encoding_binary_invariant(self, n_rows, categories):
        """Property 10: One-hot encoding binary invariant.

        For any DataFrame with categorical columns, after one-hot encoding each
        row SHALL have at most one 1 among the indicator columns derived from
        each original categorical column (when using drop_first=True), and all
        indicator values SHALL be either 0 or 1.

        **Validates: Requirements 4.5**
        """
        # Build a DataFrame with a categorical column
        # Use the generated categories, cycling to fill n_rows
        cat_values = [categories[i % len(categories)] for i in range(n_rows)]

        df = pd.DataFrame({
            "cat_col": cat_values,
            "numeric_col": range(n_rows),
        })

        result = one_hot_encode(df, columns=["cat_col"])

        # Identify the indicator columns (they start with "cat_col_")
        indicator_cols = [c for c in result.columns if c.startswith("cat_col_")]

        # All indicator values must be either 0 or 1
        for col in indicator_cols:
            unique_vals = set(result[col].unique())
            assert unique_vals.issubset({0, 1, True, False}), (
                f"Column {col} has non-binary values: {unique_vals}"
            )

        # With drop_first=True, each row should have at most one 1
        # among the indicator columns
        if indicator_cols:
            indicator_matrix = result[indicator_cols].values
            row_sums = indicator_matrix.sum(axis=1)
            assert all(row_sums <= 1), (
                f"Found rows with more than one 1 in indicator columns: "
                f"max sum = {row_sums.max()}"
            )

    @given(
        n_rows=st.integers(min_value=2, max_value=30),
        col1_values=st.lists(
            st.sampled_from(["X", "Y", "Z"]), min_size=2, max_size=30,
        ),
        col2_values=st.lists(
            st.sampled_from(["P", "Q", "R"]), min_size=2, max_size=30,
        ),
    )
    @settings(max_examples=100)
    def test_one_hot_encoding_multiple_columns(self, n_rows, col1_values, col2_values):
        """Property 10 (extended): One-hot encoding with multiple categorical columns.

        For multiple categorical columns, the binary invariant holds independently
        for each original column's indicator set.

        **Validates: Requirements 4.5**
        """
        # Ensure lists are the right length
        c1 = [col1_values[i % len(col1_values)] for i in range(n_rows)]
        c2 = [col2_values[i % len(col2_values)] for i in range(n_rows)]

        df = pd.DataFrame({
            "cat1": c1,
            "cat2": c2,
            "numeric": range(n_rows),
        })

        result = one_hot_encode(df, columns=["cat1", "cat2"])

        # Check each original column's indicators independently
        for prefix in ["cat1_", "cat2_"]:
            indicator_cols = [c for c in result.columns if c.startswith(prefix)]
            if indicator_cols:
                indicator_matrix = result[indicator_cols].values

                # All values must be 0 or 1
                unique_vals = set(np.unique(indicator_matrix))
                assert unique_vals.issubset({0, 1, True, False}), (
                    f"Non-binary values in {prefix} indicators: {unique_vals}"
                )

                # At most one 1 per row (drop_first=True)
                row_sums = indicator_matrix.sum(axis=1)
                assert all(row_sums <= 1), (
                    f"Found rows with more than one 1 in {prefix} indicators"
                )


# --- Property 11: Log transform reduces skewness ---


class TestLogTransformProperties:
    """Property-based tests for log_transform_skewed."""

    @given(
        data=st.lists(
            st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=30,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_log_transform_reduces_skewness(self, data):
        """Property 11: Log transform reduces skewness.

        For any numeric column with skewness greater than 1.0, applying the log
        transformation SHALL produce a column with skewness strictly less than
        the original skewness.

        **Validates: Requirements 4.6**
        """
        df = pd.DataFrame({"value": data})

        original_skewness = df["value"].skew()

        # Only test when skewness exceeds the threshold
        assume(original_skewness > SKEWNESS_THRESHOLD)
        # Ensure we have variance (not all same values)
        assume(df["value"].std() > 0)
        # Require enough values > 1 so log1p can meaningfully compress the range
        assume(sum(1 for v in data if v > 1.0) >= 3)

        result = log_transform_skewed(df, columns=["value"])

        new_skewness = result["value"].skew()

        assert new_skewness < original_skewness, (
            f"Log transform did not reduce skewness: "
            f"original={original_skewness:.4f}, after={new_skewness:.4f}"
        )

    @given(
        data=st.lists(
            st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=30,
            max_size=200,
        ),
    )
    @settings(max_examples=100)
    def test_log_transform_no_change_below_threshold(self, data):
        """Property 11 (complement): Columns with skewness <= threshold are unchanged.

        When skewness is at or below the threshold, log_transform_skewed SHALL
        NOT modify the column.

        **Validates: Requirements 4.6**
        """
        df = pd.DataFrame({"value": data})

        original_skewness = df["value"].skew()

        # Only test when skewness is at or below threshold
        assume(original_skewness <= SKEWNESS_THRESHOLD)
        assume(not np.isnan(original_skewness))

        result = log_transform_skewed(df, columns=["value"])

        # Column should be unchanged
        pd.testing.assert_series_equal(
            result["value"], df["value"], check_names=True
        )
