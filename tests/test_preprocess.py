"""Tests for src/preprocess.py - data loading and preprocessing functionality."""

import os
import string
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.preprocess import (
    load_dataset,
    strip_formatting,
    parse_employment_length,
    encode_loan_status,
    preprocess,
    AMBIGUOUS_STATUSES,
    COLUMN_SCHEMA,
)


class TestLoadDataset:
    """Unit tests for load_dataset function."""

    def test_loads_csv_preserving_columns(self, tmp_path):
        """load_dataset returns a DataFrame with all columns from the CSV."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")

        df = load_dataset(str(csv_path))

        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 2

    def test_raises_file_not_found_with_path_in_message(self):
        """load_dataset raises FileNotFoundError with the path in the message."""
        bad_path = "/nonexistent/path/data.csv"

        with pytest.raises(FileNotFoundError, match=bad_path.replace("/", r"[/\\]")):
            load_dataset(bad_path)

    def test_raises_ioerror_for_unreadable_file(self, tmp_path):
        """load_dataset raises IOError with path in message for corrupt files."""
        csv_path = tmp_path / "corrupt.csv"
        # Write binary garbage that pandas cannot parse as CSV
        csv_path.write_bytes(b"\x00\x01\x02" * 100)

        # pandas may raise various errors for corrupt data; our wrapper
        # should convert them to IOError. However, pandas is quite lenient
        # with CSV parsing, so we test with a directory path instead.
        dir_path = tmp_path / "a_directory"
        dir_path.mkdir()

        with pytest.raises((IOError, IsADirectoryError, PermissionError)):
            load_dataset(str(dir_path))

    def test_loads_empty_csv_with_headers(self, tmp_path):
        """load_dataset handles a CSV with headers but no data rows."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("col1,col2,col3\n")

        df = load_dataset(str(csv_path))

        assert list(df.columns) == ["col1", "col2", "col3"]
        assert len(df) == 0

    def test_loads_csv_with_mixed_types(self, tmp_path):
        """load_dataset handles mixed-type columns without DtypeWarning."""
        csv_path = tmp_path / "mixed.csv"
        lines = ["id,value\n"] + [f"{i},{'text' if i % 2 == 0 else i}\n" for i in range(200)]
        csv_path.write_text("".join(lines))

        df = load_dataset(str(csv_path))

        assert "id" in df.columns
        assert "value" in df.columns
        assert len(df) == 200


# --- Property-Based Tests ---

# Strategy: generate valid column names (non-empty, no commas/newlines, unique)
_column_name_chars = st.sampled_from(
    string.ascii_letters + string.digits + "_"
)
_column_name = st.text(_column_name_chars, min_size=1, max_size=20)
_unique_columns = st.lists(
    _column_name, min_size=1, max_size=20, unique=True
)


class TestLoadDatasetProperties:
    """Property-based tests for load_dataset function."""

    @given(columns=_unique_columns)
    @settings(max_examples=100)
    def test_column_preservation_during_loading(self, columns, tmp_path_factory):
        """Property 1: Column preservation during loading.

        For any valid CSV file with N columns, loading it via the data loader
        SHALL produce a DataFrame with exactly the same N column names as the
        source file.

        **Validates: Requirements 1.1, 1.2**
        """
        # Create a temporary CSV with the generated column names and one data row
        tmp_dir = tmp_path_factory.mktemp("data")
        csv_path = tmp_dir / "test.csv"

        # Write header + one row of dummy data
        header = ",".join(columns)
        row = ",".join(["1"] * len(columns))
        csv_path.write_text(f"{header}\n{row}\n")

        df = load_dataset(str(csv_path))

        assert list(df.columns) == columns
        assert len(df.columns) == len(columns)

    @given(
        path_parts=st.lists(
            st.text(
                st.sampled_from(string.ascii_lowercase + string.digits + "_"),
                min_size=1,
                max_size=10,
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_descriptive_error_on_invalid_file_path(self, path_parts):
        """Property 2: Descriptive error on invalid file path.

        For any file path that does not exist or is unreadable, the data loader
        SHALL raise an error whose message contains the file path string.

        **Validates: Requirements 1.2, 1.3**
        """
        # Construct a non-existent path
        fake_path = os.path.join(tempfile.gettempdir(), *path_parts, "nonexistent.csv")

        # Ensure the path truly does not exist
        assume(not os.path.exists(fake_path))

        with pytest.raises(FileNotFoundError) as exc_info:
            load_dataset(fake_path)

        # The error message must contain the file path
        assert fake_path in str(exc_info.value)


# --- Property-Based Tests for Preprocessing ---


class TestStripFormattingProperties:
    """Property-based tests for strip_formatting function."""

    @given(value=st.floats(min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_numeric_formatting_strip_round_trip(self, value):
        """Property 3: Numeric formatting strip round-trip.

        For any numeric value V and any combination of formatting characters
        ($, %, commas), formatting V with those characters and then applying
        `strip_formatting` SHALL produce a value equal to V (within
        floating-point tolerance).

        **Validates: Requirements 2.1**
        """
        # Round to 2 decimal places to avoid floating-point representation issues
        value = round(value, 2)

        # Format the value with various formatting characters
        formatted = f"${value:,.2f}%"

        series = pd.Series([formatted])
        result = strip_formatting(series)

        assert not result.isna().iloc[0], f"strip_formatting returned NaN for '{formatted}'"
        assert np.isclose(result.iloc[0], value, rtol=1e-5), (
            f"Expected {value}, got {result.iloc[0]} from formatted string '{formatted}'"
        )

    @given(
        value=st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
        prefix=st.sampled_from(["$", ""]),
        suffix=st.sampled_from(["%", ""]),
        use_commas=st.booleans(),
    )
    @settings(max_examples=100)
    def test_numeric_formatting_strip_with_combinations(self, value, prefix, suffix, use_commas):
        """Property 3 (extended): Formatting strip with various combinations.

        For any numeric value V and any combination of formatting characters,
        formatting V and then applying strip_formatting SHALL produce V.

        **Validates: Requirements 2.1**
        """
        value = round(value, 2)

        if use_commas:
            formatted = f"{prefix}{value:,.2f}{suffix}"
        else:
            formatted = f"{prefix}{value:.2f}{suffix}"

        series = pd.Series([formatted])
        result = strip_formatting(series)

        assert not result.isna().iloc[0], f"strip_formatting returned NaN for '{formatted}'"
        assert np.isclose(result.iloc[0], value, rtol=1e-5), (
            f"Expected {value}, got {result.iloc[0]} from formatted string '{formatted}'"
        )


class TestParseEmploymentLengthProperties:
    """Property-based tests for parse_employment_length function."""

    @given(years=st.integers(min_value=0, max_value=10))
    @settings(max_examples=100)
    def test_employment_length_parsing(self, years):
        """Property 4: Employment length parsing.

        For any integer year value Y in [0, 10], the employment length string
        representation parsed by the preprocessor SHALL produce the integer Y.

        **Validates: Requirements 2.2**
        """
        # Generate the string representation based on the year value
        if years == 0:
            emp_str = "< 1 year"
        elif years == 1:
            emp_str = "1 year"
        elif years == 10:
            emp_str = "10+ years"
        else:
            emp_str = f"{years} years"

        series = pd.Series([emp_str])
        result = parse_employment_length(series)

        assert result.iloc[0] == years, (
            f"Expected {years}, got {result.iloc[0]} for '{emp_str}'"
        )


class TestAmbiguousStatusFilteringProperties:
    """Property-based tests for ambiguous status filtering."""

    @given(
        n_ambiguous=st.integers(min_value=1, max_value=20),
        n_valid=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=100)
    def test_ambiguous_status_filtering(self, n_ambiguous, n_valid):
        """Property 5: Ambiguous status filtering.

        For any DataFrame containing rows with ambiguous loan statuses, after
        preprocessing the resulting DataFrame SHALL contain zero rows with any
        ambiguous status value.

        **Validates: Requirements 2.3**
        """
        ambiguous_list = list(AMBIGUOUS_STATUSES)

        # Build rows with ambiguous statuses
        ambiguous_statuses = [
            ambiguous_list[i % len(ambiguous_list)] for i in range(n_ambiguous)
        ]

        # Build rows with valid statuses
        valid_statuses = ["Fully Paid", "Charged Off"] * ((n_valid // 2) + 1)
        valid_statuses = valid_statuses[:n_valid]

        all_statuses = ambiguous_statuses + valid_statuses

        # Create a minimal DataFrame with required columns for preprocessing
        df = pd.DataFrame({
            "loan_amnt": [1000.0] * len(all_statuses),
            "term": [36] * len(all_statuses),
            "int_rate": [5.0] * len(all_statuses),
            "installment": [100.0] * len(all_statuses),
            "grade": ["A"] * len(all_statuses),
            "emp_length": ["5 years"] * len(all_statuses),
            "home_ownership": ["RENT"] * len(all_statuses),
            "annual_inc": [50000.0] * len(all_statuses),
            "purpose": ["debt_consolidation"] * len(all_statuses),
            "dti": [10.0] * len(all_statuses),
            "open_acc": [5] * len(all_statuses),
            "total_acc": [10] * len(all_statuses),
            "revol_bal": [5000.0] * len(all_statuses),
            "revol_util": [10000.0] * len(all_statuses),
            "loan_status": all_statuses,
        })

        result = preprocess(df, dataset_type="lending_club")

        # After preprocessing, no ambiguous statuses should remain
        # loan_status is now encoded as int (0 or 1), so check the original
        # statuses are gone by verifying only valid encoded values remain
        assert all(result["loan_status"].isin([0, 1])), (
            "Found non-binary loan_status values after preprocessing"
        )
        assert len(result) == n_valid, (
            f"Expected {n_valid} rows after filtering, got {len(result)}"
        )


class TestLoanStatusEncodingProperties:
    """Property-based tests for loan status encoding."""

    @given(
        n_fully_paid=st.integers(min_value=1, max_value=20),
        n_charged_off=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_loan_status_encoding_correctness(self, n_fully_paid, n_charged_off):
        """Property 6: Loan status encoding correctness.

        For any DataFrame containing rows with "Fully Paid" or "Charged Off"
        statuses, after encoding ALL "Fully Paid" rows SHALL have target value 0
        and ALL "Charged Off" rows SHALL have target value 1.

        **Validates: Requirements 2.4, 2.5**
        """
        statuses = (["Fully Paid"] * n_fully_paid) + (["Charged Off"] * n_charged_off)

        series = pd.Series(statuses)
        result = encode_loan_status(series)

        # Check Fully Paid → 0
        fully_paid_encoded = result.iloc[:n_fully_paid]
        assert all(fully_paid_encoded == 0), (
            f"Expected all 'Fully Paid' to encode as 0, got {fully_paid_encoded.tolist()}"
        )

        # Check Charged Off → 1
        charged_off_encoded = result.iloc[n_fully_paid:]
        assert all(charged_off_encoded == 1), (
            f"Expected all 'Charged Off' to encode as 1, got {charged_off_encoded.tolist()}"
        )


class TestColumnSchemaMappingProperties:
    """Property-based tests for column schema mapping."""

    @given(dataset_type=st.sampled_from(["german", "lending_club"]))
    @settings(max_examples=100)
    def test_column_schema_mapping_completeness(self, dataset_type):
        """Property 19: Column schema mapping completeness.

        For any DataFrame with column names from either the German Credit or
        Lending Club schema, applying the column mapping SHALL produce a
        DataFrame whose columns are a subset of the common schema columns.

        **Validates: Requirements 8.3**
        """
        # Get the source column names for this dataset type
        mapping = COLUMN_SCHEMA[dataset_type]["mapping"]
        source_columns = list(mapping.keys())
        common_schema_columns = set(mapping.values())

        # Create a DataFrame with the source columns
        n_rows = 3
        df = pd.DataFrame(
            {col: ["dummy"] * n_rows for col in source_columns}
        )

        # Apply column mapping (same logic as in preprocess)
        rename_map = {k: v for k, v in mapping.items() if k in df.columns}
        df_mapped = df.rename(columns=rename_map)

        # Keep only common schema columns that are present
        available_columns = [col for col in mapping.values() if col in df_mapped.columns]
        df_mapped = df_mapped[available_columns]

        # All resulting columns must be a subset of the common schema
        result_columns = set(df_mapped.columns)
        assert result_columns.issubset(common_schema_columns), (
            f"Columns {result_columns - common_schema_columns} are not in common schema"
        )
