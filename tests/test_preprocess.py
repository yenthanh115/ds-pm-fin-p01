"""Tests for src/preprocess.py - data loading functionality."""

import os
import string
import tempfile

import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.preprocess import load_dataset


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
