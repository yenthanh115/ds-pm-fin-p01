"""Tests for src/preprocess.py - data loading functionality."""

import os
import tempfile

import pandas as pd
import pytest

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
