"""Data loading and preprocessing module for LendSafe pipeline."""

import os

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
