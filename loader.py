"""
loader.py — Data Ingestion
Reads CSV / Excel / JSON into a Polars DataFrame.
"""

import io
import logging
import polars as pl

logger = logging.getLogger(__name__)


class DataLoader:
    """Load tabular data from various sources into a Polars DataFrame."""

    def __init__(self):
        self.source_name: str = ""
        self.raw_shape: tuple = (0, 0)

    # ------------------------------------------------------------------ #
    def load_file(self, uploaded_file) -> pl.DataFrame:
        """
        Load a Streamlit UploadedFile object.

        Supports .csv, .xlsx, .xls, .json.
        Returns a Polars DataFrame.
        """
        name = uploaded_file.name.lower()
        self.source_name = uploaded_file.name
        raw = uploaded_file.read()

        if name.endswith(".csv"):
            df = pl.read_csv(io.BytesIO(raw), infer_schema_length=10_000, ignore_errors=True)
        elif name.endswith((".xlsx", ".xls")):
            import pandas as _pd          # polars uses pandas under the hood for xlsx
            pdf = _pd.read_excel(io.BytesIO(raw))
            df = pl.from_pandas(pdf)
        elif name.endswith(".json"):
            df = pl.read_json(io.BytesIO(raw))
        else:
            raise ValueError(f"Unsupported file type: '{uploaded_file.name}'")

        self.raw_shape = df.shape
        logger.info("Loaded '%s' — %d rows × %d cols", uploaded_file.name, *df.shape)
        return df

    # ------------------------------------------------------------------ #
    def get_info(self, df: pl.DataFrame) -> dict:
        """Return a concise metadata dict about the DataFrame."""
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns,
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "null_counts": {c: df[c].null_count() for c in df.columns},
            "memory_mb": round(df.estimated_size("mb"), 3),
        }