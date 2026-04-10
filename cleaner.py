import logging
import polars as pl

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean a Polars DataFrame — nulls, duplicates, outliers, type coercion."""

    def __init__(self):
        self.log: list[dict] = []


    def drop_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        before = df.shape[0]
        df = df.unique()
        removed = before - df.shape[0]
        self._record("drop_duplicates", f"Removed {removed} duplicate rows")
        return df

    def handle_nulls(self, df: pl.DataFrame, strategy: str = "median") -> pl.DataFrame:
        """
        Fill or drop nulls.

        strategy: 'drop' | 'mean' | 'median' | 'mode' | 'zero'
        """
        numeric = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)]
        categorical = [c for c in df.columns if df[c].dtype == pl.Utf8]

        if strategy == "drop":
            df = df.drop_nulls()
        elif strategy in ("mean", "median"):
            fills = {}
            for c in numeric:
                val = df[c].mean() if strategy == "mean" else df[c].median()
                if val is not None:
                    fills[c] = val
            if fills:
                df = df.with_columns([pl.col(c).fill_null(v) for c, v in fills.items()])
            for c in categorical:
                mode_val = df[c].drop_nulls().mode()
                if len(mode_val) > 0:
                    df = df.with_columns(pl.col(c).fill_null(mode_val[0]))
        elif strategy == "mode":
            for c in df.columns:
                mode_val = df[c].drop_nulls().mode()
                if len(mode_val) > 0:
                    df = df.with_columns(pl.col(c).fill_null(mode_val[0]))
        elif strategy == "zero":
            df = df.with_columns([pl.col(c).fill_null(0) for c in numeric])
            df = df.with_columns([pl.col(c).fill_null("unknown") for c in categorical])

        self._record("handle_nulls", f"Strategy='{strategy}'")
        return df

  
    def remove_outliers(self, df: pl.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pl.DataFrame:
        """Remove rows containing outliers in numeric columns (IQR or Z-score)."""
        numeric = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
        before = df.shape[0]

        if method == "iqr":
            mask = pl.lit(True)
            for c in numeric:
                q1 = df[c].quantile(0.25)
                q3 = df[c].quantile(0.75)
                if q1 is None or q3 is None:
                    continue
                iqr = q3 - q1
                lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
                mask = mask & (pl.col(c) >= lo) & (pl.col(c) <= hi)
            df = df.filter(mask)

        elif method == "zscore":
            for c in numeric:
                mean = df[c].mean()
                std = df[c].std()
                if std and std > 0:
                    df = df.filter(((pl.col(c) - mean) / std).abs() <= threshold)

        self._record("remove_outliers", f"Method='{method}'; removed {before - df.shape[0]} rows")
        return df
    
    def auto_cast_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Try to cast string columns that look numeric to Float64."""
        exprs = []
        for c in df.columns:
            if df[c].dtype == pl.Utf8:
                try:
                    df[c].cast(pl.Float64)
                    exprs.append(pl.col(c).cast(pl.Float64, strict=False))
                except Exception:
                    exprs.append(pl.col(c))
            else:
                exprs.append(pl.col(c))
        df = df.with_columns(exprs)
        self._record("auto_cast_types", "Attempted numeric cast on string columns")
        return df


    def get_cleaning_report(self, df_before: pl.DataFrame, df_after: pl.DataFrame) -> dict:
        return {
            "rows_before": df_before.shape[0],
            "rows_after": df_after.shape[0],
            "rows_dropped": df_before.shape[0] - df_after.shape[0],
            "nulls_before": sum(df_before[c].null_count() for c in df_before.columns),
            "nulls_after": sum(df_after[c].null_count() for c in df_after.columns),
            "log": self.log,
        }

    def _record(self, op: str, detail: str):
        self.log.append({"operation": op, "detail": detail})
        logger.info("[Cleaner] %s — %s", op, detail)