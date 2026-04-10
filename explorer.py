"""
explorer.py — Exploratory Data Analysis
Computes descriptive statistics, correlations, and patterns using Polars.
"""

import logging
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


class DataExplorer:

    def get_summary_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """Descriptive stats (count, mean, std, min, quartiles, max, skew) for numeric columns."""
        numeric_cols = self._numeric_cols(df)
        if not numeric_cols:
            return pl.DataFrame()

        rows = []
        for c in numeric_cols:
            s = df[c].drop_nulls()
            arr = s.to_numpy()
            rows.append({
                "column": c,
                "count": len(s),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "q25": round(float(np.percentile(arr, 25)), 4),
                "median": round(float(s.median()), 4),
                "q75": round(float(np.percentile(arr, 75)), 4),
                "max": round(float(s.max()), 4),
                "skewness": round(float(self._skewness(arr)), 4),
                "nulls": df[c].null_count(),
                "null_pct": round(df[c].null_count() / df.shape[0] * 100, 2),
            })
        return pl.DataFrame(rows)

    
    def get_column_profile(self, df: pl.DataFrame) -> list[dict]:
        """Per-column metadata: dtype, null%, unique count, sample values."""
        profiles = []
        for c in df.columns:
            col = df[c]
            # Always stringify sample values — prevents PyArrow mixed-type errors
            sample_vals = [str(v) for v in col.drop_nulls().head(3).to_list()]
            profiles.append({
                "column": c,
                "dtype": str(col.dtype),
                "null_count": col.null_count(),
                "null_pct": round(col.null_count() / df.shape[0] * 100, 2),
                "unique": col.n_unique(),
                "sample": ", ".join(sample_vals),   # single string, not a list
            })
        return profiles

    
    def get_correlation_matrix(self, df: pl.DataFrame) -> pl.DataFrame:
        """Pearson correlation matrix for all numeric columns."""
        numeric_cols = self._numeric_cols(df)
        if len(numeric_cols) < 2:
            return pl.DataFrame()
        numeric_df = df.select(numeric_cols).drop_nulls()
        arr = numeric_df.to_numpy()
        corr = np.corrcoef(arr.T)
        return pl.DataFrame(
            {col: corr[i].tolist() for i, col in enumerate(numeric_cols)}
        ).with_columns(pl.Series("column", numeric_cols)).select(["column"] + numeric_cols)


    def get_high_correlations(self, df: pl.DataFrame, threshold: float = 0.7) -> list[dict]:
        """Return pairs of features with |correlation| >= threshold."""
        numeric_cols = self._numeric_cols(df)
        if len(numeric_cols) < 2:
            return []
        numeric_df = df.select(numeric_cols).drop_nulls()
        arr = numeric_df.to_numpy()
        corr = np.corrcoef(arr.T)
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                val = corr[i, j]
                if abs(val) >= threshold:
                    pairs.append({
                        "feature_a": numeric_cols[i],
                        "feature_b": numeric_cols[j],
                        "correlation": round(float(val), 4),
                    })
        return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)

   
    def detect_skewness(self, df: pl.DataFrame) -> list[dict]:
        """Skewness per numeric column."""
        results = []
        for c in self._numeric_cols(df):
            arr = df[c].drop_nulls().to_numpy()
            skew = self._skewness(arr)
            results.append({
                "column": c,
                "skewness": round(float(skew), 4),
                "highly_skewed": abs(skew) > 1,
            })
        return sorted(results, key=lambda x: abs(x["skewness"]), reverse=True)

    def detect_class_balance(self, df: pl.DataFrame, target: str) -> dict:
        """Value counts and imbalance ratio for a categorical target."""
        if target not in df.columns:
            return {}
        counts = df[target].value_counts().sort("count", descending=True)
        total = df.shape[0]
        vals = counts[target].to_list()
        cnts = counts["count"].to_list()
        ratio = cnts[0] / cnts[-1] if len(cnts) > 1 and cnts[-1] > 0 else 1.0
        return {
            "value_counts": dict(zip(vals, cnts)),
            "proportions": {v: round(c / total * 100, 2) for v, c in zip(vals, cnts)},
            "imbalance_ratio": round(ratio, 2),
            "is_imbalanced": ratio > 3,
        }

    def get_missing_summary(self, df: pl.DataFrame) -> pl.DataFrame:
        """Sorted table of columns with missing values."""
        rows = []
        for c in df.columns:
            n = df[c].null_count()
            if n > 0:
                rows.append({"column": c, "missing": n, "pct": round(n / df.shape[0] * 100, 2)})
        if not rows:
            return pl.DataFrame({"column": [], "missing": [], "pct": []})
        return pl.DataFrame(rows).sort("pct", descending=True)

   
    def full_eda(self, df: pl.DataFrame, target: str = None) -> dict:
        """Run all EDA steps and return a unified report dict."""
        report = {
            "shape": df.shape,
            "column_profiles": self.get_column_profile(df),
            "summary_stats": self.get_summary_stats(df),
            "missing_summary": self.get_missing_summary(df),
            "high_correlations": self.get_high_correlations(df),
            "skewness": self.detect_skewness(df),
            "duplicates": df.shape[0] - df.unique().shape[0],
            "numeric_cols": self._numeric_cols(df),
            "categorical_cols": self._categorical_cols(df),
        }
        if target:
            report["class_balance"] = self.detect_class_balance(df, target)
        return report

  
    # Helpers
    @staticmethod
    def _numeric_cols(df: pl.DataFrame) -> list[str]:
        return [c for c in df.columns if df[c].dtype in (
            pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32
        )]

    @staticmethod
    def _categorical_cols(df: pl.DataFrame) -> list[str]:
        return [c for c in df.columns if df[c].dtype in (pl.Utf8, pl.Categorical)]

    @staticmethod
    def _skewness(arr: np.ndarray) -> float:
        """Fisher's skewness."""
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n < 3:
            return 0.0
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))