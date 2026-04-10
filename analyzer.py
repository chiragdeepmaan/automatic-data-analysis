import logging
import numpy as np
import polars as pl
from scipy import stats as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical tests on a Polars DataFrame. All results returned as dicts."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    # Normality 
    def normality_test(self, df: pl.DataFrame, column: str) -> dict:
        data = df[column].drop_nulls().to_numpy()[:5000]
        if len(data) < 3:
            return {"column": column, "is_normal": None, "note": "Insufficient data"}
        stat, p = sp.shapiro(data)
        return {
            "column": column,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "is_normal": bool(p >= self.alpha),
            "interpretation": (
                f"{column} follows a normal distribution (p = {p:.4f} >= {self.alpha})."
                if p >= self.alpha else
                f"{column} deviates from normality (p = {p:.4f} < {self.alpha}). "
                "Non-parametric tests or transformations are recommended."
            ),
        }

    # Confidence interval 
    def confidence_interval(self, df: pl.DataFrame, column: str, confidence: float = 0.95) -> dict:
        data = df[column].drop_nulls().to_numpy()
        n = len(data)
        if n < 2:
            return {}
        mean = float(np.mean(data))
        se = float(sp.sem(data))
        lo, hi = sp.t.interval(confidence, df=n - 1, loc=mean, scale=se)
        return {
            "column": column,
            "n": n,
            "mean": round(mean, 4),
            "std_error": round(se, 4),
            "lower": round(float(lo), 4),
            "upper": round(float(hi), 4),
            "confidence": confidence,
            "interpretation": (
                f"We are {int(confidence*100)}% confident the true mean of {column} "
                f"lies between {lo:.4f} and {hi:.4f} (sample mean = {mean:.4f}, n = {n:,})."
            ),
        }

    # OLS Regression 
    def run_regression(self, df: pl.DataFrame, target: str, features: list) -> dict:
        cols   = [target] + features
        subset = df.select(cols).drop_nulls()
        X      = subset.select(features).to_numpy().astype(float)
        y      = subset[target].to_numpy().astype(float)

        model  = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        n, p  = X.shape
        r2    = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
        mse   = float(np.mean((y - y_pred) ** 2))

        # Approximate p-values
        res    = y - y_pred
        sigma2 = np.sum(res**2) / max(n - p - 1, 1)
        xtx_inv = np.linalg.pinv(X.T @ X)
        se     = np.sqrt(np.diag(sigma2 * xtx_inv))
        t_stats = model.coef_ / (se + 1e-12)
        p_vals  = [2 * (1 - sp.t.cdf(abs(t), df=n - p - 1)) for t in t_stats]

        label = "excellent" if r2 > 0.8 else "good" if r2 > 0.6 else "moderate" if r2 > 0.4 else "weak"

        return {
            "target":               target,
            "features":             features,
            "intercept":            round(float(model.intercept_), 6),
            "coefficients":         {f: round(float(c), 6) for f, c in zip(features, model.coef_)},
            "p_values":             {f: round(float(pv), 6) for f, pv in zip(features, p_vals)},
            "r_squared":            round(r2, 4),
            "adjusted_r_squared":   round(adj_r2, 4),
            "rmse":                 round(mse ** 0.5, 4),
            "mae":                  round(float(np.mean(np.abs(res))), 4),
            "n":                    n,
            # Arrays for diagnostic plots
            "y_true":               y,
            "y_pred":               y_pred,
            "coefs_array":          model.coef_,
            "interpretation": (
                f"The model explains {r2*100:.1f}% of variance in {target} "
                f"(R^2 = {r2:.4f} -- {label} fit, Adj-R^2 = {adj_r2:.4f}). "
                f"RMSE = {mse**0.5:.4f}. "
                + ("Features with p < 0.05 are statistically significant."
                   if any(pv < 0.05 for pv in p_vals) else
                   "No feature reaches significance at alpha = 0.05.")
            ),
        }

    # Full analysis pass 
    def run_full_analysis(self, df: pl.DataFrame, target: str = None) -> dict:
        numeric = [c for c in df.columns if df[c].dtype in (
            pl.Float64, pl.Float32, pl.Int64, pl.Int32
        )]
        report = {"normality": {}, "confidence_intervals": {}, "regression": None}
        for c in numeric:
            try:
                report["normality"][c]             = self.normality_test(df, c)
                report["confidence_intervals"][c]  = self.confidence_interval(df, c)
            except Exception as e:
                logger.warning("Could not analyse '%s': %s", c, e)

        if target and target in numeric:
            features = [c for c in numeric if c != target]
            if features:
                try:
                    report["regression"] = self.run_regression(df, target, features)
                except Exception as e:
                    logger.warning("Regression failed: %s", e)
        return report