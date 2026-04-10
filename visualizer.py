"""
visualizer.py — Rich Interactive Visualizations (Plotly dark theme)
Every method returns (fig, interpretation_text).
Includes: distribution, KDE, violin/box, correlation heatmap, scatter+regression,
bar, count, pie, treemap, cat×num heatmap, outlier box, pair plot,
regression diagnostics, time series, integer distribution, skewness chart.
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared palette & layout 
PALETTE = [
    "#6C63FF", "#48CAE4", "#F77F00", "#E63946", "#2DC653",
    "#FF6B9D", "#FFD166", "#06D6A0", "#118AB2", "#EF476F",
]

_BASE = dict(
    paper_bgcolor="rgba(15,17,26,0)",
    plot_bgcolor="rgba(15,17,26,0)",
    font=dict(family="DM Sans, sans-serif", color="#e2e8f0", size=13),
    title_font=dict(family="Sora, sans-serif", size=17, color="#f8fafc"),
    margin=dict(l=24, r=24, t=60, b=30),
    legend=dict(bgcolor="rgba(255,255,255,0.05)",
                bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
)


def _theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**_BASE, title=title)
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha) string safe for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig_to_png(fig: go.Figure) -> bytes:
    """Return figure as PNG bytes (requires kaleido)."""
    try:
        return fig.to_image(format="png", width=1400, height=700, scale=2)
    except Exception:
        return b""


def fig_to_html(fig: go.Figure) -> bytes:
    """Return figure as self-contained HTML bytes."""
    return fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")


class DataVisualizer:
    """All chart methods return (fig, interpretation_str)."""

    # helpers 
    @staticmethod
    def fig_png(fig): return fig_to_png(fig)

    @staticmethod
    def fig_html(fig): return fig_to_html(fig)

    #  Histogram + KDE 
    def distribution(self, df: pl.DataFrame, column: str):
        data = df[column].drop_nulls().to_numpy().astype(float)
        fig  = go.Figure()
        fig.add_trace(go.Histogram(
            x=data, nbinsx=40,
            marker=dict(color=PALETTE[0], opacity=0.70,
                        line=dict(color="rgba(255,255,255,0.08)", width=0.5)),
            name="Frequency",
        ))
        try:
            from scipy.stats import gaussian_kde
            xs  = np.linspace(data.min(), data.max(), 300)
            kde = gaussian_kde(data)
            fig.add_trace(go.Scatter(
                x=xs, y=kde(xs) * len(data) * (data.max() - data.min()) / 40,
                mode="lines", name="KDE", line=dict(color=PALETTE[1], width=2.5),
            ))
        except Exception:
            pass
        mean_v, med_v = float(np.mean(data)), float(np.median(data))
        for v, lbl, c in [(mean_v, "Mean", PALETTE[2]), (med_v, "Median", PALETTE[4])]:
            fig.add_vline(x=v, line_dash="dash", line_color=c, line_width=1.5,
                          annotation_text=f"{lbl}: {v:.2f}", annotation_font_color=c)
        _theme(fig, f"Distribution — {column}")
        skew = float(np.mean(((data - data.mean()) / (data.std() + 1e-9)) ** 3))
        dir_ = "right (positive)" if skew > 0.5 else "left (negative)" if skew < -0.5 else "symmetric"
        interp = (
            f"**{column}** spans **{data.min():.2f} – {data.max():.2f}** | "
            f"Mean = **{mean_v:.2f}**, Median = **{med_v:.2f}**, Std = **{data.std():.2f}**. "
            f"Distribution is skewed **{dir_}** (skewness = {skew:.2f}). "
            + ("Large mean–median gap suggests outliers." if abs(mean_v - med_v) > data.std() * 0.3
               else "Mean ≈ Median — well-centred.")
        )
        return fig, interp

    # KDE 
    def kde_plot(self, df: pl.DataFrame, columns: list):
        from scipy.stats import gaussian_kde
        fig = go.Figure()
        for i, col in enumerate(columns[:8]):
            data = df[col].drop_nulls().to_numpy().astype(float)
            if len(data) < 4:
                continue
            xs = np.linspace(data.min(), data.max(), 300)
            fig.add_trace(go.Scatter(
                x=xs, y=gaussian_kde(data)(xs), mode="lines", name=col,
                line=dict(color=PALETTE[i % len(PALETTE)], width=2.2),
                fill="tozeroy", fillcolor=_hex_to_rgba(PALETTE[i % len(PALETTE)], 0.12),
            ))
        _theme(fig, "KDE Density — Multi-column Comparison")
        fig.update_layout(yaxis_title="Density")
        interp = (
            "KDE smooths the histogram into a continuous probability density curve. "
            "Taller & narrower peaks = concentrated values; flat & wide = high spread. "
            "Overlapping peaks show similar distributions across features."
        )
        return fig, interp

    # Box + Violin 
    def boxplot(self, df: pl.DataFrame, columns: list):
        fig = go.Figure()
        for i, col in enumerate(columns[:8]):
            data = df[col].drop_nulls().to_numpy().astype(float)
            fig.add_trace(go.Violin(
                y=data, name=col, box_visible=True, meanline_visible=True,
                fillcolor=PALETTE[i % len(PALETTE)], opacity=0.72,
                line_color="white", marker=dict(size=2, opacity=0.3),
            ))
        _theme(fig, "Box & Violin — Distribution Spread")
        fig.update_layout(violinmode="group")
        interp = (
            "Violin width = data density. Embedded box marks Q1, median, Q3. "
            "Whiskers reach 1.5 × IQR. Dots beyond whiskers are potential outliers. "
            "Compare shapes to detect skew, bimodality, or differing spreads."
        )
        return fig, interp

    # Outlier box 
    def outlier_boxplot(self, df_before: pl.DataFrame, df_after: pl.DataFrame, column: str):
        before = df_before[column].drop_nulls().to_numpy().astype(float)
        after  = df_after[column].drop_nulls().to_numpy().astype(float)
        fig = go.Figure()
        for data, label, color in [(before, "Before Removal", PALETTE[3]),
                                   (after,  "After Removal",  PALETTE[4])]:
            fig.add_trace(go.Box(
                y=data, name=label, marker_color=color,
                boxpoints="outliers", marker=dict(size=4, opacity=0.55),
                line_color=color,
            ))
        _theme(fig, f"Outlier Analysis — {column}")
        n_out = len(before) - len(after)
        q1, q3 = np.percentile(before, [25, 75])
        iqr = q3 - q1
        interp = (
            f"**{n_out}** outliers removed from **{column}** using IQR fencing. "
            f"IQR = {iqr:.2f} (Q1 = {q1:.2f}, Q3 = {q3:.2f}). "
            f"Fence: [{q1 - 1.5*iqr:.2f}, {q3 + 1.5*iqr:.2f}]. "
            "Red = original distribution. Green = cleaned. Dots above/below whiskers were removed."
        )
        return fig, interp

    # Correlation heatmap 
    def correlation_heatmap(self, df: pl.DataFrame, corr_matrix: pl.DataFrame):
        cols   = [c for c in corr_matrix.columns if c != "column"]
        z      = corr_matrix.select(cols).to_numpy()
        labels = corr_matrix["column"].to_list()
        fig = go.Figure(go.Heatmap(
            z=z, x=cols, y=labels,
            colorscale=[[0.0, "#E63946"], [0.5, "#1d3557"], [1.0, "#6C63FF"]],
            zmid=0, zmin=-1, zmax=1,
            text=np.round(z, 2).astype(str),
            texttemplate="%{text}", textfont=dict(size=10), showscale=True,
            colorbar=dict(title="r", tickfont=dict(color="#e2e8f0")),
        ))
        _theme(fig, "Pearson Correlation Matrix")
        n_strong = int(sum(abs(z[i][j]) >= 0.7
                           for i in range(len(z)) for j in range(i + 1, len(z[0]))))
        interp = (
            f"Correlation ranges −1 to +1. Found **{n_strong}** strongly correlated pairs (|r| ≥ 0.7). "
            "Deep purple = strong positive; deep red = strong negative correlation. "
            "High |r| between features signals multicollinearity."
        )
        return fig, interp

    # Scatter + OLS trendline 
    def scatter(self, df: pl.DataFrame, x: str, y: str, color_col: str = None):
        valid_cols = set(df.columns)
        if x not in valid_cols or y not in valid_cols:
            raise ValueError(f"Column not found in DataFrame.")
        if x == y:
            raise ValueError("X and Y must be different columns.")
        use_color = color_col if (color_col and color_col in valid_cols) else None
        # Build pandas df column-by-column to avoid Polars dedup issues
        import pandas as _pd
        pdf = _pd.DataFrame({
            x: df[x].to_list(),
            **({y: df[y].to_list()} if y != x else {}),
            **({use_color: df[use_color].to_list()} if use_color else {}),
        }).dropna()
        fig = px.scatter(
            pdf, x=x, y=y,
            color=use_color, opacity=0.65,
            color_discrete_sequence=PALETTE,
            trendline="ols", trendline_color_override=PALETTE[2],
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0.3, color="white")))
        _theme(fig, f"Scatter — {x} vs {y}")
        xd = pdf[x].to_numpy().astype(float)
        yd = pdf[y].to_numpy().astype(float)
        mask = ~(np.isnan(xd) | np.isnan(yd))
        r = float(np.corrcoef(xd[mask], yd[mask])[0, 1]) if mask.sum() > 1 else 0.0
        strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
        dir_ = "positive" if r > 0 else "negative"
        interp = (
            f"**{x}** vs **{y}**: **{strength} {dir_}** relationship "
            f"(r = {r:.3f}, R² = {r**2:.3f}). "
            "Orange line = OLS best-fit. Slope reflects rate of change per unit of x."
        )
        return fig, interp

    # Regression diagnostics 
    def regression_plots(self, y_true, y_pred, feature_names, coefs):
        y_true    = np.asarray(y_true, dtype=float)
        y_pred    = np.asarray(y_pred, dtype=float)
        coefs     = np.asarray(coefs,  dtype=float)
        residuals = y_true - y_pred

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Actual vs Predicted", "Residuals vs Predicted",
                "Residual Distribution", "Feature Coefficients",
            ],
        )
        mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())

        # Actual vs predicted
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode="markers",
            marker=dict(color=PALETTE[0], size=5, opacity=0.6), name="Predicted",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color=PALETTE[2], dash="dash", width=1.5), name="Perfect fit",
        ), row=1, col=1)

        # Residuals vs predicted
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals, mode="markers",
            marker=dict(color=PALETTE[1], size=5, opacity=0.6), name="Residuals",
        ), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color=PALETTE[3],
                      line_width=1.2, row=1, col=2)

        # Residual histogram
        fig.add_trace(go.Histogram(
            x=residuals, nbinsx=30,
            marker_color=PALETTE[4], opacity=0.75, name="Residual dist",
        ), row=2, col=1)

        # Coefficient bar
        colors = [PALETTE[0] if c >= 0 else PALETTE[3] for c in coefs]
        fig.add_trace(go.Bar(
            x=list(feature_names), y=list(coefs),
            marker_color=colors, name="Coefficients",
        ), row=2, col=2)

        fig.update_layout(**_BASE, title="Regression Diagnostics", height=700, showlegend=False)
        interp = (
            "**Actual vs Predicted:** Points hugging the dashed line = accurate predictions. "
            "**Residuals vs Predicted:** Random scatter around zero = assumptions met; "
            "patterns/funnels = heteroscedasticity. "
            "**Residual Distribution:** Bell shape confirms normality. "
            "**Coefficients:** Blue = positive effect on target; red = negative."
        )
        return fig, interp

    # Best-fit line 
    def regression_fit_line(self, df: pl.DataFrame, x: str, y: str):
        plot_df = df.select([x, y]).drop_nulls().to_pandas()
        xd, yd  = plot_df[x].to_numpy(), plot_df[y].to_numpy()
        mask    = ~(np.isnan(xd) | np.isnan(yd))
        xd, yd  = xd[mask], yd[mask]

        # OLS
        coeffs = np.polyfit(xd, yd, 1)
        xs_fit = np.linspace(xd.min(), xd.max(), 300)
        ys_fit = np.polyval(coeffs, xs_fit)

        # 95 % confidence band (simplified)
        y_hat  = np.polyval(coeffs, xd)
        res    = yd - y_hat
        n      = len(xd)
        s2     = np.sum(res**2) / (n - 2)
        x_mean = xd.mean()
        se     = np.sqrt(s2 * (1/n + (xs_fit - x_mean)**2 / np.sum((xd - x_mean)**2)))
        from scipy.stats import t as t_dist
        t_val  = t_dist.ppf(0.975, df=n - 2)

        fig = go.Figure()
        # Confidence band
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs_fit, xs_fit[::-1]]),
            y=np.concatenate([ys_fit + t_val * se, (ys_fit - t_val * se)[::-1]]),
            fill="toself", fillcolor="rgba(108,99,255,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI band",
        ))
        # Data points
        fig.add_trace(go.Scatter(
            x=xd, y=yd, mode="markers",
            marker=dict(color=PALETTE[0], size=5, opacity=0.55),
            name="Observations",
        ))
        # Best-fit line
        fig.add_trace(go.Scatter(
            x=xs_fit, y=ys_fit, mode="lines",
            line=dict(color=PALETTE[2], width=2.5), name="Best-fit line",
        ))
        _theme(fig, f"Regression Best-Fit — {x} → {y}")

        r   = np.corrcoef(xd, yd)[0, 1]
        interp = (
            f"Best-fit line: **{y} = {coeffs[0]:.4f} × {x} + {coeffs[1]:.4f}**. "
            f"R² = **{r**2:.4f}** — the line explains {r**2*100:.1f}% of variance in {y}. "
            "The shaded band is the 95% confidence interval around the regression line."
        )
        return fig, interp

    # Bar chart 
    def bar_chart(self, df: pl.DataFrame, x: str, y: str, agg: str = "mean"):
        agg_fn  = {"mean": pl.mean, "sum": pl.sum, "count": pl.count, "median": pl.median}
        grouped = df.group_by(x).agg(agg_fn[agg](y).alias(y)).sort(y, descending=True).to_pandas()
        fig = px.bar(grouped, x=x, y=y, color=y,
                     color_continuous_scale=["#1d3557", "#6C63FF", "#48CAE4"], text=y)
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        _theme(fig, f"{agg.title()} of {y} by {x}")
        top = grouped.iloc[0]
        interp = (
            f"**'{top[x]}'** leads with {agg} **{top[y]:.2f}**. "
            f"Overall {agg} = {grouped[y].mean():.2f}."
        )
        return fig, interp

    # Count plot 
    def count_plot(self, df: pl.DataFrame, column: str):
        counts = df[column].value_counts().sort("count", descending=True).head(20).to_pandas()
        fig = px.bar(counts, x=column, y="count", color="count",
                     color_continuous_scale=["#1d3557", "#6C63FF", "#48CAE4"], text="count")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        fig.update_layout(xaxis_tickangle=-30)
        _theme(fig, f"Value Counts — {column}")
        total   = counts["count"].sum()
        top_lbl = counts.iloc[0][column]
        top_cnt = int(counts.iloc[0]["count"])
        interp  = (
            f"**'{top_lbl}'** is the most frequent value: **{top_cnt:,}** "
            f"({top_cnt/total*100:.1f}% of total). "
            f"Showing top {min(20, len(counts))} of {df[column].n_unique()} unique values."
        )
        return fig, interp

    # Pie 
    def pie_chart(self, df: pl.DataFrame, column: str):
        counts = df[column].value_counts().sort("count", descending=True).head(10)
        labels = counts[column].to_list()
        values = counts["count"].to_list()
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.48,
            marker=dict(colors=PALETTE, line=dict(color="rgba(255,255,255,0.08)", width=1)),
        ))
        fig.update_traces(textposition="outside", textinfo="label+percent")
        _theme(fig, f"Category Breakdown — {column}")
        pct = round(values[0] / sum(values) * 100, 1)
        interp = (
            f"**'{labels[0]}'** dominates at **{pct}%**. "
            + ("Highly concentrated — consider grouping rare categories."
               if pct > 60 else "Relatively balanced across top categories.")
        )
        return fig, interp

    #Treemap 
    def treemap(self, df: pl.DataFrame, column: str):
        counts = df[column].value_counts().sort("count", descending=True).head(25).to_pandas()
        fig = px.treemap(counts, path=[column], values="count",
                         color="count",
                         color_continuous_scale=["#1d3557", "#6C63FF", "#48CAE4"])
        fig.update_traces(textinfo="label+value+percent root")
        _theme(fig, f"Treemap — {column}")
        interp = (
            "Tile area ∝ frequency. Larger tiles = more common categories. "
            "Hover any tile for exact count and share."
        )
        return fig, interp

    # Categorical × Numeric grouped mean 
    def cat_num_heatmap(self, df: pl.DataFrame, cat_col: str, num_col: str):
        grouped = (df.group_by(cat_col)
                     .agg(pl.mean(num_col).alias("mean"))
                     .sort("mean", descending=True)
                     .head(20)
                     .to_pandas())
        fig = go.Figure(go.Bar(
            x=grouped[cat_col], y=grouped["mean"],
            marker=dict(color=grouped["mean"],
                        colorscale=[[0, "#1d3557"], [0.5, "#6C63FF"], [1, "#48CAE4"]],
                        showscale=True,
                        colorbar=dict(title="Mean", tickfont=dict(color="#e2e8f0"))),
        ))
        fig.update_layout(xaxis_tickangle=-30)
        _theme(fig, f"Mean {num_col} by {cat_col}")
        interp = (
            f"Colour encodes the mean of **{num_col}** per **{cat_col}** group. "
            "Brighter = higher average. Spot which categories drive the numeric outcome."
        )
        return fig, interp

    #  Missing values heatmap 
    def missing_heatmap(self, df: pl.DataFrame):
        sample = df.head(500)
        z = np.array([[1 if v is None else 0 for v in sample[c].to_list()]
                      for c in sample.columns], dtype=float)
        total = sum(df[c].null_count() for c in df.columns)
        fig = go.Figure(go.Heatmap(
            z=z, x=list(range(sample.shape[0])), y=sample.columns,
            colorscale=[[0, "#1a2035"], [1, "#E63946"]], showscale=False,
        ))
        _theme(fig, "Missing Values Map  (red = missing)")
        fig.update_yaxes(tickfont=dict(size=10))
        pct = round(total / (df.shape[0] * df.shape[1]) * 100, 2)
        interp = (
            f"**{total:,}** missing cells (**{pct}%**). "
            "Red bands = systematic missingness. "
            "Check if data is MAR or structural (sensor dropout, optional fields)."
        )
        return fig, interp

    # Skewness chart 
    def skewness_chart(self, skew_data: list):
        cols   = [d["column"]   for d in skew_data]
        vals   = [d["skewness"] for d in skew_data]
        colors = [PALETTE[3] if abs(v) > 1 else PALETTE[0] for v in vals]
        fig = go.Figure(go.Bar(
            y=cols, x=vals, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
        ))
        fig.add_vline(x=1,  line_dash="dot", line_color="rgba(255,255,255,0.25)")
        fig.add_vline(x=-1, line_dash="dot", line_color="rgba(255,255,255,0.25)")
        _theme(fig, "Skewness by Column  (|skew| > 1 = highly skewed)")
        n = sum(1 for v in vals if abs(v) > 1)
        interp = (
            f"**{n}** of {len(cols)} features are highly skewed (red). "
            "Apply log / sqrt / Box-Cox transforms before modelling."
        )
        return fig, interp

    # Pair plot 
    def pair_plot(self, df: pl.DataFrame, columns: list, color_col: str = None):
        cols    = columns[:6]
        sel     = cols + ([color_col] if color_col and color_col not in cols else [])
        plot_df = df.select([c for c in sel if c in df.columns]).drop_nulls().to_pandas()
        fig = px.scatter_matrix(plot_df, dimensions=cols, color=color_col,
                                color_discrete_sequence=PALETTE, opacity=0.55)
        fig.update_traces(diagonal_visible=True,
                          marker=dict(size=3, line=dict(width=0.2, color="white")))
        fig.update_layout(**_BASE, title="Pair Plot — Multivariate Analysis", height=700)
        interp = (
            "Each cell = scatter between two features. Diagonal = each feature's own distribution. "
            "Look for linear clusters (correlations), bands (categorical effects), and outlier clouds."
        )
        return fig, interp

    # Time series 
    def time_series(self, df: pl.DataFrame, date_col: str, value_col: str):
        ts = df.select([date_col, value_col]).drop_nulls().sort(date_col).to_pandas()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts[date_col].astype(str), y=ts[value_col],
            mode="lines", name=value_col,
            line=dict(color=PALETTE[0], width=2),
            fill="tozeroy", fillcolor=_hex_to_rgba(PALETTE[0], 0.15),
        ))
        _theme(fig, f"Time Series — {value_col} over {date_col}")
        vals  = ts[value_col].to_numpy()
        trend = "upward" if vals[-1] > vals[0] else "downward" if vals[-1] < vals[0] else "flat"
        interp = (
            f"**{value_col}** shows a **{trend}** trend. "
            f"Peak = **{vals.max():.2f}**, trough = **{vals.min():.2f}**. "
            "Look for seasonality or structural breaks."
        )
        return fig, interp

    # Integer value distribution 
    def int_value_distribution(self, df: pl.DataFrame, column: str):
        counts = df[column].value_counts().sort(column).to_pandas()
        fig = px.bar(counts, x=column, y="count", color="count",
                     color_continuous_scale=["#1d3557", "#6C63FF", "#48CAE4"])
        fig.update_coloraxes(showscale=False)
        _theme(fig, f"Integer Value Distribution — {column}")
        most = counts.loc[counts["count"].idxmax(), column]
        interp = (
            f"**{column}** has **{df[column].n_unique()}** unique integer values. "
            f"Most common = **{most}**. "
            "If this column acts as a category, consider encoding it."
        )
        return fig, interp

    # Overview metrics dict 
    def overview_metrics(self, df: pl.DataFrame, eda: dict) -> dict:
        return {
            "rows":          df.shape[0],
            "columns":       df.shape[1],
            "numeric":       len(eda.get("numeric_cols", [])),
            "categorical":   len(eda.get("categorical_cols", [])),
            "missing_cells": sum(df[c].null_count() for c in df.columns),
            "duplicates":    eda.get("duplicates", 0),
        }