"""
report_exporter.py — Full Report Generator
PDF: all charts rendered via Matplotlib (no kaleido needed) + all analysis.
HTML: interactive Plotly divs + all analysis.
CSV and JSON export also supported.
"""

import io
import json
import logging
import textwrap
from datetime import datetime

import polars as pl
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
:root{--bg:#0f111a;--sur:#161b2e;--brd:#2a3050;
      --txt:#e2e8f0;--mut:#94a3b8;--acc:#6C63FF;
      --red:#E63946;--grn:#2DC653;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);
     font-family:'Segoe UI',system-ui,sans-serif;
     font-size:14px;line-height:1.75;padding:48px 56px}
h1{font-size:32px;font-weight:800;
   background:linear-gradient(90deg,#818cf8,#6C63FF,#48CAE4);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
   margin-bottom:6px}
h2{font-size:19px;font-weight:700;color:#c7d2fe;
   border-left:4px solid var(--acc);padding-left:12px;margin:44px 0 16px}
h3{font-size:15px;font-weight:600;color:#a5b4fc;margin:24px 0 10px}
p,li{color:var(--mut);margin-bottom:8px}
.meta{font-size:12px;color:#475569;margin-bottom:40px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
      gap:14px;margin:18px 0 32px}
.card{background:linear-gradient(135deg,rgba(108,99,255,.10),rgba(72,202,228,.05));
      border:1px solid rgba(108,99,255,.2);border-radius:14px;
      padding:22px 14px;text-align:center;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
              background:linear-gradient(90deg,#6C63FF,#48CAE4)}
.card .val{font-size:26px;font-weight:800;
           background:linear-gradient(135deg,#818cf8,#6C63FF);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.card .lbl{font-size:10px;color:#64748b;margin-top:5px;
           text-transform:uppercase;letter-spacing:.1em;font-weight:600}
table{width:100%;border-collapse:collapse;margin:10px 0 24px;font-size:13px}
thead tr{background:linear-gradient(90deg,#6C63FF,#5a52e0)}
thead th{padding:10px 14px;text-align:left;color:#fff;font-weight:600}
tbody tr:nth-child(even){background:rgba(255,255,255,.025)}
td{padding:8px 14px;border-bottom:1px solid var(--brd)}
.ok{color:var(--grn);font-weight:700} .warn{color:var(--red);font-weight:700}
.interp{background:linear-gradient(135deg,rgba(108,99,255,.08),rgba(72,202,228,.04));
        border:1px solid rgba(108,99,255,.18);border-left:3px solid var(--acc);
        border-radius:0 12px 12px 0;padding:14px 20px;margin:8px 0 22px;
        color:var(--txt);font-size:13px}
.chart-wrap{background:var(--sur);border:1px solid var(--brd);
            border-radius:16px;padding:14px;margin:16px 0 8px;overflow:hidden}
hr{border:none;border-top:1px solid rgba(108,99,255,.12);margin:44px 0}
"""

# ─────────────────────────────────────────────────────────────────────────────
def _plotly_to_png_bytes(fig, width=1200, height=600) -> bytes:
    """
    Convert a Plotly figure to PNG bytes without kaleido.
    Strategy:
      1. Try kaleido (fastest, best quality)
      2. Fall back to matplotlib recreation of a simplified version
    """
    # ── Try kaleido first ────────────────────────────────────────────────
    try:
        return fig.to_image(format="png", width=width, height=height, scale=1.5)
    except Exception:
        pass

    # ── Fallback: render via matplotlib ─────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig_mpl, ax = plt.subplots(figsize=(width/100, height/100),
                                   facecolor="#161b2e")
        ax.set_facecolor("#161b2e")

        title = ""
        try:
            title = fig.layout.title.text or ""
        except Exception:
            pass

        # Try to extract traces and render them
        rendered = False
        for trace in fig.data:
            try:
                ttype = trace.type if hasattr(trace, "type") else str(type(trace).__name__)

                if ttype in ("bar", "Bar"):
                    x = list(trace.x) if trace.x is not None else []
                    y = list(trace.y) if trace.y is not None else []
                    if x and y:
                        colors_bar = ["#6C63FF"] * len(x)
                        ax.bar(range(len(x)), y, color=colors_bar, alpha=0.82,
                               edgecolor="none")
                        ax.set_xticks(range(len(x)))
                        ax.set_xticklabels([str(v) for v in x],
                                           rotation=30, ha="right",
                                           fontsize=8, color="#94a3b8")
                        rendered = True

                elif ttype in ("scatter", "Scatter"):
                    x = list(trace.x) if trace.x is not None else []
                    y = list(trace.y) if trace.y is not None else []
                    if x and y:
                        mode = getattr(trace, "mode", "lines")
                        if "markers" in str(mode):
                            ax.scatter(x, y, color="#6C63FF", alpha=0.55, s=12)
                        else:
                            ax.plot(x, y, color="#6C63FF", linewidth=1.8, alpha=0.85)
                        rendered = True

                elif ttype in ("histogram", "Histogram"):
                    x = list(trace.x) if trace.x is not None else []
                    if x:
                        ax.hist(x, bins=40, color="#6C63FF", alpha=0.75,
                                edgecolor="none")
                        rendered = True

                elif ttype in ("heatmap", "Heatmap"):
                    z = trace.z
                    if z is not None:
                        z_arr = np.array(z, dtype=float)
                        ax.imshow(z_arr, cmap="RdPu", aspect="auto",
                                  interpolation="nearest")
                        rendered = True

                elif ttype in ("violin", "Violin"):
                    y = list(trace.y) if trace.y is not None else []
                    name = getattr(trace, "name", "") or ""
                    if y:
                        ax.violinplot([y], showmedians=True)
                        rendered = True

                elif ttype in ("box", "Box"):
                    y = list(trace.y) if trace.y is not None else []
                    if y:
                        ax.boxplot([y], patch_artist=True,
                                   boxprops=dict(facecolor="#6C63FF", alpha=0.7))
                        rendered = True

            except Exception:
                continue

        if not rendered:
            ax.text(0.5, 0.5,
                    f"Chart: {title}\n(visual preview unavailable)",
                    ha="center", va="center",
                    color="#6C63FF", fontsize=13,
                    transform=ax.transAxes)

        # Style axes
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a3050")
        ax.yaxis.label.set_color("#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")

        if title:
            ax.set_title(title, color="#c7d2fe", fontsize=11,
                         fontweight="bold", pad=10)

        plt.tight_layout(pad=1.5)
        buf = io.BytesIO()
        fig_mpl.savefig(buf, format="png", dpi=150,
                        facecolor="#161b2e", edgecolor="none")
        plt.close(fig_mpl)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        logger.warning("Fig to PNG fallback also failed: %s", e)
        return b""


# ─────────────────────────────────────────────────────────────────────────────
class ReportExporter:
    """Full report exporter: PDF (all charts + all analysis), HTML, CSV, JSON."""

    def __init__(self, project_name: str = "DataLens Analysis Report"):
        self.project_name = project_name
        self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── CSV ──────────────────────────────────────────────────────────────── #
    def to_csv_bytes(self, df: pl.DataFrame) -> bytes:
        buf = io.BytesIO()
        df.write_csv(buf)
        return buf.getvalue()

    # ── JSON ─────────────────────────────────────────────────────────────── #
    def to_json_bytes(self, report: dict) -> bytes:
        def _s(obj):
            if isinstance(obj, np.ndarray):    return obj.tolist()
            if isinstance(obj, np.integer):    return int(obj)
            if isinstance(obj, np.floating):   return float(obj)
            if isinstance(obj, dict):          return {k: _s(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [_s(i) for i in obj]
            if isinstance(obj, pl.DataFrame):  return obj.to_dicts()
            if isinstance(obj, (int,float,str,bool)) or obj is None: return obj
            return str(obj)
        return json.dumps(_s(report), indent=2, ensure_ascii=False).encode("utf-8")

    # ── Figure → image bytes (public, used by app.py buttons) ────────────── #
    @staticmethod
    def fig_to_image_bytes(fig, fmt: str = "png") -> bytes:
        """Export a Plotly figure as PNG or JPEG. Uses kaleido if available."""
        try:
            return fig.to_image(format=fmt, width=1400, height=700, scale=2)
        except Exception as e:
            raise RuntimeError(
                f"Image export failed ({e}). pip install kaleido")

    # ── HTML ─────────────────────────────────────────────────────────────── #
    def to_html_bytes(self, df: pl.DataFrame, eda: dict, stats: dict,
                      figures: dict = None, analysis_text: dict = None) -> bytes:
        import plotly.io as pio

        shape        = eda.get("shape", (0, 0))
        numeric_cols = eda.get("numeric_cols", [])
        cat_cols     = eda.get("categorical_cols", [])
        miss_total   = sum(df[c].null_count() for c in df.columns)
        high_corr    = eda.get("high_correlations", [])
        skewness     = eda.get("skewness", [])
        at           = analysis_text or {}

        def tbl(headers, rows, badge_col=None, good_val=None):
            ths = "".join(f"<th>{h}</th>" for h in headers)
            trs = ""
            for row in rows:
                tds = ""
                for ci, cell in enumerate(row):
                    s = str(cell)
                    if badge_col is not None and ci == badge_col:
                        cls = "ok" if s == str(good_val) else "warn"
                        tds += f'<td class="{cls}">{s}</td>'
                    else:
                        tds += f"<td>{s}</td>"
                trs += f"<tr>{tds}</tr>"
            return f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"

        def charts_html(key):
            if not figures or key not in figures:
                return ""
            out = ""
            for fig, interp in figures[key]:
                try:
                    div = pio.to_html(fig, full_html=False,
                                      include_plotlyjs=False)
                    out += (f'<div class="chart-wrap">{div}</div>'
                            f'<p class="interp">{interp}</p>')
                except Exception as e:
                    out += f'<p class="interp">Chart unavailable: {e}</p>'
            return out

        def texts(section):
            return "".join(
                f'<p class="interp">{v}</p>'
                for v in at.get(section, {}).values()
            )

        # Tables
        ss = eda.get("summary_stats")
        ss_tbl = ""
        if ss is not None and ss.shape[0] > 0:
            ss_rows = [
                [r["column"], r["count"], r["mean"], r["std"],
                 r["min"], r["median"], r["max"], r["skewness"],
                 f"{r['nulls']} ({r['null_pct']}%)"]
                for r in ss.iter_rows(named=True)
            ]
            ss_tbl = tbl(
                ["Column","Count","Mean","Std","Min","Median","Max","Skewness","Nulls"],
                ss_rows)

        profiles = eda.get("column_profiles", [])
        prof_tbl = tbl(
            ["Column","Dtype","Nulls","Null%","Unique","Sample"],
            [[p["column"], p["dtype"], p["null_count"],
              f"{p['null_pct']}%", p["unique"], str(p["sample"])[:50]]
             for p in profiles]) if profiles else ""

        miss_df  = eda.get("missing_summary")
        miss_tbl = (
            tbl(["Column","Missing","Pct"],
                [[r["column"], r["missing"], f"{r['pct']}%"]
                 for r in miss_df.iter_rows(named=True)])
            if miss_df is not None and miss_df.shape[0] > 0
            else "<p>No missing values found.</p>"
        )
        corr_tbl = (
            tbl(["Feature A","Feature B","r"],
                [[r["feature_a"],r["feature_b"],r["correlation"]]
                 for r in high_corr])
            if high_corr else "<p>No strongly correlated pairs.</p>"
        )
        skew_tbl = tbl(
            ["Column","Skewness","Flag"],
            [[s["column"], s["skewness"], "High" if s["highly_skewed"] else "OK"]
             for s in skewness],
            badge_col=2, good_val="OK")

        norm = stats.get("normality", {})
        norm_tbl = (
            tbl(["Column","Statistic","p-value","Normal?","Interpretation"],
                [[c, r.get("statistic","—"), r.get("p_value","—"),
                  "Yes" if r.get("is_normal") else "No",
                  r.get("interpretation","")]
                 for c, r in norm.items()],
                badge_col=3, good_val="Yes")
            if norm else "<p>No normality results.</p>"
        )
        ci = stats.get("confidence_intervals", {})
        ci_tbl = (
            tbl(["Column","n","Mean","Lower 95%","Upper 95%","Interpretation"],
                [[c, r.get("n"), r.get("mean"), r.get("lower"),
                  r.get("upper"), r.get("interpretation","")]
                 for c, r in ci.items()])
            if ci else "<p>No CI results.</p>"
        )

        reg = stats.get("regression") or {}
        reg_html = ""
        if reg:
            coef_rows = [
                [f, c, reg["p_values"].get(f,"—"),
                 "Yes" if reg["p_values"].get(f,1) < 0.05 else "No"]
                for f, c in reg.get("coefficients",{}).items()
            ]
            reg_html = f"""
            <hr><h2>Linear Regression</h2>
            <p>Target: <strong>{reg.get('target','—')}</strong> &nbsp;|&nbsp;
               R&sup2; = <strong>{reg.get('r_squared','—')}</strong> &nbsp;|&nbsp;
               Adj-R&sup2; = <strong>{reg.get('adjusted_r_squared','—')}</strong> &nbsp;|&nbsp;
               RMSE = <strong>{reg.get('rmse','—')}</strong> &nbsp;|&nbsp;
               MAE = <strong>{reg.get('mae','—')}</strong> &nbsp;|&nbsp;
               n = <strong>{reg.get('n','—')}</strong></p>
            <p class="interp">{reg.get('interpretation','')}</p>
            {tbl(['Feature','Coefficient','p-value','Significant?'],
                 coef_rows, badge_col=3, good_val='Yes')}
            {charts_html('Regression')}"""

        html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{self.project_name}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>{_CSS}</style>
</head><body>
<h1>&#128300; {self.project_name}</h1>
<p class="meta">Generated: {self.generated_at} &nbsp;|&nbsp;
{shape[0]:,} rows &times; {shape[1]} columns</p>

<h2>Dataset Overview</h2>
<div class="grid">
<div class="card"><div class="val">{shape[0]:,}</div><div class="lbl">Rows</div></div>
<div class="card"><div class="val">{shape[1]}</div><div class="lbl">Columns</div></div>
<div class="card"><div class="val">{len(numeric_cols)}</div><div class="lbl">Numeric</div></div>
<div class="card"><div class="val">{len(cat_cols)}</div><div class="lbl">Categorical</div></div>
<div class="card"><div class="val">{miss_total:,}</div><div class="lbl">Missing</div></div>
<div class="card"><div class="val">{eda.get('duplicates',0):,}</div><div class="lbl">Duplicates</div></div>
</div>
{texts('Overview')}{charts_html('Overview')}
<hr><h2>Column Profiles</h2>{prof_tbl}
<hr><h2>Summary Statistics</h2>{ss_tbl}
<hr><h2>Missing Values</h2>{miss_tbl}
<hr><h2>Skewness</h2>{skew_tbl}
{texts('Distributions')}{charts_html('Distributions')}
<hr><h2>KDE &amp; Pair Plot</h2>
{texts('KDE & Pair Plot')}{charts_html('KDE & Pair Plot')}
<hr><h2>Correlations</h2>{corr_tbl}
{charts_html('Correlations')}
<hr><h2>Category Analysis</h2>{charts_html('Categories')}
<hr><h2>Outlier Analysis</h2>
{texts('Outliers')}{charts_html('Outliers')}
<hr><h2>Normality Tests</h2>{norm_tbl}
{texts('Statistics')}{charts_html('Statistics')}
<hr><h2>High Correlations</h2>{corr_tbl}
<hr><h2>Confidence Intervals</h2>{ci_tbl}
{reg_html}
<p class="meta" style="margin-top:60px;text-align:center">
&mdash; End of Report &mdash; {self.project_name} &mdash; {self.generated_at} &mdash;
</p></body></html>"""
        return html.encode("utf-8")

    # ── PDF — all charts as images, all analysis, all tables ─────────────── #
    def to_pdf_bytes(self, df: pl.DataFrame, eda: dict, stats: dict,
                     figures: dict = None, analysis_text: dict = None) -> bytes:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, Image as RLImage,
            )
            from reportlab.lib.units import mm
        except ImportError:
            raise ImportError("pip install reportlab")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=18*mm, rightMargin=18*mm,
            topMargin=20*mm, bottomMargin=20*mm,
        )

        # ── Color palette ─────────────────────────────────────────────────── #
        ACCENT = colors.HexColor("#6C63FF")
        MUTED  = colors.HexColor("#94a3b8")
        LIGHT  = colors.HexColor("#e2e8f0")
        BORDER = colors.HexColor("#2a3050")
        DARK   = colors.HexColor("#0f111a")
        SURF   = colors.HexColor("#161b2e")
        SURF2  = colors.HexColor("#1a1f36")

        # ── Text styles ───────────────────────────────────────────────────── #
        title_s = ParagraphStyle(
            "T", fontSize=22, textColor=ACCENT,
            fontName="Helvetica-Bold", spaceAfter=4)
        meta_s = ParagraphStyle(
            "M", fontSize=9, textColor=MUTED,
            fontName="Helvetica", spaceAfter=14)
        h2_s = ParagraphStyle(
            "H2", fontSize=13, textColor=colors.HexColor("#c7d2fe"),
            fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=7)
        h3_s = ParagraphStyle(
            "H3", fontSize=11, textColor=colors.HexColor("#a5b4fc"),
            fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
        body_s = ParagraphStyle(
            "B", fontSize=9, textColor=MUTED,
            fontName="Helvetica", spaceAfter=5, leading=14)
        interp_s = ParagraphStyle(
            "I", fontSize=9, textColor=LIGHT,
            fontName="Helvetica-Oblique", spaceAfter=8,
            leading=14, leftIndent=10, rightIndent=4,
            backColor=SURF2, borderPad=6)
        caption_s = ParagraphStyle(
            "C", fontSize=8, textColor=MUTED,
            fontName="Helvetica-Oblique", spaceAfter=10,
            alignment=1)  # centre

        def hr():
            return HRFlowable(
                width="100%", thickness=0.5, color=BORDER,
                spaceAfter=10, spaceBefore=4)

        def mk_tbl(data, col_widths=None):
            if not data:
                return Spacer(1, 1)
            t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
                ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",       (0, 0), (-1, -1), 8),
                ("GRID",           (0, 0), (-1, -1), 0.25, BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SURF, DARK]),
                ("TEXTCOLOR",      (0, 1), (-1, -1), LIGHT),
                ("PADDING",        (0, 0), (-1, -1), 5),
                ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
                ("VALIGN",         (0, 0), (-1, -1), "TOP"),
            ]))
            return t

        # ── Chart → ReportLab Image ───────────────────────────────────────── #
        PW = A4[0] - 36*mm   # usable page width

        def add_figure(fig, story, w_frac=1.0, h_mm=90):
            """Render a Plotly figure and append as an image to story."""
            title_text = ""
            try:
                title_text = fig.layout.title.text or ""
            except Exception:
                pass

            png = _plotly_to_png_bytes(fig, width=1200, height=600)
            if png:
                img_buf = io.BytesIO(png)
                img = RLImage(img_buf,
                              width=PW * w_frac,
                              height=h_mm * mm)
                story.append(img)
                if title_text:
                    story.append(Paragraph(title_text, caption_s))
            else:
                story.append(
                    Paragraph(f"[Chart unavailable: {title_text}]", body_s))

        def add_figures(key, story):
            if not figures or key not in figures:
                return
            for fig, interp in figures[key]:
                add_figure(fig, story)
                story.append(Paragraph(interp, interp_s))
                story.append(Spacer(1, 6))

        def add_texts(section, story):
            for txt in (analysis_text or {}).get(section, {}).values():
                story.append(Paragraph(txt, interp_s))

        # ── Build story ───────────────────────────────────────────────────── #
        story = []
        shape        = eda.get("shape", (0, 0))
        numeric_cols = eda.get("numeric_cols", [])
        cat_cols     = eda.get("categorical_cols", [])
        miss_total   = sum(df[c].null_count() for c in df.columns)
        high_corr    = eda.get("high_correlations", [])
        skewness     = eda.get("skewness", [])

        # ── Cover ─────────────────────────────────────────────────────────── #
        story.append(Paragraph(self.project_name, title_s))
        story.append(Paragraph(
            f"Generated: {self.generated_at}  |  "
            f"{shape[0]:,} rows × {shape[1]} columns  |  "
            f"{len(numeric_cols)} numeric  |  {len(cat_cols)} categorical",
            meta_s))
        story.append(hr())

        # ── Overview table ────────────────────────────────────────────────── #
        story.append(Paragraph("Dataset Overview", h2_s))
        ov = [
            ["Metric",               "Value"],
            ["Total rows",           f"{shape[0]:,}"],
            ["Total columns",        str(shape[1])],
            ["Numeric columns",      str(len(numeric_cols))],
            ["Categorical columns",  str(len(cat_cols))],
            ["Missing cells",        f"{miss_total:,}"],
            ["Duplicate rows",       str(eda.get("duplicates", 0))],
        ]
        story.append(mk_tbl(ov, col_widths=[90*mm, 70*mm]))
        story.append(Spacer(1, 8))
        add_texts("Overview", story)
        add_figures("Overview", story)
        story.append(hr())

        # ── Column profiles ───────────────────────────────────────────────── #
        profiles = eda.get("column_profiles", [])
        if profiles:
            story.append(Paragraph("Column Profiles", h2_s))
            data = [["Column", "Dtype", "Nulls", "Null%", "Unique", "Sample"]] + [
                [str(p["column"]) or "(unnamed)",
                 str(p["dtype"]),
                 str(p["null_count"]),
                 f"{p['null_pct']}%",
                 str(p["unique"]),
                 str(p["sample"])[:35]]
                for p in profiles
            ]
            story.append(mk_tbl(data))
            story.append(hr())

        # ── Summary statistics ────────────────────────────────────────────── #
        ss = eda.get("summary_stats")
        if ss is not None and ss.shape[0] > 0:
            story.append(Paragraph("Summary Statistics", h2_s))
            data = [["Column","Count","Mean","Std","Min","Median","Max","Skewness"]] + [
                [str(r["column"]) or "(unnamed)",
                 r["count"], r["mean"], r["std"],
                 r["min"], r["median"], r["max"], r["skewness"]]
                for r in ss.iter_rows(named=True)
            ]
            story.append(mk_tbl(data))
            n_sk = sum(1 for s in skewness if s["highly_skewed"])
            story.append(Paragraph(
                f"{n_sk} of {ss.shape[0]} features are highly skewed (|skew| > 1) "
                "and may benefit from log/sqrt/Box-Cox transformation.", interp_s))
            story.append(hr())

        # ── Missing values ────────────────────────────────────────────────── #
        story.append(Paragraph("Missing Values", h2_s))
        miss_df = eda.get("missing_summary")
        if miss_df is not None and miss_df.shape[0] > 0:
            data = [["Column", "Missing Count", "Missing %"]] + [
                [r["column"], str(r["missing"]), f"{r['pct']}%"]
                for r in miss_df.iter_rows(named=True)
            ]
            story.append(mk_tbl(data))
        else:
            story.append(Paragraph("No missing values found in this dataset.", body_s))
        story.append(hr())

        # ── Skewness ──────────────────────────────────────────────────────── #
        if skewness:
            story.append(Paragraph("Skewness Analysis", h2_s))
            data = [["Column", "Skewness", "Flag"]] + [
                [str(s["column"]) or "(unnamed)",
                 str(s["skewness"]),
                 "HIGH" if s["highly_skewed"] else "OK"]
                for s in skewness
            ]
            story.append(mk_tbl(data))
            story.append(hr())

        # ── High correlations ─────────────────────────────────────────────── #
        story.append(Paragraph("High Correlations", h2_s))
        if high_corr:
            data = [["Feature A", "Feature B", "Pearson r"]] + [
                [r["feature_a"], r["feature_b"], str(r["correlation"])]
                for r in high_corr
            ]
            story.append(mk_tbl(data))
            story.append(Paragraph(
                f"{len(high_corr)} pairs with |r| ≥ threshold. "
                "High correlation signals potential multicollinearity.", interp_s))
        else:
            story.append(Paragraph(
                "No strongly correlated feature pairs detected.", body_s))
        story.append(hr())

        # ── Distribution charts ───────────────────────────────────────────── #
        if figures and ("Distributions" in figures or "KDE & Pair Plot" in figures):
            story.append(Paragraph("Distributions & KDE", h2_s))
            add_texts("Distributions", story)
            add_figures("Distributions", story)
            story.append(Paragraph("KDE & Pair Plot", h3_s))
            add_figures("KDE & Pair Plot", story)
            story.append(hr())

        # ── Correlation charts ────────────────────────────────────────────── #
        if figures and "Correlations" in figures:
            story.append(Paragraph("Correlation Analysis", h2_s))
            add_figures("Correlations", story)
            story.append(hr())

        # ── Category charts ───────────────────────────────────────────────── #
        if figures and "Categories" in figures:
            story.append(Paragraph("Category Analysis", h2_s))
            add_figures("Categories", story)
            story.append(hr())

        # ── Outlier charts ────────────────────────────────────────────────── #
        if figures and "Outliers" in figures:
            story.append(Paragraph("Outlier Analysis", h2_s))
            add_texts("Outliers", story)
            add_figures("Outliers", story)
            story.append(hr())

        # ── Normality tests ───────────────────────────────────────────────── #
        norm = stats.get("normality", {})
        if norm:
            story.append(Paragraph("Normality Tests (Shapiro-Wilk, α = 0.05)", h2_s))
            data = [["Column","Statistic","p-value","Normal?","Interpretation"]] + [
                [str(c) or "(unnamed)",
                 str(r.get("statistic","—")),
                 str(r.get("p_value","—")),
                 "Yes" if r.get("is_normal") else "No",
                 textwrap.shorten(r.get("interpretation",""), 100)]
                for c, r in norm.items()
            ]
            story.append(mk_tbl(data))
            n_normal = sum(1 for r in norm.values() if r.get("is_normal"))
            story.append(Paragraph(
                f"{n_normal} of {len(norm)} columns follow a normal distribution. "
                "Non-normal features may require non-parametric tests or transformations.",
                interp_s))
            add_texts("Statistics", story)
            add_figures("Statistics", story)
            story.append(hr())

        # ── Confidence intervals ──────────────────────────────────────────── #
        ci = stats.get("confidence_intervals", {})
        if ci:
            story.append(Paragraph("95% Confidence Intervals (Mean)", h2_s))
            data = [["Column","n","Mean","Lower 95%","Upper 95%","Interpretation"]] + [
                [str(c) or "(unnamed)",
                 str(r.get("n","")),
                 str(r.get("mean","")),
                 str(r.get("lower","")),
                 str(r.get("upper","")),
                 textwrap.shorten(r.get("interpretation",""), 90)]
                for c, r in ci.items()
            ]
            story.append(mk_tbl(data))
            story.append(hr())

        # ── Regression ────────────────────────────────────────────────────── #
        reg = stats.get("regression") or {}
        if reg:
            story.append(Paragraph("Linear Regression", h2_s))
            meta_data = [
                ["Metric",      "Value"],
                ["Target",      str(reg.get("target","—"))],
                ["R²",          str(reg.get("r_squared","—"))],
                ["Adj R²",      str(reg.get("adjusted_r_squared","—"))],
                ["RMSE",        str(reg.get("rmse","—"))],
                ["MAE",         str(reg.get("mae","—"))],
                ["n",           str(reg.get("n","—"))],
            ]
            story.append(mk_tbl(meta_data, col_widths=[80*mm, 90*mm]))
            story.append(Paragraph(reg.get("interpretation",""), interp_s))
            story.append(Spacer(1, 6))
            story.append(Paragraph("Coefficients & Significance", h3_s))
            coef_data = [["Feature","Coefficient","p-value","Significant?"]] + [
                [str(f),
                 str(c),
                 str(reg["p_values"].get(f,"—")),
                 "Yes" if reg["p_values"].get(f,1) < 0.05 else "No"]
                for f, c in reg.get("coefficients",{}).items()
            ]
            story.append(mk_tbl(coef_data))
            add_texts("Regression", story)
            add_figures("Regression", story)
            story.append(hr())

        # ── Footer ────────────────────────────────────────────────────────── #
        story.append(Paragraph(
            f"End of Report  ·  {self.project_name}  ·  {self.generated_at}",
            meta_s))

        doc.build(story)
        return buf.getvalue()
