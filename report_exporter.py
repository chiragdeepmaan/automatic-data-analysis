"""
report_exporter.py — Full Report Generator
PDF: every chart as image + all written analysis + all tables.
HTML: every chart as interactive Plotly div + all analysis.
Also: CSV and JSON export.
"""

import io
import json
import logging
import textwrap
from datetime import datetime

import polars as pl
import numpy as np

logger = logging.getLogger(__name__)

# ── Shared HTML report CSS ───────────────────────────────────────────────────
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
   border-left:4px solid var(--acc);padding-left:12px;
   margin:44px 0 16px}
h3{font-size:15px;font-weight:600;color:#a5b4fc;margin:24px 0 10px}
p,li{color:var(--mut);margin-bottom:8px}
.meta{font-size:12px;color:#475569;margin-bottom:40px;letter-spacing:.03em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
      gap:14px;margin:18px 0 32px}
.card{background:linear-gradient(135deg,rgba(108,99,255,0.10),rgba(72,202,228,0.05));
      border:1px solid rgba(108,99,255,0.2);border-radius:14px;
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
tbody tr:nth-child(even){background:rgba(255,255,255,0.025)}
td{padding:8px 14px;border-bottom:1px solid var(--brd)}
.ok{color:var(--grn);font-weight:700}
.warn{color:var(--red);font-weight:700}
.interp{background:linear-gradient(135deg,rgba(108,99,255,0.08),rgba(72,202,228,0.04));
        border:1px solid rgba(108,99,255,0.18);
        border-left:3px solid var(--acc);
        border-radius:0 12px 12px 0;
        padding:14px 20px;margin:8px 0 22px;color:var(--txt);font-size:13px}
.chart-wrap{background:var(--sur);border:1px solid var(--brd);
            border-radius:16px;padding:14px;margin:16px 0 8px;overflow:hidden}
hr{border:none;border-top:1px solid rgba(108,99,255,0.12);margin:44px 0}
.section-badge{display:inline-block;padding:3px 12px;border-radius:20px;
               font-size:11px;font-weight:700;letter-spacing:.06em;
               background:rgba(108,99,255,0.15);color:#818cf8;margin-bottom:12px}
"""


class ReportExporter:
    """Full report exporter: PDF (all charts + analysis), HTML, CSV, JSON."""

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
            if isinstance(obj, np.ndarray):   return obj.tolist()
            if isinstance(obj, np.integer):   return int(obj)
            if isinstance(obj, np.floating):  return float(obj)
            if isinstance(obj, dict):         return {k: _s(v) for k, v in obj.items()}
            if isinstance(obj, (list,tuple)): return [_s(i) for i in obj]
            if isinstance(obj, pl.DataFrame): return obj.to_dicts()
            if isinstance(obj, (int,float,str,bool)) or obj is None: return obj
            return str(obj)
        return json.dumps(_s(report), indent=2, ensure_ascii=False).encode("utf-8")

    # ── Figure → PNG bytes ───────────────────────────────────────────────── #
    @staticmethod
    def fig_to_image_bytes(fig, fmt: str = "png") -> bytes:
        """Export a Plotly figure as PNG or JPEG. Requires kaleido."""
        try:
            return fig.to_image(format=fmt, width=1400, height=700, scale=2)
        except Exception as e:
            raise RuntimeError(f"Image export failed: {e} — pip install kaleido")

    # ── HTML ─────────────────────────────────────────────────────────────── #
    def to_html_bytes(self, df: pl.DataFrame, eda: dict, stats: dict,
                      figures: dict = None, analysis_text: dict = None) -> bytes:
        import plotly.io as pio

        shape        = eda.get("shape", (0,0))
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

        def charts(key):
            if not figures or key not in figures: return ""
            out = ""
            for fig, interp in figures[key]:
                try:
                    div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                    out += (f'<div class="chart-wrap">{div}</div>'
                            f'<p class="interp">{interp}</p>')
                except Exception as e:
                    out += f'<p class="interp">Chart unavailable: {e}</p>'
            return out

        def texts(section):
            s = at.get(section, {})
            if not s: return ""
            return "".join(f'<p class="interp">{v}</p>' for v in s.values())

        # Stats tables
        ss = eda.get("summary_stats")
        ss_tbl = ""
        if ss is not None and ss.shape[0] > 0:
            ss_rows = [[r["column"],r["count"],r["mean"],r["std"],
                        r["min"],r["median"],r["max"],r["skewness"],
                        f"{r['nulls']} ({r['null_pct']}%)"]
                       for r in ss.iter_rows(named=True)]
            ss_tbl = tbl(["Column","Count","Mean","Std","Min","Median","Max","Skewness","Nulls"],
                          ss_rows)

        profiles = eda.get("column_profiles",[])
        prof_tbl = tbl(["Column","Dtype","Nulls","Null%","Unique","Sample"],
                       [[p["column"],p["dtype"],p["null_count"],
                         f"{p['null_pct']}%",p["unique"],str(p["sample"])[:50]]
                        for p in profiles]) if profiles else ""

        miss_df = eda.get("missing_summary")
        miss_tbl = (tbl(["Column","Missing","Pct"],
                        [[r["column"],r["missing"],f"{r['pct']}%"]
                         for r in miss_df.iter_rows(named=True)])
                    if miss_df is not None and miss_df.shape[0]>0
                    else "<p>No missing values.</p>")

        corr_tbl = (tbl(["Feature A","Feature B","r"],
                        [[r["feature_a"],r["feature_b"],r["correlation"]]
                         for r in high_corr])
                    if high_corr else "<p>No strongly correlated pairs.</p>")

        skew_tbl = tbl(["Column","Skewness","Flag"],
                        [[s["column"],s["skewness"],"High" if s["highly_skewed"] else "OK"]
                         for s in skewness],
                        badge_col=2, good_val="OK")

        norm = stats.get("normality",{})
        norm_tbl = (tbl(["Column","Statistic","p-value","Normal?","Interpretation"],
                         [[c,r.get("statistic","—"),r.get("p_value","—"),
                           "Yes" if r.get("is_normal") else "No",
                           r.get("interpretation","")]
                          for c,r in norm.items()],
                         badge_col=3, good_val="Yes")
                    if norm else "<p>No normality results.</p>")

        ci = stats.get("confidence_intervals",{})
        ci_tbl = (tbl(["Column","n","Mean","Lower 95%","Upper 95%","Interpretation"],
                       [[c,r.get("n"),r.get("mean"),r.get("lower"),
                         r.get("upper"),r.get("interpretation","")]
                        for c,r in ci.items()])
                  if ci else "<p>No CI results.</p>")

        reg = stats.get("regression") or {}
        reg_html = ""
        if reg:
            coef_rows = [[f,c,reg["p_values"].get(f,"—"),
                          "Yes" if reg["p_values"].get(f,1)<0.05 else "No"]
                         for f,c in reg.get("coefficients",{}).items()]
            reg_html = f"""
            <hr><h2>Linear Regression</h2>
            <p>Target: <strong>{reg.get('target')}</strong> &nbsp;|&nbsp;
               R&sup2; = <strong>{reg.get('r_squared')}</strong> &nbsp;|&nbsp;
               Adj-R&sup2; = <strong>{reg.get('adjusted_r_squared')}</strong> &nbsp;|&nbsp;
               RMSE = <strong>{reg.get('rmse')}</strong> &nbsp;|&nbsp;
               MAE = <strong>{reg.get('mae')}</strong> &nbsp;|&nbsp;
               n = <strong>{reg.get('n')}</strong></p>
            <p class="interp">{reg.get('interpretation','')}</p>
            {tbl(['Feature','Coefficient','p-value','Significant?'],
                 coef_rows, badge_col=3, good_val='Yes')}
            {charts('Regression')}
            """

        html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{self.project_name}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>{_CSS}</style>
</head><body>
<h1>&#128300; {self.project_name}</h1>
<p class="meta">Generated: {self.generated_at} &nbsp;|&nbsp; {shape[0]:,} rows &times; {shape[1]} columns</p>

<h2>Dataset Overview</h2>
<div class="grid">
<div class="card"><div class="val">{shape[0]:,}</div><div class="lbl">Rows</div></div>
<div class="card"><div class="val">{shape[1]}</div><div class="lbl">Columns</div></div>
<div class="card"><div class="val">{len(numeric_cols)}</div><div class="lbl">Numeric</div></div>
<div class="card"><div class="val">{len(cat_cols)}</div><div class="lbl">Categorical</div></div>
<div class="card"><div class="val">{miss_total:,}</div><div class="lbl">Missing</div></div>
<div class="card"><div class="val">{eda.get('duplicates',0):,}</div><div class="lbl">Duplicates</div></div>
</div>
{texts('Overview')}
{charts('Overview')}

<hr><h2>Column Profiles</h2>{prof_tbl}

<hr><h2>Summary Statistics</h2>{ss_tbl}

<hr><h2>Missing Values</h2>{miss_tbl}

<hr><h2>Skewness Analysis</h2>{skew_tbl}
{texts('Distributions')}
{charts('Distributions')}

<hr><h2>KDE &amp; Pair Plot</h2>
{texts('KDE & Pair Plot')}
{charts('KDE & Pair Plot')}

<hr><h2>Correlations</h2>{corr_tbl}
{charts('Correlations')}

<hr><h2>Category Analysis</h2>
{charts('Categories')}

<hr><h2>Outlier Analysis</h2>
{charts('Outliers')}

<hr><h2>Normality Tests</h2>{norm_tbl}
{texts('Statistics')}
{charts('Statistics')}

<hr><h2>Confidence Intervals</h2>{ci_tbl}

{reg_html}

<p class="meta" style="margin-top:60px;text-align:center">
&mdash; End of Report &mdash; {self.project_name} &mdash; {self.generated_at} &mdash;
</p></body></html>"""
        return html.encode("utf-8")

    # ── PDF — all charts + all analysis ──────────────────────────────────── #
    def to_pdf_bytes(self, df: pl.DataFrame, eda: dict, stats: dict,
                     figures: dict = None, analysis_text: dict = None) -> bytes:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, Image as RLImage, KeepTogether,
            )
            from reportlab.lib.units import mm
        except ImportError:
            raise ImportError("pip install reportlab")

        W, H   = A4
        buf    = io.BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4,
                                   leftMargin=18*mm, rightMargin=18*mm,
                                   topMargin=20*mm, bottomMargin=20*mm)

        # ── Styles ───────────────────────────────────────────────────────── #
        ACCENT  = colors.HexColor("#6C63FF")
        CYAN    = colors.HexColor("#48CAE4")
        DARK    = colors.HexColor("#0f111a")
        SURF    = colors.HexColor("#161b2e")
        MUTED   = colors.HexColor("#94a3b8")
        LIGHT   = colors.HexColor("#e2e8f0")
        GREEN   = colors.HexColor("#2DC653")
        RED     = colors.HexColor("#E63946")
        BORDER  = colors.HexColor("#2a3050")

        title_s  = ParagraphStyle("T", fontSize=24, textColor=ACCENT,
                                  fontName="Helvetica-Bold", spaceAfter=4)
        meta_s   = ParagraphStyle("M", fontSize=9,  textColor=MUTED,
                                  fontName="Helvetica",       spaceAfter=16)
        h2_s     = ParagraphStyle("H2", fontSize=14, textColor=colors.HexColor("#c7d2fe"),
                                  fontName="Helvetica-Bold",
                                  spaceBefore=20, spaceAfter=8)
        h3_s     = ParagraphStyle("H3", fontSize=11, textColor=colors.HexColor("#a5b4fc"),
                                  fontName="Helvetica-Bold",
                                  spaceBefore=12, spaceAfter=5)
        body_s   = ParagraphStyle("B", fontSize=9,  textColor=MUTED,
                                  fontName="Helvetica",       spaceAfter=5,
                                  leading=14)
        interp_s = ParagraphStyle("I", fontSize=9,  textColor=LIGHT,
                                  fontName="Helvetica-Oblique", spaceAfter=8,
                                  leading=14, leftIndent=10,
                                  borderPad=6, backColor=colors.HexColor("#1a1f36"),
                                  borderColor=ACCENT, borderWidth=0)

        def hr():
            return HRFlowable(width="100%", thickness=0.5, color=BORDER,
                              spaceAfter=10, spaceBefore=4)

        def mk_tbl(data, col_widths=None):
            if not data: return Spacer(1,1)
            t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,0), ACCENT),
                ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
                ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0),(-1,-1), 8),
                ("GRID",          (0,0),(-1,-1), 0.25, BORDER),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),
                 [colors.HexColor("#161b2e"), colors.HexColor("#0f111a")]),
                ("TEXTCOLOR",     (0,1),(-1,-1), LIGHT),
                ("PADDING",       (0,0),(-1,-1), 5),
                ("FONTNAME",      (0,1),(-1,-1), "Helvetica"),
                ("WORDWRAP",      (0,0),(-1,-1), True),
                ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ]))
            return t

        def fig_image(fig, w_mm=170, h_mm=85):
            """Convert Plotly figure to a ReportLab Image flowable."""
            try:
                png = fig.to_image(format="png", width=1400, height=700, scale=1.5)
                img_buf = io.BytesIO(png)
                return RLImage(img_buf, width=w_mm*mm, height=h_mm*mm)
            except Exception:
                return Paragraph("[Chart: install kaleido for image export]", body_s)

        def add_figures(key, story):
            if not figures or key not in figures: return
            for fig, interp in figures[key]:
                title = getattr(fig.layout, "title", None)
                t_text = title.text if title and title.text else ""
                if t_text:
                    story.append(Paragraph(t_text, h3_s))
                story.append(fig_image(fig))
                story.append(Paragraph(interp, interp_s))
                story.append(Spacer(1, 6))

        def add_texts(section, story):
            at = analysis_text or {}
            for txt in at.get(section, {}).values():
                story.append(Paragraph(txt, interp_s))

        # ── Build story ──────────────────────────────────────────────────── #
        story = []
        shape = eda.get("shape", (0,0))
        numeric_cols = eda.get("numeric_cols", [])
        cat_cols     = eda.get("categorical_cols", [])
        miss_total   = sum(df[c].null_count() for c in df.columns)

        # Cover / header
        story.append(Paragraph(self.project_name, title_s))
        story.append(Paragraph(
            f"Generated: {self.generated_at}  |  "
            f"File: {shape[0]:,} rows × {shape[1]} columns", meta_s))
        story.append(hr())

        # ── Overview ─────────────────────────────────────────────────────── #
        story.append(Paragraph("Dataset Overview", h2_s))
        ov_data = [
            ["Metric","Value"],
            ["Total rows",          f"{shape[0]:,}"],
            ["Total columns",       str(shape[1])],
            ["Numeric columns",     str(len(numeric_cols))],
            ["Categorical columns", str(len(cat_cols))],
            ["Missing cells",       f"{miss_total:,}"],
            ["Duplicate rows",      str(eda.get("duplicates",0))],
        ]
        story.append(mk_tbl(ov_data, col_widths=[90*mm, 70*mm]))
        story.append(Spacer(1,8))
        add_texts("Overview", story)
        add_figures("Overview", story)
        story.append(hr())

        # ── Column profiles ───────────────────────────────────────────────── #
        profiles = eda.get("column_profiles",[])
        if profiles:
            story.append(Paragraph("Column Profiles", h2_s))
            prof_data = [["Column","Dtype","Nulls","Null%","Unique","Sample"]] + [
                [p["column"], p["dtype"], str(p["null_count"]),
                 f"{p['null_pct']}%", str(p["unique"]), str(p["sample"])[:35]]
                for p in profiles
            ]
            story.append(mk_tbl(prof_data))
            story.append(hr())

        # ── Summary stats ─────────────────────────────────────────────────── #
        ss = eda.get("summary_stats")
        if ss is not None and ss.shape[0] > 0:
            story.append(Paragraph("Summary Statistics", h2_s))
            ss_data = [["Column","Count","Mean","Std","Min","Median","Max","Skewness"]] + [
                [r["column"],r["count"],r["mean"],r["std"],
                 r["min"],r["median"],r["max"],r["skewness"]]
                for r in ss.iter_rows(named=True)
            ]
            story.append(mk_tbl(ss_data))
            n_sk = sum(1 for s in eda.get("skewness",[]) if s["highly_skewed"])
            story.append(Paragraph(
                f"{n_sk} of {ss.shape[0]} features are highly skewed (|skew|>1) "
                "and may benefit from transformation before modelling.", interp_s))
            story.append(hr())

        # ── Missing values ────────────────────────────────────────────────── #
        miss_df = eda.get("missing_summary")
        story.append(Paragraph("Missing Values", h2_s))
        if miss_df is not None and miss_df.shape[0] > 0:
            m_data = [["Column","Missing Count","Missing %"]] + [
                [r["column"], str(r["missing"]), f"{r['pct']}%"]
                for r in miss_df.iter_rows(named=True)
            ]
            story.append(mk_tbl(m_data, col_widths=[80*mm,50*mm,40*mm]))
        else:
            story.append(Paragraph("No missing values found.", body_s))
        story.append(hr())

        # ── Skewness ─────────────────────────────────────────────────────── #
        skewness = eda.get("skewness",[])
        if skewness:
            story.append(Paragraph("Skewness Analysis", h2_s))
            sk_data = [["Column","Skewness","Flag"]] + [
                [s["column"], str(s["skewness"]),
                 "HIGH" if s["highly_skewed"] else "OK"]
                for s in skewness
            ]
            story.append(mk_tbl(sk_data, col_widths=[80*mm,50*mm,40*mm]))
            story.append(hr())

        # ── High correlations ─────────────────────────────────────────────── #
        hc = eda.get("high_correlations",[])
        story.append(Paragraph("High Correlations", h2_s))
        if hc:
            hc_data = [["Feature A","Feature B","Pearson r"]] + [
                [r["feature_a"], r["feature_b"], str(r["correlation"])]
                for r in hc
            ]
            story.append(mk_tbl(hc_data))
            story.append(Paragraph(
                f"{len(hc)} feature pairs with |r| ≥ threshold detected. "
                "Highly correlated features may cause multicollinearity.", interp_s))
        else:
            story.append(Paragraph("No strongly correlated pairs found.", body_s))
        story.append(hr())

        # ── Distributions & KDE charts ────────────────────────────────────── #
        if figures and ("Distributions" in figures or "KDE & Pair Plot" in figures):
            story.append(Paragraph("Distributions & KDE", h2_s))
            add_texts("Distributions", story)
            add_figures("Distributions", story)
            add_figures("KDE & Pair Plot", story)
            story.append(hr())

        # ── Correlations charts ───────────────────────────────────────────── #
        if figures and "Correlations" in figures:
            story.append(Paragraph("Correlation Charts", h2_s))
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
        norm = stats.get("normality",{})
        if norm:
            story.append(Paragraph("Normality Tests (Shapiro-Wilk, α=0.05)", h2_s))
            n_data = [["Column","Statistic","p-value","Normal?","Interpretation"]] + [
                [c, str(r.get("statistic","—")), str(r.get("p_value","—")),
                 "Yes" if r.get("is_normal") else "No",
                 textwrap.shorten(r.get("interpretation",""), 120)]
                for c, r in norm.items()
            ]
            story.append(mk_tbl(n_data))
            n_normal = sum(1 for r in norm.values() if r.get("is_normal"))
            story.append(Paragraph(
                f"{n_normal} of {len(norm)} columns follow a normal distribution. "
                "Non-normal features may need non-parametric tests.", interp_s))
            add_texts("Statistics", story)
            add_figures("Statistics", story)
            story.append(hr())

        # ── Confidence intervals ──────────────────────────────────────────── #
        ci = stats.get("confidence_intervals",{})
        if ci:
            story.append(Paragraph("95% Confidence Intervals (Mean)", h2_s))
            ci_data = [["Column","n","Mean","Lower 95%","Upper 95%","Interpretation"]] + [
                [c, str(r.get("n")), str(r.get("mean")),
                 str(r.get("lower")), str(r.get("upper")),
                 textwrap.shorten(r.get("interpretation",""), 100)]
                for c, r in ci.items()
            ]
            story.append(mk_tbl(ci_data))
            story.append(hr())

        # ── Regression ────────────────────────────────────────────────────── #
        reg = stats.get("regression") or {}
        if reg:
            story.append(Paragraph("Linear Regression", h2_s))
            meta_data = [
                ["Metric","Value"],
                ["Target",       str(reg.get("target","—"))],
                ["R²",           str(reg.get("r_squared","—"))],
                ["Adj R²",       str(reg.get("adjusted_r_squared","—"))],
                ["RMSE",         str(reg.get("rmse","—"))],
                ["MAE",          str(reg.get("mae","—"))],
                ["n (samples)",  str(reg.get("n","—"))],
            ]
            story.append(mk_tbl(meta_data, col_widths=[80*mm,90*mm]))
            story.append(Paragraph(reg.get("interpretation",""), interp_s))
            story.append(Spacer(1,6))
            story.append(Paragraph("Coefficients & Significance", h3_s))
            coef_data = [["Feature","Coefficient","p-value","Significant?"]] + [
                [f, str(c), str(reg["p_values"].get(f,"—")),
                 "Yes" if reg["p_values"].get(f,1) < 0.05 else "No"]
                for f, c in reg.get("coefficients",{}).items()
            ]
            story.append(mk_tbl(coef_data))
            add_texts("Regression", story)
            add_figures("Regression", story)
            story.append(hr())

        # ── Footer ────────────────────────────────────────────────────────── #
        story.append(Paragraph(
            f"End of Report  ·  {self.project_name}  ·  {self.generated_at}", meta_s))

        doc.build(story)
        return buf.getvalue()