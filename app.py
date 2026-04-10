import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from loader import DataLoader
from cleaner import DataCleaner
from explorer import DataExplorer
from visualizer import DataVisualizer, fig_to_html, fig_to_png
from analyzer import StatisticalAnalyzer
from report_exporter import ReportExporter

# page configuration 
st.set_page_config(
    page_title="DataLens · Smart Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# css 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

/* ── Root & body ─────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
  background: radial-gradient(ellipse at 0% 0%, #0d0f1f 0%, #080a14 60%, #060810 100%);
  min-height: 100vh;
}

/* animated mesh overlay */
.stApp::before {
  content: '';
  position: fixed; inset: 0;
  background:
    radial-gradient(ellipse 900px 600px at 10% 20%, rgba(108,99,255,0.07) 0%, transparent 70%),
    radial-gradient(ellipse 600px 400px at 85% 75%, rgba(72,202,228,0.05) 0%, transparent 70%),
    radial-gradient(ellipse 500px 300px at 60% 10%, rgba(247,127,0,0.03) 0%, transparent 70%);
  pointer-events: none; z-index: 0;
}

/* ── Hide Streamlit chrome ───────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0e1020 0%, #0c0e1c 100%) !important;
  border-right: 1px solid rgba(108,99,255,0.18) !important;
  box-shadow: 4px 0 32px rgba(0,0,0,0.5);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
  font-family: 'Sora', sans-serif;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #6C63FF;
  margin: 20px 0 8px;
}

/* ── Logo badge ──────────────────────────────────────── */
.logo-wrap {
  text-align: center;
  padding: 28px 0 16px;
}
.logo-icon {
  width: 52px; height: 52px;
  background: linear-gradient(135deg, #6C63FF, #48CAE4);
  border-radius: 14px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 24px; margin-bottom: 10px;
  box-shadow: 0 8px 24px rgba(108,99,255,0.4);
}
.logo-title {
  font-family: 'Sora', sans-serif;
  font-size: 20px; font-weight: 800;
  background: linear-gradient(90deg, #6C63FF, #48CAE4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.logo-sub { font-size: 11px; color: #475569; margin-top: 2px; letter-spacing: 0.06em; }

/* ── Metric cards ────────────────────────────────────── */
.metric-card {
  background: linear-gradient(135deg,
    rgba(108,99,255,0.10) 0%,
    rgba(72,202,228,0.05) 100%);
  border: 1px solid rgba(108,99,255,0.22);
  border-radius: 16px;
  padding: 22px 16px;
  text-align: center;
  transition: transform .25s cubic-bezier(.34,1.56,.64,1), box-shadow .25s;
  position: relative; overflow: hidden;
}
.metric-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, #6C63FF, #48CAE4);
  opacity: 0; transition: opacity .25s;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(108,99,255,0.25); }
.metric-card:hover::before { opacity: 1; }
.metric-card .val {
  font-family: 'Sora', sans-serif;
  font-size: 30px; font-weight: 800;
  background: linear-gradient(135deg, #818cf8, #6C63FF);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1.1;
}
.metric-card .lbl {
  font-size: 10px; color: #64748b; margin-top: 6px;
  text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
}
.metric-card .icon { font-size: 20px; margin-bottom: 8px; }

/* ── Section headers ─────────────────────────────────── */
.section-header {
  font-family: 'Sora', sans-serif;
  font-size: 18px; font-weight: 700;
  color: #e2e8f0;
  display: flex; align-items: center; gap: 10px;
  margin: 36px 0 18px;
}
.section-header::before {
  content: '';
  display: inline-block;
  width: 4px; height: 22px;
  background: linear-gradient(180deg, #6C63FF, #48CAE4);
  border-radius: 2px; flex-shrink: 0;
}

/* ── Insight / interpretation box ───────────────────── */
.interp-box {
  background: linear-gradient(135deg,
    rgba(108,99,255,0.08) 0%,
    rgba(72,202,228,0.04) 100%);
  border: 1px solid rgba(108,99,255,0.18);
  border-left: 3px solid #6C63FF;
  border-radius: 0 12px 12px 0;
  padding: 14px 20px;
  color: #cbd5e1;
  font-size: 13.5px;
  line-height: 1.75;
  margin: 6px 0 20px;
  position: relative;
}
.interp-box::before {
  content: '💡';
  position: absolute; top: 12px; right: 16px;
  font-size: 16px; opacity: 0.6;
}

/* ── Upload hero ─────────────────────────────────────── */
.upload-hero {
  text-align: center;
  padding: 80px 48px;
  border: 2px dashed rgba(108,99,255,0.30);
  border-radius: 28px;
  background: radial-gradient(ellipse at center, rgba(108,99,255,0.05) 0%, transparent 70%);
  margin: 60px auto;
  max-width: 640px;
  position: relative;
}
.upload-hero-icon {
  font-size: 64px; margin-bottom: 20px;
  animation: float 3s ease-in-out infinite;
}
@keyframes float {
  0%,100% { transform: translateY(0); }
  50%      { transform: translateY(-10px); }
}
.upload-hero h2 {
  font-family: 'Sora', sans-serif;
  font-size: 28px; font-weight: 700;
  background: linear-gradient(90deg, #e2e8f0, #a5b4fc);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 10px;
}
.upload-hero p { color: #64748b; font-size: 14px; line-height: 1.7; }
.feature-pill {
  display: inline-block;
  padding: 5px 14px; border-radius: 20px;
  font-size: 12px; font-weight: 500;
  margin: 4px;
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.02);
  border-radius: 12px; padding: 4px;
  border: 1px solid rgba(255,255,255,0.06);
  gap: 2px;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important; font-weight: 500 !important;
  color: #64748b !important;
  border-radius: 8px !important;
  padding: 8px 16px !important;
  transition: all .2s !important;
  border: none !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #a5b4fc !important; background: rgba(108,99,255,0.08) !important; }
.stTabs [aria-selected="true"] {
  color: #fff !important;
  background: linear-gradient(135deg, #6C63FF, #5a52e0) !important;
  box-shadow: 0 4px 16px rgba(108,99,255,0.35) !important;
}

/* ── Download buttons ────────────────────────────────── */
.stDownloadButton > button {
  background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 100%) !important;
  color: white !important; border: none !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important; font-size: 13px !important;
  padding: 9px 20px !important;
  transition: all .2s !important;
  box-shadow: 0 4px 16px rgba(108,99,255,0.25) !important;
  letter-spacing: 0.02em;
}
.stDownloadButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px rgba(108,99,255,0.4) !important;
  opacity: 0.92 !important;
}

/* ── Primary buttons ─────────────────────────────────── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #6C63FF, #48CAE4) !important;
  color: white !important; border: none !important;
  border-radius: 10px !important; font-weight: 600 !important;
  box-shadow: 0 4px 20px rgba(108,99,255,0.3) !important;
  transition: all .2s !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 28px rgba(108,99,255,0.45) !important;
}

/* ── Form elements ───────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(108,99,255,0.2) !important;
  border-radius: 10px !important;
}
.stSlider > div > div > div { background: rgba(108,99,255,0.2) !important; }
.stSlider > div > div > div > div { background: linear-gradient(90deg,#6C63FF,#48CAE4) !important; }
.stCheckbox > label > div:first-child {
  border-color: rgba(108,99,255,0.4) !important;
  border-radius: 5px !important;
}

/* ── DataFrames ──────────────────────────────────────── */
.stDataFrame {
  border-radius: 14px !important;
  border: 1px solid rgba(108,99,255,0.15) !important;
  overflow: hidden;
}

/* ── Alerts ──────────────────────────────────────────── */
.stAlert { border-radius: 12px !important; }
.stSuccess { border-left-color: #2DC653 !important; }
.stInfo    { border-left-color: #6C63FF !important; }
.stWarning { border-left-color: #F77F00 !important; }

/* ── Spinner ─────────────────────────────────────────── */
.stSpinner > div { border-top-color: #6C63FF !important; }

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: rgba(108,99,255,0.12) !important; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(108,99,255,0.5); }
</style>
""", unsafe_allow_html=True)

# session state 
def _init():
    defaults = {
        "df_raw": None, "df_clean": None, "df_no_outliers": None,
        "eda": None, "stats": None, "file_name": "", "all_figures": {},
        "analysis_text": {},   # stores written analysis per section
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# sidebar
with st.sidebar:
    st.markdown("""
    <div class="logo-wrap">
      <div class="logo-icon">🔬</div>
      <div class="logo-title">DataLens</div>
      <div class="logo-sub">BY CHIRAG MAAN</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader(
        "📂 Upload your dataset",
        type=["csv", "xlsx", "xls", "json"],
        help="Supported: CSV, Excel, JSON — up to 200 MB",
    )

    st.markdown("### ⚙️ Cleaning")
    null_strategy  = st.selectbox("Null handling", ["median","mean","mode","zero","drop"], label_visibility="visible")
    do_outliers    = st.checkbox("Remove outliers (IQR)", value=False)
    outlier_thresh = st.slider("IQR threshold", 1.0, 3.0, 1.5, 0.1) if do_outliers else 1.5

    st.markdown("### 📊 Analysis")
    corr_threshold = st.slider("Correlation threshold", 0.5, 0.95, 0.7, 0.05)

    st.divider()
    st.markdown("""
    <div style='font-size:11px;color:#334155;text-align:center;line-height:1.8'>
      CSV &middot; Excel &middot; JSON<br>
      Polars &middot; Plotly &middot; Streamlit<br>
      <span style='color:#6C63FF'>v2.0</span>
    </div>""", unsafe_allow_html=True)


# data pipeline 
if uploaded and uploaded.name != st.session_state.file_name:
    with st.spinner("🔄 Loading & analysing your data…"):
        try:
            loader   = DataLoader()
            cleaner  = DataCleaner()
            explorer = DataExplorer()

            df_raw = loader.load_file(uploaded)
            df = cleaner.drop_duplicates(df_raw)
            df = cleaner.auto_cast_types(df)
            df = cleaner.handle_nulls(df, strategy=null_strategy)

            df_no_out = df.clone()
            if do_outliers:
                df_no_out = cleaner.remove_outliers(df, threshold=outlier_thresh)

            eda   = explorer.full_eda(df_no_out)
            eda["high_correlations"] = explorer.get_high_correlations(df_no_out, threshold=corr_threshold)
            stats = StatisticalAnalyzer().run_full_analysis(df_no_out)

            st.session_state.update({
                "df_raw": df_raw, "df_clean": df, "df_no_outliers": df_no_out,
                "eda": eda, "stats": stats,
                "file_name": uploaded.name, "all_figures": {}, "analysis_text": {},
            })
            st.success(f"✅ **{uploaded.name}** — {df_no_out.shape[0]:,} rows × {df_no_out.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ {e}")


# welcome screen 
df      = st.session_state.df_no_outliers
df_clean= st.session_state.df_clean
df_raw  = st.session_state.df_raw
eda     = st.session_state.eda
stats   = st.session_state.stats

if df is None:
    st.markdown("""
    <div class="upload-hero">
      <div class="upload-hero-icon">📊</div>
      <h2>Drop your data file to begin</h2>
      <p>Upload a CSV, Excel, or JSON file using the sidebar.<br>
         Get a complete automated analysis in seconds — no code needed.</p>
      <div style='margin-top:28px'>
        <span class="feature-pill" style='background:rgba(108,99,255,0.15);color:#a5b4fc'>📈 EDA</span>
        <span class="feature-pill" style='background:rgba(72,202,228,0.12);color:#67e8f9'>〰️ KDE</span>
        <span class="feature-pill" style='background:rgba(247,127,0,0.12);color:#fdba74'>🔗 Correlations</span>
        <span class="feature-pill" style='background:rgba(44,214,83,0.10);color:#86efac'>⚠️ Outliers</span>
        <span class="feature-pill" style='background:rgba(230,57,70,0.12);color:#fca5a5'>📐 Statistics</span>
        <span class="feature-pill" style='background:rgba(255,107,157,0.12);color:#f9a8d4'>📈 Regression</span>
        <span class="feature-pill" style='background:rgba(255,209,102,0.12);color:#fde68a'>⬇️ PDF Report</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# helper 
viz          = DataVisualizer()
exp          = DataExplorer()
numeric_cols = eda.get("numeric_cols", [])
cat_cols     = eda.get("categorical_cols", [])
all_figures  = st.session_state.all_figures
analysis_text= st.session_state.analysis_text


def _save_fig(section: str, fig, interp: str):
    all_figures.setdefault(section, []).append((fig, interp))


def _save_text(section: str, key: str, text: str):
    analysis_text.setdefault(section, {})[key] = text


def _chart_row(fig, interp: str, download_key: str, section: str = None):
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{download_key}")
    st.markdown(f'<div class="interp-box">{interp}</div>', unsafe_allow_html=True)
    if section:
        _save_fig(section, fig, interp)
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ HTML", fig_to_html(fig),
                       f"{download_key}.html", "text/html",
                       key=f"dl_html_{download_key}")
    try:
        from report_exporter import ReportExporter as _RE
        c2.download_button("⬇️ PNG", _RE.fig_to_image_bytes(fig, "png"),
                           f"{download_key}.png", "image/png",
                           key=f"dl_png_{download_key}")
        c3.download_button("⬇️ JPEG", _RE.fig_to_image_bytes(fig, "jpg"),
                           f"{download_key}.jpg", "image/jpeg",
                           key=f"dl_jpg_{download_key}")
    except Exception:
        c2.caption("Install kaleido for PNG/JPEG")


def _sh(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# file banner 
st.markdown(f"""
<div style='
  background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(72,202,228,0.06));
  border: 1px solid rgba(108,99,255,0.2);
  border-radius: 16px; padding: 18px 24px;
  display: flex; align-items: center; gap: 14px; margin-bottom: 24px;
'>
  <div style='font-size:28px'>📁</div>
  <div>
    <div style='font-family:Sora,sans-serif;font-size:16px;font-weight:700;color:#e2e8f0'>
      {st.session_state.file_name}
    </div>
    <div style='font-size:12px;color:#64748b;margin-top:3px'>
      {df.shape[0]:,} rows &nbsp;×&nbsp; {df.shape[1]} columns &nbsp;·&nbsp;
      {len(numeric_cols)} numeric &nbsp;·&nbsp; {len(cat_cols)} categorical
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# metric cards 
metrics = viz.overview_metrics(df, eda)
items = [
    ("🗂️", "Rows",        f"{metrics['rows']:,}"),
    ("📊", "Columns",     metrics["columns"]),
    ("🔢", "Numeric",     metrics["numeric"]),
    ("🏷️", "Categorical", metrics["categorical"]),
    ("❓", "Missing",     f"{metrics['missing_cells']:,}"),
    ("♊", "Duplicates",  metrics["duplicates"]),
]
cols = st.columns(6)
for col, (icon, lbl, val) in zip(cols, items):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="icon">{icon}</div>
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# tabs 
(tab_overview, tab_dist, tab_kde, tab_corr,
 tab_cats, tab_outliers, tab_stats, tab_reg,
 tab_data, tab_download) = st.tabs([
    "📋 Overview", "📊 Distributions", "〰️ KDE & Pair Plot",
    "🔗 Correlations", "🏷️ Categories", "⚠️ Outliers",
    "📐 Statistics", "📈 Regression", "🗃️ Data", "⬇️ Download",
])


# overview 
with tab_overview:
    _sh("Column Profiles")
    profiles = eda.get("column_profiles", [])
    if profiles:
        pf = pd.DataFrame(profiles)
        for c in pf.columns:
            pf[c] = pf[c].astype(str)
        st.dataframe(pf, use_container_width=True, height=300)
        txt = (f"Dataset has {df.shape[1]} columns: {len(numeric_cols)} numeric and "
               f"{len(cat_cols)} categorical. "
               f"Total missing cells: {metrics['missing_cells']:,}. "
               f"Duplicate rows: {metrics['duplicates']}.")
        st.markdown(f'<div class="interp-box">{txt}</div>', unsafe_allow_html=True)
        _save_text("Overview", "column_profiles", txt)

    miss = eda.get("missing_summary")
    if miss is not None and miss.shape[0] > 0:
        _sh("Missing Values Heatmap")
        fig, interp = viz.missing_heatmap(df)
        _chart_row(fig, interp, "missing_heatmap", "Overview")

    skew_data = eda.get("skewness", [])
    if skew_data:
        _sh("Feature Skewness")
        fig, interp = viz.skewness_chart(skew_data)
        _chart_row(fig, interp, "skewness_chart", "Overview")

    high_corr = eda.get("high_correlations", [])
    if high_corr:
        _sh("Highly Correlated Pairs")
        st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
        txt2 = (f"Found {len(high_corr)} feature pairs with |r| ≥ {corr_threshold}. "
                "Strongly correlated features may cause multicollinearity in models.")
        st.markdown(f'<div class="interp-box">{txt2}</div>', unsafe_allow_html=True)
        _save_text("Overview", "high_corr", txt2)


# distributions 
with tab_dist:
    if not numeric_cols:
        st.info("No numeric columns found.")
    else:
        _sh("Histogram + KDE")
        sel_col = st.selectbox("Select column", numeric_cols, key="dist_col")
        fig, interp = viz.distribution(df, sel_col)
        _chart_row(fig, interp, f"dist_{sel_col}", "Distributions")

        _sh("Box & Violin Plots")
        box_cols = st.multiselect("Columns (up to 8)", numeric_cols,
                                  default=numeric_cols[:min(4, len(numeric_cols))],
                                  key="box_cols")
        if box_cols:
            fig, interp = viz.boxplot(df, box_cols)
            _chart_row(fig, interp, "violin_box", "Distributions")

        _sh("Summary Statistics")
        ss = eda.get("summary_stats")
        if ss is not None and ss.shape[0] > 0:
            st.dataframe(ss.to_pandas(), use_container_width=True)
            n_skewed = sum(1 for s in eda.get("skewness",[]) if s["highly_skewed"])
            txt = (f"Summary statistics for {ss.shape[0]} numeric features. "
                   f"{n_skewed} features are highly skewed (|skew| > 1) and may "
                   "benefit from log or sqrt transformation before modelling.")
            st.markdown(f'<div class="interp-box">{txt}</div>', unsafe_allow_html=True)
            _save_text("Distributions", "summary_stats", txt)

        int_cols = [c for c in df.columns
                    if df[c].dtype in (pl.Int64,pl.Int32,pl.Int16,pl.Int8)
                    and df[c].n_unique() <= 30]
        if int_cols:
            _sh("Integer Value Distributions")
            int_sel = st.selectbox("Column", int_cols, key="int_col")
            fig, interp = viz.int_value_distribution(df, int_sel)
            _chart_row(fig, interp, f"int_{int_sel}", "Distributions")


# kde and pair plots 
with tab_kde:
    if not numeric_cols:
        st.info("No numeric columns.")
    else:
        _sh("KDE Density Comparison")
        kde_cols = st.multiselect("Columns (up to 8)", numeric_cols,
                                  default=numeric_cols[:min(5,len(numeric_cols))],
                                  key="kde_cols")
        if kde_cols:
            try:
                fig, interp = viz.kde_plot(df, kde_cols)
                _chart_row(fig, interp, "kde_multi", "KDE & Pair Plot")
            except Exception as e:
                st.warning(f"KDE error: {e}")

        _sh("Pair Plot — Multivariate Scatter Matrix")
        pair_cols = st.multiselect("Features (2–6)", numeric_cols,
                                   default=numeric_cols[:min(4,len(numeric_cols))],
                                   key="pair_cols")
        pair_hue = st.selectbox("Color by", ["None"] + cat_cols, key="pair_hue")
        hue = None if pair_hue == "None" else pair_hue
        if pair_cols and len(pair_cols) >= 2:
            with st.spinner("Rendering pair plot…"):
                fig, interp = viz.pair_plot(df, pair_cols, color_col=hue)
            _chart_row(fig, interp, "pair_plot", "KDE & Pair Plot")
            txt = (f"Pair plot of {len(pair_cols)} features. "
                   "Each scatter shows the bivariate relationship; diagonal shows each "
                   "feature's distribution. Clustered patterns indicate correlations.")
            _save_text("KDE & Pair Plot", "pair_plot", txt)
        elif pair_cols:
            st.info("Select at least 2 columns.")


# correlation 
with tab_corr:
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns.")
    else:
        _sh("Pearson Correlation Heatmap")
        corr_matrix = exp.get_correlation_matrix(df)
        fig, interp = viz.correlation_heatmap(df, corr_matrix)
        _chart_row(fig, interp, "corr_heatmap", "Correlations")

        _sh("Scatter Plot + OLS Trendline")
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis", numeric_cols, key="sc_x")
        y_col = c2.selectbox("Y axis", numeric_cols,
                             index=min(1,len(numeric_cols)-1), key="sc_y")
        hue_c = c3.selectbox("Color by", ["None"] + cat_cols, key="sc_hue")
        hue   = None if hue_c == "None" else hue_c
        if x_col == y_col:
            st.info("Select different X and Y columns.")
        else:
            try:
                fig, interp = viz.scatter(df, x_col, y_col, color_col=hue)
                _chart_row(fig, interp, f"scatter_{x_col}_{y_col}", "Correlations")
            except Exception as e:
                st.warning(f"Scatter error: {e}")

        _sh("Best-Fit Regression Line (with 95% CI)")
        c1, c2 = st.columns(2)
        bf_x = c1.selectbox("X (predictor)", numeric_cols, key="bf_x")
        bf_y = c2.selectbox("Y (outcome)", numeric_cols,
                            index=min(1,len(numeric_cols)-1), key="bf_y")
        if bf_x == bf_y:
            st.info("Select different columns.")
        else:
            try:
                fig, interp = viz.regression_fit_line(df, bf_x, bf_y)
                _chart_row(fig, interp, f"bestfit_{bf_x}_{bf_y}", "Correlations")
            except Exception as e:
                st.warning(f"Best-fit error: {e}")


# categorical analysis 
with tab_cats:
    if not cat_cols:
        st.info("No categorical columns found.")
    else:
        cat_sel = st.selectbox("Categorical column", cat_cols, key="cat_sel")

        c1, c2 = st.columns(2)
        with c1:
            _sh("Count Plot")
            fig, interp = viz.count_plot(df, cat_sel)
            _chart_row(fig, interp, f"count_{cat_sel}", "Categories")
        with c2:
            _sh("Donut Chart")
            fig, interp = viz.pie_chart(df, cat_sel)
            _chart_row(fig, interp, f"pie_{cat_sel}", "Categories")

        _sh("Treemap")
        fig, interp = viz.treemap(df, cat_sel)
        _chart_row(fig, interp, f"treemap_{cat_sel}", "Categories")

        if numeric_cols:
            _sh("Grouped Bar Chart")
            c1, c2, c3 = st.columns(3)
            bar_x   = c1.selectbox("Group by", cat_cols, key="bar_x")
            bar_y   = c2.selectbox("Value", numeric_cols, key="bar_y")
            bar_agg = c3.selectbox("Aggregation", ["mean","sum","count","median"], key="bar_agg")
            try:
                fig, interp = viz.bar_chart(df, bar_x, bar_y, agg=bar_agg)
                _chart_row(fig, interp, f"bar_{bar_x}_{bar_y}", "Categories")
            except Exception as e:
                st.warning(f"Bar chart error: {e}")

            _sh("Category × Numeric Mean Heatmap")
            c1, c2 = st.columns(2)
            cn_cat = c1.selectbox("Categorical", cat_cols, key="cn_cat")
            cn_num = c2.selectbox("Numeric", numeric_cols, key="cn_num")
            try:
                fig, interp = viz.cat_num_heatmap(df, cn_cat, cn_num)
                _chart_row(fig, interp, f"catnum_{cn_cat}_{cn_num}", "Categories")
            except Exception as e:
                st.warning(f"Heatmap error: {e}")

        date_cols = [c for c in df.columns
                     if any(k in c.lower() for k in ["date","time","year","month"])]
        if date_cols and numeric_cols:
            _sh("Time Series")
            c1, c2 = st.columns(2)
            ts_d = c1.selectbox("Date column", date_cols, key="ts_d")
            ts_v = c2.selectbox("Value column", numeric_cols, key="ts_v")
            try:
                fig, interp = viz.time_series(df, ts_d, ts_v)
                _chart_row(fig, interp, f"ts_{ts_d}_{ts_v}", "Categories")
            except Exception as e:
                st.warning(f"Time series error: {e}")


# outliers 
with tab_outliers:
    if not numeric_cols:
        st.info("No numeric columns.")
    else:
        _sh("Outlier Detection & Removal")
        out_col    = st.selectbox("Column", numeric_cols, key="out_col")
        out_thresh = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, key="out_thresh")

        data_arr = df[out_col].drop_nulls().to_numpy().astype(float)
        q1, q3   = np.percentile(data_arr, [25,75])
        iqr      = q3 - q1
        lo, hi   = q1 - out_thresh*iqr, q3 + out_thresh*iqr
        n_out    = int(np.sum((data_arr < lo) | (data_arr > hi)))
        pct_out  = round(n_out/len(data_arr)*100, 2)

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total rows",      f"{len(data_arr):,}")
        m2.metric("Outliers found",  f"{n_out:,}")
        m3.metric("Outlier %",       f"{pct_out}%")
        m4.metric("IQR fence",       f"[{lo:.2f}, {hi:.2f}]")

        txt_out = (f"**{out_col}**: {n_out} outliers detected ({pct_out}%) using IQR method. "
                   f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}. "
                   f"Valid range: [{lo:.2f}, {hi:.2f}].")
        st.markdown(f'<div class="interp-box">{txt_out}</div>', unsafe_allow_html=True)
        _save_text("Outliers", f"detection_{out_col}", txt_out)

        cleaner_tmp   = DataCleaner()
        df_out_removed= cleaner_tmp.remove_outliers(df, method="iqr", threshold=out_thresh)

        fig, interp = viz.outlier_boxplot(df, df_out_removed, out_col)
        _chart_row(fig, interp, f"outlier_{out_col}", "Outliers")

        _sh("Distribution: Before vs After Removal")
        import plotly.graph_objects as _go
        from visualizer import _theme as _vtheme
        fig2 = _go.Figure()
        fig2.add_trace(_go.Histogram(
            x=df[out_col].drop_nulls().to_numpy(), nbinsx=40,
            name="Before", opacity=0.65, marker_color="#E63946"))
        fig2.add_trace(_go.Histogram(
            x=df_out_removed[out_col].drop_nulls().to_numpy(), nbinsx=40,
            name="After", opacity=0.65, marker_color="#2DC653"))
        _vtheme(fig2, f"Outlier Removal Effect — {out_col}")
        fig2.update_layout(barmode="overlay")
        interp2 = (f"Red = original ({len(data_arr):,} values). "
                   f"Green = cleaned ({len(data_arr)-n_out:,} values). "
                   "Overlap shows retained values.")
        _chart_row(fig2, interp2, f"outlier_dist_{out_col}", "Outliers")

        _sh("Apply to Full Dataset")
        if st.button("🗑️ Remove Outliers from All Numeric Columns", type="primary"):
            with st.spinner("Removing outliers…"):
                c2_ = DataCleaner()
                df_new = c2_.remove_outliers(df, threshold=out_thresh)
                e2 = DataExplorer()
                new_eda = e2.full_eda(df_new)
                new_eda["high_correlations"] = e2.get_high_correlations(df_new, corr_threshold)
                new_stats = StatisticalAnalyzer().run_full_analysis(df_new)
                st.session_state.df_no_outliers = df_new
                st.session_state.eda            = new_eda
                st.session_state.stats          = new_stats
            st.success(f"Done! Dataset now has {df_new.shape[0]:,} rows.")


# statistics 
with tab_stats:
    norm_results = stats.get("normality", {})
    if norm_results:
        _sh("Normality Tests (Shapiro-Wilk, α = 0.05)")
        norm_rows = [{"Column": c,
                      "Statistic": r.get("statistic","—"),
                      "p-value":   r.get("p_value","—"),
                      "Normal?":   "✅ Yes" if r.get("is_normal") else "⚠️ No"}
                     for c, r in norm_results.items()]
        st.dataframe(pd.DataFrame(norm_rows), use_container_width=True)
        n_normal = sum(1 for r in norm_results.values() if r.get("is_normal"))
        txt_norm = (f"{n_normal} of {len(norm_results)} numeric columns follow a normal distribution. "
                    "Non-normal features may require non-parametric tests or transformations.")
        st.markdown(f'<div class="interp-box">{txt_norm}</div>', unsafe_allow_html=True)
        _save_text("Statistics", "normality_summary", txt_norm)

        sel_norm = st.selectbox("Details for", list(norm_results.keys()), key="norm_sel")
        if sel_norm:
            st.markdown(f'<div class="interp-box">'
                        f'{norm_results[sel_norm].get("interpretation","")}</div>',
                        unsafe_allow_html=True)

    ci_results = stats.get("confidence_intervals", {})
    if ci_results:
        _sh("95% Confidence Intervals")
        ci_rows = [{"Column": c, "n": r.get("n"), "Mean": r.get("mean"),
                    "Lower 95%": r.get("lower"), "Upper 95%": r.get("upper")}
                   for c, r in ci_results.items()]
        st.dataframe(pd.DataFrame(ci_rows), use_container_width=True)

        if len(ci_results) >= 2:
            import plotly.graph_objects as _go
            from visualizer import _theme as _vt
            ci_fig = _go.Figure()
            for col_ci, r in ci_results.items():
                ci_fig.add_trace(_go.Scatter(
                    x=[r["lower"], r["mean"], r["upper"]],
                    y=[col_ci]*3, mode="markers+lines",
                    marker=dict(color=["#E63946","#6C63FF","#E63946"], size=[8,14,8]),
                    line=dict(color="#6C63FF", width=2),
                    name=col_ci, showlegend=False,
                ))
            _vt(ci_fig, "95% Confidence Intervals — Forest Plot")
            ci_fig.update_layout(xaxis_title="Value",
                                 height=max(300, len(ci_results)*45))
            ci_interp = ("Each bar = 95% CI for the column mean. "
                         "Purple dot = sample mean; red dots = bounds. "
                         "Wider bars = more uncertainty (larger std or smaller n).")
            _chart_row(ci_fig, ci_interp, "ci_forest", "Statistics")
            _save_text("Statistics", "ci_summary", ci_interp)

# regression  
with tab_reg:
    if not numeric_cols or len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns.")
    else:
        _sh("Multi-Feature Linear Regression")
        c1, c2 = st.columns(2)
        reg_target   = c1.selectbox("Target (Y)", numeric_cols, key="reg_target")
        feat_options = [c for c in numeric_cols if c != reg_target]
        reg_features = c2.multiselect("Features (X)",
                                      feat_options,
                                      default=feat_options[:min(3,len(feat_options))],
                                      key="reg_features")

        if reg_features and st.button("▶ Run Regression", type="primary", key="run_reg"):
            with st.spinner("Fitting model…"):
                reg = StatisticalAnalyzer().run_regression(df, reg_target, reg_features)

            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("R²",     reg["r_squared"])
            mc2.metric("Adj R²", reg["adjusted_r_squared"])
            mc3.metric("RMSE",   reg["rmse"])
            mc4.metric("MAE",    reg["mae"])
            st.markdown(f'<div class="interp-box">{reg["interpretation"]}</div>',
                        unsafe_allow_html=True)
            _save_text("Regression", "multi_reg", reg["interpretation"])

            _sh("Coefficients & Statistical Significance")
            coef_df = pd.DataFrame([
                {"Feature": f, "Coefficient": c,
                 "p-value": reg["p_values"].get(f,"—"),
                 "Significant": "✅ Yes" if reg["p_values"].get(f,1) < 0.05 else "⚠️ No"}
                for f, c in reg["coefficients"].items()
            ])
            st.dataframe(coef_df, use_container_width=True)

            _sh("Regression Diagnostics (4-panel)")
            y_true    = np.asarray(reg["y_true"])
            y_pred    = np.asarray(reg["y_pred"])
            coefs_arr = np.asarray(reg["coefs_array"])
            fig, interp = viz.regression_plots(y_true, y_pred, reg_features, coefs_arr)
            _chart_row(fig, interp, "reg_diagnostics", "Regression")

            _sh("Per-Feature Best-Fit Lines")
            for feat in reg_features[:4]:
                fig, interp = viz.regression_fit_line(df, feat, reg_target)
                _chart_row(fig, interp, f"regfit_{feat}_{reg_target}", "Regression")

        _sh("Quick Single-Feature Regression")
        c1, c2 = st.columns(2)
        qx = c1.selectbox("Predictor (X)", numeric_cols, key="qreg_x")
        qy = c2.selectbox("Outcome (Y)", numeric_cols,
                          index=min(1,len(numeric_cols)-1), key="qreg_y")
        if qx == qy:
            st.info("Select different columns.")
        else:
            try:
                fig, interp = viz.regression_fit_line(df, qx, qy)
                _chart_row(fig, interp, f"quick_reg_{qx}_{qy}", "Regression")
            except Exception as e:
                st.warning(f"Best-fit error: {e}")
            try:
                fig2, interp2 = viz.scatter(df, qx, qy)
                _chart_row(fig2, interp2, f"quick_scatter_{qx}_{qy}", "Regression")
            except Exception as e:
                st.warning(f"Scatter error: {e}")


# data 
with tab_data:
    _sh("Cleaned Dataset Preview")
    n_rows = st.slider("Rows to display", 10, min(1000,df.shape[0]), 50)
    st.dataframe(df.head(n_rows).to_pandas(), use_container_width=True, height=460)
    st.caption(f"Showing {n_rows:,} of {df.shape[0]:,} rows · "
               f"{df.shape[1]} columns · "
               f"Est. memory: {df.estimated_size('mb'):.2f} MB")


# download 
with tab_download:
    _sh("Export Full Analysis Report")
    st.markdown("""
    <div style='color:#64748b;font-size:13.5px;line-height:1.8;margin-bottom:28px'>
      The HTML and PDF reports include <strong style='color:#a5b4fc'>every chart generated
      this session</strong>, all written analysis, statistical tables, regression results,
      and interpretations — ready to share with stakeholders.
    </div>""", unsafe_allow_html=True)

    exporter = ReportExporter(project_name=f"DataLens — {st.session_state.file_name}")

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown("##### 📄 CSV")
        st.caption("Cleaned dataset as flat file")
        st.download_button("⬇️ Download CSV",
                           exporter.to_csv_bytes(df),
                           "cleaned_data.csv", "text/csv",
                           use_container_width=True)
    with c2:
        st.markdown("##### 📊 JSON")
        st.caption("Full analysis as structured JSON")
        report_dict = {
            **{k:v for k,v in eda.items() if k!="summary_stats"},
            "statistical_analysis": {k:v for k,v in stats.items() if k!="regression"},
        }
        st.download_button("⬇️ Download JSON",
                           exporter.to_json_bytes(report_dict),
                           "report.json", "application/json",
                           use_container_width=True)
    with c3:
        st.markdown("##### 🌐 HTML")
        st.caption("Interactive charts embedded — open in browser")
        html_bytes = exporter.to_html_bytes(
            df, eda, stats,
            figures=st.session_state.all_figures,
            analysis_text=st.session_state.analysis_text,
        )
        st.download_button("⬇️ Download HTML", html_bytes,
                           "analysis_report.html", "text/html",
                           use_container_width=True)
    with c4:
        st.markdown("##### 📑 PDF")
        st.caption("All charts + full analysis — print ready")
        with st.spinner("Building PDF…"):
            try:
                pdf_bytes = exporter.to_pdf_bytes(
                    df, eda, stats,
                    figures=st.session_state.all_figures,
                    analysis_text=st.session_state.analysis_text,
                )
                st.download_button("⬇️ Download PDF", pdf_bytes,
                                   "report.pdf", "application/pdf",
                                   use_container_width=True)
            except Exception as e:
                st.warning(f"PDF error: {e}\n\n`pip install reportlab kaleido`")

    st.divider()
    st.markdown("##### 📋 Inline HTML Preview")
    with st.expander("Preview full report"):
        st.components.v1.html(html_bytes.decode("utf-8"), height=800, scrolling=True)

    if st.session_state.all_figures:
        st.divider()
        _sh("Download Individual Charts")
        st.caption("Every chart generated this session — HTML, PNG, or JPEG")
        from report_exporter import ReportExporter as _RE
        idx = 0
        for section, fig_list in st.session_state.all_figures.items():
            with st.expander(f"📂 {section} ({len(fig_list)} charts)", expanded=False):
                for fig, interp in fig_list:
                    title = fig.layout.title.text or f"Chart {idx+1}"
                    c1,c2,c3,c4 = st.columns([4,1,1,1])
                    c1.markdown(f"<span style='color:#94a3b8;font-size:13px'>"
                                f"{'📊'} {title}</span>", unsafe_allow_html=True)
                    c2.download_button("HTML", fig_to_html(fig),
                                       f"chart_{idx}.html", "text/html",
                                       key=f"dl_all_html_{idx}")
                    try:
                        c3.download_button("PNG", _RE.fig_to_image_bytes(fig,"png"),
                                           f"chart_{idx}.png","image/png",
                                           key=f"dl_all_png_{idx}")
                        c4.download_button("JPEG", _RE.fig_to_image_bytes(fig,"jpg"),
                                           f"chart_{idx}.jpg","image/jpeg",
                                           key=f"dl_all_jpg_{idx}")
                    except Exception:
                        c3.caption("kaleido needed")
                    idx += 1