import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss
import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  FUTURISTIC DARK THEME — CSS injection
# ─────────────────────────────────────────────
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
    --neon-cyan:   #00f5d4;
    --neon-blue:   #0ea5e9;
    --neon-purple: #a855f7;
    --neon-amber:  #f59e0b;
    --neon-red:    #ef4444;
    --neon-green:  #22c55e;
    --bg-primary:  #020817;
    --bg-surface:  #0d1525;
    --bg-card:     #0f1e35;
    --border:      rgba(0,245,212,0.15);
    --border-glow: rgba(0,245,212,0.4);
    --text-primary: #e2e8f0;
    --text-muted:   #64748b;
    --font-mono:   'Share Tech Mono', monospace;
    --font-ui:     'Rajdhani', sans-serif;
}

html, body, [class*="st-"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-ui) !important; }
[data-testid="stSidebar"] label { color: var(--neon-cyan) !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] textarea {
    background: #071120 !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 8px rgba(0,245,212,0.3) !important;
}
.stSlider [data-baseweb="slider"] { accent-color: var(--neon-cyan); }

/* ── Main header ── */
h1 { font-family: var(--font-ui) !important; font-weight: 700 !important; letter-spacing: 0.04em; color: var(--neon-cyan) !important; }
h2, h3 { font-family: var(--font-ui) !important; color: var(--text-primary) !important; letter-spacing: 0.03em; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--neon-cyan) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { font-family: var(--font-mono) !important; font-size: 0.65rem !important; color: var(--neon-cyan) !important; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-family: var(--font-ui) !important; font-size: 1.8rem !important; font-weight: 700 !important; color: var(--text-primary) !important; }
[data-testid="stMetricDelta"] { font-family: var(--font-mono) !important; font-size: 0.7rem !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-blue)) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.stDataFrame * { font-family: var(--font-mono) !important; font-size: 0.78rem !important; }

/* ── Info / success boxes ── */
[data-testid="stInfo"] { background: rgba(0,245,212,0.06) !important; border-left: 3px solid var(--neon-cyan) !important; border-radius: 4px !important; }
[data-testid="stSuccess"] { background: rgba(34,197,94,0.06) !important; border-left: 3px solid var(--neon-green) !important; }
[data-testid="stWarning"] { background: rgba(245,158,11,0.06) !important; border-left: 3px solid var(--neon-amber) !important; }
[data-testid="stError"] { background: rgba(239,68,68,0.06) !important; border-left: 3px solid var(--neon-red) !important; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Section separator ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--neon-cyan);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Signal badges ── */
.badge-buy  { background: rgba(34,197,94,0.12); color: #22c55e; border: 1px solid #22c55e44; padding: 4px 14px; border-radius: 3px; font-family: var(--font-mono); font-size: 0.75rem; letter-spacing: 0.08em; }
.badge-wait { background: rgba(239,68,68,0.10); color: #ef4444; border: 1px solid #ef444444; padding: 4px 14px; border-radius: 3px; font-family: var(--font-mono); font-size: 0.75rem; letter-spacing: 0.08em; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 2px; }
</style>
"""

# ─────────────────────────────────────────────
#  PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#071120",
    font=dict(family="'Share Tech Mono', monospace", color="#64748b", size=11),
    xaxis=dict(
        gridcolor="rgba(0,245,212,0.05)",
        linecolor="rgba(0,245,212,0.15)",
        tickcolor="rgba(0,245,212,0.15)",
    ),
    yaxis=dict(
        gridcolor="rgba(0,245,212,0.05)",
        linecolor="rgba(0,245,212,0.15)",
        tickcolor="rgba(0,245,212,0.15)",
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0d1525",
        bordercolor="#00f5d4",
        font=dict(color="#e2e8f0", size=12, family="'Share Tech Mono', monospace"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,245,212,0.2)",
        borderwidth=1,
        font=dict(size=11),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=50, r=30, t=60, b=40),
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QUANT HODL v14",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⬡",
)
st.markdown(STYLE, unsafe_allow_html=True)

#
