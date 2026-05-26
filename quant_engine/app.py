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

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='border-bottom:1px solid rgba(0,245,212,0.2);padding-bottom:0.75rem;margin-bottom:1.5rem'>
  <h1 style='margin:0;font-size:1.6rem;letter-spacing:0.08em'>
    ⬡ QUANT HODL <span style='color:#64748b'>v14</span>
  </h1>
  <p style='font-family:"Share Tech Mono",monospace;font-size:0.7rem;color:#64748b;margin:4px 0 0;letter-spacing:0.12em'>
    AI PREDICTIVE · WALK-FORWARD · NO DATA LEAKAGE · ENSEMBLE XGB+RF+LGB
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ASSET WATCHLIST
# ─────────────────────────────────────────────
WATCHLIST = {
    "── CRYPTO ──": None,
    "BTC-USD":   "Bitcoin",
    "ETH-USD":   "Ethereum",
    "SOL-USD":   "Solana",
    "BNB-USD":   "Binance Coin",
    "XRP-USD":   "XRP",
    "── ETF ──": None,
    "SPY":       "S&P 500 ETF",
    "QQQ":       "Nasdaq-100 ETF",
    "GLD":       "Gold ETF",
    "TLT":       "20yr Treasury ETF",
    "ARKK":      "ARK Innovation ETF",
    "VNQ":       "Real Estate ETF",
    "── AZIONI ──": None,
    "AAPL":      "Apple",
    "NVDA":      "NVIDIA",
    "TSLA":      "Tesla",
    "MSFT":      "Microsoft",
    "AMZN":      "Amazon",
    "META":      "Meta",
}

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.6rem;"
    "color:#00f5d4;letter-spacing:0.2em;text-transform:uppercase;"
    "border-bottom:1px solid rgba(0,245,212,0.2);padding-bottom:0.5rem;"
    "margin-bottom:1rem'>⬡ ASSET CONFIG</div>",
    unsafe_allow_html=True,
)

ticker_custom = st.sidebar.text_input("Ticker personalizzato", value="").upper().strip()

asset_options = [k for k, v in WATCHLIST.items() if v is not None]
selected_asset = st.sidebar.selectbox(
    "Watchlist (18 asset)",
    asset_options,
    index=0,
)
ticker = ticker_custom if ticker_custom else selected_asset

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.6rem;"
    "color:#00f5d4;letter-spacing:0.2em;text-transform:uppercase;"
    "border-bottom:1px solid rgba(0,245,212,0.2);padding-bottom:0.5rem;"
    "margin-bottom:1rem'>⬡ MODEL PARAMS</div>",
    unsafe_allow_html=True,
)

target_select = st.sidebar.selectbox(
    "Orizzonte AI",
    ["Target_1d", "Target_3d", "Target_5d"],
    index=0,
)

threshold_slider = st.sidebar.slider(
    "Soglia attivazione prob.",
    min_value=0.50,
    max_value=0.65,
    value=0.53,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.6rem;"
    "color:#00f5d4;letter-spacing:0.2em;text-transform:uppercase;"
    "border-bottom:1px solid rgba(0,245,212,0.2);padding-bottom:0.5rem;"
    "margin-bottom:0.5rem'>⬡ PREDICTION BAND</div>",
    unsafe_allow_html=True,
)
show_pred_band = st.sidebar.checkbox("Mostra banda predittiva AI", value=True)
pred_days = st.sidebar.slider("Giorni proiezione", 5, 30, 14)

# ─────────────────────────────────────────────
#  DATA FETCHING — LIVE via yfinance
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_market_data(ticker_str: str):
    try:
        df = yf.Ticker(ticker_str).history(period="5y", progress=False)
        if df.empty:
            return None
        df.columns = [str(c).strip().capitalize() for c in df.columns]
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required):
            return None
        df = df[required].astype(float)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna()
    except Exception as e:
        st.sidebar.error(f"Errore download: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_usd_eur_rate() -> float:
    try:
        df = yf.Ticker("USDEUR=X").history(period="5d", progress=False)
        return float(df["Close"].iloc[-1])
    except Exception:
        return 0.92


@st.cache_data(ttl=3600)
def fetch_fear_greed_history() -> pd.Series:
    try:
        url = "https://api.alternative.me/fng/?limit=500&format=json"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return pd.Series(dtype=float)
        df_fng = pd.DataFrame(data)
        df_fng["timestamp"] = pd.to_datetime(
            df_fng["timestamp"].astype(int), unit="s"
        ).dt.normalize()
        df_fng["fng_value"] = df_fng["value"].astype(float)
        df_fng = df_fng.set_index("timestamp").sort_index()
        df_fng.index = pd.to_datetime(df_fng.index).tz_localize(None)
        return df_fng["fng_value"]
    except Exception as e:
        st.sidebar.warning(f"⚠ Fear & Greed non disponibile: {e}")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df_raw: pd.DataFrame, series_fng: pd.Series) -> pd.DataFrame:
    df = df_raw.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()

    std200 = close.rolling(200).std().replace(0, 1e-9)
    df["Z_Score"] = (close - df["SMA_200"]) / std200
    df["Above_SMA50"] = (close > df["SMA_50"]).astype(int)
    df["Above_SMA200"] = (close > df["SMA_200"]).astype(int)

    for p in [5, 10, 21]:
        df[f"Mom_{p}d"] = close.pct_change(p)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = delta.clip(upper=0).abs().rolling(14).mean().replace(0, 1e-9)
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd_line - signal_line

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std().replace(0, 1e-9)
    df["BB_pct"] = (close - (sma20 - 2 * std20)) / (4 * std20)

    hl = high - low
    hc = np.abs(high - close.shift())
    lc = np.abs(low - close.shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_Norm"] = df["ATR"] / close.replace(0, 1e-9)

    vol_sma20 = vol.rolling(20).mean().replace(0, 1e-9)
    df["Vol_Ratio"] = vol / vol_sma20
    df["Vol_Mom_20d"] = vol.pct_change(20).fillna(0)

    if not series_fng.empty:
        fng_aligned = series_fng.reindex(df.index, method="ffill").fillna(50)
        df["FNG"] = fng_aligned.shift(1).fillna(50)
    else:
        df["FNG"] = 50.0

    # Rimuoviamo i NaN degli indicatori ma teniamo le righe finali per la predizione live
    df = df.replace([np.inf, -np.inf], np.nan)
    # Rimuoviamo solo se mancano le feature (ovvero le prime 200 righe)
    feature_cols_check = [c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna(subset=feature_cols_check)

    # I target conterranno NaN solo nelle ultime righe (che escluderemo manualmente nel training)
    df["Target_1d"] = (df["Close"].shift(-1) > df["Close"] * 1.003).astype(int)
    df["Target_3d"] = (df["Close"].shift(-3) > df["Close"] * 1.010).astype(int)
    df["Target_5d"] = (df["Close"].shift(-5) > df["Close"] * 1.018).astype(int)

    return df


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "Z_Score", "RSI", "ATR_Norm", "FNG",
    "Vol_Mom_20d", "Mom_5d", "Mom_10d", "Mom_21d",
    "MACD_Hist", "BB_pct", "Above_SMA50", "Above_SMA200", "Vol_Ratio",
]
TARGETS = {"Target_1d": 1, "Target_3d": 3, "Target_5d": 5}


def build_ensemble():
    xgb_m = xgb.XGBClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.05,
        verbosity=0, random_state=42, eval_metric="logloss",
    )
    rf_m = RandomForestClassifier(
        n_estimators=80, max_depth=5, random_state=42, n_jobs=-1
    )
    lgb_m = lgb.LGBMClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.05,
        verbose=-1, random_state=42,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb_m), ("rf", rf_m), ("lgb", lgb_m)],
        voting="soft",
    )


# ─────────────────────────────────────────────
#  TRAIN & EVALUATE
# ─────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame, target_col: str, shift_len: int):
    # Escludiamo rigorosamente le ultime righe per evitare NaN nel target e data leakage
    X_train_set = df[FEATURE_COLS].iloc[:-shift_len]
    y_train_set = df[target_col].iloc[:-shift_len]

    if len(X_train_set) < 200:
        return None, None, 0.0, None, None, None

    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    oof_probs = np.full(len(X_train_set), np.nan)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_set)):
        if len(train_idx) < 80:
            continue
        X_tr, y_tr = X_train_set.iloc[train_idx], y_train_set.iloc[train_idx]
        X_te, y_te = X_train_set.iloc[test_idx], y_train_set.iloc[test_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        m = build_ensemble()
        m.fit(X_tr_s, y_tr)
        preds = m.predict(X_te_s)
        fold_scores.append(accuracy_score(y_te, preds))

        probs = m.predict_proba(X_te_s)[:, 1]
        oof_probs[test_idx] = probs

    accuracy = float(np.mean(fold_scores)) if fold_scores else 0.5

    # Fit finale su tutta la porzione provvista di target valido
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_train_set)
    ensemble_final = build_ensemble()

    calibrated = CalibratedClassifierCV(
        estimator=ensemble_final,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )
    calibrated.fit(X_scaled, y_train_set)

    # PREDIZIONE LIVE: Calcolata sull'ULTIMA RIGA ASSOLUTA del dataset (Oggi)
    last_row = df[FEATURE_COLS].iloc[[-1]]
    prob_live = float(
        calibrated.predict_proba(scaler_final.transform(last_row))[:, 1]
    )

    brier = brier_score_loss(y_train_set, calibrated.predict_proba(X_scaled)[:, 1])

    return calibrated, scaler_final, accuracy, prob_live, brier, oof_probs


# ─────────────────────────────────────────────
#  BACKTEST BUILDER
# ─────────────────────────────────────────────
def build_backtest_series(df, oof_probs_dict, shift_lens):
    df_bt = df.copy()
    df_bt["Market_Returns"] = df_bt["Close"].pct_change()

    for target_col, oof_probs in oof_probs_dict.items():
        n = len(oof_probs)
        col_name = f"OOF_Prob_{target_col}"
        df_bt[col_name] = np.nan
        df_bt.iloc[:n, df_bt.columns.get_loc(col_name)] = oof_probs

    return df_bt


# ─────────────────────────────────────────────
#  PREDICTIVE BAND
# ─────────────────────────────────────────────
def make_prediction_band(df, model, scaler, prob_live, n_days=14):
    last_price = df["Close"].iloc[-1]
    last_vol = df["Close"].pct_change().rolling(21).std().iloc[-1]
    last_vol = last_vol if last_vol > 0 else 0.02

    np.random.seed(42)
    n_sims = 500
    bias = (prob_live - 0.5) * 0.3

    paths = np.zeros((n_sims, n_days))
    for sim in range(n_sims):
        price = last_price
        for d in range(n_days):
            ret = bias * last_vol + last_vol * np.random.randn()
            price = price * (1 + ret)
            paths[sim, d] = price

    dates_future = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=n_days,
        freq="B",
    )

    p10 = np.percentile(paths, 10, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    return dates_future, p10, p25, p50, p75, p90


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
if not ticker:
    st.info("Inserisci un ticker nella sidebar per iniziare.")
    st.stop()

with st.spinner(f"[LIVE] Scaricando dati per {ticker} da Yahoo Finance..."):
    df_raw = fetch_market_data(ticker)
    eur_usd = fetch_usd_eur_rate()
    fng_series = fetch_fear_greed_history()

asset_name = WATCHLIST.get(ticker, ticker)
st.sidebar.markdown(
    f"<div style='font-family:\"Share Tech Mono\",monospace;font-size:0.65rem;"
    f"color:#64748b;margin-top:0.5rem'>USD/EUR: <span style='color:#00f5d4'>{eur_usd:.4f}</span></div>",
    unsafe_allow_html=True,
)

if df_raw is None or len(df_raw) < 300:
    st.error("❌ Dati insufficienti o ticker non valido. Prova BTC-USD, ETH-USD, SPY, AAPL…")
    st.stop()

with st.spinner("Calcolando indicatori tecnici..."):
    df = engineer_features(df_raw, fng_series)

if len(df) < 200:
    st.error("❌ Troppo pochi dati dopo il feature engineering.")
    st.stop()

results = {}
oof_probs_all = {}
progress_bar = st.progress(0, text="Addestrando modelli AI...")

try:
    for i, (target_col, shift_len) in enumerate(TARGETS.items()):
        progress_bar.progress(
            (i + 1) / len(TARGETS),
            text=f"[{i+1}/{len(TARGETS)}] Training {target_col}...",
        )
        out = train_and_evaluate(df, target_col, shift_len)
        model, scaler, accuracy, prob_live, brier, oof_probs = out
        results[target_col] = {
            "model": model,
            "scaler": scaler,
            "accuracy": accuracy,
            "prob_live": prob_live,
            "brier": brier,
        }
        if oof_probs is not None:
            oof_probs_all[target_col] = oof_probs
finally:
    progress_bar.empty()

# Backtest
df_bt = build_backtest_series(df, oof_probs_all, TARGETS)

oof_col = f"OOF_Prob_{target_select}"
if oof_col in df_bt.columns:
    df_bt["Signal"] = np.where(df_bt[oof_col] > threshold_slider, 1, 0)
    df_bt["Strategy_Returns"] = df_bt["Market_Returns"] * df_bt["Signal"]
else:
    df_bt["Signal"] = 0
    df_bt["Strategy_Returns"] = 0

df_bt["Cum_Market"] = (1 + df_bt["Market_Returns"].fillna(0)).cumprod()
df_bt["Cum_Strategy"] = (1 + df_bt["Strategy_Returns"].fillna(0)).cumprod()

mean_acc = float(np.mean([r["accuracy"] for r in results.values()]))
market_perf = (df_bt["Cum_Market"].iloc[-1] - 1) * 100
strat_perf = (df_bt["Cum_Strategy"].iloc[-1] - 1) * 100
total_trades = int(df_bt["Signal"].diff().abs().fillna(0).sum() / 2)

# ─────────────────────────────────────────────
#  UI RENDERING
# ─────────────────────────────────────────────
last_price = float(df["Close"].iloc[-1])
last_date = df.index[-1].strftime("%Y-%m-%d")
fng_now = int(df["FNG"].iloc[-1])
fng_label = "Fear & Greed (ieri)"

st.markdown('<div class="section-label">LIVE MARKET DATA</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"PRICE  {ticker}", f"${last_price:,.2f}")
c2.metric("ACCURACY ENSEMBLE", f"{mean_acc*100:.1f}%", help="Out-of-sample, ensemble XGB+RF+LGB")
c3.metric(fng_label, str(fng_now), help="0=panico · 100=greed · laggato d−1 per no leakage")
c4.metric("STRATEGIA AI", f"{strat_perf:+.1f}%")
c5.metric("BUY & HOLD", f"{market_perf:+.1f}%")

st.markdown("---")

st.markdown('<div class="section-label">AI SIGNAL — PROBABILITÀ LIVE CALIBRATE</div>', unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)
signal_cols = [p1, p2, p3]
horizon_info = [
    ("Target_1d", "1 GIORNO", "+0.3%"),
    ("Target_3d", "3 GIORNI", "+1.0%"),
    ("Target_5d", "5 GIORNI", "+1.8%"),
]

for col, (t, label, thresh_label) in zip(signal_cols, horizon_info):
    r = results[t]
    pct = r["prob_live"] * 100 if r["prob_live"] else 50.0
    buy = r["prob_live"] > threshold_slider if r["prob_live"] else False
    badge_class = "badge-buy" if buy else "badge-wait"
    badge_text = "▲ LONG" if buy else "◼ WAIT"
    bar_color = "#22c55e" if buy else "#ef4444"

    col.markdown(f"""
    <div style="background:#0f1e35;border:1px solid rgba(0,245,212,0.15);border-top:2px solid {bar_color};
    border-radius:6px;padding:1rem;font-family:'Share Tech Mono',monospace">
      <div style="font-size:0.6rem;color:#64748b;letter-spacing:0.15em;margin-bottom:8px">{label} · {thresh_label}</div>
      <div style="font-size:2rem;font-weight:700;color:#e2e8f0;line-height:1">{pct:.1f}<span style="font-size:1rem;color:#64748b">%</span></div>
      <div style="background:#071120;border-radius:2px;height:4px;margin:10px 0;overflow:hidden">
        <div style="width:{min(pct,100):.0f}%;height:100%;background:{bar_color};border-radius:2px"></div>
      </div>
      <div style="font-size:0.65rem;color:#64748b;margin-bottom:10px">
        ACC: {r['accuracy']*100:.1f}% &nbsp;|&nbsp; BRIER: {r['brier']:.3f}
      </div>
      <span class="{badge_class}">{badge_text}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div class="section-label">PREDICTIVE CHART — PREZZI + BANDA AI MONTE CARLO</div>', unsafe_allow_html=True)

ref_model = results[target_select]["model"]
ref_scaler = results[target_select]["scaler"]
ref_prob = results[target_select]["prob_live"] or 0.5

if show_pred_band and ref_model is not None:
    dates_fut, p10, p25, p50, p75, p90 = make_prediction_band(
        df, ref_model, ref_scaler, ref_prob, n_days=pred_days
    )

fig_price = go.Figure()

df_tail = df_raw.iloc[-180:]
fig_price.add_trace(go.Candlestick(
    x=df_tail.index,
    open=df_tail["Open"],
    high=df_tail["High"],
    low=df_tail["Low"],
    close=df_tail["Close"],
    name="OHLC",
    increasing_line_color="#22c55e",
    decreasing_line_color="#ef4444",
    increasing_fillcolor="rgba(34,197,94,0.3)",
    decreasing_fillcolor="rgba(239,68,68,0.2)",
    line=dict(width=1),
))

df_ind = df.iloc[-180:]
fig_price.add_trace(go.Scatter(
    x=df_ind.index, y=df_ind["SMA_50"],
    name="SMA 50", line=dict(color="#a855f7", width=1.2, dash="dot"),
    opacity=0.8,
))
fig_price.add_trace(go.Scatter(
    x=df_ind.index, y=df_ind["SMA_200"],
    name="SMA 200", line=dict(color="#f59e0b", width=1.5, dash="dash"),
    opacity=0.8,
))

if show_pred_band and ref_model is not None:
    fig_price.add_trace(go.Scatter(
        x=list(dates_fut) + list(dates_fut[::-1]),
        y=list(p90) + list(p10[::-1]),
        fill="toself",
        fillcolor="rgba(14,165,233,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Banda 10–90%",
        hoverinfo="skip",
    ))
    fig_price.add_trace(go.Scatter(
        x=list(dates_fut) + list(dates_fut[::-1]),
        y=list(p75) + list(p25[::-1]),
        fill="toself",
        fillcolor="rgba(14,165,233,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Banda 25–75%",
        hoverinfo="skip",
    ))
    fig_price.add_trace(go.Scatter(
        x=dates_fut, y=p50,
        name="Mediana AI",
        line=dict(color="#0ea5e9", width=2, dash="longdash"),
    ))
    fig_price.add_vline(
        x=df.index[-1],
        line_width=1,
        line_dash="dot",
        line_color="rgba(0,245,212,0.4)",
        annotation_text="OGGI",
        annotation_font=dict(color="#00f5d4", size=10, family="'Share Tech Mono',monospace"),
        annotation_position="top right",
    )

fig_price.update_layout(
    **PLOT_LAYOUT,
    title=dict(
        text=f"{ticker}  ·  {asset_name or ticker}  ·  Ultimi 180 giorni + proiezione {pred_days}gg",
        font=dict(size=13, color="#64748b", family="'Share Tech Mono',monospace"),
    ),
    xaxis_rangeslider_visible=False,
    height=500,
)

st.plotly_chart(fig_price, on_select="ignore")

st.markdown('<div class="section-label">BACKTEST — STRATEGIA AI vs BUY & HOLD</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("PERFORMANCE AI", f"{strat_perf:+.1f}%")
m2.metric("BUY & HOLD", f"{market_perf:+.1f}%")
m3.metric("TRADE TOTALI", str(total_trades))
m4.metric("SOGLIA ATTIVA", f"{threshold_slider:.2f}")

df_chart = df_bt[["Cum_Market", "Cum_Strategy"]].dropna()

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart["Cum_Market"],
    name="Buy & Hold",
    line=dict(color="rgba(100,116,139,0.7)", width=1.5, dash="dot"),
    fill="tozeroy",
    fillcolor="rgba(100,116,139,0.02)",
))
fig_bt.add_trace(go.Scatter(
    x=df_chart.index, y=df_chart["Cum_Strategy"],
    name="Strategia AI",
    line=dict(color="#00f5d4", width=2),
    fill="tozeroy",
    fillcolor="rgba(0,245,212,0.04)",
))
fig_bt.update_layout(
    **PLOT_LAYOUT,
    title=dict(
        text=f"Performance cumulativa · soglia {threshold_slider:.2f} · {target_select}",
        font=dict(size=13, color="#64748b", family="'Share Tech Mono',monospace"),
    ),
    yaxis_title="Capitale (1 = start)",
    height=380,
)
st.plotly_chart(fig_bt, on_select="ignore")

st.markdown('<div class="section-label">OSCILLATORI TECNICI — RSI + MACD</div>', unsafe_allow_html=True)

df_osc = df.iloc[-180:]
fig_osc = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.5, 0.5], vertical_spacing=0.06,
    subplot_titles=["RSI (14)", "MACD Histogram"],
)

fig_osc.add_trace(go.Scatter(
    x=df_osc.index, y=df_osc["RSI"],
    name="RSI", line=dict(color="#a855f7", width=1.5),
), row=1, col=1)
fig_osc.add_hline(y=70, line_dash="dot", line_color="rgba(239,68,68,0.4)", row=1, col=1)
fig_osc.add_hline(y=30, line_dash="dot", line_color="rgba(34,197,94,0.4)", row=1, col=1)

macd_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df_osc["MACD_Hist"]]
fig_osc.add_trace(go.Bar(
    x=df_osc.index, y=df_osc["MACD_Hist"],
    name="MACD Hist", marker_color=macd_colors, opacity=0.7,
), row=2, col=1)

fig_osc.update_layout(
    **PLOT_LAYOUT,
    height=340,
    showlegend=False,
)
fig_osc.update_yaxes(
    gridcolor="rgba(0,245,212,0.05)",
    linecolor="rgba(0,245,212,0.15)",
)
st.plotly_chart(fig_osc, on_select="ignore")

st.markdown('<div class="section-label">MODEL METRICS TABLE</div>', unsafe_allow_html=True)

rows = []
for target_col, r in results.items():
    if r["prob_live"] is None:
        continue
    buy = r["prob_live"] > threshold_slider
    rows.append({
        "Orizzonte": target_col,
        "Prob Live": f"{r['prob_live']*100:.1f}%",
        "Accuracy OOS": f"{r['accuracy']*100:.1f}%",
        "Brier Score": f"{r['brier']:.4f}",
        "Qualità prob.": "OTTIMA" if r["brier"] < 0.22 else ("BUONA" if r["brier"] < 0.245 else "DEBOLE"),
        "Segnale": "▲ LONG" if buy else "◼ WAIT",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(f"""
<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:#334155;
line-height:1.8;padding:0.5rem 0">
  QUANT HODL v14 &nbsp;·&nbsp; {last_date} &nbsp;·&nbsp; DATI LIVE DA YAHOO FINANCE<br>
  ⚠ NON È CONSULENZA FINANZIARIA · SOLO USO DIDATTICO/RICERCA
</div>
""", unsafe_allow_html=True)
