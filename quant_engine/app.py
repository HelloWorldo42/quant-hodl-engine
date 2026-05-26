import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    xaxis=dict(gridcolor="rgba(0,245,212,0.05)", linecolor="rgba(0,245,212,0.15)", tickcolor="rgba(0,245,212,0.15)"),
    yaxis=dict(gridcolor="rgba(0,245,212,0.05)", linecolor="rgba(0,245,212,0.15)", tickcolor="rgba(0,245,212,0.15)"),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#0d1525", bordercolor="#00f5d4",
                    font=dict(color="#e2e8f0", size=12, family="'Share Tech Mono', monospace")),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,245,212,0.2)", borderwidth=1,
                font=dict(size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=50, r=30, t=60, b=40),
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="QUANT HODL v14", layout="wide",
                   initial_sidebar_state="expanded", page_icon="⬡")
st.markdown(STYLE, unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  DATA LAYER
# ═════════════════════════════════════════════
ASSETS = {
    "Bitcoin":  "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana":   "SOL-USD",
    "Cardano":  "ADA-USD",
    "Polkadot": "DOT-USD",
    "Chainlink": "LINK-USD",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    """Download daily OHLCV. Robust to yfinance MultiIndex / empty returns."""
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Recent yfinance wraps single-ticker frames in a MultiIndex column level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()
    return df


# ═════════════════════════════════════════════
#  FEATURE ENGINEERING  (pure pandas, no TA-Lib)
# ═════════════════════════════════════════════
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame, horizon: int):
    out = pd.DataFrame(index=df.index)
    close, vol = df["Close"], df["Volume"]

    out["ret_1"]  = close.pct_change(1)
    out["ret_5"]  = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    out["rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd - signal

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out["sma20_ratio"] = close / sma20 - 1
    out["sma50_ratio"] = close / sma50 - 1

    std20 = close.rolling(20).std()
    out["bb_pct"] = (close - (sma20 - 2 * std20)) / (4 * std20)

    out["volatility"] = out["ret_1"].rolling(20).std()
    out["momentum"]   = close / close.shift(10) - 1

    vol_sma = vol.rolling(20).mean()
    out["vol_ratio"] = vol / vol_sma.replace(0, np.nan) - 1

    # Forward target: did price rise over the next `horizon` days?
    fwd = close.shift(-horizon) / close - 1
    out["target"] = (fwd > 0).astype(int)

    features = [c for c in out.columns if c != "target"]
    out = out.replace([np.inf, -np.inf], np.nan)
    labeled = out.dropna()                 # rows with known target (for training)
    live = out[features].replace([np.inf, -np.inf], np.nan).dropna()  # incl. last bar
    return out, labeled, live, features


# ═════════════════════════════════════════════
#  MODEL  +  WALK-FORWARD BACKTEST
# ═════════════════════════════════════════════
def _make_ensemble(seed: int = 42) -> VotingClassifier:
    xgb_clf = xgb.XGBClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=seed, n_jobs=1, verbosity=0,
    )
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=120, max_depth=4, num_leaves=15, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_jobs=1, verbose=-1,
    )
    rf_clf = RandomForestClassifier(
        n_estimators=120, max_depth=6, random_state=seed, n_jobs=1,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("rf", rf_clf)],
        voting="soft", n_jobs=1,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def train_pipeline(ticker: str, period: str, horizon: int, seed: int = 42):
    raw = load_prices(ticker, period)
    if raw.empty or len(raw) < 200:
        return {"ok": False, "reason": "Dati insufficienti per questo asset/periodo."}

    full, labeled, live, feats = build_features(raw, horizon)
    if len(labeled) < 150:
        return {"ok": False, "reason": "Storico troppo corto dopo il calcolo degli indicatori."}

    X = labeled[feats].values
    y = labeled["target"].values
    idx = labeled.index

    # ── Walk-forward out-of-fold predictions (honest backtest) ──
    tscv = TimeSeriesSplit(n_splits=5)
    oof_proba = np.full(len(y), np.nan)
    for tr, te in tscv.split(X):
        model = Pipeline([("sc", StandardScaler()), ("clf", _make_ensemble(seed))])
        model.fit(X[tr], y[tr])
        oof_proba[te] = model.predict_proba(X[te])[:, 1]

    mask = ~np.isnan(oof_proba)
    oof_pred = (oof_proba[mask] >= 0.5).astype(int)
    acc   = accuracy_score(y[mask], oof_pred)
    brier = brier_score_loss(y[mask], oof_proba[mask])
    baseline = max(y.mean(), 1 - y.mean())   # naive "always majority class"

    # ── Final calibrated model on ALL labeled data → live signal ──
    base = Pipeline([("sc", StandardScaler()), ("clf", _make_ensemble(seed))])
    calib = CalibratedClassifierCV(base, method="sigmoid",
                                   cv=TimeSeriesSplit(n_splits=3))
    calib.fit(X, y)

    live_row = live[feats].iloc[[-1]].values
    live_proba = float(calib.predict_proba(live_row)[0, 1])
    live_date = live.index[-1]

    # ── Feature importance (avg gain across the 3 fitted RF/boosters) ──
    fit_ref = base.fit(X, y)
    importances = {}
    clf = fit_ref.named_steps["clf"]
    for name, est in clf.named_estimators_.items():
        if hasattr(est, "feature_importances_"):
            importances[name] = est.feature_importances_
    if importances:
        imp = np.mean([v / (v.sum() + 1e-9) for v in importances.values()], axis=0)
    else:
        imp = np.zeros(len(feats))

    return {
        "ok": True,
        "price": raw["Close"], "dates": raw.index,
        "oof_idx": idx[mask], "oof_proba": oof_proba[mask],
        "oof_price": labeled["Close"][mask] if "Close" in labeled else raw["Close"].reindex(idx[mask]),
        "acc": acc, "brier": brier, "baseline": baseline,
        "live_proba": live_proba, "live_date": live_date,
        "last_close": float(raw["Close"].iloc[-1]),
        "feats": feats, "importance": imp,
        "n_train": int(len(y)), "up_rate": float(y.mean()),
    }


# ═════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='section-label'>// CONTROL DECK</div>", unsafe_allow_html=True)
    asset_name = st.selectbox("Asset", list(ASSETS.keys()), index=0)
    ticker = ASSETS[asset_name]

    period = st.selectbox("Storico", ["1y", "2y", "3y", "5y", "max"], index=2)
    horizon = st.slider("Orizzonte previsione (giorni)", 1, 30, 7)
    threshold = st.slider("Soglia segnale BUY", 0.50, 0.80, 0.55, 0.01)

    st.markdown("---")
    st.caption("Come funziona: il programma guarda anni di prezzi passati, impara "
               "qualche schema e prova a indovinare se il prezzo salirà. "
               "Poi ti dice da solo quanto è (poco) affidabile.")
    if st.button("⬡  ESEGUI ANALISI", use_container_width=True):
        st.session_state["ran"] = True


# ═════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════
st.markdown("# ⬡ QUANT HODL v14")
st.markdown(
    f"<div class='section-label'>// {asset_name.upper()} · {ticker} · "
    f"HORIZON {horizon}D · SOGLIA {threshold:.0%}</div>",
    unsafe_allow_html=True,
)

if not st.session_state.get("ran", False):
    st.info("Scegli una criptovaluta nel menu a sinistra e premi **ESEGUI ANALISI**.")
    st.caption("⚠️ App per imparare. Nessun programma prevede davvero il prezzo delle "
               "criptovalute: quello che vedi NON è un consiglio su come investire.")
    st.stop()

with st.spinner("Sto scaricando i prezzi e facendo i calcoli…"):
    res = train_pipeline(ticker, period, horizon)

if not res.get("ok"):
    st.error(res.get("reason", "Errore sconosciuto."))
    st.stop()

# Quanto è affidabile il modello su questo asset, in parole semplici
edge = res["acc"] - res["baseline"]
if edge <= 0.01:
    affidabile = False
    voto = "SCARSA"
else:
    affidabile = True
    voto = "DISCRETA" if edge < 0.05 else "BUONA"

# ─── KPI ROW ───
signal_buy = res["live_proba"] >= threshold and affidabile
c1, c2, c3, c4 = st.columns(4)
c1.metric("Prezzo attuale", f"${res['last_close']:,.2f}")
c2.metric("Fiducia che salga", f"{res['live_proba']:.0%}",
          delta=f"{(res['live_proba'] - 0.5) * 100:+.0f} punti sopra/sotto il 50%",
          help="Quanto il modello è ottimista sui prossimi giorni. "
               "50% = come lanciare una moneta.")
c3.metric("Quante volte ci azzecca", f"{res['acc']:.0%}",
          delta=f"{edge * 100:+.0f} punti vs tirare a caso",
          help="Percentuale di volte in cui, nei test sul passato, ha indovinato "
               "la direzione. Va confrontata con il 'tirare a caso'.")
c4.metric("Voto di affidabilità", voto,
          help="Sintesi: il modello batte il caso? SCARSA = no, fidati poco.")

if not affidabile:
    badge = "<span class='badge-wait'>● VERDETTO: NON FIDARTI</span>"
elif signal_buy:
    badge = "<span class='badge-buy'>● VERDETTO: SEGNALE POSITIVO</span>"
else:
    badge = "<span class='badge-wait'>● VERDETTO: MEGLIO ASPETTARE</span>"
st.markdown(badge, unsafe_allow_html=True)

# ─── RIQUADRO "COSA SIGNIFICA" (in italiano corrente) ───
if not affidabile:
    st.warning(
        f"**Cosa significa, in parole semplici.** Su {asset_name} questo modello "
        f"ci azzecca circa **{res['acc']:.0%}** delle volte: praticamente come **tirare "
        f"una monetina**, anzi un po' peggio. Quindi i suoi 'segnali' qui **non valgono nulla** "
        "e la riga viola che salta su e giù è solo rumore. Non è un errore del programma: "
        "è la prova, nera su bianco, che i prezzi delle crypto a breve termine **non si "
        "possono prevedere**. Usa l'app per capire come funziona, non per decidere dove "
        "mettere dei soldi."
    )
else:
    direzione = "in salita" if res["live_proba"] >= 0.5 else "in discesa"
    st.info(
        f"**Cosa significa, in parole semplici.** Sui test passati questo modello ci ha "
        f"azzeccato **{res['acc']:.0%}** delle volte, un po' meglio del caso. Adesso si "
        f"sbilancia leggermente **{direzione}** ({res['live_proba']:.0%} di fiducia). "
        "Resta comunque un vantaggio piccolissimo e fragile: i mercati crypto cambiano "
        "in fretta e nessuno li prevede davvero. Trattalo come un'indicazione curiosa, "
        "**non** come un consiglio su cui investire."
    )
st.markdown("")

# ─── GRAFICO PREZZO + FIDUCIA ───
st.markdown("<div class='section-label'>// PREZZO REALE vs FIDUCIA DEL MODELLO</div>",
            unsafe_allow_html=True)
st.caption("Linea verde = prezzo vero in dollari (scala a sinistra). "
           "Linea viola = quanto il modello era ottimista quel giorno, da 0% a 100% "
           "(scala a destra). Più la viola sembra impazzita, meno è affidabile.")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=res["dates"], y=res["price"], name="Prezzo reale ($)",
                         line=dict(color="#00f5d4", width=1.6)), secondary_y=False)
fig.add_trace(go.Scatter(x=res["oof_idx"], y=res["oof_proba"] * 100,
                         name="Fiducia che salga (%)",
                         line=dict(color="#a855f7", width=1.2), opacity=0.8),
              secondary_y=True)
fig.add_hline(y=threshold * 100, line=dict(color="#f59e0b", width=1, dash="dot"),
              secondary_y=True, annotation_text=f"soglia {threshold:.0%}",
              annotation_font_color="#f59e0b")
buy_idx = res["oof_idx"][res["oof_proba"] >= threshold]
buy_px = res["price"].reindex(buy_idx)
fig.add_trace(go.Scatter(x=buy_idx, y=buy_px, mode="markers",
                         name="Giorni 'segnale positivo'",
                         marker=dict(color="#22c55e", size=5, symbol="triangle-up")),
              secondary_y=False)
fig.update_layout(**PLOT_LAYOUT, height=440)
fig.update_yaxes(title_text="Prezzo ($)", secondary_y=False)
fig.update_yaxes(title_text="Fiducia (%)", range=[0, 100], secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

# ─── COSA GUARDA IL MODELLO + RIEPILOGO ───
LABELS = {
    "ret_1": "Variazione di ieri", "ret_5": "Andamento ultimi 5 giorni",
    "ret_10": "Andamento ultimi 10 giorni", "rsi_14": "Ipercomprato/ipervenduto (RSI)",
    "macd_hist": "Forza del trend (MACD)", "sma20_ratio": "Distanza dalla media a 20gg",
    "sma50_ratio": "Distanza dalla media a 50gg", "bb_pct": "Posizione nelle bande",
    "volatility": "Quanto è agitato il prezzo", "momentum": "Spinta recente",
    "vol_ratio": "Volume di scambi",
}
col_a, col_b = st.columns([3, 2])
with col_a:
    st.markdown("<div class='section-label'>// SU COSA SI BASA IL MODELLO</div>",
                unsafe_allow_html=True)
    st.caption("Gli ingredienti che pesano di più nelle sue decisioni. "
               "Più la barra è lunga, più quel dato conta.")
    order = np.argsort(res["importance"])
    fimp = go.Figure(go.Bar(
        x=res["importance"][order],
        y=[LABELS.get(res["feats"][i], res["feats"][i]) for i in order],
        orientation="h", marker=dict(color="#0ea5e9")))
    fimp.update_layout(**PLOT_LAYOUT, height=360)
    st.plotly_chart(fimp, use_container_width=True)

with col_b:
    st.markdown("<div class='section-label'>// IN BREVE</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Cosa": ["Giorni analizzati", "Quante volte sale di solito",
                 "Quante volte ci azzecca", "Se tirasse a caso", "Voto"],
        "Valore": [res["n_train"], f"{res['up_rate']:.0%}", f"{res['acc']:.0%}",
                   f"{res['baseline']:.0%}", voto],
    }), hide_index=True, use_container_width=True)
    if not affidabile:
        st.warning("In parole povere: qui **non ha capacità di previsione**.")
    else:
        st.success("Qui fa **un pelo meglio del caso** — ma resta poco affidabile.")

st.caption(f"Calcolato il {res['live_date'].date()}. "
           "⚠️ Questa è un'app per imparare, NON un consiglio finanziario. "
           "Il prezzo delle criptovalute è imprevedibile e puoi perdere soldi.")
