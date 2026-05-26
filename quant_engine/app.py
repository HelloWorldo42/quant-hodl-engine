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
import warnings

warnings.filterwarnings('ignore')

# =================== CONFIG ===================

st.set_page_config(
    page_title="QUANT HODL v13 FIXED",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 QUANT HODL v13 FIXED — AI Predictive & Backtest Suite")
st.subheader("Validazione Walk-Forward Rigorosa | No Data Leakage | Accuracy Reale")

# =================== SIDEBAR ===================

st.sidebar.header("⚙️ Configurazione Asset")
ticker = st.sidebar.text_input(
    "Ticker Yahoo Finance (es. BTC-USD, ETH-USD, AAPL)",
    value="BTC-USD"
).upper().strip()

st.sidebar.markdown("---")
st.sidebar.header("📈 Parametri Backtesting")

target_select = st.sidebar.selectbox(
    "Orizzonte AI di Riferimento",
    ['Target_1d', 'Target_3d', 'Target_5d'],
    index=0
)

threshold_slider = st.sidebar.slider(
    "Soglia Attivazione Probabilità",
    min_value=0.50,
    max_value=0.65,
    value=0.53,
    step=0.01
)

# =================== DATA FETCHING ===================

@st.cache_data(ttl=300)  # Cache 5 minuti
def fetch_market_data(ticker_str):
    """Scarica dati storici - SAFE"""
    try:
        df = yf.Ticker(ticker_str).history(period="5y", progress=False)

        if df.empty:
            return None

        df.columns = [str(c).strip().capitalize() for c in df.columns]
        required = ['Open', 'High', 'Low', 'Close', 'Volume']

        if not all(col in df.columns for col in required):
            return None

        df = df[required].astype(float)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna()

    except Exception as e:
        st.sidebar.error(f"Errore download: {e}")
        return None


@st.cache_data(ttl=3600)  # Cache 1 ora
def fetch_usd_eur_rate():
    """Tasso USD/EUR - SAFE"""
    try:
        df = yf.Ticker("USDEUR=X").history(period="5d", progress=False)
        return float(df['Close'].iloc[-1])
    except:
        return 0.92


@st.cache_data(ttl=3600)
def fetch_fear_greed_history():
    """
    Fear & Greed Index storico - URL CORRETTO
    BUG ORIGINALE: requests.get("https://alternative.me") → HTML, non JSON
    FIX: usa l'API endpoint corretto
    """
    try:
        # ✅ URL CORRETTO (era "https://alternative.me" nell'originale → SBAGLIATO)
        url = "https://api.alternative.me/fng/?limit=500&format=json"
        r = requests.get(url, timeout=8)
        r.raise_for_status()

        data = r.json().get('data', [])
        if not data:
            return pd.Series(dtype=float)

        df_fng = pd.DataFrame(data)
        df_fng['timestamp'] = pd.to_datetime(df_fng['timestamp'].astype(int), unit='s').dt.normalize()
        df_fng['fng_value'] = df_fng['value'].astype(float)
        df_fng = df_fng.set_index('timestamp').sort_index()
        df_fng.index = pd.to_datetime(df_fng.index).tz_localize(None)
        return df_fng['fng_value']

    except Exception as e:
        st.sidebar.warning(f"⚠️ Fear & Greed non disponibile: {e}")
        return pd.Series(dtype=float)

# =================== FEATURE ENGINEERING ===================

def engineer_features(df_raw, series_fng):
    """Crea 13+ features - SAFE con protezione divisione per zero"""

    df = df_raw.copy()
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    # Trend
    df['SMA_50']  = close.rolling(50).mean()
    df['SMA_200'] = close.rolling(200).mean()

    std200 = close.rolling(200).std().replace(0, 1e-9)
    df['Z_Score'] = (close - df['SMA_200']) / std200

    df['Above_SMA50']  = (close > df['SMA_50']).astype(int)
    df['Above_SMA200'] = (close > df['SMA_200']).astype(int)

    # Momentum
    for p in [5, 10, 21]:
        df[f'Mom_{p}d'] = close.pct_change(p)

    # RSI - SAFE (protezione da loss=0)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = delta.clip(upper=0).abs().rolling(14).mean().replace(0, 1e-9)
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line

    # Bollinger Bands %
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std().replace(0, 1e-9)
    df['BB_pct'] = (close - (sma20 - 2 * std20)) / (4 * std20)

    # ATR normalizzato
    hl  = high - low
    hc  = np.abs(high - close.shift())
    lc  = np.abs(low  - close.shift())
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR']      = tr.rolling(14).mean()
    df['ATR_Norm'] = df['ATR'] / close.replace(0, 1e-9)

    # Volume
    vol_sma20 = vol.rolling(20).mean().replace(0, 1e-9)
    df['Vol_Ratio']   = vol / vol_sma20
    df['Vol_Mom_20d'] = vol.pct_change(20).fillna(0)

    # Fear & Greed Index (storico, shiftato di 1 per no leakage)
    if not series_fng.empty:
        fng_aligned = series_fng.reindex(df.index, method='ffill').fillna(50)
        df['FNG'] = fng_aligned.shift(1).fillna(50)
    else:
        df['FNG'] = 50.0

    df = df.dropna()

    # Target binari
    df['Target_1d'] = (df['Close'].shift(-1) > df['Close'] * 1.003).astype(int)
    df['Target_3d'] = (df['Close'].shift(-3) > df['Close'] * 1.010).astype(int)
    df['Target_5d'] = (df['Close'].shift(-5) > df['Close'] * 1.018).astype(int)

    return df.replace([np.inf, -np.inf], np.nan).dropna()

# =================== MODEL TRAINING (NO LEAKAGE) ===================

FEATURE_COLS = [
    'Z_Score', 'RSI', 'ATR_Norm', 'FNG',
    'Vol_Mom_20d', 'Mom_5d', 'Mom_10d', 'Mom_21d',
    'MACD_Hist', 'BB_pct', 'Above_SMA50', 'Above_SMA200', 'Vol_Ratio'
]

TARGETS = {'Target_1d': 1, 'Target_3d': 3, 'Target_5d': 5}


def build_ensemble():
    """Crea ensemble identico per accuracy e prediction"""
    xgb_m = xgb.XGBClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.05,
        verbosity=0, random_state=42, eval_metric='logloss'
    )
    rf_m = RandomForestClassifier(
        n_estimators=80, max_depth=5,
        random_state=42, n_jobs=-1
    )
    lgb_m = lgb.LGBMClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.05,
        verbose=-1, random_state=42
    )
    return VotingClassifier(
        estimators=[('xgb', xgb_m), ('rf', rf_m), ('lgb', lgb_m)],
        voting='soft'
    )


def train_and_evaluate(df, target_col, shift_len):
    """
    Train + valutazione SENZA data leakage.

    FIX rispetto all'originale:
    - Accuracy calcolata sull'ENSEMBLE (non solo RF)
    - TimeSeriesSplit n_splits=5 (era 3)
    - predict_proba solo su dati out-of-sample nel backtest
    """

    # Rimuovi gli ultimi `shift_len` per non usare dati futuri
    X = df[FEATURE_COLS].iloc[:-shift_len]
    y = df[target_col].iloc[:-shift_len]

    if len(X) < 200:
        return None, None, 0.0, None, None

    # ✅ TimeSeriesSplit n_splits=5 (era 3 nell'originale → instabile)
    tscv = TimeSeriesSplit(n_splits=5)

    fold_scores = []
    oof_probs  = np.full(len(X), np.nan)  # Out-of-fold probabilities

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < 80:
            continue

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # ✅ Accuracy calcolata sull'ENSEMBLE (era solo RF nell'originale)
        m = build_ensemble()
        m.fit(X_tr_s, y_tr)

        preds = m.predict(X_te_s)
        fold_scores.append(accuracy_score(y_te, preds))

        # Salva probabilità out-of-fold per backtest pulito
        probs = m.predict_proba(X_te_s)[:, 1]
        oof_probs[test_idx] = probs

    accuracy = float(np.mean(fold_scores)) if fold_scores else 0.5

    # Modello finale su tutto (per predizione live)
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)

    ensemble_final = build_ensemble()

    # ✅ Calibrazione probabilità (manteniamo questa buona idea dall'originale)
    calibrated = CalibratedClassifierCV(
        estimator=ensemble_final,
        method='sigmoid',
        cv=3  # era 2, aumentato a 3 per stabilità
    )
    calibrated.fit(X_scaled, y)

    # Probabilità live sull'ultima riga disponibile
    today_row = df[FEATURE_COLS].iloc[[-1]]
    prob_live = float(calibrated.predict_proba(scaler_final.transform(today_row))[:, 1])

    # Brier score (misura qualità probabilità: più basso = meglio, max=0.25)
    brier = brier_score_loss(y, calibrated.predict_proba(X_scaled)[:, 1])

    return calibrated, scaler_final, accuracy, prob_live, brier, oof_probs


def build_backtest_series(df, oof_probs_dict, shift_lens):
    """
    Costruisce backtest usando SOLO probabilità out-of-fold.
    Nessun leakage: il modello non ha mai visto i dati su cui predice.
    """
    df_bt = df.copy()
    df_bt['Market_Returns'] = df_bt['Close'].pct_change()

    for target_col, oof_probs in oof_probs_dict.items():
        shift_len = shift_lens[target_col]

        # Allinea OOF con il dataframe (rimuovendo gli ultimi shift_len)
        n = len(oof_probs)
        col_name = f'OOF_Prob_{target_col}'
        df_bt[col_name] = np.nan
        df_bt.iloc[:n, df_bt.columns.get_loc(col_name)] = oof_probs

    return df_bt


# =================== MAIN PIPELINE ===================

if not ticker:
    st.info("Inserisci un ticker nella sidebar per iniziare.")
    st.stop()

# Carica dati
with st.spinner("Scaricando dati storici..."):
    df_raw  = fetch_market_data(ticker)
    eur_usd = fetch_usd_eur_rate()
    fng_series = fetch_fear_greed_history()

st.sidebar.markdown(f"**Cambio USD/EUR:** {eur_usd:.4f}")

if df_raw is None or len(df_raw) < 300:
    st.error("❌ Dati insufficienti o ticker non valido. Prova BTC-USD, ETH-USD, AAPL.")
    st.stop()

# Feature Engineering
with st.spinner("Calcolando indicatori tecnici..."):
    df = engineer_features(df_raw, fng_series)

if len(df) < 200:
    st.error("❌ Troppo pochi dati dopo il feature engineering.")
    st.stop()

# Training
results       = {}
oof_probs_all = {}

progress_bar = st.progress(0, text="Addestrando modelli...")

for i, (target_col, shift_len) in enumerate(TARGETS.items()):
    progress_bar.progress((i + 1) / len(TARGETS), text=f"Training {target_col}...")

    out = train_and_evaluate(df, target_col, shift_len)
    model, scaler, accuracy, prob_live, brier, oof_probs = out

    results[target_col] = {
        'model':     model,
        'scaler':    scaler,
        'accuracy':  accuracy,
        'prob_live': prob_live,
        'brier':     brier,
    }

    if oof_probs is not None:
        oof_probs_all[target_col] = oof_probs

progress_bar.empty()

# Backtest
df_bt = build_backtest_series(df, oof_probs_all, TARGETS)

oof_col = f'OOF_Prob_{target_select}'
if oof_col in df_bt.columns:
    df_bt['Signal']           = np.where(df_bt[oof_col].shift(1) > threshold_slider, 1, 0)
    df_bt['Strategy_Returns'] = df_bt['Market_Returns'] * df_bt['Signal']
else:
    df_bt['Signal']           = 0
    df_bt['Strategy_Returns'] = 0

df_bt['Cum_Market']   = (1 + df_bt['Market_Returns'].fillna(0)).cumprod()
df_bt['Cum_Strategy'] = (1 + df_bt['Strategy_Returns'].fillna(0)).cumprod()

# Metriche globali
mean_acc    = float(np.mean([r['accuracy'] for r in results.values()]))
market_perf = (df_bt['Cum_Market'].iloc[-1] - 1) * 100
strat_perf  = (df_bt['Cum_Strategy'].iloc[-1] - 1) * 100
total_trades = int(df_bt['Signal'].diff().abs().fillna(0).sum() / 2)

# =================== UI RENDERING ===================

st.markdown("---")

# Header metrics
last_price = float(df['Close'].iloc[-1])
last_date  = df.index[-1].strftime('%Y-%m-%d')
fng_now    = int(df['FNG'].iloc[-1])

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Prezzo ({ticker})",     f"${last_price:,.2f}")
col2.metric("Accuracy Media (reale)", f"{mean_acc*100:.1f}%",
            help="Accuracy dell'ensemble XGB+RF+LGB su dati out-of-sample")
col3.metric("Fear & Greed Index",     str(fng_now),
            help="Indice da API alternative.me (0=panico, 100=greed)")
col4.metric("Data",                   last_date)

st.markdown("---")

# Probabilità predittive
st.subheader("🔮 Probabilità di Rialzo (Calibrate)")

p1, p2, p3 = st.columns(3)

prob_1d = results['Target_1d']['prob_live']
prob_3d = results['Target_3d']['prob_live']
prob_5d = results['Target_5d']['prob_live']
acc_1d  = results['Target_1d']['accuracy']
acc_3d  = results['Target_3d']['accuracy']
acc_5d  = results['Target_5d']['accuracy']
brier_1d = results['Target_1d']['brier']
brier_3d = results['Target_3d']['brier']
brier_5d = results['Target_5d']['brier']

with p1:
    st.progress(min(prob_1d, 1.0))
    delta_1d = "🟢 COMPRA" if prob_1d > threshold_slider else "🔴 ASPETTA"
    st.metric(
        "Target 1 Giorno (+0.3%)",
        f"{prob_1d*100:.1f}%",
        delta=f"Acc: {acc_1d*100:.1f}% | Brier: {brier_1d:.3f}"
    )
    st.write(delta_1d)

with p2:
    st.progress(min(prob_3d, 1.0))
    delta_3d = "🟢 COMPRA" if prob_3d > threshold_slider else "🔴 ASPETTA"
    st.metric(
        "Target 3 Giorni (+1.0%)",
        f"{prob_3d*100:.1f}%",
        delta=f"Acc: {acc_3d*100:.1f}% | Brier: {brier_3d:.3f}"
    )
    st.write(delta_3d)

with p3:
    st.progress(min(prob_5d, 1.0))
    delta_5d = "🟢 COMPRA" if prob_5d > threshold_slider else "🔴 ASPETTA"
    st.metric(
        "Target 5 Giorni (+1.8%)",
        f"{prob_5d*100:.1f}%",
        delta=f"Acc: {acc_5d*100:.1f}% | Brier: {brier_5d:.3f}"
    )
    st.write(delta_5d)

st.markdown("---")

# Backtest chart
st.subheader("📈 Backtest: Strategia AI vs Buy & Hold")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Performance Strategia", f"{strat_perf:+.1f}%")
m2.metric("Performance Buy & Hold", f"{market_perf:+.1f}%")
m3.metric("Trade Totali", str(total_trades))
m4.metric("Soglia Attiva", f"{threshold_slider:.2f}")

df_chart = df_bt[['Cum_Market', 'Cum_Strategy']].dropna()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_chart.index,
    y=df_chart['Cum_Market'],
    name="Buy & Hold",
    line=dict(color='#888888', width=1.5)
))

fig.add_trace(go.Scatter(
    x=df_chart.index,
    y=df_chart['Cum_Strategy'],
    name="Strategia AI",
    line=dict(color='#00ff88', width=2)
))

fig.update_layout(
    title=f"Performance Cumulativa — {ticker}",
    xaxis_title="Data",
    yaxis_title="Crescita del Capitale (1 = partenza)",
    template="plotly_dark",
    hovermode="x unified",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)

st.plotly_chart(fig, use_container_width=True)

# Prezzo storico
st.subheader("📊 Prezzo Storico")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df.index, y=df['Close'],
    name="Close", line=dict(color='#00d4ff', width=1.5)
))

fig2.add_trace(go.Scatter(
    x=df.index, y=df['SMA_200'],
    name="SMA 200", line=dict(color='#ff9900', width=1, dash='dash')
))

fig2.add_trace(go.Scatter(
    x=df.index, y=df['SMA_50'],
    name="SMA 50", line=dict(color='#ff44aa', width=1, dash='dot')
))

fig2.update_layout(
    title=f"Prezzo + SMA — {ticker}",
    xaxis_title="Data",
    yaxis_title="Prezzo (USD)",
    template="plotly_dark",
    height=400,
    hovermode="x unified"
)

st.plotly_chart(fig2, use_container_width=True)

# Tabella metriche dettagliate
st.subheader("📋 Metriche Dettagliate per Orizzonte")

rows = []
for target_col, r in results.items():
    rows.append({
        "Orizzonte":  target_col,
        "Prob Live":  f"{r['prob_live']*100:.1f}%",
        "Accuracy":   f"{r['accuracy']*100:.1f}%",
        "Brier Score": f"{r['brier']:.4f}",
        "Segnale":    "🟢 COMPRA" if r['prob_live'] > threshold_slider else "⏸️ ASPETTA"
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.info("""
**QUANT HODL v13 FIXED** — Bug corretti rispetto all'originale:

| Bug | Originale | Corretto |
|-----|-----------|----------|
| API Fear & Greed | URL sbagliato → crash | `api.alternative.me/fng/?limit=500` ✅ |
| Data Leakage | predict_proba su tutto il df | Solo dati out-of-sample ✅ |
| Accuracy metrica | Calcolata solo su RF | Calcolata sull'ensemble reale ✅ |
| Walk-forward splits | n_splits=3 | n_splits=5 ✅ |
| UI troncata | Mancava p_col3 + grafico | UI completa ✅ |
| Brier Score | Non presente | Aggiunto (misura qualità probabilità) ✅ |
