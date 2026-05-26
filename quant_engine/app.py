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
import plotly.graph_objects as graph_objects
import warnings

warnings.filterwarnings('ignore')

# 1. CONFIGURAZIONE INTERFACCIA IMMEDIATA
st.set_page_config(page_title="QUANT HODL v13 - GENUINE ACCURACY", layout="wide")

st.title("📊 QUANT HODL v13 — AI Predictive & Backtest Suite")
st.subheader("Validazione Walk-Forward Rigorosa & Simulazione Performance Storica")

# 2. SIDEBAR DI CONTROLLO IMMEDIATA
st.sidebar.header("⚙️ Configurazione Asset")
ticker = st.sidebar.text_input("Ticker Yahoo Finance (es. BTC-USD, ETH-USD)", value="BTC-USD").upper().strip()

st.sidebar.markdown("---")
st.sidebar.header("📈 Parametri Backtesting AI")
target_select = st.sidebar.selectbox("Orizzonte AI di Riferimento", ['Target_1d', 'Target_3d', 'Target_5d'], index=0)
threshold_slider = st.sidebar.slider("Soglia di Attivazione Probabilità", min_value=0.50, max_value=0.60, value=0.53, step=0.01)

def fetch_market_data_safe(ticker_str):
    try:
        # Scarichiamo i dati base
        df = yf.download(ticker_str, period="5y", progress=False)
        if df.empty:
            return None
            
        # FIX FINALE PER LE NUOVE VERSIONI DI YFINANCE (2025/2026)
        # Se le colonne sono un MultiIndex, estraiamo solo il primo livello (Open, High, Close, ecc.)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Forziamo i nomi delle colonne in stringhe pulite con iniziale maiuscola
        df.columns = [str(c).strip().capitalize() for c in df.columns]
        
        # Rimappatura di sicurezza delle colonne
        rename_dict = {'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume', 'Adj close': 'Close'}
        df = df.rename(columns=rename_dict)
        
        # Rimuoviamo colonne duplicate o non necessarie nate dall'appiattimento
        df = df.loc[:, ~df.columns.duplicated()]
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            st.error(f"Colonne mancanti nel dataset scaricato. Trovate solo: {list(df.columns)}")
            return None
            
        df_cleaned = df[required].astype(float)
        df_cleaned.index = pd.to_datetime(df_cleaned.index).tz_localize(None).astype('datetime64[ns]')
        return df_cleaned
    except Exception as e:
        st.error(f"Errore critico durante il download da Yahoo Finance: {str(e)}")
        return None

def fetch_usd_eur_rate_safe():
    try:
        df = yf.download("USDEUR=X", period="5d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df['Close'].iloc[-1])
    except Exception:
        return 0.92

def fetch_historical_sentiment_safe():
    try:
        r = requests.get("https://alternative.me", timeout=5)
        data = r.json()['data']
        df_fng = pd.DataFrame(data)
        df_fng['timestamp'] = pd.to_datetime(df_fng['timestamp'], unit='s').dt.normalize()
        df_fng['fng_value'] = df_fng['value'].astype(float)
        df_fng.set_index('timestamp', inplace=True)
        df_fng.index = pd.to_datetime(df_fng.index).tz_localize(None).astype('datetime64[ns]')
        return df_fng['fng_value'].sort_index()
    except Exception:
        return pd.Series(dtype=float)

# Esecuzione della Pipeline con feedback immediato a schermo
if ticker:
    df_raw = fetch_market_data_safe(ticker)
    eur_usd_rate = fetch_usd_eur_rate_safe()
    series_fng = fetch_historical_sentiment_safe()
    
    st.sidebar.markdown(f"**Cambio USD/EUR Corrente:** {eur_usd_rate:.4f}")

    if df_raw is None or len(df_raw) < 300:
        st.error("❌ Storico dati insufficiente o errore strutturale nei dati scaricati. Verifica il Ticker inserito.")
    else:
        # --- FEATURE ENGINEERING ---
        df = df_raw.copy()
        close = df['Close']
        high  = df['High']
        low   = df['Low']
        vol   = df['Volume']

        df['SMA_50']  = close.rolling(50).mean()
        df['SMA_200'] = close.rolling(200).mean()
        std200 = close.rolling(200).std().replace(0, 1e-9)
        df['Z_Score'] = (close - df['SMA_200']) / std200
        df['Above_SMA50']  = (close > df['SMA_50']).astype(int)
        df['Above_SMA200'] = (close > df['SMA_200']).astype(int)

        # Correzione definitiva della sintassi dei cicli del momentum
        for p in [5, 10, 21]:
            df[f'Mom_{p}d'] = close.pct_change(p)

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = delta.clip(upper=0).abs().rolling(14).mean().replace(0, 1e-9)
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = macd_line - signal_line

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std().replace(0, 1e-9)
        df['BB_pct'] = (close - (sma20 - 2 * std20)) / (4 * std20)
        df['ATR'] = (high - low).rolling(14).mean()
        df['ATR_Norm'] = df['ATR'] / close
        df['Vol_Mom_20d'] = vol.pct_change(20).fillna(0)
        vol_sma20 = vol.rolling(20).mean().replace(0, 1e-9)
        df['Vol_Ratio'] = vol / vol_sma20

        if not series_fng.empty:
            fng_aligned = series_fng.reindex(df.index, method='ffill').fillna(50)
            df['FNG'] = fng_aligned.shift(1).fillna(50)
        else:
            df['FNG'] = 50

        df = df.dropna()

        df['Target_1d'] = (df['Close'].shift(-1) > df['Close'] * 1.003).astype(int)
        df['Target_3d'] = (df['Close'].shift(-3) > df['Close'] * 1.010).astype(int)
        df['Target_5d'] = (df['Close'].shift(-5) > df['Close'] * 1.018).astype(int)

        # --- CORE ADDESTRAMENTO E PREDIZIONE ---
        FEATURE_COLS = ['Z_Score', 'RSI', 'ATR_Norm', 'FNG', 'Vol_Mom_20d', 'Mom_5d', 'Mom_10d', 'Mom_21d', 'MACD_Hist', 'BB_pct', 'Above_SMA50', 'Above_SMA200', 'Vol_Ratio']
        TARGETS = {'Target_1d': 1, 'Target_3d': 3, 'Target_5d': 5}
        
        today_row = df[FEATURE_COLS].iloc[[-1]]
        probabilities = {}
        accuracies = {}
        df_backtest = df.copy()

        for target_col, shift_len in TARGETS.items():
            X_clean = df[FEATURE_COLS].iloc[:-shift_len]
            y_clean = df[target_col].iloc[:-shift_len]

            tscv = TimeSeriesSplit(n_splits=3)
            fold_scores = []
            for train_idx, test_idx in tscv.split(X_clean):
                if len(train_idx) < 50: continue
                X_tr, y_tr = X_clean.iloc[train_idx], y_clean.iloc[train_idx]
                X_te, y_te = X_clean.iloc[test_idx], y_clean.iloc[test_idx]
                
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)
                
                rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
                rf.fit(X_tr_s, y_tr)
                fold_scores.append(accuracy_score(y_te, rf.predict(X_te_s)))
            
            accuracies[target_col] = float(np.mean(fold_scores)) if fold_scores else 0.52

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            xgb_m = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0, random_state=42)
            rf_m = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
            lgb_m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=-1, random_state=42)
            
            ensemble = VotingClassifier(estimators=[('xgb', xgb_m), ('rf', rf_m), ('lgb', lgb_m)], voting='soft')
            calibrated = CalibratedClassifierCV(estimator=ensemble, method='sigmoid', cv=2)
            calibrated.fit(X_scaled, y_clean)

            df_backtest[f'Prob_{target_col}'] = calibrated.predict_proba(scaler.transform(df[FEATURE_COLS]))[:, 1]
            probabilities[target_col] = float(calibrated.predict_proba(scaler.transform(today_row))[:, 1])

        # --- ENGINE BACKTEST ---
        df_backtest['Market_Returns'] = df_backtest['Close'].pct_change()
        prob_col = f'Prob_{target_select}'
        df_backtest['Signal'] = np.where(df_backtest[prob_col].shift(1) > threshold_slider, 1, 0)
        df_backtest['Strategy_Returns'] = df_backtest['Market_Returns'] * df_backtest['Signal']
        
        df_backtest['Cum_Market'] = (1 + df_backtest['Market_Returns'].fillna(0)).cumprod()
        df_backtest['Cum_Strategy'] = (1 + df_backtest['Strategy_Returns'].fillna(0)).cumprod()
        
        total_trades = int(df_backtest['Signal'].diff().abs().sum() / 2)
        market_perf = (df_backtest['Cum_Market'].iloc[-1] - 1) * 100
        strat_perf = (df_backtest['Cum_Strategy'].iloc[-1] - 1) * 100
        mean_acc = float(np.mean(list(accuracies.values())))

        # --- RENDERING FINALE INTERFACCIA ---
        st.markdown("---")
        last_close_usd = float(df['Close'].iloc[-1])
        last_date = df.index[-1].strftime('%Y-%m-%d')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label=f"Ultimo Prezzo ({ticker})", value=f"\${last_close_usd:,.2f}")
        with col2:
            st.metric(label="Accuratezza Modello Media", value=f"{mean_acc * 100:.1f}%")
