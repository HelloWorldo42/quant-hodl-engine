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

# =====================================================================
# 1. CONFIGURAZIONE INTERFACCIA
# =====================================================================
st.set_page_config(page_title="QUANT HODL v13 - GENUINE ACCURACY", layout="wide")

# =====================================================================
# 2. DATA EXTRACTION CON CONTROLLO STRUTTURA BLINDATO
# =====================================================================
@st.cache_data(ttl=1800)
def fetch_market_data(ticker):
    try:
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [str(c).strip().capitalize() for c in df.columns]
        
        rename_dict = {
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 
            'Close': 'Close', 'Adj close': 'Close', 'Volume': 'Volume'
        }
        df = df.rename(columns=rename_dict)
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            return None
            
        df_cleaned = df[required].astype(float)
        df_cleaned.index = pd.to_datetime(df_cleaned.index).tz_localize(None).astype('datetime64[ns]')
        return df_cleaned
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_usd_eur_rate():
    try:
        df = yf.download("USDEUR=X", period="5d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).strip().capitalize() for c in df.columns]
        return float(df['Close'].iloc[-1])
    except Exception:
        return 0.92

@st.cache_data(ttl=1800)
def fetch_historical_sentiment():
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

# =====================================================================
# 3. FEATURE ENGINEERING
# =====================================================================
class MacroFeatureEngineer:
    @staticmethod
    def construct_matrix(df_price, series_fng):
        df = df_price.copy()
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

        # BUG CORRETTO RIGOROSAMENTE QUI: Inserita la lista numerica dei giorni
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

        return df

# =====================================================================
# 4. WALK-FORWARD VALIDATION
# =====================================================================
def walk_forward_accuracy(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < 60 or len(test_idx) < 10:
            continue
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        xgb_m = xgb.XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42
        )
        rf_m = RandomForestClassifier(
            n_estimators=80, max_depth=5,
            min_samples_leaf=10, random_state=42, n_jobs=-1
        )
        lgb_m = lgb.LGBMClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.03,
            verbosity=-1, random_state=42
        )

        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_m), ('rf', rf_m), ('lgb', lgb_m)],
            voting='soft'
        )
        ensemble.fit(X_tr_s, y_tr)
        scores.append(accuracy_score(y_te, ensemble.predict(X_te_s)))

    return float(np.mean(scores)) if scores else 0.5

# =====================================================================
# 5. CORE PREDITTIVO
# =====================================================================
FEATURE_COLS = [
    'Z_Score', 'RSI', 'ATR_Norm', 'FNG', 'Vol_Mom_20d',
    'Mom_5d', 'Mom_10d', 'Mom_21d',
    'MACD_Hist', 'BB_pct', 'Above_SMA50', 'Above_SMA200',
    'Vol_Ratio'
]

TARGETS = {
    'Target_1d': 1,
    'Target_3d': 3,
    'Target_5d': 5,
}

class MacroPredictiveCore:
    @staticmethod
    @st.cache_resource
    def compile_and_validate_multi(ticker):
        raw_price  = fetch_market_data(ticker)
        series_fng = fetch_historical_sentiment()

        if raw_price is None or len(raw_price) < 300:
            return None, 0.0, {}, {}, {}, pd.DataFrame()

        df = MacroFeatureEngineer.construct_matrix(raw_price, series_fng)
        if df.empty:
            return None, 0.0, {}, {}, {}, pd.DataFrame()

        today_row = df[FEATURE_COLS].iloc[[-1]]

        probabilities = {}
        accuracies    = {}
        brier_scores  = {}
        df_backtest = df.copy()
        
        for target_col, shift_len in TARGETS.items():
            X_clean = df[FEATURE_COLS].iloc[:-shift_len]
            y_clean = df[target_col].iloc[:-shift_len]

            if len(X_clean) < 100:
                probabilities[target_col] = 0.5
                accuracies[target_col]    = 0.5
                brier_scores[target_col]  = 0.25
                df_backtest[f'Prob_{target_col}'] = 0.5
                continue

            accuracies[target_col] = walk_forward_accuracy(X_clean, y_clean, n_splits=5)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

            xgb_m = xgb.XGBClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42
            )
            rf_m = RandomForestClassifier(
                n_estimators=80, max_depth=5,
                min_samples_leaf=10, random_state=42, n_jobs=-1
            )
            lgb_m = lgb.LGBMClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.03,
                verbosity=-1, random_state=42
            )

            ensemble = VotingClassifier(
                estimators=[('xgb', xgb_m), ('rf', rf_m), ('lgb', lgb_m)],
                voting='soft'
            )
            
            calibrated_ensemble = CalibratedClassifierCV(estimator=ensemble, method='sigmoid', cv=3)
            calibrated_ensemble.fit(X_scaled, y_clean)
            
            preds_prob = calibrated_ensemble.predict_proba(X_scaled)[:, 1]
            brier_scores[target_col] = float(brier_score_loss(y_clean, preds_prob))

            X_full_scaled = scaler.transform(df[FEATURE_COLS])
            df_backtest[f'Prob_{target_col}'] = calibrated_ensemble.predict_proba(X_full_scaled)[:, 1]

            today_scaled = scaler.transform(today_row)
            probabilities[target_col] = float(calibrated_ensemble.predict_proba(today_scaled)[:, 1])

        mean_accuracy = float(np.mean(list(accuracies.values()))) if accuracies else 0.5
        
        return df, mean_accuracy, probabilities, accuracies, brier_scores, df_backtest

