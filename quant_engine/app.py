import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURAZIONE INTERFACCIA
# =====================================================================
st.set_page_config(page_title="QUANT HODL v12.1 - PURIFIED ENGINE", layout="wide")

# =====================================================================
# 2. SEZIONE API & DATA EXTRACTION (CON CACHING)
# =====================================================================
@st.cache_data(ttl=3600)
def fetch_market_data(ticker):
    try:
        df = yf.download(ticker, period="3y", progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None).astype('datetime64[ns]')
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_usd_eur_rate():
    try:
        df = yf.download("USDEUR=X", period="5d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df['Close'].iloc[-1])
    except Exception:
        return 0.92  # Fallback approssimativo in caso di errore API

@st.cache_data(ttl=3600)
def fetch_historical_sentiment():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=100", timeout=5)
        data = r.json()['data']
        df_fng = pd.DataFrame(data)
        df_fng['timestamp'] = pd.to_datetime(df_fng['timestamp'], unit='s')
        df_fng['fng_value'] = df_fng['value'].astype(float)
        df_fng.set_index('timestamp', inplace=True)
        df_fng.index = pd.to_datetime(df_fng.index).tz_localize(None).astype('datetime64[ns]')
        return df_fng['fng_value'].sort_index()
    except Exception:
        return pd.Series(dtype=float)

# =====================================================================
# 3. ADVANCED FEATURE ENGINEERING (TECNICO + MACRO + SENTIMENT)
# =====================================================================
class MacroFeatureEngineer:
    @staticmethod
    def construct_matrix(df_price, series_fng):
        df = df_price.copy()
        close = df['Close']
        
        df['SMA_200'] = close.rolling(200).mean()
        std200 = close.rolling(200).std().replace(0, 1e-9)
        df['Z_Score'] = (close - df['SMA_200']) / std200
        
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = delta.clip(upper=0).abs().rolling(14).mean().replace(0, 1e-9)
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        
        if not series_fng.empty:
            df['FNG_Feature'] = series_fng.reindex(df.index, method='ffill').fillna(50)
        else:
            df['FNG_Feature'] = 50
            
        df['Macro_Volume_Momentum'] = df['Volume'].pct_change(20).fillna(0)
        
        df = df.dropna()
        df['Target'] = (df['Close'].shift(-5) > df['Close'] * 1.025).astype(int)
        
        return df

# =====================================================================
# 4. ENGINE DI TRAINING E CONVALIDA WALK-FORWARD
# =====================================================================
class MacroPredictiveCore:
    @staticmethod
    @st.cache_resource
    def compile_and_validate(ticker):
        raw_price = fetch_market_data(ticker)
        series_fng = fetch_historical_sentiment()
        
        if raw_price is None or len(raw_price) < 250:
            return None, 0.0, 0.0, None
            
        df = MacroFeatureEngineer.construct_matrix(raw_price, series_fng)
        if df.empty:
            return None, 0.0, 0.0, None
            
        train_df = df.iloc[:-5] 
        today_features = df.iloc[[-1]] 
        
        feature_cols = ['Z_Score', 'RSI', 'ATR', 'FNG_Feature', 'Macro_Volume_Momentum']
        X = train_df[feature_cols]
        y = train_df['Target']
        
        n_splits = 5
        split_size = len(X) // (n_splits + 1)
        accuracies = []
        
        for i in range(n_splits):
            train_end = (i + 1) * split_size
            test_end = train_end + min(split_size, len(X) - train_end)
            if test_end > len(X): break
                
            X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
            X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            
            xgb_m = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0, random_state=42)
            lgb_m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
            rf_m = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
            
            ensemble = VotingClassifier(estimators=[('xgb', xgb_m), ('lgb', lgb_m), ('rf', rf_m)], voting='soft')
            ensemble.fit(X_tr_s, y_tr)
            accuracies.append(accuracy_score(y_te, ensemble.predict(X_te_s)))
            
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X)
        final_ensemble = VotingClassifier(estimators=[
            ('xgb', xgb.XGBClassifier(n_estimators=60, max_depth=3, verbosity=0, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=60, max_depth=3, verbose=-1, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=-1))
        ], voting='soft')
        final_ensemble.fit(X_scaled, y)
        
        latest_vector = final_scaler.transform(today_features[feature_cols])
        prob_up = final_ensemble.predict_proba(latest_vector)[0][1]
        
        return df, np.mean(accuracies), prob_up, today_features

# =====================================================================
# 5. LOGICA OPERATIVA COMBINATA (ACCUMULO + ASSET REBALANCING)
# =====================================================================
def calculate_hodl_matrix(z_score, prob_up, base_quota):
    if z_score > 2.3:
        mult = 0.0; status = "🛑 STOP ACCUMULO (Rischio Bolla Estrema)"
        sell_action = "🚨 VENDITA FRAZIONATA: Vendi il 20% della posizione per incassare profitto."
    elif z_score > 1.2:
        mult = 0.4; status = "⚠️ DISTRIBUZIONE (Rallenta gradualmente)"
        sell_action = "💵 ALLEGGERIMENTO: Valuta di liquidare il 10% se hai sovraesposizione."
    elif z_score < -1.6:
        mult = 2.5; status = "🔥 SUPER SCONTO (Massimo Accumulo Storico)"
        sell_action = "💎 STRONG HODL (Vietato vendere)"
    elif z_score < -0.7:
        mult = 1.6; status = "📈 ACQUISTO CON VANTAGGIO (Prezzo Sotto Valore)"
        sell_action = "💎 STRONG HODL (Vietato vendere)"
    else:
        mult = 1.0; status = "⚖️ BILANCIAMENTO STANDARD (Prezzo di Equilibrio)"
        sell_action = "💎 HODL"
        
    if prob_up > 0.60 and mult > 0:
        mult *= 1.25
    elif prob_up < 0.40 and mult > 0:
        mult *= 0.5
        
    return round(base_quota * mult, 2), status, sell_action

# =====================================================================
# 6. DASHBOARD PRINCIPALE
# =====================================================================
def main():
    st.title("🎯 QUANT HODL ENGINE v12.1")
    st.subheader("Architettura Purificata con Doppia Valuta USD/EUR")
    st.divider()
    
    base_quota = st.sidebar.number_input("Quota PAC Base (€)", min_value=10, value=100, step=10)
    
    ticker_input = st.sidebar.text_input("Inserisci i Ticker separati da virgola", value="BTC-USD, ETH-USD, AAPL, NVDA")
    assets = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
    
    # Recupera il tasso di cambio live USD -> EUR
    usd_eur_rate = fetch_usd_eur_rate()
    
    if not assets:
        st.info("Inserisci almeno un ticker valido nella barra laterale (es: BTC-USD, TSLA).")
        return
        
    for asset in assets:
        df, acc, prob_up, today_features = MacroPredictiveCore.compile_and_validate(asset)
        if df is None or today_features is None or today_features.empty: 
            continue
        
        z_now = today_features['Z_Score'].iloc[-1]
        price_now = today_features['Close'].iloc[-1]
        target_quota, state, sell_instruction = calculate_hodl_matrix(z_now, prob_up, base_quota)
        
        # Gestione intelligente della valuta visiva
        if "EUR" in asset:
            price_display = f"{price_now:,.2f} €"
        else:
            price_eur = price_now * usd_eur_rate
            price_display = f"${price_now:,.2f} ({price_eur:,.2f} €)"
        
        with st.expander(f"🔮 MATRICE QUANTITATIVA: {asset}", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Prezzo Attuale", price_display)
            col2.metric("Accuratezza Predittiva Reale (WF)", f"{acc * 100:.1f}%")
            col3.metric("Confidenza Algoritmo (Prob Up)", f"{prob_up * 100:.1f}%")
            
            c_box1, c_box2 = st.columns(2)
            with c_box1:
                st.info(f"**Strategia d'Ingresso:**\n{state}\n\n**PAC Dinamico:** **{target_quota} €**")
            with c_box2:
                st.warning(f"**Strategia d'Uscita (Rebalancing):**\n{sell_instruction}")

if __name__ == "__main__":
    main()
