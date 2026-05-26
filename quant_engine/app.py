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
st.set_page_config(page_title="QUANT HODL v12.5 - MULTI-TIMEFRAME ENGINE", layout="wide")

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
        return 0.92

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
# 3. ADVANCED MULTI-TARGET FEATURE ENGINEERING
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
        
        # Tre Target distinti: 1 giorno, 3 giorni, 5 giorni
        df['Target_1d'] = (df['Close'].shift(-1) > df['Close'] * 1.005).astype(int)
        df['Target_3d'] = (df['Close'].shift(-3) > df['Close'] * 1.015).astype(int)
        df['Target_5d'] = (df['Close'].shift(-5) > df['Close'] * 1.025).astype(int)
        
        return df

# =====================================================================
# 4. MULTI-TIMEFRAME PREDICTIVE CORE
# =====================================================================
class MacroPredictiveCore:
    @staticmethod
    @st.cache_resource
    def compile_and_validate_multi(ticker):
        raw_price = fetch_market_data(ticker)
        series_fng = fetch_historical_sentiment()
        
        if raw_price is None or len(raw_price) < 250:
            return None, 0.0, {}, None
            
        df = MacroFeatureEngineer.construct_matrix(raw_price, series_fng)
        if df.empty:
            return None, 0.0, {}, None
            
        feature_cols = ['Z_Score', 'RSI', 'ATR', 'FNG_Feature', 'Macro_Volume_Momentum']
        today_features = df.iloc[[-1]]
        
        # Validazione e predizione per i 3 orizzonti temporali
        probabilities = {}
        accuracies = []
        
        for horizon, target_col in [("1d", "Target_1d"), ("3d", "Target_3d"), ("5d", "Target_5d")]:
            train_df = df.iloc[:-5] # Evita data leakage
            X = train_df[feature_cols]
            y = train_df[target_col]
            
            # Singolo split walk-forward veloce per stabilità
            split_idx = int(len(X) * 0.8)
            X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
            X_te, y_te = X.iloc[split_idx:], y.iloc[split_idx:]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            
            xgb_m = xgb.XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.05, verbosity=0, random_state=42)
            rf_m = RandomForestClassifier(n_estimators=40, max_depth=4, random_state=42, n_jobs=-1)
            
            ensemble = VotingClassifier(estimators=[('xgb', xgb_m), ('rf', rf_m)], voting='soft')
            ensemble.fit(X_tr_s, y_tr)
            accuracies.append(accuracy_score(y_te, ensemble.predict(X_te_s)))
            
            # Predizione finale
            final_scaler = StandardScaler()
            X_scaled = final_scaler.fit_transform(X)
            final_ensemble = VotingClassifier(estimators=[('xgb', xgb_m), ('rf', rf_m)], voting='soft')
            final_ensemble.fit(X_scaled, y)
            
            latest_vector = final_scaler.transform(today_features[feature_cols])
            probabilities[horizon] = final_ensemble.predict_proba(latest_vector)[0][1]
            
        return df, np.mean(accuracies), probabilities, today_features

# =====================================================================
# 5. LOGICA OPERATIVA COMBINATA CON IA MULTI-TEMPORALE
# =====================================================================
def calculate_hodl_matrix(z_score, probs, base_quota):
    # Base deterministica dallo Z-Score
    if z_score > 2.3:
        mult = 0.0
        status = ":red[🛑 NON COMPRARE (Prezzo Troppo Alto)]"
        sell_action = "🚨 **VENDI IL 20%** per incassare profitto."
    elif z_score > 1.2:
        mult = 0.4
        status = ":orange[⚠️ COMPRA POCO (Prezzo Alto)]"
        sell_action = "💵 **ALLEGGERISCI**: Valuta di ridurre se hai troppa esposizione."
    elif z_score < -0.7:
        mult = 1.6
        status = ":green[🔥 COMPRA COMPRA (Forte Sconto)]"
        sell_action = "💎 **HODL** (Vietato vendere)"
    else:
        mult = 1.0
        status = ":green[⚖️ COMPRA (Accumulo Standard)]"
        sell_action = "💎 **HODL**"
        
    # Calcolo della spinta aggregata (Media pesata o trend)
    avg_prob = np.mean([probs['1d'], probs['3d'], probs['5d']])
    
    if avg_prob > 0.58 and mult > 0:
        mult *= 1.25
    elif avg_prob < 0.42 and mult > 0:
        mult *= 0.5
        
    # Definizione del testo del Trend visivo
    if probs['1d'] > 0.52 and probs['5d'] > 0.52:
        trend_label = "🟢 RIALZO COMPATTO"
    elif probs['1d'] < 0.45 and probs['5d'] < 0.45:
        trend_label = "🔴 RIBASSO COMPATTO"
    else:
        trend_label = "🟡 LATERALE / INCERTO"
        
    return round(base_quota * mult, 2), status, sell_action, trend_label

# =====================================================================
# 6. DASHBOARD PRINCIPALE
# =====================================================================
def main():
    st.title("🎯 QUANT HODL ENGINE v12.5")
    st.subheader("Analisi Quantitativa Integrata su Orizzonti Organizzati (1d, 3d, 5d)")
    st.divider()
    
    base_quota = st.sidebar.number_input("Quota PAC Base (€)", min_value=10, value=100, step=10)
    ticker_input = st.sidebar.text_input("Inserisci i Ticker separati da virgola", value="BTC-USD, TSLA, AAPL")
    assets = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
    
    usd_eur_rate = fetch_usd_eur_rate()
    
    if not assets:
        st.info("Inserisci almeno un ticker valido nella barra laterale.")
        return
        
    summary_data = []
        
    for asset in assets:
        df, acc, probs, today_features = MacroPredictiveCore.compile_and_validate_multi(asset)
        
        if df is None or today_features is None or today_features.empty: 
            st.sidebar.error(f"❌ Impossibile caricare '{asset}'.")
            continue
        
        z_now = today_features['Z_Score'].iloc[-1]
        price_now = today_features['Close'].iloc[-1]
        
        target_quota, state, sell_instruction, trend_label = calculate_hodl_matrix(z_now, probs, base_quota)
        
        summary_data.append({"Asset": asset, "Quota da Comprare": target_quota, "Stato": state.split("]")[0].split("[")[-1] if "[" in state else state})
        
        if "EUR" in asset:
            price_display = f"{price_now:,.2f} €"
        else:
            price_eur = price_now * usd_eur_rate
            price_display = f"${price_now:,.2f} ({price_eur:,.2f} €)"
        
        # Stringa di aiuto per il popup del mouse
        help_string = f"Dettaglio Probabilità:\n• 1 Giorno: {probs['1d']*100:.1f}%\n• 3 Giorni: {probs['3d']*100:.1f}%\n• 5 Giorni: {probs['5d']*100:.1f}%"
        
        with st.expander(f"🔮 SEGNO CORRENTE: {asset}", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Prezzo Attuale", price_display)
            col2.metric("Affidabilità Motore (Media)", f"{acc * 100:.1f}%")
            col3.metric("Trend IA Multi-Timeframe", trend_label, help=help_string)
            
            c_box1, c_box2 = st.columns(2)
            with c_box1:
                st.markdown(f"### **Cosa fare oggi:**\n## {state}")
                st.markdown(f"### 💵 **Quota da investire nel PAC:** **{target_quota} €**")
            with c_box2:
                st.markdown("### **Se hai bisogno di ribilanciare:**")
                st.info(f"{sell_instruction}")
                
    if summary_data:
        st.divider()
        st.markdown("## 📊 ORDINE DI ACQUISTO COMPLESSIVO")
        
        df_summary = pd.DataFrame(summary_data)
        total_pac = df_summary["Quota da Comprare"].sum()
        
        col_table, col_total = st.columns([2, 1])
        with col_table:
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
        with col_total:
            st.metric(label="💰 BUDGET TOTALE RICHIESTO", value=f"{total_pac:,.2f} €")
            st.caption("Esegui gli ordini sul tuo broker usando queste quote esatte.")

if __name__ == "__main__":
    main()
