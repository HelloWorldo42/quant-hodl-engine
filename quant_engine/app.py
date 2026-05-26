                verbosity=-1, random_state=42
            )

            ensemble = VotingClassifier(
                estimators=[('xgb', xgb_m), ('rf', rf_m), ('lgb', lgb_m)],
                voting='soft'
            )

            # Calibrazione e addestramento finale
            calibrated_ensemble = CalibratedClassifierCV(estimator=ensemble, method='sigmoid', cv=3)
            calibrated_ensemble.fit(X_scaled, y_clean)

            # Calcolo del Brier Score storico
            preds_prob = calibrated_ensemble.predict_proba(X_scaled)[:, 1]
            brier_scores[target_col] = float(brier_score_loss(y_clean, preds_prob))

            # Predizione live sulla riga di oggi
            today_scaled = scaler.transform(today_row)
            probabilities[target_col] = float(calibrated_ensemble.predict_proba(today_scaled)[:, 1])

        mean_accuracy = float(np.mean(list(accuracies.values()))) if accuracies else 0.5

        return df, mean_accuracy, probabilities, accuracies, brier_scores

# =====================================================================
# 6. INTERFACCIA UTENTE (STREAMLIT UI)
# =====================================================================
def main():
    st.title("📊 QUANT HODL v13 — AI Predictive Suite")
    st.subheader("Validazione Walk-Forward Rigorosa & Probabilità di Mercato Calibrate")

    st.sidebar.header("⚙️ Configurazione Asset")
    ticker = st.sidebar.text_input("Ticker Yahoo Finance (es. BTC-USD, ETH-USD)", value="BTC-USD").upper().strip()

    eur_usd_rate = fetch_usd_eur_rate()
    st.sidebar.markdown(f"**Cambio USD/EUR Corrente:** {eur_usd_rate:.4f}")

    if not ticker:
        st.warning("Inserisci un ticker valido per iniziare l'analisi quantitativa.")
        return

    with st.spinner(f"Estrazione dati e calibrazione modelli matematici per {ticker}..."):
        res = MacroPredictiveCore.compile_and_validate_multi(ticker)
        
    if res[0] is None:
        st.error("Errore nel caricamento dei dati o storico insufficiente (richiesti almeno 300 giorni).")
        return

    df, mean_acc, probs, accs, briers = res

    last_close_usd = float(df['Close'].iloc[-1])
    last_close_eur = last_close_usd * eur_usd_rate
    last_date = df.index[-1].strftime('%Y-%m-%d')

    # Row 1: KPI Principali
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=f"Ultimo Prezzo ({ticker})", value=f"${last_close_usd:,.2f}", delta=f"€{last_close_eur:,.2f} EUR")
    with col2:
        st.metric(label="Accuratezza Genuina Media", value=f"{mean_acc * 100:.2f}%", help="Calcolata tramite TimeSeriesSplit (Walk-Forward).")
    with col3:
        st.metric(label="Fear & Greed Index (Ieri)", value=int(df['FNG'].iloc[-1]), delta="Alimentato da Alternative.me")
    with col4:
        st.metric(label="Data Ultimo Aggiornamento", value=last_date)

    st.markdown("---")

    # Row 2: Tabellone Predittivo
    st.write("### 🔮 Tabellone Predittivo Direzionale (Probabilità di Rialzo Calibrate)")
    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        st.progress(probs['Target_1d'])
        st.metric(label="Target 1 Giorno (+0.3%)", value=f"{probs['Target_1d']*100:.1f}% prob.", 
                  delta=f"Accuratezza: {accs['Target_1d']*100:.1f}% (Brier: {briers['Target_1d']:.3f})")

    with p_col2:
        st.progress(probs['Target_3d'])
        st.metric(label="Target 3 Giorni (+1.0%)", value=f"{probs['Target_3d']*100:.1f}% prob.", 
                  delta=f"Accuratezza: {accs['Target_3d']*100:.1f}% (Brier: {briers['Target_3d']:.3f})")

    with p_col3:
        st.progress(probs['Target_5d'])
        st.metric(label="Target 5 Giorni (+1.8%)", value=f"{probs['Target_5d']*100:.1f}% prob.", 
                  delta=f"Accuratezza: {accs['Target_5d']*100:.1f}% (Brier: {briers['Target_5d']:.3f})")

    st.markdown("---")

    # Row 3: Matrice Feature Recenti
    st.write("### 🛠️ Matrice delle Feature Quantitative Recenti (Ultime 5 Sessioni)")
    display_cols = ['Close', 'Z_Score', 'RSI', 'MACD_Hist', 'BB_pct', 'ATR_Norm', 'Vol_Ratio', 'FNG']
    st.dataframe(df[display_cols].tail(5).style.format({
        'Close': '${:,.2f}', 'Z_Score': '{:.2f}', 'RSI': '{:.1f}',
        'MACD_Hist': '{:.2f}', 'BB_pct': '{:.2f}', 'ATR_Norm': '{:.4f}',
        'Vol_Ratio': '{:.2f}', 'FNG': '{:.0f}'
    }), use_container_width=True)

    # Row 4: Analisi Logica
    st.write("### 🧠 Analisi Logica del Regime di Mercato")
    exp1, exp2 = st.columns(2)
    with exp1:
        st.markdown(f"""
        **Stato dei Trend di Fondo:**
        * Prezzo vs SMA 50: {"🟢 Sopra (Rialzista)" if df['Above_SMA50'].iloc[-1] == 1 else "🔴 Sotto (Ribassista)"}
        * Prezzo vs SMA 200: {"🟢 Sopra (Lungo Termine)" if df['Above_SMA200'].iloc[-1] == 1 else "🔴 Sotto (Lungo Termine)"}
        * Deviazione Standard (Z-Score 200d): **{df['Z_Score'].iloc[-1]:.2f}**
        """)
    with exp2:
        st.markdown(f"""
        **Analisi degli Oscillatori & Volatilità:**
        * Relative Strength Index (RSI): **{df['RSI'].iloc[-1]:.1f}** ({'Ipercomprato' if df['RSI'].iloc[-1] > 70 else 'Ipervenduto' if df['RSI'].iloc[-1] < 30 else 'Neutrale'})
        * Posizione Bande Bollinger (%B): **{df['BB_pct'].iloc[-1]:.2f}**
        * Volatilità Relativa (ATR Norm): **{df['ATR_Norm'].iloc[-1]*100:.2f}%** del prezzo spot.
        """)

if __name__ == '__main__':
    main()
