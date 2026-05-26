from textwrap import dedent

fixed = dedent("""
# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

import streamlit as st
import numpy as np
import pandas as pd
import requests
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit


# ─────────────────────────────────────────────
# SAFE INIT (evita crash)
# ─────────────────────────────────────────────

df = globals().get("df", pd.DataFrame())
close = df.get("Close") if "Close" in df.columns else None

FEATURE_COLS = globals().get("FEATURE_COLS", [])


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────

@st.cache_data(ttl=21600)
def fetch_fear_greed_history():
    try:
        url = "https://api.alternative.me/fng/?limit=500&format=json"
        r = requests.get(url, timeout=8)
        r.raise_for_status()

        data = r.json().get("data", [])
        if not data:
            return pd.Series(dtype=float)

        df_fng = pd.DataFrame(data)

        df_fng["timestamp"] = pd.to_datetime(
            df_fng["timestamp"].astype(int),
            unit="s"
        ).dt.normalize()

        df_fng["fng_value"] = df_fng["value"].astype(float)

        df_fng = df_fng.set_index("timestamp").sort_index()
        df_fng.index = pd.to_datetime(df_fng.index).tz_localize(None)

        return df_fng["fng_value"]

    except Exception:
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (SAFE)
# ─────────────────────────────────────────────

if not df.empty and close is not None:

    df["Trend_Regime"] = np.where(
        df["SMA_50"] > df["SMA_200"],
        1,
        0
    )

    df["Vol_Regime"] = (
        df["Volatility_20d"] >
        df["Volatility_20d"].rolling(100).mean()
    ).astype(int)

    rolling_ath = close.cummax()
    rolling_max = close.cummax()

    df["ATH_Distance"] = (close / rolling_ath) - 1
    df["Drawdown"] = (close / rolling_max) - 1


# ─────────────────────────────────────────────
# ENSEMBLE
# ─────────────────────────────────────────────

def build_ensemble():

    xgb_m = xgb.XGBClassifier(
        n_estimators=40,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        random_state=42,
    )

    rf_m = RandomForestClassifier(
        n_estimators=40,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    return StackingClassifier(
        estimators=[("xgb", xgb_m), ("rf", rf_m)],
        final_estimator=LogisticRegression(),
        stack_method="predict_proba",
        cv=3,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────
# SAFE AUC
# ─────────────────────────────────────────────

def safe_auc(y_te=None, proba=None):
    try:
        if y_te is None or proba is None:
            return 0.5
        return roc_auc_score(y_te, proba[:, 1])
    except Exception:
        return 0.5


# ─────────────────────────────────────────────
# DYNAMIC THRESHOLD
# ─────────────────────────────────────────────

if "Trend_Regime" in df.columns and not df.empty:
    trend_regime = df["Trend_Regime"].iloc[-1]
    dynamic_threshold = 0.52 if trend_regime == 1 else 0.58
else:
    dynamic_threshold = 0.55


# ─────────────────────────────────────────────
# BACKTEST SAFE
# ─────────────────────────────────────────────

if "df_bt" in globals():

    fee = 0.001

    df_bt["Strategy_Returns"] = (
        df_bt["Market_Returns"] * df_bt["Signal"]
    )

    df_bt["Strategy_Returns"] -= (
        fee * df_bt["Signal"].diff().abs().fillna(0)
    )


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────

threshold_slider = dynamic_threshold
""")

path = "/mnt/data/fixed_streamlit_patch.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(fixed)

print(path)
