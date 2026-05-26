# ─────────────────────────────────────────────
# IMPORT
# ─────────────────────────────────────────────

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────
# CACHE FNG
# ─────────────────────────────────────────────

@st.cache_data(ttl=21600)
def fetch_fear_greed_history():
    pass


# ─────────────────────────────────────────────
# FEATURE EXTRA
# ─────────────────────────────────────────────

df["Trend_Regime"] = np.where(
    df["SMA_50"] > df["SMA_200"], 1, 0
)

df["Vol_Regime"] = (
    df["Volatility_20d"] >
    df["Volatility_20d"].rolling(100).mean()
).astype(int)

rolling_ath = close.cummax()
df["ATH_Distance"] = close / rolling_ath - 1

rolling_max = close.cummax()
df["Drawdown"] = close / rolling_max - 1


# ─────────────────────────────────────────────
# FIX TARGET LEAKAGE
# ─────────────────────────────────────────────

df["Target_1d"] = np.where(
    df["Close"].shift(-1).notna(),
    (df["Close"].shift(-1) > df["Close"] * 1.003).astype(float),
    np.nan
)

df["Target_3d"] = np.where(
    df["Close"].shift(-3).notna(),
    (df["Close"].shift(-3) > df["Close"] * 1.010).astype(float),
    np.nan
)

df["Target_5d"] = np.where(
    df["Close"].shift(-5).notna(),
    (df["Close"].shift(-5) > df["Close"] * 1.018).astype(float),
    np.nan
)


# ─────────────────────────────────────────────
# FEATURE COLS
# ─────────────────────────────────────────────

FEATURE_COLS += [
    "Trend_Regime",
    "Vol_Regime",
    "ATH_Distance",
    "Drawdown",
]


# ─────────────────────────────────────────────
# BUILD ENSEMBLE
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
        n_jobs=1,
    )

    estimators = [
        ("xgb", xgb_m),
        ("rf", rf_m),
    ]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        stack_method="predict_proba",
        passthrough=False,
        cv=3,
        n_jobs=1,
    )

    return stack


# ─────────────────────────────────────────────
# ROC AUC
# ─────────────────────────────────────────────

try:
    auc = roc_auc_score(y_te, proba[:,1])
except:
    auc = 0.5


# ─────────────────────────────────────────────
# THRESHOLD DINAMICO
# ─────────────────────────────────────────────

trend_regime = df["Trend_Regime"].iloc[-1]

if trend_regime == 1:
    dynamic_threshold = 0.52
else:
    dynamic_threshold = 0.58


# ─────────────────────────────────────────────
# BACKTEST REALISTICO
# ─────────────────────────────────────────────

fee = 0.001

df_bt["Strategy_Returns"] = (
    df_bt["Market_Returns"] *
    df_bt["Signal"]
)

df_bt["Strategy_Returns"] -= (
    fee *
    df_bt["Signal"].diff().abs().fillna(0)
)


# ─────────────────────────────────────────────
# CALIBRAZIONE
# ─────────────────────────────────────────────

calibrated = CalibratedClassifierCV(
    estimator=ensemble_final,
    method="isotonic",
    cv=TimeSeriesSplit(n_splits=3),
)
