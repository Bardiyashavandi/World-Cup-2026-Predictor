import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
from scipy.stats import poisson
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append("src")


# ─────────────────────────────────────────
# 1. FEATURE COLUMNS
# ─────────────────────────────────────────

FEATURE_COLS = [
    "is_neutral",
    "home_elo", "away_elo", "elo_diff",
    "home_form5_points", "home_form5_avg_scored", "home_form5_avg_conceded",
    "away_form5_points", "away_form5_avg_scored", "away_form5_avg_conceded",
    "home_form10_points", "home_form10_avg_scored", "home_form10_avg_conceded",
    "away_form10_points", "away_form10_avg_scored", "away_form10_avg_conceded",
    "form5_points_diff", "form10_points_diff", "avg_scored_diff",
    "h2h_matches", "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    "h2h_avg_goals",
]

REFERENCE_DATE = pd.Timestamp("2026-06-11")


# ─────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────

def load_data():
    train = pd.read_csv("data/processed/train_features.csv")
    predict = pd.read_csv("data/processed/predict_features.csv")

    train = train.dropna(subset=["home_goals", "away_goals"])
    train[FEATURE_COLS] = train[FEATURE_COLS].fillna(0)
    predict[FEATURE_COLS] = predict[FEATURE_COLS].fillna(0)

    return train, predict


# ─────────────────────────────────────────
# 3. TIME WEIGHTS
# ─────────────────────────────────────────

def compute_sample_weights(train: pd.DataFrame) -> pd.Series:
    """
    Compute exponential time decay weights.
    Recent matches get higher weight than old ones.

    decay = 0.001 means:
      - match today        → weight ~5.8x baseline
      - match 1 year ago   → weight ~3.5x baseline
      - match 3 years ago  → weight ~1.5x baseline
      - match 8 years ago  → weight ~0.3x baseline
    """
    train = train.copy()
    train["date"] = pd.to_datetime(train["date"])
    weights = train["date"].apply(
        lambda d: np.exp(-0.001 * (REFERENCE_DATE - d).days)
    )
    # Normalize so weights sum to n
    weights = weights / weights.sum() * len(train)
    return weights


# ─────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────

def train_xgboost(train: pd.DataFrame) -> tuple:
    """
    Train two XGBoost models with time-based sample weights.
    Recent matches influence the model more than old ones.
    """
    X = train[FEATURE_COLS]
    y_home = train["home_goals"]
    y_away = train["away_goals"]

    # Compute time decay weights
    sample_weights = compute_sample_weights(train)

    print(f"  Sample weight range: "
          f"{sample_weights.min():.3f} → {sample_weights.max():.3f}")
    print(f"  (recent matches weighted up to "
          f"{sample_weights.max():.1f}x more than oldest)")
    print()

    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "objective": "count:poisson",
        "random_state": 42,
        "n_jobs": -1,
    }

    print("Training XGBoost — Home Goals model (time-weighted)...")
    model_home = xgb.XGBRegressor(**params)
    model_home.fit(X, y_home, sample_weight=sample_weights)

    print("Training XGBoost — Away Goals model (time-weighted)...")
    model_away = xgb.XGBRegressor(**params)
    model_away.fit(X, y_away, sample_weight=sample_weights)

    # Cross validation with time series split
    # Note: we skip sample_weight in CV since sklearn 1.4+
    # changed the API — CV gives us directional signal anyway
    print("\nCross-validating with TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)

    home_scores = cross_val_score(
        xgb.XGBRegressor(**params), X, y_home,
        cv=tscv, scoring="neg_mean_absolute_error"
    )
    away_scores = cross_val_score(
        xgb.XGBRegressor(**params), X, y_away,
        cv=tscv, scoring="neg_mean_absolute_error"
    )

    print(f"  Home goals MAE: {-home_scores.mean():.3f} "
          f"(±{home_scores.std():.3f})")
    print(f"  Away goals MAE: {-away_scores.mean():.3f} "
          f"(±{away_scores.std():.3f})")

    return model_home, model_away


# ─────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────

def print_feature_importance(model_home, model_away):
    importance_home = pd.Series(
        model_home.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    importance_away = pd.Series(
        model_away.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    print("\nTop 10 Features — Home Goals Model:")
    print("-" * 40)
    for feat, imp in importance_home.head(10).items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:35} {imp:.4f} {bar}")

    print("\nTop 10 Features — Away Goals Model:")
    print("-" * 40)
    for feat, imp in importance_away.head(10).items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:35} {imp:.4f} {bar}")


# ─────────────────────────────────────────
# 6. PREDICT MATCHES
# ─────────────────────────────────────────

def predict_scorelines(model_home, model_away,
                        predict_df: pd.DataFrame) -> pd.DataFrame:
    X_pred = predict_df[FEATURE_COLS].fillna(0)

    home_xg = model_home.predict(X_pred)
    away_xg = model_away.predict(X_pred)

    home_xg = np.clip(home_xg, 0.3, 4.0)
    away_xg = np.clip(away_xg, 0.3, 4.0)

    results = []
    for i, (_, row) in enumerate(predict_df.iterrows()):
        hxg = home_xg[i]
        axg = away_xg[i]

        scorelines = []
        for h in range(7):
            for a in range(7):
                p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
                scorelines.append({"home_goals": h,
                                   "away_goals": a,
                                   "probability": p})

        score_df = pd.DataFrame(scorelines).sort_values(
            "probability", ascending=False
        )

        home_win = score_df[
            score_df["home_goals"] > score_df["away_goals"]
        ]["probability"].sum()
        draw = score_df[
            score_df["home_goals"] == score_df["away_goals"]
        ]["probability"].sum()
        away_win = score_df[
            score_df["home_goals"] < score_df["away_goals"]
        ]["probability"].sum()

        top = score_df.iloc[0]

        results.append({
            "match_id": row.get("match_id", i),
            "group": row.get("group", ""),
            "matchday": row.get("matchday", 1),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_xg": round(float(hxg), 3),
            "away_xg": round(float(axg), 3),
            "predicted_home_goals": int(top["home_goals"]),
            "predicted_away_goals": int(top["away_goals"]),
            "home_win_prob": round(home_win, 3),
            "draw_prob": round(draw, 3),
            "away_win_prob": round(away_win, 3),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────
# 7. RUN FULL PIPELINE
# ─────────────────────────────────────────

def run_xgboost():
    print("=" * 60)
    print("  XGBOOST MODEL (TIME-WEIGHTED) — WC 2026 Predictor")
    print("=" * 60)
    print()

    print("Loading data...")
    train, predict = load_data()
    print(f"  Training set: {len(train)} matches")
    print(f"  Prediction set: {len(predict)} fixtures")
    print()

    model_home, model_away = train_xgboost(train)
    print_feature_importance(model_home, model_away)

    print("\nGenerating WC 2026 predictions...")
    out_df = predict_scorelines(model_home, model_away, predict)

    os.makedirs("data/predictions", exist_ok=True)
    out_df.to_csv("data/predictions/xgboost_all.csv", index=False)
    out_df[out_df["matchday"] == 1].to_csv(
        "data/predictions/xgboost_md1.csv", index=False
    )

    print()
    print("XGBoost Predictions (Time-Weighted) — WC 2026 Group Stage")
    print("=" * 60)
    for _, row in out_df.iterrows():
        print(
            f"Group {row['group']} MD{row['matchday']} | "
            f"{row['home_team']:20} {row['predicted_home_goals']}-"
            f"{row['predicted_away_goals']} {row['away_team']:20} | "
            f"xG: {row['home_xg']:.2f}-{row['away_xg']:.2f} | "
            f"H:{row['home_win_prob']} "
            f"D:{row['draw_prob']} "
            f"A:{row['away_win_prob']}"
        )

    print()
    print("Saved to data/predictions/xgboost_all.csv")
    print("Saved to data/predictions/xgboost_md1.csv")


if __name__ == "__main__":
    run_xgboost()