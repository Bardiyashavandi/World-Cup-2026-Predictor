import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
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
# 3. TRAIN MODEL
# ─────────────────────────────────────────

def train_lightgbm(train: pd.DataFrame) -> tuple:
    """
    Train two LightGBM models:
    - model_home: predicts home goals
    - model_away: predicts away goals

    Key difference from XGBoost:
    - objective='poisson' uses Poisson log-loss
    - grows trees leaf-wise (faster, more accurate on tabular)
    - num_leaves controls complexity instead of max_depth
    """
    X = train[FEATURE_COLS]
    y_home = train["home_goals"]
    y_away = train["away_goals"]

    params = {
        "n_estimators": 300,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "objective": "poisson",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    print("Training LightGBM — Home Goals model...")
    model_home = lgb.LGBMRegressor(**params)
    model_home.fit(X, y_home)

    print("Training LightGBM — Away Goals model...")
    model_away = lgb.LGBMRegressor(**params)
    model_away.fit(X, y_away)

    # Cross validation
    print("\nCross-validating with TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)

    home_scores = cross_val_score(
        lgb.LGBMRegressor(**params), X, y_home,
        cv=tscv, scoring="neg_mean_absolute_error"
    )
    away_scores = cross_val_score(
        lgb.LGBMRegressor(**params), X, y_away,
        cv=tscv, scoring="neg_mean_absolute_error"
    )

    print(f"  Home goals MAE: {-home_scores.mean():.3f} "
          f"(±{home_scores.std():.3f})")
    print(f"  Away goals MAE: {-away_scores.mean():.3f} "
          f"(±{away_scores.std():.3f})")

    return model_home, model_away


# ─────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────

def print_feature_importance(model_home, model_away):
    """Print top 10 most important features for each model."""
    importance_home = pd.Series(
        model_home.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    importance_away = pd.Series(
        model_away.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    print("\nTop 10 Features — Home Goals Model:")
    print("-" * 40)
    for feat, imp in importance_home.head(10).items():
        bar = "█" * int(imp / 20)
        print(f"  {feat:35} {imp:6.0f} {bar}")

    print("\nTop 10 Features — Away Goals Model:")
    print("-" * 40)
    for feat, imp in importance_away.head(10).items():
        bar = "█" * int(imp / 20)
        print(f"  {feat:35} {imp:6.0f} {bar}")


# ─────────────────────────────────────────
# 5. PREDICT MATCHES
# ─────────────────────────────────────────

def predict_scorelines(model_home, model_away,
                        predict_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate scoreline predictions for all WC 2026 fixtures.
    """
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
# 6. RUN FULL PIPELINE
# ─────────────────────────────────────────

def run_lightgbm():
    print("=" * 60)
    print("  LIGHTGBM MODEL — WC 2026 Predictor")
    print("=" * 60)
    print()

    print("Loading data...")
    train, predict = load_data()
    print(f"  Training set: {len(train)} matches")
    print(f"  Prediction set: {len(predict)} fixtures")
    print()

    # Train
    model_home, model_away = train_lightgbm(train)

    # Feature importance
    print_feature_importance(model_home, model_away)

    # Predict
    print("\nGenerating WC 2026 predictions...")
    out_df = predict_scorelines(model_home, model_away, predict)

    # Save
    os.makedirs("data/predictions", exist_ok=True)
    out_df.to_csv("data/predictions/lightgbm_all.csv", index=False)
    out_df[out_df["matchday"] == 1].to_csv(
        "data/predictions/lightgbm_md1.csv", index=False
    )

    # Print
    print()
    print("LightGBM Predictions — WC 2026 Group Stage")
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
    print("Saved to data/predictions/lightgbm_all.csv")
    print("Saved to data/predictions/lightgbm_md1.csv")


if __name__ == "__main__":
    run_lightgbm()