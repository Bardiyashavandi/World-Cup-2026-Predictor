import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from scipy.stats import poisson
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append("src")
sys.path.append("src/features")

from group_standings import compute_standings, compute_stakes_features


# ─────────────────────────────────────────
# 1. FEATURE COLUMNS
# ─────────────────────────────────────────

BASE_FEATURES = [
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

STAKES_FEATURES = [
    "home_points", "home_position", "home_matches_played",
    "home_gd", "home_can_qualify", "home_already_qualified",
    "home_must_win", "home_elimination_risk", "home_points_needed",
    "away_points", "away_position", "away_matches_played",
    "away_gd", "away_can_qualify", "away_already_qualified",
    "away_must_win", "away_elimination_risk", "away_points_needed",
]

ALL_FEATURES = BASE_FEATURES + STAKES_FEATURES + ["matchday"]

REFERENCE_DATE = pd.Timestamp("2026-06-11")


# ─────────────────────────────────────────
# 2. BUILD HISTORICAL STAKES FEATURES
# ─────────────────────────────────────────

def build_historical_stakes(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build stakes features for historical World Cup MD2/MD3 matches.
    Replays each tournament chronologically to compute group context.
    """
    wc_results = results[
        results["tournament"].str.contains("FIFA World Cup", na=False) &
        ~results["tournament"].str.contains(
            "qualification", case=False, na=False
        )
    ].copy()

    print(f"Found {len(wc_results)} World Cup group stage matches")

    wc_results["year"] = wc_results["date"].dt.year
    wc_years = wc_results["year"].unique()
    print(f"World Cup years: {sorted(wc_years)}")

    all_stakes_rows = []

    for year in sorted(wc_years):
        year_matches = wc_results[
            wc_results["year"] == year
        ].sort_values("date").copy()

        if len(year_matches) < 10:
            continue

        dates = sorted(year_matches["date"].unique())
        n_dates = len(dates)
        third = max(1, n_dates // 3)

        date_to_md = {}
        for i, d in enumerate(dates):
            if i < third:
                date_to_md[d] = 1
            elif i < 2 * third:
                date_to_md[d] = 2
            else:
                date_to_md[d] = 3

        year_matches["matchday"] = year_matches["date"].map(date_to_md)

        for matchday in [2, 3]:
            md_matches = year_matches[
                year_matches["matchday"] == matchday
            ]
            prior_results = year_matches[
                year_matches["matchday"] < matchday
            ].copy()

            for _, match in md_matches.iterrows():
                home_team = match["home_team"]
                away_team = match["away_team"]

                home_prior = prior_results[
                    (prior_results["home_team"] == home_team) |
                    (prior_results["away_team"] == home_team)
                ]
                away_prior = prior_results[
                    (prior_results["home_team"] == away_team) |
                    (prior_results["away_team"] == away_team)
                ]

                def calc_pts_gd(team_matches, team):
                    pts = gd = mp = 0
                    for _, m in team_matches.iterrows():
                        if m["home_team"] == team:
                            s, c = m["home_goals"], m["away_goals"]
                        else:
                            s, c = m["away_goals"], m["home_goals"]
                        gd += s - c
                        mp += 1
                        if s > c:
                            pts += 3
                        elif s == c:
                            pts += 1
                    return pts, gd, mp

                h_pts, h_gd, h_mp = calc_pts_gd(home_prior, home_team)
                a_pts, a_gd, a_mp = calc_pts_gd(away_prior, away_team)

                h_pos = 1 if h_pts >= 3 else (2 if h_pts == 1 else 3)
                a_pos = 1 if a_pts >= 3 else (2 if a_pts == 1 else 3)

                row = {
                    "home_team": home_team,
                    "away_team": away_team,
                    "date": match["date"],
                    "tournament": match["tournament"],
                    "home_goals": match["home_goals"],
                    "away_goals": match["away_goals"],
                    "matchday": int(matchday),
                    "year": year,
                    "home_points": h_pts,
                    "home_position": h_pos,
                    "home_matches_played": h_mp,
                    "home_gd": h_gd,
                    "home_can_qualify": int(
                        h_pts + (3 - h_mp) * 3 >= 4
                    ),
                    "home_already_qualified": int(
                        h_pts >= 6 and h_mp >= 2
                    ),
                    "home_must_win": int(
                        h_pts <= 1 and matchday == 3
                    ),
                    "home_elimination_risk": (
                        1 if h_pos >= 3 and h_pts <= 1
                        else 0.5 if h_pos >= 3
                        else 0
                    ),
                    "home_points_needed": max(0, 4 - h_pts),
                    "away_points": a_pts,
                    "away_position": a_pos,
                    "away_matches_played": a_mp,
                    "away_gd": a_gd,
                    "away_can_qualify": int(
                        a_pts + (3 - a_mp) * 3 >= 4
                    ),
                    "away_already_qualified": int(
                        a_pts >= 6 and a_mp >= 2
                    ),
                    "away_must_win": int(
                        a_pts <= 1 and matchday == 3
                    ),
                    "away_elimination_risk": (
                        1 if a_pos >= 3 and a_pts <= 1
                        else 0.5 if a_pos >= 3
                        else 0
                    ),
                    "away_points_needed": max(0, 4 - a_pts),
                }
                all_stakes_rows.append(row)

    return pd.DataFrame(all_stakes_rows)


# ─────────────────────────────────────────
# 3. LOAD AND MERGE DATA
# ─────────────────────────────────────────

def load_data():
    train = pd.read_csv(
        "data/processed/train_features.csv", parse_dates=["date"]
    )
    predict = pd.read_csv(
        "data/processed/predict_features.csv", parse_dates=["date"]
    )
    results = pd.read_csv(
        "data/processed/results_clean.csv", parse_dates=["date"]
    )

    train = train.dropna(subset=["home_goals", "away_goals"])

    print("Building historical stakes features...")
    stakes = build_historical_stakes(results)
    print(f"Built stakes for {len(stakes)} historical MD2/MD3 matches")

    stakes_merge_cols = [
        "home_team", "away_team", "date", "matchday"
    ] + STAKES_FEATURES

    train_with_stakes = train.merge(
        stakes[stakes_merge_cols],
        on=["home_team", "away_team", "date"],
        how="inner",
        suffixes=("", "_stakes")
    )

    if "matchday_stakes" in train_with_stakes.columns:
        train_with_stakes["matchday"] = train_with_stakes[
            "matchday_stakes"
        ].fillna(train_with_stakes.get("matchday", 2))
        train_with_stakes = train_with_stakes.drop(
            columns=["matchday_stakes"]
        )

    print(f"Training matches with stakes: {len(train_with_stakes)}")

    if "matchday" not in train_with_stakes.columns:
        train_with_stakes["matchday"] = 2

    for col in ALL_FEATURES:
        if col not in train_with_stakes.columns:
            train_with_stakes[col] = 0
        if col not in predict.columns:
            predict[col] = 0

    train_with_stakes[ALL_FEATURES] = train_with_stakes[
        ALL_FEATURES
    ].fillna(0)
    predict[BASE_FEATURES] = predict[BASE_FEATURES].fillna(0)

    return train_with_stakes, predict, results


# ─────────────────────────────────────────
# 4. TIME WEIGHTS
# ─────────────────────────────────────────

def compute_sample_weights(train: pd.DataFrame) -> np.ndarray:
    weights = train["date"].apply(
        lambda d: np.exp(-0.001 * (REFERENCE_DATE - d).days)
    )
    weights = weights / weights.sum() * len(train)
    return weights.values


# ─────────────────────────────────────────
# 5. TRAIN MODEL
# ─────────────────────────────────────────

def train_stakes_model(train: pd.DataFrame) -> tuple:
    X = train[ALL_FEATURES].copy()
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.astype(float)

    y_home = train["home_goals"].values.astype(float)
    y_away = train["away_goals"].values.astype(float)

    sample_weights = compute_sample_weights(train)

    print(f"  Training on {len(train)} MD2/MD3 matches")
    print(f"  Features: {len(ALL_FEATURES)} "
          f"(24 base + {len(STAKES_FEATURES)} stakes + matchday)")
    print()

    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "objective": "count:poisson",
        "random_state": 42,
        "n_jobs": -1,
    }

    print("Training Stakes XGBoost — Home Goals model...")
    model_home = xgb.XGBRegressor(**params)
    model_home.fit(X, y_home, sample_weight=sample_weights)

    print("Training Stakes XGBoost — Away Goals model...")
    model_away = xgb.XGBRegressor(**params)
    model_away.fit(X, y_away, sample_weight=sample_weights)

    tscv = TimeSeriesSplit(n_splits=3)
    home_scores = cross_val_score(
        xgb.XGBRegressor(**params), X, y_home,
        cv=tscv, scoring="neg_mean_absolute_error"
    )
    away_scores = cross_val_score(
        xgb.XGBRegressor(**params), X, y_away,
        cv=tscv, scoring="neg_mean_absolute_error"
    )

    print(f"\n  Home goals MAE: {-home_scores.mean():.3f} "
          f"(±{home_scores.std():.3f})")
    print(f"  Away goals MAE: {-away_scores.mean():.3f} "
          f"(±{away_scores.std():.3f})")

    return model_home, model_away


# ─────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────

def print_feature_importance(model_home, model_away):
    feat_names = ALL_FEATURES[:len(model_home.feature_importances_)]

    importance_home = pd.Series(
        model_home.feature_importances_, index=feat_names
    ).sort_values(ascending=False)

    importance_away = pd.Series(
        model_away.feature_importances_, index=feat_names
    ).sort_values(ascending=False)

    print("\nTop 10 Features — Home Goals (Stakes Model):")
    print("-" * 50)
    for feat, imp in importance_home.head(10).items():
        bar = "█" * int(imp * 200)
        tag = " ← STAKES" if feat in STAKES_FEATURES + ["matchday"] else ""
        print(f"  {feat:35} {imp:.4f} {bar}{tag}")

    print("\nTop 10 Features — Away Goals (Stakes Model):")
    print("-" * 50)
    for feat, imp in importance_away.head(10).items():
        bar = "█" * int(imp * 200)
        tag = " ← STAKES" if feat in STAKES_FEATURES + ["matchday"] else ""
        print(f"  {feat:35} {imp:.4f} {bar}{tag}")


# ─────────────────────────────────────────
# 7. PREDICT MATCHES
# ─────────────────────────────────────────

def predict_scorelines(model_home, model_away,
                        predict_df: pd.DataFrame,
                        standings: pd.DataFrame,
                        fixtures: pd.DataFrame,
                        matchday: int) -> pd.DataFrame:

    stakes_df = compute_stakes_features(standings, fixtures, matchday)

    merged = predict_df.merge(
        stakes_df[["match_id"] + STAKES_FEATURES],
        on="match_id",
        how="left"
    )
    merged["matchday"] = matchday

    for col in ALL_FEATURES:
        if col not in merged.columns:
            merged[col] = 0
    merged[ALL_FEATURES] = merged[ALL_FEATURES].fillna(0)

    X_pred = merged[ALL_FEATURES].copy()
    X_pred = X_pred.loc[:, ~X_pred.columns.duplicated()]
    X_pred = X_pred.astype(float)

    home_xg = np.clip(model_home.predict(X_pred), 0.3, 4.0)
    away_xg = np.clip(model_away.predict(X_pred), 0.3, 4.0)

    results = []
    for i, (_, row) in enumerate(merged.iterrows()):
        hxg = home_xg[i]
        axg = away_xg[i]

        scorelines = []
        for h in range(7):
            for a in range(7):
                p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
                scorelines.append({
                    "home_goals": h,
                    "away_goals": a,
                    "probability": p
                })

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
            "match_id": row["match_id"],
            "group": row["group"],
            "matchday": matchday,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_xg": round(float(hxg), 3),
            "away_xg": round(float(axg), 3),
            "predicted_home_goals": int(top["home_goals"]),
            "predicted_away_goals": int(top["away_goals"]),
            "home_win_prob": round(home_win, 3),
            "draw_prob": round(draw, 3),
            "away_win_prob": round(away_win, 3),
            "home_must_win": int(row.get("home_must_win", 0)),
            "away_must_win": int(row.get("away_must_win", 0)),
            "home_already_qualified": int(
                row.get("home_already_qualified", 0)
            ),
            "away_already_qualified": int(
                row.get("away_already_qualified", 0)
            ),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────
# 8. RUN FULL PIPELINE
# ─────────────────────────────────────────

def run_stakes_model(matchday: int = 2):
    """
    Run stakes model for a specific matchday.
    NOTE: Stakes model is designed for MD2 and MD3 only.
          For MD1 use xgboost_model.py instead.
    """
    print("=" * 60)
    print("  STAKES MODEL — WC 2026 Predictor")
    print("=" * 60)
    print()

    if matchday == 1:
        print("⚠️  Stakes model not designed for MD1.")
        print("   Stakes features are all equal at MD1 — no context yet.")
        print("   For MD1 predictions use:")
        print("   python3 src/models/ml/xgboost_model.py")
        print()
        print("   Stakes model activates from MD2 onwards when real")
        print("   results create meaningful group context.")
        return

    print("Loading data...")
    train, predict, results = load_data()
    fixtures = pd.read_csv("data/raw/wc_2026_fixtures.csv")
    print()

    print("Training stakes model...")
    model_home, model_away = train_stakes_model(train)
    print_feature_importance(model_home, model_away)

    # Load standings from previous matchday
    standings_path = (
        f"data/processed/standings_after_md{matchday-1}.csv"
    )

    standings = pd.DataFrame()
    if os.path.exists(standings_path):
        try:
            standings = pd.read_csv(standings_path)
            if len(standings) == 0:
                raise ValueError("Empty file")
            print(f"\nLoaded standings after MD{matchday-1} "
                  f"({len(standings)} teams)")
        except Exception:
            standings = pd.DataFrame()
            print(f"\n⚠️  Standings file empty or invalid.")
            print(f"   Why: No MD{matchday-1} results entered yet.")
            print(f"   Fix: Fill in data/raw/wc_2026_results.csv")
            print(f"        then run: python3 src/features/"
                  f"group_standings.py")
            print("   Proceeding with empty stakes for now...")
    else:
        print(f"\n⚠️  No standings file found for MD{matchday-1}.")
        print(f"   Why: save_standings({matchday-1}) hasn't been run.")
        print(f"   Fix: Enter MD{matchday-1} results and run standings.")
        print("   Proceeding with empty stakes for now...")

    # Predict
    print(f"\nPredicting MD{matchday} with stakes features...")
    out_df = predict_scorelines(
        model_home, model_away,
        predict, standings, fixtures, matchday
    )
    out_df = out_df[out_df["matchday"] == matchday]

    # Save
    os.makedirs("data/predictions", exist_ok=True)
    out_path = f"data/predictions/stakes_model_md{matchday}.csv"
    out_df.to_csv(out_path, index=False)

    # Print
    print()
    print(f"Stakes Model Predictions — WC 2026 MD{matchday}")
    print("=" * 60)
    for _, row in out_df.iterrows():
        flags = ""
        if row["home_must_win"]:
            flags += f" [{row['home_team']} MUST WIN]"
        if row["away_must_win"]:
            flags += f" [{row['away_team']} MUST WIN]"
        if row["home_already_qualified"]:
            flags += f" [{row['home_team']} QUALIFIED]"
        if row["away_already_qualified"]:
            flags += f" [{row['away_team']} QUALIFIED]"

        print(
            f"Group {row['group']} MD{row['matchday']} | "
            f"{row['home_team']:20} {row['predicted_home_goals']}-"
            f"{row['predicted_away_goals']} {row['away_team']:20} | "
            f"xG: {row['home_xg']:.2f}-{row['away_xg']:.2f} | "
            f"H:{row['home_win_prob']} "
            f"D:{row['draw_prob']} "
            f"A:{row['away_win_prob']}"
            f"{flags}"
        )

    print()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    md = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run_stakes_model(matchday=md)