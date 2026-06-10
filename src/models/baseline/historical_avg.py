import pandas as pd
import numpy as np
from scipy.stats import poisson
import os
import sys

sys.path.append("src")


# ─────────────────────────────────────────
# 1. TEAM STATS
# ─────────────────────────────────────────

def get_team_stats(results: pd.DataFrame, team: str,
                   before_date, n: int = 10) -> dict:
    """
    Get a team's average goals scored and conceded
    in their last n matches before a given date.
    """
    mask = (
        ((results["home_team"] == team) |
         (results["away_team"] == team)) &
        (results["date"] < before_date)
    )
    matches = results[mask].sort_values(
        "date", ascending=False
    ).head(n)

    if len(matches) == 0:
        return {"avg_scored": 1.2, "avg_conceded": 1.2, "n_matches": 0}

    scored = []
    conceded = []

    for _, m in matches.iterrows():
        if m["home_team"] == team:
            scored.append(m["home_goals"])
            conceded.append(m["away_goals"])
        else:
            scored.append(m["away_goals"])
            conceded.append(m["home_goals"])

    return {
        "avg_scored": np.mean(scored),
        "avg_conceded": np.mean(conceded),
        "n_matches": len(matches),
    }


# ─────────────────────────────────────────
# 2. EXPECTED GOALS
# ─────────────────────────────────────────

def get_expected_goals(results: pd.DataFrame,
                        home_team: str, away_team: str,
                        match_date, n: int = 10) -> tuple:
    """
    Combine home attack vs away defense and vice versa
    to get expected goals for each team.

    Formula:
        home_xg = (home_avg_scored + away_avg_conceded) / 2
        away_xg = (away_avg_scored + home_avg_conceded) / 2
    """
    home_stats = get_team_stats(results, home_team, match_date, n)
    away_stats = get_team_stats(results, away_team, match_date, n)

    home_xg = (home_stats["avg_scored"] + away_stats["avg_conceded"]) / 2
    away_xg = (away_stats["avg_scored"] + home_stats["avg_conceded"]) / 2

    # Clip to reasonable range
    home_xg = np.clip(home_xg, 0.3, 4.0)
    away_xg = np.clip(away_xg, 0.3, 4.0)

    return round(home_xg, 3), round(away_xg, 3)


# ─────────────────────────────────────────
# 3. PREDICT MATCH
# ─────────────────────────────────────────

def predict_match_ha(results: pd.DataFrame,
                      home_team: str, away_team: str,
                      match_date, n: int = 10) -> dict:
    """
    Predict a single match using historical averages.
    """
    home_xg, away_xg = get_expected_goals(
        results, home_team, away_team, match_date, n
    )

    # Poisson scoreline probabilities
    scorelines = []
    for h in range(7):
        for a in range(7):
            p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            scorelines.append({
                "home_goals": h,
                "away_goals": a,
                "probability": round(p, 6)
            })

    score_df = pd.DataFrame(scorelines).sort_values(
        "probability", ascending=False
    ).reset_index(drop=True)

    home_win = score_df[
        score_df["home_goals"] > score_df["away_goals"]
    ]["probability"].sum()
    draw = score_df[
        score_df["home_goals"] == score_df["away_goals"]
    ]["probability"].sum()
    away_win = score_df[
        score_df["home_goals"] < score_df["away_goals"]
    ]["probability"].sum()

    total = home_win + draw + away_win
    if total > 0:
        home_win, draw, away_win = (
            home_win / total, draw / total, away_win / total
        )

    top = score_df.iloc[0]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "predicted_home_goals": int(top["home_goals"]),
        "predicted_away_goals": int(top["away_goals"]),
        "home_win_prob": round(home_win, 3),
        "draw_prob": round(draw, 3),
        "away_win_prob": round(away_win, 3),
    }


# ─────────────────────────────────────────
# 4. RUN FULL PIPELINE
# ─────────────────────────────────────────

def run_historical_avg():
    print("=" * 60)
    print("  HISTORICAL AVERAGE MODEL — WC 2026 Predictor")
    print("=" * 60)
    print()

    print("Loading data...")
    results = pd.read_csv(
        "data/processed/results_clean.csv", parse_dates=["date"]
    )
    fixtures = pd.read_csv(
        "data/processed/fixtures_clean.csv", parse_dates=["date"]
    )

    print(f"Loaded {len(results)} historical matches")
    print(f"Predicting {len(fixtures)} WC 2026 fixtures...")
    print()

    predictions = []
    reference_date = pd.Timestamp("2026-06-11")

    for i, (_, fixture) in enumerate(fixtures.iterrows()):
        if i % 10 == 0:
            print(f"  Processing fixture {i+1}/{len(fixtures)}...")

        pred = predict_match_ha(
            results=results,
            home_team=fixture["home_team"],
            away_team=fixture["away_team"],
            match_date=reference_date,
            n=10,
        )
        pred["match_id"] = fixture["match_id"]
        pred["group"] = fixture["group"]
        pred["matchday"] = fixture["matchday"]
        pred["city"] = fixture["city"]
        predictions.append(pred)

    out_df = pd.DataFrame(predictions)

    # Save
    os.makedirs("data/predictions", exist_ok=True)
    out_df.to_csv("data/predictions/historical_avg_all.csv", index=False)
    out_df[out_df["matchday"] == 1].to_csv(
        "data/predictions/historical_avg_md1.csv", index=False
    )

    # Print
    print()
    print("Historical Average Predictions — WC 2026 Group Stage")
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
    print("Saved to data/predictions/historical_avg_all.csv")
    print("Saved to data/predictions/historical_avg_md1.csv")


if __name__ == "__main__":
    run_historical_avg()