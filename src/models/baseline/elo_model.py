import pandas as pd
import numpy as np
import os
import sys

sys.path.append("src")


# Average goals per match in international football
# Based on World Cup data
AVG_GOALS = 1.35


def elo_to_expected_goals(elo_diff: float) -> tuple:
    """
    Convert ELO difference to expected goals for each team.
    
    Logic:
    - A team with higher ELO is expected to score more
    - ELO diff of 0 = both teams score AVG_GOALS
    - Each 100 ELO points ~ 0.2 goal difference
    
    Returns:
        (home_xg, away_xg) — expected goals for each team
    """
    scale = 0.001

    home_xg = AVG_GOALS + (elo_diff * scale)
    away_xg = AVG_GOALS - (elo_diff * scale)

    # Clip to reasonable range — no negative goals
    home_xg = np.clip(home_xg, 0.3, 4.0)
    away_xg = np.clip(away_xg, 0.3, 4.0)

    return round(home_xg, 3), round(away_xg, 3)


def predict_scoreline(home_xg: float, away_xg: float,
                       max_goals: int = 6) -> pd.DataFrame:
    """
    Given expected goals for each team, compute probability
    for every possible scoreline using Poisson distribution.
    
    Returns:
        DataFrame with columns: home_goals, away_goals, probability
    """
    from scipy.stats import poisson

    scorelines = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            scorelines.append({
                "home_goals": h,
                "away_goals": a,
                "probability": round(prob, 6)
            })

    df = pd.DataFrame(scorelines)
    df = df.sort_values("probability", ascending=False).reset_index(drop=True)
    return df


def get_match_probabilities(score_matrix: pd.DataFrame) -> dict:
    """
    From a scoreline probability matrix, compute
    win/draw/loss probabilities.
    """
    home_win = score_matrix[
        score_matrix["home_goals"] > score_matrix["away_goals"]
    ]["probability"].sum()

    draw = score_matrix[
        score_matrix["home_goals"] == score_matrix["away_goals"]
    ]["probability"].sum()

    away_win = score_matrix[
        score_matrix["home_goals"] < score_matrix["away_goals"]
    ]["probability"].sum()

    # Normalize: the truncated scoreline grid drops a little tail mass,
    # so rescale the three outcomes into a proper distribution.
    total = home_win + draw + away_win
    if total > 0:
        home_win, draw, away_win = (
            home_win / total, draw / total, away_win / total
        )

    return {
        "home_win_prob": round(home_win, 3),
        "draw_prob": round(draw, 3),
        "away_win_prob": round(away_win, 3),
    }


def predict_match(home_team: str, away_team: str,
                   elo_diff: float) -> dict:
    """
    Full prediction for a single match.
    """
    home_xg, away_xg = elo_to_expected_goals(elo_diff)
    score_matrix = predict_scoreline(home_xg, away_xg)
    probs = get_match_probabilities(score_matrix)

    top_scoreline = score_matrix.iloc[0]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "predicted_home_goals": int(top_scoreline["home_goals"]),
        "predicted_away_goals": int(top_scoreline["away_goals"]),
        **probs,
        "top_5_scorelines": score_matrix.head(5).to_dict("records"),
    }


def run_predictions(pred_path: str = "data/processed/predict_features.csv",
                    out_path: str = "data/processed/predictions_elo.csv"):
    """Run ELO model predictions on all WC 2026 fixtures."""

    df = pd.read_csv(pred_path)
    os.makedirs("data/processed", exist_ok=True)

    results = []
    for _, row in df.iterrows():
        pred = predict_match(
            home_team=row["home_team"],
            away_team=row["away_team"],
            elo_diff=row["elo_diff"],
        )
        pred["match_id"] = row["match_id"]
        pred["group"] = row["group"]
        pred["matchday"] = row["matchday"]
        pred.pop("top_5_scorelines")
        results.append(pred)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)

    print(f"ELO Model Predictions — WC 2026 Group Stage")
    print("=" * 55)
    for _, row in out_df.iterrows():
        print(
            f"Group {row['group']} MD{row['matchday']} | "
            f"{row['home_team']} {row['predicted_home_goals']}-"
            f"{row['predicted_away_goals']} {row['away_team']} | "
            f"H:{row['home_win_prob']} "
            f"D:{row['draw_prob']} "
            f"A:{row['away_win_prob']}"
        )


if __name__ == "__main__":
    run_predictions()