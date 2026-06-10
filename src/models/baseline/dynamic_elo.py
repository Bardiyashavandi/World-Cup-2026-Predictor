import pandas as pd
import numpy as np
from scipy.stats import poisson
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append("src")


# ─────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────

DEFAULT_ELO = 1500
HOME_ADVANTAGE = 100  # ELO points added for home team

# K-factors by tournament type
# Higher K = rating changes more after each match
K_FACTORS = {
    "world_cup":        60,
    "continental":      50,
    "qualifier":        40,
    "nations_league":   40,
    "friendly":         20,
    "other":            30,
}

# Tournament name → category mapping
TOURNAMENT_CATEGORIES = {
    "FIFA World Cup":                   "world_cup",
    "UEFA Euro":                        "continental",
    "Copa América":                     "continental",
    "Africa Cup of Nations":            "continental",
    "AFC Asian Cup":                    "continental",
    "CONCACAF Gold Cup":                "continental",
    "FIFA World Cup qualification":     "qualifier",
    "UEFA Euro qualification":          "qualifier",
    "Copa América qualification":       "qualifier",
    "UEFA Nations League":              "nations_league",
    "Friendly":                         "friendly",
    "International friendly":           "friendly",
}


# ─────────────────────────────────────────
# 2. TOURNAMENT CATEGORY
# ─────────────────────────────────────────

def get_tournament_category(tournament: str) -> str:
    """Map tournament name to category for K-factor lookup."""
    if not isinstance(tournament, str):
        return "other"
    tournament_lower = tournament.lower()
    for key, category in TOURNAMENT_CATEGORIES.items():
        if key.lower() in tournament_lower:
            return category
    return "other"


def get_k_factor(tournament: str) -> float:
    """Get K-factor for a given tournament."""
    category = get_tournament_category(tournament)
    return K_FACTORS[category]


# ─────────────────────────────────────────
# 3. ELO CORE FORMULAS
# ─────────────────────────────────────────

def expected_score(elo_a: float, elo_b: float) -> float:
    """
    Expected score for team A against team B.
    Returns probability that A wins (or draws weighted).
    Standard Elo formula.
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def goal_difference_multiplier(goal_diff: int) -> float:
    """
    Multiply K-factor based on goal difference.
    Bigger wins = bigger rating changes.
    FIFA uses this approach since 2018.

    1 goal difference  → 1.0x
    2 goal difference  → 1.5x
    3+ goal difference → 1.75x + extra
    """
    goal_diff = abs(goal_diff)
    if goal_diff == 0 or goal_diff == 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return 1.75 + (goal_diff - 3) * 0.1


def actual_score(home_goals: int, away_goals: int) -> tuple:
    """
    Convert match result to Elo score values.
    Win=1, Draw=0.5, Loss=0
    """
    if home_goals > away_goals:
        return 1.0, 0.0
    elif home_goals == away_goals:
        return 0.5, 0.5
    else:
        return 0.0, 1.0


def update_elo(home_elo: float, away_elo: float,
               home_goals: int, away_goals: int,
               tournament: str, is_neutral: bool = True) -> tuple:
    """
    Update ELO ratings after a match.

    Returns:
        (new_home_elo, new_away_elo)
    """
    k = get_k_factor(tournament)
    gd_mult = goal_difference_multiplier(home_goals - away_goals)

    # Apply home advantage if not neutral venue
    home_elo_adj = home_elo + (0 if is_neutral else HOME_ADVANTAGE)

    # Expected scores
    exp_home = expected_score(home_elo_adj, away_elo)
    exp_away = 1 - exp_home

    # Actual scores
    act_home, act_away = actual_score(home_goals, away_goals)

    # Update ratings
    new_home = home_elo + k * gd_mult * (act_home - exp_home)
    new_away = away_elo + k * gd_mult * (act_away - exp_away)

    return new_home, new_away


# ─────────────────────────────────────────
# 4. BUILD DYNAMIC RATINGS
# ─────────────────────────────────────────

def build_dynamic_elo(matches: pd.DataFrame,
                       start_date: str = "2006-01-01") -> dict:
    """
    Replay all matches chronologically to build dynamic ELO ratings.

    Returns:
        dict of {team_name: current_elo_rating}
    """
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    # Initialize all teams at DEFAULT_ELO
    ratings = {}

    print(f"Building dynamic ELO from {len(matches)} matches...")

    for i, match in matches.iterrows():
        if i % 2000 == 0:
            print(f"  Processing match {i}/{len(matches)}...")

        home = match["home_team"]
        away = match["away_team"]

        # Initialize new teams
        if home not in ratings:
            ratings[home] = DEFAULT_ELO
        if away not in ratings:
            ratings[away] = DEFAULT_ELO

        home_elo = ratings[home]
        away_elo = ratings[away]

        try:
            home_goals = int(match["home_goals"])
            away_goals = int(match["away_goals"])
        except (ValueError, TypeError):
            continue

        tournament = match.get("tournament", "other")
        is_neutral = bool(match.get("neutral", True))

        new_home, new_away = update_elo(
            home_elo, away_elo,
            home_goals, away_goals,
            tournament, is_neutral
        )

        ratings[home] = new_home
        ratings[away] = new_away

    print(f"  Done. {len(ratings)} teams rated.")
    return ratings


# ─────────────────────────────────────────
# 5. ELO TO EXPECTED GOALS
# ─────────────────────────────────────────

def elo_to_xg(home_elo: float, away_elo: float,
               avg_goals: float = 1.35) -> tuple:
    """
    Convert dynamic ELO ratings to expected goals.
    Uses win probability to scale expected goals.
    """
    exp_home = expected_score(home_elo, away_elo)
    exp_away = 1 - exp_home

    # Scale around average goals
    # exp=0.5 → avg_goals, exp=0.8 → higher, exp=0.2 → lower
    home_xg = avg_goals * (exp_home / 0.5) * 0.85
    away_xg = avg_goals * (exp_away / 0.5) * 0.85

    home_xg = np.clip(home_xg, 0.3, 4.0)
    away_xg = np.clip(away_xg, 0.3, 4.0)

    return round(home_xg, 3), round(away_xg, 3)


# ─────────────────────────────────────────
# 6. PREDICT MATCH
# ─────────────────────────────────────────

def predict_match_dynamic(home_team: str, away_team: str,
                           ratings: dict,
                           max_goals: int = 6) -> dict:
    """Predict a match using dynamic ELO ratings."""

    home_elo = ratings.get(home_team, DEFAULT_ELO)
    away_elo = ratings.get(away_team, DEFAULT_ELO)

    home_xg, away_xg = elo_to_xg(home_elo, away_elo)

    scorelines = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            scorelines.append({
                "home_goals": h,
                "away_goals": a,
                "probability": max(p, 0)
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
        "home_elo": round(home_elo, 1),
        "away_elo": round(away_elo, 1),
        "elo_diff": round(home_elo - away_elo, 1),
        "home_xg": home_xg,
        "away_xg": away_xg,
        "predicted_home_goals": int(top["home_goals"]),
        "predicted_away_goals": int(top["away_goals"]),
        "home_win_prob": round(home_win, 3),
        "draw_prob": round(draw, 3),
        "away_win_prob": round(away_win, 3),
    }


# ─────────────────────────────────────────
# 7. RUN FULL PIPELINE
# ─────────────────────────────────────────

def run_dynamic_elo():
    print("=" * 60)
    print("  DYNAMIC ELO MODEL — WC 2026 Predictor")
    print("=" * 60)
    print()

    print("Loading data...")
    results = pd.read_csv(
        "data/processed/results_clean.csv", parse_dates=["date"]
    )
    fixtures = pd.read_csv(
        "data/processed/fixtures_clean.csv", parse_dates=["date"]
    )
    print(f"  Loaded {len(results)} historical matches")
    print()

    # Build dynamic ratings by replaying all matches
    ratings = build_dynamic_elo(results)

    # Print top 20 teams by current dynamic ELO
    print()
    print("Top 20 Teams by Dynamic ELO:")
    print("-" * 40)
    top_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for i, (team, elo) in enumerate(top_teams[:20], 1):
        bar = "█" * int((elo - 1400) / 20)
        print(f"  {i:2}. {team:25} {elo:7.1f} {bar}")

    # Predict all WC 2026 fixtures
    print()
    print("Predicting WC 2026 Group Stage...")
    predictions = []

    for _, fixture in fixtures.iterrows():
        pred = predict_match_dynamic(
            home_team=fixture["home_team"],
            away_team=fixture["away_team"],
            ratings=ratings,
        )
        pred["match_id"] = fixture["match_id"]
        pred["group"] = fixture["group"]
        pred["matchday"] = fixture["matchday"]
        pred["city"] = fixture["city"]
        predictions.append(pred)

    out_df = pd.DataFrame(predictions)

    # Save
    os.makedirs("data/predictions", exist_ok=True)
    out_df.to_csv("data/predictions/dynamic_elo_all.csv", index=False)
    out_df[out_df["matchday"] == 1].to_csv(
        "data/predictions/dynamic_elo_md1.csv", index=False
    )

    # Print
    print()
    print("Dynamic ELO Predictions — WC 2026 Group Stage")
    print("=" * 60)
    for _, row in out_df.iterrows():
        print(
            f"Group {row['group']} MD{row['matchday']} | "
            f"{row['home_team']:20} {row['predicted_home_goals']}-"
            f"{row['predicted_away_goals']} {row['away_team']:20} | "
            f"ELO: {row['home_elo']:.0f}-{row['away_elo']:.0f} | "
            f"xG: {row['home_xg']:.2f}-{row['away_xg']:.2f} | "
            f"H:{row['home_win_prob']} "
            f"D:{row['draw_prob']} "
            f"A:{row['away_win_prob']}"
        )

    print()
    print("Saved to data/predictions/dynamic_elo_all.csv")
    print("Saved to data/predictions/dynamic_elo_md1.csv")


if __name__ == "__main__":
    run_dynamic_elo()