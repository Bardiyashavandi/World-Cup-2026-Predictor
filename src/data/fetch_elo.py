import pandas as pd
import os

def load_elo_ratings(path: str = "data/raw/elo_ratings_wc2026.csv") -> pd.DataFrame:
    """
    Load the full historical ELO dataset for all 48 WC 2026 teams.
    Source: Kaggle - 2026 FIFA World Cup Historical ELO Ratings
    """
    df = pd.read_csv(path)
    return df


def get_latest_elo(path: str = "data/raw/elo_ratings_wc2026.csv") -> pd.DataFrame:
    """
    Get the most recent ELO snapshot for each team.
    This is what we'll use as our pre-tournament strength baseline.
    """
    df = load_elo_ratings(path)

    # Get the latest snapshot per team
    latest = (
        df.sort_values("snapshot_date", ascending=False)
        .groupby("country")
        .first()
        .reset_index()
    )

    # Keep only what we need for modeling
    cols = [
        "country", "country_code", "confederation",
        "rating", "rank",
        "matches_total", "wins", "losses", "draws",
        "goals_for", "goals_against",
        "is_host", "snapshot_date"
    ]
    latest = latest[cols].sort_values("rating", ascending=False).reset_index(drop=True)

    # Add derived features useful for modeling
    latest["win_rate"] = latest["wins"] / latest["matches_total"]
    latest["goals_per_match"] = latest["goals_for"] / latest["matches_total"]
    latest["goals_against_per_match"] = latest["goals_against"] / latest["matches_total"]
    latest["goal_difference"] = latest["goals_for"] - latest["goals_against"]

    return latest


def save_latest_elo(out_path: str = "data/processed/elo_latest.csv"):
    """Save the latest ELO snapshot to processed folder."""
    os.makedirs("data/processed", exist_ok=True)
    df = get_latest_elo()
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} teams to {out_path}")
    print()
    print(df[["country", "rating", "rank", "win_rate", "goals_per_match"]].head(15).to_string())


if __name__ == "__main__":
    save_latest_elo()