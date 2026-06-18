import pandas as pd
import os
from normalize_teams import normalize_dataframe

def process_results(path: str = "data/raw/results.csv") -> pd.DataFrame:
    """
    Clean and filter the main international results dataset.
    Keeps only matches from 1990 onwards and relevant tournaments.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Filter from 1990 onwards — older data is less relevant
    df = df[df["date"] >= "2006-01-01"].copy()

    # Normalize team names
    df = normalize_dataframe(df, ["home_team", "away_team"])

    # Add useful derived columns
    df["home_goals"] = df["home_score"]
    df["away_goals"] = df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_difference"] = df["home_score"] - df["away_score"]

    df["result"] = df["goal_difference"].apply(
        lambda x: "H" if x > 0 else ("A" if x < 0 else "D")
    )

    # Flag tournament type
    df["is_world_cup"] = df["tournament"].str.contains(
        "FIFA World Cup", na=False
    ).astype(int)

    df["is_friendly"] = df["tournament"].str.contains(
        "Friendly", na=False
    ).astype(int)

    df = df.sort_values("date").reset_index(drop=True)

    return df


def merge_played_wc_results(
    results: pd.DataFrame,
    path: str = "data/raw/wc_2026_results.csv",
) -> pd.DataFrame:
    """Fold real, played WC 2026 results into the historical match set.

    Without this, form / head-to-head / ELO features never reflect what
    actually happened during the tournament — so live updating after each
    matchday had no real effect. Played rows are converted to the same
    schema as the historical results and appended (de-duplicated so the
    step is safe to re-run with updated scores).
    """
    if not os.path.exists(path):
        return results

    wc = pd.read_csv(path)
    wc = wc[wc["played"].astype(str).str.lower() == "true"].copy()
    wc = wc.dropna(subset=["home_goals", "away_goals"])
    if len(wc) == 0:
        return results

    wc["date"] = pd.to_datetime(wc["date"])
    wc = normalize_dataframe(wc, ["home_team", "away_team"])
    wc["home_score"] = wc["home_goals"].astype(int)
    wc["away_score"] = wc["away_goals"].astype(int)
    wc["home_goals"] = wc["home_score"]
    wc["away_goals"] = wc["away_score"]
    wc["tournament"] = "FIFA World Cup"
    wc["country"] = wc.get("city", "")
    wc["neutral"] = True
    wc["total_goals"] = wc["home_score"] + wc["away_score"]
    wc["goal_difference"] = wc["home_score"] - wc["away_score"]
    wc["result"] = wc["goal_difference"].apply(
        lambda x: "H" if x > 0 else ("A" if x < 0 else "D")
    )
    wc["is_world_cup"] = 1
    wc["is_friendly"] = 0

    # Align to the historical schema, then append.
    for col in results.columns:
        if col not in wc.columns:
            wc[col] = None
    wc = wc[results.columns]

    combined = pd.concat([results, wc], ignore_index=True)
    key = (combined["date"].dt.strftime("%Y-%m-%d") + "|"
           + combined["home_team"].astype(str) + "|"
           + combined["away_team"].astype(str))
    combined = combined[~key.duplicated(keep="last")]
    combined = combined.sort_values("date").reset_index(drop=True)

    n_added = len(combined) - len(results)
    print(f"  Folded in {len(wc)} played WC 2026 matches "
          f"(+{n_added} new rows)")
    return combined


def process_fixtures(path: str = "data/raw/wc_2026_fixtures.csv") -> pd.DataFrame:
    """Clean the 2026 WC fixtures."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = normalize_dataframe(df, ["home_team", "away_team"])
    return df


def process_elo(path: str = "data/processed/elo_latest.csv") -> pd.DataFrame:
    """Load the latest ELO ratings."""
    df = pd.read_csv(path)
    df = normalize_dataframe(df, ["country"])
    return df


def save_all(out_dir: str = "data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    print("Processing results...")
    results = process_results()
    results = merge_played_wc_results(results)
    results.to_csv(f"{out_dir}/results_clean.csv", index=False)
    print(f"  Saved {len(results)} matches to results_clean.csv")

    print("Processing fixtures...")
    fixtures = process_fixtures()
    fixtures.to_csv(f"{out_dir}/fixtures_clean.csv", index=False)
    print(f"  Saved {len(fixtures)} fixtures to fixtures_clean.csv")

    print("Processing ELO...")
    elo = process_elo()
    elo.to_csv(f"{out_dir}/elo_clean.csv", index=False)
    print(f"  Saved {len(elo)} teams to elo_clean.csv")

    print("\nAll done. Processed files:")
    for f in os.listdir(out_dir):
        print(f"  data/processed/{f}")


if __name__ == "__main__":
    save_all()