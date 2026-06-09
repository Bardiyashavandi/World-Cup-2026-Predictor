import pandas as pd
import numpy as np
import os
import sys

sys.path.append("src")


# ─────────────────────────────────────────
# 1. COMPUTE GROUP STANDINGS
# ─────────────────────────────────────────

def compute_standings(results: pd.DataFrame,
                       fixtures: pd.DataFrame,
                       after_matchday: int) -> pd.DataFrame:
    """
    Compute group standings after a given matchday.

    Args:
        results: WC results so far (with home_goals, away_goals filled)
        fixtures: full fixture list with group and matchday info
        after_matchday: compute standings after this matchday (1 or 2)

    Returns:
        DataFrame with one row per team with standing info
    """
    # Get played matches up to and including after_matchday
    played_fixtures = fixtures[
        fixtures["matchday"] <= after_matchday
    ].copy()

    # Merge with results
    played = played_fixtures.merge(
        results[["match_id", "home_goals", "away_goals", "played"]],
        on="match_id",
        how="left"
    )
    played = played[played["played"] == True].copy()

    if len(played) == 0:
        return pd.DataFrame()

    # Build standings
    standings = []

    groups = fixtures["group"].unique()
    for group in sorted(groups):
        group_fixtures = fixtures[fixtures["group"] == group]
        teams = pd.concat([
            group_fixtures["home_team"],
            group_fixtures["away_team"]
        ]).unique()

        for team in teams:
            team_matches = played[
                (played["group"] == group) &
                (
                    (played["home_team"] == team) |
                    (played["away_team"] == team)
                )
            ]

            pts = gf = ga = wins = draws = losses = 0

            for _, m in team_matches.iterrows():
                if m["home_team"] == team:
                    scored = m["home_goals"]
                    conceded = m["away_goals"]
                else:
                    scored = m["away_goals"]
                    conceded = m["home_goals"]

                gf += scored
                ga += conceded

                if scored > conceded:
                    pts += 3
                    wins += 1
                elif scored == conceded:
                    pts += 1
                    draws += 1
                else:
                    losses += 1

            standings.append({
                "group": group,
                "team": team,
                "matches_played": len(team_matches),
                "points": pts,
                "goals_for": gf,
                "goals_against": ga,
                "goal_difference": gf - ga,
                "wins": wins,
                "draws": draws,
                "losses": losses,
            })

    standings_df = pd.DataFrame(standings)

    # Add group position
    standings_df = standings_df.sort_values(
        ["group", "points", "goal_difference", "goals_for"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)

    standings_df["group_position"] = standings_df.groupby(
        "group"
    ).cumcount() + 1

    return standings_df


# ─────────────────────────────────────────
# 2. COMPUTE STAKES FEATURES
# ─────────────────────────────────────────

def compute_stakes_features(standings: pd.DataFrame,
                              fixtures: pd.DataFrame,
                              matchday: int) -> pd.DataFrame:
    """
    For each fixture in the given matchday, compute stakes features
    for both teams based on current standings.

    Stakes features:
    - points: current points
    - group_position: current position (1-4)
    - matches_played: matches played so far
    - gd: current goal difference
    - can_qualify: can still mathematically reach top 2?
    - already_qualified: mathematically through already?
    - must_win: must win to have any chance?
    - elimination_risk: high if currently 3rd/4th with few points
    - points_needed: minimum points needed to qualify
    - matchday: which matchday this is
    """
    target_fixtures = fixtures[
        fixtures["matchday"] == matchday
    ].copy()

    if len(standings) == 0:
        # MD1 — no standings yet, everyone equal
        target_fixtures["home_points"] = 0
        target_fixtures["home_position"] = 2
        target_fixtures["home_matches_played"] = 0
        target_fixtures["home_gd"] = 0
        target_fixtures["home_can_qualify"] = 1
        target_fixtures["home_already_qualified"] = 0
        target_fixtures["home_must_win"] = 0
        target_fixtures["home_elimination_risk"] = 0
        target_fixtures["home_points_needed"] = 4
        target_fixtures["away_points"] = 0
        target_fixtures["away_position"] = 2
        target_fixtures["away_matches_played"] = 0
        target_fixtures["away_gd"] = 0
        target_fixtures["away_can_qualify"] = 1
        target_fixtures["away_already_qualified"] = 0
        target_fixtures["away_must_win"] = 0
        target_fixtures["away_elimination_risk"] = 0
        target_fixtures["away_points_needed"] = 4
        target_fixtures["matchday"] = matchday
        return target_fixtures

    rows = []
    for _, fixture in target_fixtures.iterrows():
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]
        group = fixture["group"]

        # Get team standings
        home_row = standings[
            (standings["team"] == home_team) &
            (standings["group"] == group)
        ]
        away_row = standings[
            (standings["team"] == away_team) &
            (standings["group"] == group)
        ]

        def get_stakes(row, md):
            if len(row) == 0:
                return {
                    "points": 0, "position": 2,
                    "matches_played": 0, "gd": 0,
                    "can_qualify": 1, "already_qualified": 0,
                    "must_win": 0, "elimination_risk": 0,
                    "points_needed": 4
                }

            pts = int(row["points"].values[0])
            pos = int(row["group_position"].values[0])
            mp = int(row["matches_played"].values[0])
            gd = int(row["goal_difference"].values[0])

            remaining = 3 - mp
            max_possible = pts + remaining * 3
            points_needed = max(0, 4 - pts)

            already_qualified = int(pts >= 6 and mp >= 2)
            can_qualify = int(max_possible >= 4)
            must_win = int(
                can_qualify and
                pts <= 1 and
                md == 3
            )

            if pos >= 3 and pts <= 1:
                elimination_risk = 1
            elif pos >= 3 and pts <= 3:
                elimination_risk = 0.5
            else:
                elimination_risk = 0

            return {
                "points": pts,
                "position": pos,
                "matches_played": mp,
                "gd": gd,
                "can_qualify": can_qualify,
                "already_qualified": already_qualified,
                "must_win": must_win,
                "elimination_risk": elimination_risk,
                "points_needed": points_needed,
            }

        home_stakes = get_stakes(home_row, matchday)
        away_stakes = get_stakes(away_row, matchday)

        row_data = fixture.to_dict()
        for k, v in home_stakes.items():
            row_data[f"home_{k}"] = v
        for k, v in away_stakes.items():
            row_data[f"away_{k}"] = v
        row_data["matchday"] = matchday

        rows.append(row_data)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 3. SAVE STANDINGS
# ─────────────────────────────────────────

def save_standings(matchday: int = 1):
    """
    Compute and save standings after a given matchday.
    Run this after each matchday with real results.
    """
    fixtures = pd.read_csv("data/raw/wc_2026_fixtures.csv")
    results = pd.read_csv("data/raw/wc_2026_results.csv")

    standings = compute_standings(results, fixtures, matchday)

    os.makedirs("data/processed", exist_ok=True)
    out_path = f"data/processed/standings_after_md{matchday}.csv"

    if len(standings) == 0:
        print(f"No results yet for matchday {matchday} — standings empty.")
        print("Fill in data/raw/wc_2026_results.csv with real results first.")
        return standings

    standings.to_csv(out_path, index=False)

    print(f"Standings after Matchday {matchday}:")
    print("=" * 55)
    for group in sorted(standings["group"].unique()):
        print(f"\nGroup {group}:")
        group_df = standings[
            standings["group"] == group
        ].sort_values("group_position")
        for _, row in group_df.iterrows():
            print(
                f"  {row['group_position']}. {row['team']:25} "
                f"Pts:{row['points']} "
                f"GD:{row['goal_difference']:+d} "
                f"GF:{row['goals_for']}"
            )

    print(f"\nSaved to {out_path}")
    return standings


if __name__ == "__main__":
    save_standings(matchday=1)