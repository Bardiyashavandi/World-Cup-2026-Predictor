"""
Bayesian Updater — updates team strength estimates after each matchday.

After real WC results come in, this script:
1. Loads prior team strength estimates (from historical data)
2. Updates them based on observed goals scored/conceded
3. Saves updated estimates for use in next matchday predictions

This is Bayesian updating in its simplest form:
    posterior = (prior * prior_weight + observed * obs_weight) / total_weight

The more matches we observe, the more we trust the observed data
and the less we rely on the historical prior.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append("src")
sys.path.append("src/data")

from normalize_teams import normalize_team_name


# ─────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────

PRIOR_WEIGHT = 10.0


# ─────────────────────────────────────────
# 2. BUILD PRIOR STRENGTH ESTIMATES
# ─────────────────────────────────────────

def build_team_priors(results: pd.DataFrame,
                       teams: list,
                       lookback_matches: int = 20) -> pd.DataFrame:
    """
    Build prior attack/defense strength for each team
    from their last N historical matches.

    Attack strength  = average goals scored per match
    Defense weakness = average goals conceded per match

    Note: We drop NaN goals rows to exclude unplayed fixtures
    that may have been added to results_clean.csv.
    """
    priors = []

    for team in teams:
        mask = (
            (results["home_team"] == team) |
            (results["away_team"] == team)
        )
        # Drop NaN goals — excludes unplayed future fixtures
        team_matches = results[mask].dropna(
            subset=["home_goals", "away_goals"]
        ).sort_values(
            "date", ascending=False
        ).head(lookback_matches)

        if len(team_matches) == 0:
            priors.append({
                "team": team,
                "attack_prior": 1.2,
                "defense_prior": 1.2,
                "n_matches": 0,
            })
            continue

        goals_scored = []
        goals_conceded = []

        for _, m in team_matches.iterrows():
            if m["home_team"] == team:
                goals_scored.append(float(m["home_goals"]))
                goals_conceded.append(float(m["away_goals"]))
            else:
                goals_scored.append(float(m["away_goals"]))
                goals_conceded.append(float(m["home_goals"]))

        priors.append({
            "team": team,
            "attack_prior": round(np.mean(goals_scored), 4),
            "defense_prior": round(np.mean(goals_conceded), 4),
            "n_matches": len(team_matches),
        })

    return pd.DataFrame(priors)


# ─────────────────────────────────────────
# 3. BAYESIAN UPDATE
# ─────────────────────────────────────────

def bayesian_update(prior: float,
                     observed: float,
                     n_observed: int,
                     prior_weight: float = PRIOR_WEIGHT) -> float:
    """
    Update a strength estimate using Bayesian updating.

    Formula:
        posterior = (prior * prior_weight + observed * n_observed)
                    / (prior_weight + n_observed)

    Example:
        prior = 1.8 goals/match (from 20 historical matches)
        observed = 2.5 goals/match (from 1 WC match)
        prior_weight = 10

        posterior = (1.8 * 10 + 2.5 * 1) / (10 + 1)
                  = (18 + 2.5) / 11
                  = 1.864 ← moves toward observed but not fully
    """
    posterior = (
        (prior * prior_weight + observed * n_observed) /
        (prior_weight + n_observed)
    )
    return round(posterior, 4)


def update_team_strengths(priors: pd.DataFrame,
                           wc_results: pd.DataFrame,
                           matchday: int) -> pd.DataFrame:
    """
    Update team strength estimates using actual WC results
    up to and including the given matchday.
    """
    played = wc_results[
        (wc_results["matchday"] <= matchday) &
        (wc_results["played"] == True)
    ].copy()

    if len(played) == 0:
        print(f"  No played matches found for MD{matchday}")
        updated = priors.copy()
        updated["attack_posterior"] = updated["attack_prior"]
        updated["defense_posterior"] = updated["defense_prior"]
        updated["wc_matches"] = 0
        updated["wc_goals_scored"] = 0.0
        updated["wc_goals_conceded"] = 0.0
        return updated

    print(f"  Updating from {len(played)} played WC matches...")

    updated = priors.copy()
    updated["attack_posterior"] = updated["attack_prior"]
    updated["defense_posterior"] = updated["defense_prior"]
    updated["wc_matches"] = 0
    updated["wc_goals_scored"] = 0.0
    updated["wc_goals_conceded"] = 0.0

    for idx, row in updated.iterrows():
        team = row["team"]

        team_matches = played[
            (played["home_team"] == team) |
            (played["away_team"] == team)
        ]

        if len(team_matches) == 0:
            continue

        wc_scored = []
        wc_conceded = []

        for _, m in team_matches.iterrows():
            if m["home_team"] == team:
                wc_scored.append(float(m["home_goals"]))
                wc_conceded.append(float(m["away_goals"]))
            else:
                wc_scored.append(float(m["away_goals"]))
                wc_conceded.append(float(m["home_goals"]))

        obs_attack = np.mean(wc_scored)
        obs_defense = np.mean(wc_conceded)
        n_obs = len(team_matches)

        new_attack = bayesian_update(
            row["attack_prior"], obs_attack, n_obs
        )
        new_defense = bayesian_update(
            row["defense_prior"], obs_defense, n_obs
        )

        updated.at[idx, "attack_posterior"] = new_attack
        updated.at[idx, "defense_posterior"] = new_defense
        updated.at[idx, "wc_matches"] = n_obs
        updated.at[idx, "wc_goals_scored"] = sum(wc_scored)
        updated.at[idx, "wc_goals_conceded"] = sum(wc_conceded)

    return updated


# ─────────────────────────────────────────
# 4. PRINT UPDATES
# ─────────────────────────────────────────

def print_strength_updates(updated: pd.DataFrame):
    """Print teams whose strength estimates changed most."""

    if "attack_posterior" not in updated.columns:
        print("  No updates yet — no matches played")
        return

    updated = updated.copy()
    updated["attack_change"] = (
        updated["attack_posterior"] - updated["attack_prior"]
    ).abs()
    updated["defense_change"] = (
        updated["defense_posterior"] - updated["defense_prior"]
    ).abs()
    updated["total_change"] = (
        updated["attack_change"] + updated["defense_change"]
    )

    changed = updated[updated["wc_matches"] > 0].sort_values(
        "total_change", ascending=False
    )

    if len(changed) == 0:
        print("  No updates yet — no matches played")
        return

    print(
        f"\n  {'Team':25} {'Atk Prior':>10} {'Atk Post':>10} "
        f"{'Def Prior':>10} {'Def Post':>10} {'WC':>4}"
    )
    print(
        f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*4}"
    )

    for _, row in changed.iterrows():
        atk_arrow = (
            "↑" if row["attack_posterior"] > row["attack_prior"]
            else "↓"
        )
        def_arrow = (
            "↑" if row["defense_posterior"] > row["defense_prior"]
            else "↓"
        )
        print(
            f"  {row['team']:25} "
            f"{row['attack_prior']:>10.3f} "
            f"{row['attack_posterior']:>9.3f}{atk_arrow} "
            f"{row['defense_prior']:>10.3f} "
            f"{row['defense_posterior']:>9.3f}{def_arrow} "
            f"{int(row['wc_matches']):>4}"
        )


# ─────────────────────────────────────────
# 5. SHOW PRIORS SUMMARY
# ─────────────────────────────────────────

def print_priors_summary(priors: pd.DataFrame):
    """Print team priors sorted by attack strength."""
    valid = priors.dropna(subset=["attack_prior"]).copy()

    print(f"\n  {'Team':25} {'Attack':>8} {'Defense':>8} {'Matches':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")

    for _, row in valid.sort_values(
        "attack_prior", ascending=False
    ).head(20).iterrows():
        print(
            f"  {row['team']:25} "
            f"{row['attack_prior']:>8.3f} "
            f"{row['defense_prior']:>8.3f} "
            f"{int(row['n_matches']):>8}"
        )


# ─────────────────────────────────────────
# 6. RUN FULL PIPELINE
# ─────────────────────────────────────────

def run_bayesian_update(matchday: int = 1):
    """
    Run Bayesian update after a given matchday.
    Saves updated team strengths for use in next predictions.
    """
    print("=" * 60)
    print("  BAYESIAN UPDATER — WC 2026 Predictor")
    print("=" * 60)
    print()

    print("Loading data...")
    results = pd.read_csv(
        "data/processed/results_clean.csv", parse_dates=["date"]
    )

    # Use cleaned fixtures — normalized team names
    fixtures = pd.read_csv("data/processed/fixtures_clean.csv")

    wc_results = pd.read_csv("data/raw/wc_2026_results.csv")

    # Normalize team names in wc_results to match results_clean
    wc_results["home_team"] = wc_results["home_team"].apply(
        normalize_team_name
    )
    wc_results["away_team"] = wc_results["away_team"].apply(
        normalize_team_name
    )

    teams = sorted(set(
        fixtures["home_team"].tolist() +
        fixtures["away_team"].tolist()
    ))
    print(f"  Teams: {len(teams)}")

    # Build priors
    print("\nBuilding team priors from last 20 historical matches...")
    priors = build_team_priors(results, teams, lookback_matches=20)
    print(f"  Built priors for {len(priors)} teams")

    print("\nTop 20 teams by attack prior:")
    print_priors_summary(priors)

    # Update with WC results
    print(f"\nApplying Bayesian update from MD{matchday} results...")
    updated = update_team_strengths(priors, wc_results, matchday)

    # Print changes
    if updated["wc_matches"].sum() > 0:
        print("\nStrength updates after real WC matches:")
        print_strength_updates(updated)
    else:
        print("\n  No matches played yet — priors saved as baseline")
        print("  Run again after filling in real results")

    # Save
    os.makedirs("data/processed", exist_ok=True)
    out_path = (
        f"data/processed/team_strengths_after_md{matchday}.csv"
    )
    updated.to_csv(out_path, index=False)

    latest_path = "data/processed/team_strengths_latest.csv"
    updated.to_csv(latest_path, index=False)

    print(f"\nSaved to {out_path}")
    print(f"Saved as latest: {latest_path}")

    return updated


if __name__ == "__main__":
    md = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_bayesian_update(matchday=md)