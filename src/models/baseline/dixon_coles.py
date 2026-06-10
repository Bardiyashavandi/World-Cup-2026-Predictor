import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append("src")


# ─────────────────────────────────────────
# 1. TIME WEIGHTING
# ─────────────────────────────────────────

def time_weight(date, reference_date, xi: float = 0.003) -> float:
    days_ago = (reference_date - date).days
    return np.exp(-xi * days_ago)


# ─────────────────────────────────────────
# 2. DIXON-COLES CORRECTION
# ─────────────────────────────────────────

def dc_correction_vectorized(h, a, home_xg, away_xg, rho):
    """Vectorized Dixon-Coles correction for all matches at once."""
    correction = np.ones(len(h))
    mask_00 = (h == 0) & (a == 0)
    mask_10 = (h == 1) & (a == 0)
    mask_01 = (h == 0) & (a == 1)
    mask_11 = (h == 1) & (a == 1)
    correction[mask_00] = 1 - home_xg[mask_00] * away_xg[mask_00] * rho
    correction[mask_10] = 1 + away_xg[mask_10] * rho
    correction[mask_01] = 1 + home_xg[mask_01] * rho
    correction[mask_11] = 1 - rho
    return correction


def dc_correction_single(home_goals, away_goals, home_xg, away_xg, rho):
    """Single match Dixon-Coles correction for prediction."""
    if home_goals == 0 and away_goals == 0:
        return 1 - home_xg * away_xg * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + away_xg * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + home_xg * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0


# ─────────────────────────────────────────
# 3. MODEL PARAMETERS
# ─────────────────────────────────────────

def build_params(teams: list) -> tuple:
    n = len(teams)
    params = np.array(
        [1.0] * n +
        [1.0] * n +
        [1.1] +
        [0.1]
    )
    idx = {team: i for i, team in enumerate(teams)}
    return params, idx


def extract_params(params: np.ndarray, idx: dict, team: str) -> tuple:
    n = len(idx)
    attack = params[idx[team]]
    defense = params[n + idx[team]]
    return attack, defense


# ─────────────────────────────────────────
# 4. LOG LIKELIHOOD — VECTORIZED
# ─────────────────────────────────────────

iteration_count = [0]


def neg_log_likelihood(params: np.ndarray, matches: pd.DataFrame,
                        idx: dict, reference_date) -> float:
    iteration_count[0] += 1
    n = len(idx)
    home_adv = params[-2]
    rho = params[-1]

    if home_adv < 0 or rho < -1 or rho > 1:
        return 1e10

    # Vectorized — all matches at once
    home_idx = matches["home_idx"].values
    away_idx = matches["away_idx"].values
    h = matches["home_goals"].values.astype(int)
    a = matches["away_goals"].values.astype(int)
    weights = matches["weight"].values

    alpha_h = params[home_idx]
    beta_h = params[n + home_idx]
    alpha_a = params[away_idx]
    beta_a = params[n + away_idx]

    home_xg = alpha_h * beta_a * home_adv
    away_xg = alpha_a * beta_h

    if np.any(home_xg <= 0) or np.any(away_xg <= 0):
        return 1e10

    p_home = poisson.pmf(h, home_xg)
    p_away = poisson.pmf(a, away_xg)
    correction = dc_correction_vectorized(h, a, home_xg, away_xg, rho)

    prob = p_home * p_away * correction
    prob = np.clip(prob, 1e-10, None)

    loss = -np.sum(weights * np.log(prob))

    if iteration_count[0] % 5 == 0:
        bar_len = 30
        progress = min(iteration_count[0] / 100, 1.0)
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  Iteration {iteration_count[0]:>4} [{bar}] "
            f"Loss: {loss:>12.2f} | "
            f"home_adv: {home_adv:.3f} | rho: {rho:.3f}",
            end="", flush=True
        )

    return loss


# ─────────────────────────────────────────
# 5. MODEL FITTING
# ─────────────────────────────────────────

def fit_model(matches: pd.DataFrame, reference_date=None) -> tuple:
    if reference_date is None:
        reference_date = matches["date"].max()

    matches = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    matches["date"] = pd.to_datetime(matches["date"])

    teams = sorted(set(
        matches["home_team"].unique().tolist() +
        matches["away_team"].unique().tolist()
    ))

    print(f"Fitting Dixon-Coles on {len(matches)} matches, {len(teams)} teams...")
    print(f"Optimizing {len(teams) * 2 + 2} parameters...")
    print()

    iteration_count[0] = 0

    params, idx = build_params(teams)

    # Precompute indices and weights — key to vectorized speed
    matches["home_idx"] = matches["home_team"].map(idx)
    matches["away_idx"] = matches["away_team"].map(idx)
    matches["weight"] = matches["date"].apply(
        lambda d: time_weight(d, reference_date)
    )
    matches = matches.dropna(subset=["home_idx", "away_idx"])
    matches["home_idx"] = matches["home_idx"].astype(int)
    matches["away_idx"] = matches["away_idx"].astype(int)

    n = len(teams)
    bounds = (
        [(0.01, 5.0)] * n +
        [(0.01, 5.0)] * n +
        [(0.5, 2.0)] +
        [(-0.99, 0.99)]
    )

    result = minimize(
        neg_log_likelihood,
        params,
        args=(matches, idx, reference_date),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100, "disp": False}
    )

    print()
    print()
    print(f"  ✓ Optimization complete in {iteration_count[0]} iterations")
    print(f"  ✓ Converged: {result.success}")
    print(f"  ✓ Home advantage: {result.x[-2]:.3f}")
    print(f"  ✓ Rho (DC correction): {result.x[-1]:.3f}")

    return result.x, idx, teams


# ─────────────────────────────────────────
# 6. PREDICTION WITH ELO FALLBACK
# ─────────────────────────────────────────

def predict_match_dc(home_team: str, away_team: str,
                      params: np.ndarray, idx: dict,
                      elo_features: dict = None,
                      max_goals: int = 6) -> dict:
    """
    Predict a single match using fitted Dixon-Coles parameters.
    Falls back to ELO-based xG if team not in training data.
    """
    home_adv = params[-2]
    rho = params[-1]

    home_known = home_team in idx
    away_known = away_team in idx

    if home_known and away_known:
        # Both teams in training data — full Dixon-Coles
        alpha_h, beta_h = extract_params(params, idx, home_team)
        alpha_a, beta_a = extract_params(params, idx, away_team)
        home_xg = alpha_h * beta_a * home_adv
        away_xg = alpha_a * beta_h
        method = "dixon_coles"

    elif elo_features is not None:
        # At least one team unknown — fall back to ELO
        elo_diff = elo_features.get("elo_diff", 0)
        avg_goals = 1.35
        scale = 0.001
        home_xg = float(np.clip(avg_goals + elo_diff * scale, 0.3, 4.0))
        away_xg = float(np.clip(avg_goals - elo_diff * scale, 0.3, 4.0))
        method = "elo_fallback"

    else:
        # No ELO data either — use global average
        home_xg = 1.35
        away_xg = 1.10
        method = "global_average"

    scorelines = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = (poisson.pmf(h, home_xg) *
                 poisson.pmf(a, away_xg) *
                 dc_correction_single(h, a, home_xg, away_xg, rho))
            scorelines.append({
                "home_goals": h,
                "away_goals": a,
                "probability": max(p, 0)
            })

    score_df = pd.DataFrame(scorelines)
    score_df = score_df.sort_values(
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

    # The Dixon-Coles low-score correction and the truncated scoreline
    # grid leave the three outcome probabilities summing to slightly
    # under 1. Normalize so they form a proper distribution.
    total = home_win + draw + away_win
    if total > 0:
        home_win, draw, away_win = (
            home_win / total, draw / total, away_win / total
        )

    top = score_df.iloc[0]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": round(home_xg, 3),
        "away_xg": round(away_xg, 3),
        "predicted_home_goals": int(top["home_goals"]),
        "predicted_away_goals": int(top["away_goals"]),
        "home_win_prob": round(home_win, 3),
        "draw_prob": round(draw, 3),
        "away_win_prob": round(away_win, 3),
        "method": method,
    }


# ─────────────────────────────────────────
# 7. RUN FULL PIPELINE
# ─────────────────────────────────────────

def run_dixon_coles():
    print("=" * 60)
    print("  DIXON-COLES MODEL — WC 2026 Predictor")
    print("=" * 60)
    print()

    print("Loading data...")
    results = pd.read_csv(
        "data/processed/results_clean.csv", parse_dates=["date"]
    )
    fixtures = pd.read_csv(
        "data/processed/fixtures_clean.csv", parse_dates=["date"]
    )

    # Load ELO features for fallback
    pred_features = pd.read_csv("data/processed/predict_features.csv")
    elo_lookup = pred_features.set_index(
        ["home_team", "away_team"]
    )["elo_diff"].to_dict()

    # Filter to teams with at least 200 matches
    all_teams = pd.concat([results["home_team"], results["away_team"]])
    team_counts = all_teams.value_counts()
    active_teams = team_counts[team_counts >= 200].index.tolist()
    results = results[
        results["home_team"].isin(active_teams) &
        results["away_team"].isin(active_teams)
    ].copy()

    print(f"Filtered to {len(active_teams)} teams, {len(results)} matches")
    print()

    # Fit model
    reference_date = pd.Timestamp("2026-06-11")
    params, idx, teams = fit_model(results, reference_date)

    # Predict all WC 2026 fixtures
    print()
    print("Predicting WC 2026 Group Stage...")
    predictions = []

    for _, fixture in fixtures.iterrows():
        elo_diff = elo_lookup.get(
            (fixture["home_team"], fixture["away_team"]), 0
        )
        pred = predict_match_dc(
            home_team=fixture["home_team"],
            away_team=fixture["away_team"],
            params=params,
            idx=idx,
            elo_features={"elo_diff": elo_diff},
        )
        pred["match_id"] = fixture["match_id"]
        pred["group"] = fixture["group"]
        pred["matchday"] = fixture["matchday"]
        pred["city"] = fixture["city"]
        predictions.append(pred)

    out_df = pd.DataFrame(predictions)

    # Save
    os.makedirs("data/predictions", exist_ok=True)
    out_df.to_csv("data/predictions/dixon_coles_all.csv", index=False)
    out_df[out_df["matchday"] == 1].to_csv(
        "data/predictions/dixon_coles_md1.csv", index=False
    )

    # Print results
    print()
    print("Dixon-Coles Predictions — WC 2026 Group Stage")
    print("=" * 60)
    for _, row in out_df.iterrows():
        print(
            f"Group {row['group']} MD{row['matchday']} | "
            f"{row['home_team']:20} {row['predicted_home_goals']}-"
            f"{row['predicted_away_goals']} {row['away_team']:20} | "
            f"xG: {row['home_xg']:.2f}-{row['away_xg']:.2f} | "
            f"H:{row['home_win_prob']} "
            f"D:{row['draw_prob']} "
            f"A:{row['away_win_prob']} "
            f"[{row['method']}]"
        )

    print()
    print("Saved to data/predictions/dixon_coles_all.csv")
    print("Saved to data/predictions/dixon_coles_md1.csv")


if __name__ == "__main__":
    run_dixon_coles()