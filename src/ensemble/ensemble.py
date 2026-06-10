"""
Ensemble — combines all model predictions into final predictions.

For each match:
1. Load predictions from all available models
2. Apply weighted average of xG values
3. Use Poisson distribution for final scoreline
4. Save final predictions

Weights are higher for MD2/MD3 on stakes model
since it has real group context then.
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append("src")


# ─────────────────────────────────────────
# 1. MODEL WEIGHTS
# ─────────────────────────────────────────

# Weights for each matchday
# Stakes model gets 0 weight for MD1 (no context yet)
# increases for MD2/MD3

# Weights rebalanced after backtesting (WC 2018 + 2022). The original
# weights gave Logistic Regression — the single strongest model — only
# 0.04, while over-weighting weak baselines (ELO, Historical Average).
# Boosting the top three (Logistic, XGBoost, LightGBM) lifted the
# ensemble to match the best individual model on result accuracy while
# keeping the best scoreline error and Brier score.
WEIGHTS = {
    1: {
        "elo":          0.06,
        "dixon_coles":  0.12,
        "historical_avg": 0.06,
        "dynamic_elo":  0.08,
        "xgboost":      0.20,
        "lightgbm":     0.20,
        "neural_net":   0.10,
        "logistic":     0.18,
        "stakes_model": 0.00,
    },
    2: {
        "elo":          0.05,
        "dixon_coles":  0.10,
        "historical_avg": 0.05,
        "dynamic_elo":  0.07,
        "xgboost":      0.17,
        "lightgbm":     0.17,
        "neural_net":   0.09,
        "logistic":     0.15,
        "stakes_model": 0.15,
    },
    3: {
        "elo":          0.04,
        "dixon_coles":  0.09,
        "historical_avg": 0.05,
        "dynamic_elo":  0.06,
        "xgboost":      0.15,
        "lightgbm":     0.15,
        "neural_net":   0.08,
        "logistic":     0.13,
        "stakes_model": 0.25,
    },
}

# Map model name to prediction file
MODEL_FILES = {
    "elo":            "data/predictions/elo_all.csv",
    "dixon_coles":    "data/predictions/dixon_coles_all.csv",
    "historical_avg": "data/predictions/historical_avg_all.csv",
    "dynamic_elo":    "data/predictions/dynamic_elo_all.csv",
    "xgboost":        "data/predictions/xgboost_all.csv",
    "lightgbm":       "data/predictions/lightgbm_all.csv",
    "neural_net":     "data/predictions/neural_net_all.csv",
    "logistic":       "data/predictions/logistic_all.csv",
}


# ─────────────────────────────────────────
# 2. LOAD PREDICTIONS
# ─────────────────────────────────────────

def load_model_predictions(matchday: int) -> dict:
    """
    Load predictions from all available models for a given matchday.
    Returns dict of {model_name: dataframe}
    """
    predictions = {}
    weights = WEIGHTS[matchday]

    for model_name, weight in weights.items():
        if weight == 0:
            continue

        # Stakes model has per-matchday files
        if model_name == "stakes_model":
            path = f"data/predictions/stakes_model_md{matchday}.csv"
        else:
            path = MODEL_FILES.get(model_name)

        if path and os.path.exists(path):
            df = pd.read_csv(path)
            # Filter to requested matchday
            if "matchday" in df.columns:
                df = df[df["matchday"] == matchday].copy()
            predictions[model_name] = df
            print(f"  ✅ Loaded {model_name}: {len(df)} matches")
        else:
            print(f"  ⚠️  Missing {model_name}: {path}")

    return predictions


# ─────────────────────────────────────────
# 3. NORMALIZE WEIGHTS
# ─────────────────────────────────────────

def get_active_weights(predictions: dict,
                        matchday: int) -> dict:
    """
    Get normalized weights for only the available models.
    If a model is missing its weight is redistributed.
    """
    base_weights = WEIGHTS[matchday]
    active = {
        k: v for k, v in base_weights.items()
        if k in predictions and v > 0
    }

    # Normalize to sum to 1
    total = sum(active.values())
    normalized = {k: v / total for k, v in active.items()}

    return normalized


# ─────────────────────────────────────────
# 4. ENSEMBLE PREDICTIONS
# ─────────────────────────────────────────

def ensemble_predictions(predictions: dict,
                          weights: dict,
                          matchday: int) -> pd.DataFrame:
    """
    Combine model predictions using weighted average of xG values.

    For logistic regression we use win probabilities directly
    since it doesn't predict xG.
    """
    # Get fixture list from first available model
    base_model = list(predictions.keys())[0]
    base_df = predictions[base_model][
        ["match_id", "group", "matchday",
         "home_team", "away_team"]
    ].copy()

    results = []

    for _, fixture in base_df.iterrows():
        match_id = fixture["match_id"]
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]

        weighted_home_xg = 0.0
        weighted_away_xg = 0.0
        weighted_home_win = 0.0
        weighted_draw = 0.0
        weighted_away_win = 0.0
        total_xg_weight = 0.0
        total_prob_weight = 0.0

        for model_name, weight in weights.items():
            if model_name not in predictions:
                continue

            model_df = predictions[model_name]
            match_row = model_df[
                model_df["match_id"] == match_id
            ]

            if len(match_row) == 0:
                continue

            row = match_row.iloc[0]

            # Logistic regression — use probabilities only
            if model_name == "logistic":
                weighted_home_win += weight * row["home_win_prob"]
                weighted_draw += weight * row["draw_prob"]
                weighted_away_win += weight * row["away_win_prob"]
                total_prob_weight += weight

            # All other models — use xG
            elif "home_xg" in row and "away_xg" in row:
                weighted_home_xg += weight * row["home_xg"]
                weighted_away_xg += weight * row["away_xg"]
                weighted_home_win += weight * row["home_win_prob"]
                weighted_draw += weight * row["draw_prob"]
                weighted_away_win += weight * row["away_win_prob"]
                total_xg_weight += weight
                total_prob_weight += weight

        # Normalize
        if total_xg_weight > 0:
            home_xg = weighted_home_xg / total_xg_weight
            away_xg = weighted_away_xg / total_xg_weight
        else:
            home_xg = 1.2
            away_xg = 1.0

        if total_prob_weight > 0:
            home_win_prob = weighted_home_win / total_prob_weight
            draw_prob = weighted_draw / total_prob_weight
            away_win_prob = weighted_away_win / total_prob_weight
        else:
            home_win_prob = 0.45
            draw_prob = 0.25
            away_win_prob = 0.30

        # Clip xG to reasonable range
        home_xg = np.clip(home_xg, 0.3, 4.0)
        away_xg = np.clip(away_xg, 0.3, 4.0)

        # Get scoreline from Poisson
        scorelines = []
        for h in range(7):
            for a in range(7):
                p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                scorelines.append({
                    "home_goals": h,
                    "away_goals": a,
                    "probability": p
                })

        score_df = pd.DataFrame(scorelines).sort_values(
            "probability", ascending=False
        )
        top = score_df.iloc[0]

        results.append({
            "match_id": match_id,
            "group": fixture["group"],
            "matchday": matchday,
            "home_team": home_team,
            "away_team": away_team,
            "home_xg": round(home_xg, 3),
            "away_xg": round(away_xg, 3),
            "predicted_home_goals": int(top["home_goals"]),
            "predicted_away_goals": int(top["away_goals"]),
            "home_win_prob": round(home_win_prob, 3),
            "draw_prob": round(draw_prob, 3),
            "away_win_prob": round(away_win_prob, 3),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────
# 5. MODEL AGREEMENT
# ─────────────────────────────────────────

def compute_model_agreement(predictions: dict,
                              ensemble_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match compute how many models agree on the outcome.
    High agreement = more confident prediction.
    """
    agreement = []

    for _, row in ensemble_df.iterrows():
        match_id = row["match_id"]

        # Ensemble predicted result
        if row["predicted_home_goals"] > row["predicted_away_goals"]:
            ensemble_result = "H"
        elif row["predicted_home_goals"] == row["predicted_away_goals"]:
            ensemble_result = "D"
        else:
            ensemble_result = "A"

        # Count model agreement
        agree_count = 0
        total_models = 0

        for model_name, model_df in predictions.items():
            match_row = model_df[model_df["match_id"] == match_id]
            if len(match_row) == 0:
                continue

            r = match_row.iloc[0]
            if "predicted_home_goals" in r:
                if r["predicted_home_goals"] > r["predicted_away_goals"]:
                    model_result = "H"
                elif r["predicted_home_goals"] == r["predicted_away_goals"]:
                    model_result = "D"
                else:
                    model_result = "A"

                if model_result == ensemble_result:
                    agree_count += 1
                total_models += 1
            elif "predicted_result" in r:
                if r["predicted_result"] == ensemble_result:
                    agree_count += 1
                total_models += 1

        agreement.append({
            "match_id": match_id,
            "model_agreement": agree_count,
            "total_models": total_models,
            "agreement_pct": round(
                agree_count / total_models * 100
                if total_models > 0 else 0, 1
            ),
        })

    return pd.DataFrame(agreement)


# ─────────────────────────────────────────
# 6. RUN ENSEMBLE
# ─────────────────────────────────────────

def run_ensemble(matchday: int = 1):
    print("=" * 60)
    print(f"  ENSEMBLE MODEL — WC 2026 MD{matchday}")
    print("=" * 60)
    print()

    # Load predictions
    print(f"Loading model predictions for MD{matchday}...")
    predictions = load_model_predictions(matchday)
    print(f"  Loaded {len(predictions)} models")

    if len(predictions) == 0:
        print("❌ No predictions found. Run individual models first.")
        sys.exit(1)

    # Get normalized weights
    weights = get_active_weights(predictions, matchday)
    print(f"\nActive model weights (MD{matchday}):")
    for model, weight in sorted(
        weights.items(), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(weight * 100)
        print(f"  {model:20} {weight:.3f} {bar}")

    # Ensemble
    print(f"\nComputing ensemble predictions...")
    ensemble_df = ensemble_predictions(predictions, weights, matchday)

    # Model agreement
    print("Computing model agreement...")
    agreement_df = compute_model_agreement(predictions, ensemble_df)
    ensemble_df = ensemble_df.merge(
        agreement_df[["match_id", "agreement_pct"]],
        on="match_id", how="left"
    )

    # Save
    os.makedirs("data/predictions", exist_ok=True)
    out_path = f"data/predictions/ensemble_md{matchday}.csv"
    ensemble_df.to_csv(out_path, index=False)

    # Print
    print()
    print(f"Ensemble Predictions — WC 2026 MD{matchday}")
    print("=" * 60)
    for _, row in ensemble_df.iterrows():
        agreement = row.get("agreement_pct", 0)
        confidence = (
            "🔥HIGH" if agreement >= 70
            else "📊MED" if agreement >= 50
            else "❓LOW"
        )
        print(
            f"Group {row['group']} | "
            f"{row['home_team']:20} "
            f"{row['predicted_home_goals']}-"
            f"{row['predicted_away_goals']} "
            f"{row['away_team']:20} | "
            f"xG:{row['home_xg']:.2f}-{row['away_xg']:.2f} | "
            f"H:{row['home_win_prob']} "
            f"D:{row['draw_prob']} "
            f"A:{row['away_win_prob']} | "
            f"{confidence} {agreement:.0f}%"
        )

    print()
    print(f"Saved to {out_path}")

    # Summary stats
    print()
    home_wins = (
        ensemble_df["predicted_home_goals"] >
        ensemble_df["predicted_away_goals"]
    ).sum()
    draws = (
        ensemble_df["predicted_home_goals"] ==
        ensemble_df["predicted_away_goals"]
    ).sum()
    away_wins = (
        ensemble_df["predicted_home_goals"] <
        ensemble_df["predicted_away_goals"]
    ).sum()
    avg_agreement = ensemble_df["agreement_pct"].mean()

    print(f"Prediction summary:")
    print(f"  Home wins:  {home_wins}")
    print(f"  Draws:      {draws}")
    print(f"  Away wins:  {away_wins}")
    print(f"  Avg model agreement: {avg_agreement:.1f}%")

    return ensemble_df


# ─────────────────────────────────────────
# 7. RUN ALL MATCHDAYS
# ─────────────────────────────────────────

def run_all_matchdays():
    """Run ensemble for all 3 matchdays."""
    for md in [1, 2, 3]:
        print()
        run_ensemble(matchday=md)
        print()


if __name__ == "__main__":
    md = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if sys.argv[1:] == ["all"]:
        run_all_matchdays()
    else:
        run_ensemble(matchday=md)