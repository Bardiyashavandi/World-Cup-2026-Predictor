"""
Market-blended predictions — combine the model's ensemble with bookmaker
odds via a logarithmic opinion pool.

Why: betting markets price in information the model can't see (injuries,
line-ups, sharp money) and sit at ~55% result accuracy. Blending two
forecasters with uncorrelated errors usually beats either alone — as long
as you don't simply copy the market.

How:
  1. Read decimal odds (home/draw/away) from data/raw/market_odds.csv.
  2. De-vig them: implied prob = 1/odds, then renormalize so the three
     outcomes sum to 1 (removes the bookmaker's overround).
  3. Log-opinion-pool with the ensemble probabilities:
        p_blend ∝ p_model^w · p_market^(1-w)      (then renormalize)
     w = how much to trust the model (default 0.5). w=1 → pure model,
     w=0 → pure (de-vigged) market.
  4. Matches with no odds fall back to the pure ensemble.

Usage:
    python3 src/ensemble/market_blend.py 2            # MD2, w=0.5
    python3 src/ensemble/market_blend.py 2 --weight 0.4
"""

import argparse
import os
import numpy as np
import pandas as pd

ODDS_PATH = "data/raw/market_odds.csv"
PROB_COLS = ["home_win_prob", "draw_prob", "away_win_prob"]


def devig(odds_home, odds_draw, odds_away):
    """Decimal odds -> de-vigged implied probabilities (sum to 1)."""
    imp = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    return imp / imp.sum()


def log_pool(p_model, p_market, w):
    """Weighted geometric mean of two probability vectors, renormalized."""
    p_model = np.clip(np.asarray(p_model, float), 1e-9, 1)
    p_market = np.clip(np.asarray(p_market, float), 1e-9, 1)
    blended = (p_model ** w) * (p_market ** (1 - w))
    return blended / blended.sum()


def load_odds():
    if not os.path.exists(ODDS_PATH):
        return {}
    df = pd.read_csv(ODDS_PATH)
    odds = {}
    for _, r in df.iterrows():
        try:
            mid = int(r["match_id"])
            h, d, a = (float(r["home_odds"]), float(r["draw_odds"]),
                       float(r["away_odds"]))
            if h > 1 and d > 1 and a > 1:
                odds[mid] = (h, d, a)
        except (ValueError, TypeError, KeyError):
            continue
    return odds


def blend_matchday(md, weight):
    ens_path = f"data/predictions/ensemble_md{md}.csv"
    if not os.path.exists(ens_path):
        raise FileNotFoundError(ens_path)
    df = pd.read_csv(ens_path)
    odds = load_odds()

    rows = []
    n_blended = 0
    for _, r in df.iterrows():
        p_model = [r["home_win_prob"], r["draw_prob"], r["away_win_prob"]]
        mid = int(r["match_id"])
        if mid in odds:
            p_market = devig(*odds[mid])
            p = log_pool(p_model, p_market, weight)
            source = "blended"
            n_blended += 1
        else:
            p = np.asarray(p_model, float)
            p = p / p.sum()
            source = "ensemble"
        rows.append({
            "match_id": mid, "group": r["group"], "matchday": md,
            "home_team": r["home_team"], "away_team": r["away_team"],
            "predicted_home_goals": r["predicted_home_goals"],
            "predicted_away_goals": r["predicted_away_goals"],
            "home_win_prob": round(float(p[0]), 4),
            "draw_prob": round(float(p[1]), 4),
            "away_win_prob": round(float(p[2]), 4),
            "result": ["H", "D", "A"][int(np.argmax(p))],
            "source": source,
        })

    out = pd.DataFrame(rows)
    out_path = f"data/predictions/market_blended_md{md}.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path}  (w={weight}, "
          f"{n_blended}/{len(out)} matches blended with odds)")
    if n_blended == 0:
        print("   No odds found — output equals the ensemble. Fill "
              f"{ODDS_PATH} (match_id,home_odds,draw_odds,away_odds).")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("matchday", type=int, choices=[1, 2, 3])
    ap.add_argument("--weight", type=float, default=0.5,
                    help="model weight 0..1 (1=pure model, 0=pure market)")
    args = ap.parse_args()
    blend_matchday(args.matchday, args.weight)
