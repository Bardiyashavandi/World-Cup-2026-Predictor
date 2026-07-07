"""
Betting-edge simulation — would following the model have beaten the market?

For every match where we have both bookmaker odds (data/raw/market_odds.csv)
and a real result, "bet" 1 unit and tally the return under three strategies:

  • Model      — back the model's most-likely outcome at the market's price
  • Market fav — back the bookmakers' favourite (baseline)
  • Value only — back the model's pick ONLY when the model rates it higher
                 than the de-vigged market does (i.e. the model sees an edge)

Return on investment = (total returned − total staked) / total staked.
A positive Model/Value ROI is the real test: it means the model found
prices the market mispriced. (Sample size matters — add more odds rows to
market_odds.csv for a more reliable read.)

    python3 src/analysis/betting_edge.py
"""

import os
import numpy as np
import pandas as pd


def devig(h, d, a):
    imp = np.array([1 / h, 1 / d, 1 / a])
    return imp / imp.sum()


def outcome(hg, ag):
    return "H" if hg > ag else ("D" if hg == ag else "A")


def main():
    odds = pd.read_csv("data/raw/market_odds.csv")
    res = pd.read_csv("data/raw/wc_2026_results.csv").set_index("match_id")
    preds = {}
    for md in (1, 2, 3):
        p = f"data/predictions/ensemble_md{md}.csv"
        if os.path.exists(p):
            for _, r in pd.read_csv(p).iterrows():
                preds[int(r["match_id"])] = r

    idx = {"H": 0, "D": 1, "A": 2}
    strat = {k: {"staked": 0.0, "ret": 0.0, "n": 0, "won": 0}
             for k in ("model", "market", "value")}
    log = []

    for _, o in odds.iterrows():
        mid = int(o["match_id"])
        if mid not in preds or mid not in res.index:
            continue
        row = res.loc[mid]
        if row["played"] != True:
            continue
        dec = {"H": float(o["home_odds"]), "D": float(o["draw_odds"]),
               "A": float(o["away_odds"])}
        mkt = devig(dec["H"], dec["D"], dec["A"])
        pr = preds[mid]
        model = np.array([pr["home_win_prob"], pr["draw_prob"],
                          pr["away_win_prob"]])
        act = outcome(int(row["home_goals"]), int(row["away_goals"]))
        model_pick = "HDA"[int(np.argmax(model))]
        market_pick = "HDA"[int(np.argmax(mkt))]

        def place(name, pick):
            s = strat[name]
            s["staked"] += 1.0
            s["n"] += 1
            if pick == act:
                s["ret"] += dec[pick]
                s["won"] += 1

        place("model", model_pick)
        place("market", market_pick)
        # value: model rates its pick above the market's de-vigged prob
        if model[idx[model_pick]] > mkt[idx[model_pick]]:
            place("value", model_pick)

        log.append((o["home_team"], o["away_team"], model_pick, act,
                    dec[model_pick], model_pick == act))

    print(f"{'Match':32} {'Pick':>4} {'Act':>4} {'Odds':>5} {'W/L':>4}")
    for h, a, pk, act, od, win in log:
        print(f"{h+' v '+a:32} {pk:>4} {act:>4} {od:>5.2f} "
              f"{'WON' if win else 'lost':>4}")

    print("\nStrategy       Bets  Hit%   Staked  Return   ROI")
    print("-" * 52)
    for name in ("model", "market", "value"):
        s = strat[name]
        if s["n"] == 0:
            print(f"{name:12}   (no bets)")
            continue
        roi = (s["ret"] - s["staked"]) / s["staked"] * 100
        print(f"{name:12} {s['n']:5d} {s['won']/s['n']*100:5.0f}% "
              f"{s['staked']:7.1f} {s['ret']:7.2f} {roi:+6.1f}%")


if __name__ == "__main__":
    main()
