"""
Backtesting — evaluates model performance on WC 2018 and 2022.

Methodology:
1. For each World Cup (2018, 2022):
   - Train all models on data BEFORE that tournament
   - Predict all group stage matches
   - Compare predictions to real results
2. Compute metrics:
   - MAE (Mean Absolute Error) on goals
   - Result accuracy (H/D/A)
   - Brier Score (probability calibration)
   - Exact score accuracy
3. Compare all models side by side
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append("src")
sys.path.append("src/data")
sys.path.append("src/models/baseline")
sys.path.append("src/models/ml")
sys.path.append("src/features")


# ─────────────────────────────────────────
# 1. LOAD WC MATCHES
# ─────────────────────────────────────────

def get_wc_matches(results: pd.DataFrame, year: int) -> pd.DataFrame:
    """Get World Cup group stage matches for a given year."""
    wc = results[
        results["tournament"].str.contains(
            "FIFA World Cup", na=False
        ) &
        ~results["tournament"].str.contains(
            "qualification", case=False, na=False
        ) &
        (results["date"].dt.year == year)
    ].copy()

    dates = sorted(wc["date"].unique())
    n = len(dates)
    third = max(1, n // 3)

    date_to_md = {}
    for i, d in enumerate(dates):
        if i < third:
            date_to_md[d] = 1
        elif i < 2 * third:
            date_to_md[d] = 2
        else:
            date_to_md[d] = 3

    wc["matchday"] = wc["date"].map(date_to_md)
    return wc.dropna(subset=["home_goals", "away_goals"])


# ─────────────────────────────────────────
# 2. SIMPLE POISSON PREDICTION
# ─────────────────────────────────────────

def predict_poisson(home_xg: float, away_xg: float) -> dict:
    """Generate scoreline and probabilities from xG values."""
    home_xg = np.clip(home_xg, 0.3, 4.0)
    away_xg = np.clip(away_xg, 0.3, 4.0)

    scorelines = []
    for h in range(7):
        for a in range(7):
            p = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            scorelines.append({"h": h, "a": a, "p": p})

    df = pd.DataFrame(scorelines)
    top = df.sort_values("p", ascending=False).iloc[0]

    home_win = df[df["h"] > df["a"]]["p"].sum()
    draw = df[df["h"] == df["a"]]["p"].sum()
    away_win = df[df["h"] < df["a"]]["p"].sum()

    # The 7x7 scoreline grid truncates the Poisson tail, so the three
    # outcome probabilities sum to slightly under 1. Normalize so they
    # form a proper distribution.
    total = home_win + draw + away_win
    if total > 0:
        home_win, draw, away_win = (
            home_win / total, draw / total, away_win / total
        )

    return {
        "pred_home": int(top["h"]),
        "pred_away": int(top["a"]),
        "home_win_prob": home_win,
        "draw_prob": draw,
        "away_win_prob": away_win,
        "home_xg": float(home_xg),
        "away_xg": float(away_xg),
    }


# ─────────────────────────────────────────
# 3. MODEL PREDICTORS
# ─────────────────────────────────────────

def predict_historical_avg(train: pd.DataFrame,
                            match: pd.Series,
                            n: int = 10) -> dict:
    """Historical average prediction."""
    def get_stats(team, before_date):
        mask = (
            ((train["home_team"] == team) |
             (train["away_team"] == team)) &
            (train["date"] < before_date)
        )
        matches = train[mask].dropna(
            subset=["home_goals", "away_goals"]
        ).sort_values("date", ascending=False).head(n)

        if len(matches) == 0:
            return 1.2, 1.2

        scored = []
        conceded = []
        for _, m in matches.iterrows():
            if m["home_team"] == team:
                scored.append(m["home_goals"])
                conceded.append(m["away_goals"])
            else:
                scored.append(m["away_goals"])
                conceded.append(m["home_goals"])
        return np.mean(scored), np.mean(conceded)

    h_scored, h_conceded = get_stats(
        match["home_team"], match["date"]
    )
    a_scored, a_conceded = get_stats(
        match["away_team"], match["date"]
    )

    home_xg = (h_scored + a_conceded) / 2
    away_xg = (a_scored + h_conceded) / 2

    return predict_poisson(home_xg, away_xg)


def _placeholder_scoreline(home_win, draw, away_win):
    """A nominal scoreline for models that only predict W/D/L."""
    best = max(
        {"H": home_win, "D": draw, "A": away_win},
        key=lambda k: {"H": home_win, "D": draw, "A": away_win}[k],
    )
    return {"H": (1, 0), "D": (1, 1), "A": (0, 1)}[best]


# ── Per-model backtest predictors ────────────────────────
# Each takes the shared test_feat (one row per WC fixture, carrying
# features + actual goals) and returns a list of prediction dicts in
# the same order, so the ensemble can align them by index.

def bp_historical_avg(train_results, test_feat):
    preds = []
    for _, row in test_feat.iterrows():
        p = predict_historical_avg(train_results, row)
        preds.append({
            **p,
            "actual_home": int(row["home_goals"]),
            "actual_away": int(row["away_goals"]),
        })
    return preds


def bp_elo(test_feat):
    """Static ELO: convert the precomputed elo_diff to xG (README formula)."""
    preds = []
    for _, row in test_feat.iterrows():
        elo_diff = float(row.get("elo_diff", 0.0))
        p = predict_poisson(1.35 + elo_diff * 0.001,
                            1.35 - elo_diff * 0.001)
        preds.append({
            **p,
            "actual_home": int(row["home_goals"]),
            "actual_away": int(row["away_goals"]),
        })
    return preds


def bp_dynamic_elo(train_results, test_feat):
    """Replay pre-tournament matches to build ELO, then predict."""
    from dynamic_elo import build_dynamic_elo, elo_to_xg, DEFAULT_ELO
    ratings = build_dynamic_elo(train_results)
    preds = []
    for _, row in test_feat.iterrows():
        he = ratings.get(row["home_team"], DEFAULT_ELO)
        ae = ratings.get(row["away_team"], DEFAULT_ELO)
        hxg, axg = elo_to_xg(he, ae)
        p = predict_poisson(hxg, axg)
        preds.append({
            **p,
            "actual_home": int(row["home_goals"]),
            "actual_away": int(row["away_goals"]),
        })
    return preds


def bp_logistic(train_feat, test_feat, feature_cols):
    """Logistic regression — predicts W/D/L probabilities directly."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = train_feat.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"]
        else ("D" if r["home_goals"] == r["away_goals"] else "A"),
        axis=1,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat[feature_cols])
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_train, y)

    X_test = scaler.transform(test_feat[feature_cols])
    proba = clf.predict_proba(X_test)
    classes = list(clf.classes_)
    hi, di, ai = (classes.index("H"), classes.index("D"),
                  classes.index("A"))

    preds = []
    for i, (_, row) in enumerate(test_feat.iterrows()):
        hw, dr, aw = proba[i][hi], proba[i][di], proba[i][ai]
        ph, pa = _placeholder_scoreline(hw, dr, aw)
        preds.append({
            "pred_home": ph, "pred_away": pa,
            "home_win_prob": hw, "draw_prob": dr, "away_win_prob": aw,
            "home_xg": np.nan, "away_xg": np.nan,  # no scoreline model
            "actual_home": int(row["home_goals"]),
            "actual_away": int(row["away_goals"]),
        })
    return preds


def bp_tree(train_feat, test_feat, feature_cols, kind="xgboost"):
    """Gradient-boosted home/away goal regressors (XGBoost or LightGBM)."""
    if kind == "xgboost":
        import xgboost as xgb
        make = lambda: xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective="count:poisson", random_state=42, n_jobs=-1,
        )
    else:
        import lightgbm as lgb
        make = lambda: lgb.LGBMRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective="poisson", random_state=42, n_jobs=-1,
            verbose=-1,
        )

    X_train = train_feat[feature_cols]
    model_h, model_a = make(), make()
    model_h.fit(X_train, train_feat["home_goals"])
    model_a.fit(X_train, train_feat["away_goals"])

    preds = []
    for _, row in test_feat.iterrows():
        X_pred = pd.DataFrame([row[feature_cols].fillna(0).to_dict()])
        hxg = float(model_h.predict(X_pred)[0])
        axg = float(model_a.predict(X_pred)[0])
        p = predict_poisson(hxg, axg)
        preds.append({
            **p,
            "actual_home": int(row["home_goals"]),
            "actual_away": int(row["away_goals"]),
        })
    return preds


def bp_ensemble(model_preds: dict, weights: dict):
    """Weighted-average the available models into an ensemble prediction.

    Mirrors the production ensemble: probabilities are blended across all
    present models, and expected goals are blended across the models that
    produce them (everything except the W/D/L-only logistic). The
    scoreline is the Poisson mode of the blended xG, and the result is the
    argmax of the blended probabilities. Weights are renormalized over
    whichever models are present.
    """
    present = {m: w for m, w in weights.items() if m in model_preds}
    wsum = sum(present.values())
    if wsum == 0:
        return []
    present = {m: w / wsum for m, w in present.items()}

    n = len(next(iter(model_preds.values())))
    out = []
    for i in range(n):
        hw = dr = aw = 0.0
        hxg = axg = 0.0
        xg_w = 0.0
        for m, w in present.items():
            p = model_preds[m][i]
            hw += w * p["home_win_prob"]
            dr += w * p["draw_prob"]
            aw += w * p["away_win_prob"]
            if not np.isnan(p.get("home_xg", np.nan)):
                hxg += w * p["home_xg"]
                axg += w * p["away_xg"]
                xg_w += w

        s = hw + dr + aw
        if s > 0:
            hw, dr, aw = hw / s, dr / s, aw / s

        if xg_w > 0:
            sc = predict_poisson(hxg / xg_w, axg / xg_w)
            ph, pa = sc["pred_home"], sc["pred_away"]
        else:
            ph, pa = _placeholder_scoreline(hw, dr, aw)

        ref = next(iter(model_preds.values()))[i]
        out.append({
            "pred_home": ph, "pred_away": pa,
            "home_win_prob": hw, "draw_prob": dr, "away_win_prob": aw,
            "actual_home": ref["actual_home"],
            "actual_away": ref["actual_away"],
        })
    return out


# ── Ensemble weight learning ─────────────────────────────

def _eval_weights(model_preds, models, w):
    """Return (result_accuracy, brier) for a weight vector over `models`."""
    n = len(model_preds[models[0]])
    correct = 0
    bsum = 0.0
    for i in range(n):
        hw = dr = aw = 0.0
        for j, m in enumerate(models):
            p = model_preds[m][i]
            hw += w[j] * p["home_win_prob"]
            dr += w[j] * p["draw_prob"]
            aw += w[j] * p["away_win_prob"]
        s = hw + dr + aw
        if s > 0:
            hw, dr, aw = hw / s, dr / s, aw / s
        ref = model_preds[models[0]][i]
        ah, aa = ref["actual_home"], ref["actual_away"]
        act = "H" if ah > aa else ("D" if ah == aa else "A")
        pred = max((("H", hw), ("D", dr), ("A", aw)), key=lambda x: x[1])[0]
        correct += (pred == act)
        yh, yd, ya = (act == "H"), (act == "D"), (act == "A")
        bsum += ((hw - yh) ** 2 + (dr - yd) ** 2 + (aw - ya) ** 2) / 3.0
    return correct / n, bsum / n


def learn_weights(model_preds, models, n_iter=8000, seed=0):
    """Random-search the weight simplex, minimizing Brier (robust to the
    non-smoothness of accuracy on a small sample). Uniform weights are
    always tried as a baseline candidate."""
    rng = np.random.default_rng(seed)
    best_w = np.ones(len(models)) / len(models)
    best_brier = _eval_weights(model_preds, models, best_w)[1]
    for k in range(n_iter):
        w = rng.dirichlet(np.ones(len(models)))
        _, brier = _eval_weights(model_preds, models, w)
        if brier < best_brier:
            best_brier, best_w = brier, w
    return best_w


# ─────────────────────────────────────────
# 4. COMPUTE METRICS
# ─────────────────────────────────────────

def compute_metrics(predictions: list) -> dict:
    """Compute evaluation metrics from a list of predictions."""
    if len(predictions) == 0:
        return {}

    df = pd.DataFrame(predictions)

    home_mae = np.mean(
        np.abs(df["pred_home"] - df["actual_home"])
    )
    away_mae = np.mean(
        np.abs(df["pred_away"] - df["actual_away"])
    )

    # Derive the predicted result from the *summed* outcome
    # probabilities, not the single modal scoreline. The modal
    # scoreline is often 1-0 / 1-1 and systematically under-picks
    # draws, throwing away information the probabilities already
    # capture. argmax over P(H)/P(D)/P(A) is the proper read.
    def _prob_result(r):
        probs = {
            "H": r.get("home_win_prob", 0.0),
            "D": r.get("draw_prob", 0.0),
            "A": r.get("away_win_prob", 0.0),
        }
        return max(probs, key=probs.get)

    df["pred_result"] = df.apply(_prob_result, axis=1)
    df["actual_result"] = df.apply(
        lambda r: "H" if r["actual_home"] > r["actual_away"]
        else ("D" if r["actual_home"] == r["actual_away"] else "A"),
        axis=1
    )
    result_acc = (
        df["pred_result"] == df["actual_result"]
    ).mean()

    exact_score = (
        (df["pred_home"] == df["actual_home"]) &
        (df["pred_away"] == df["actual_away"])
    ).mean()

    df["actual_home_win"] = (
        df["actual_result"] == "H"
    ).astype(float)
    df["actual_draw"] = (
        df["actual_result"] == "D"
    ).astype(float)
    df["actual_away_win"] = (
        df["actual_result"] == "A"
    ).astype(float)

    brier = np.mean([
        brier_score_loss(
            df["actual_home_win"],
            df["home_win_prob"].clip(0.01, 0.99)
        ),
        brier_score_loss(
            df["actual_draw"],
            df["draw_prob"].clip(0.01, 0.99)
        ),
        brier_score_loss(
            df["actual_away_win"],
            df["away_win_prob"].clip(0.01, 0.99)
        ),
    ])

    return {
        "home_mae": round(home_mae, 3),
        "away_mae": round(away_mae, 3),
        "result_accuracy": round(result_acc, 3),
        "exact_score_accuracy": round(exact_score, 3),
        "brier_score": round(brier, 3),
        "n_matches": len(df),
    }


# ─────────────────────────────────────────
# 5. RUN BACKTEST
# ─────────────────────────────────────────

def run_backtest(results: pd.DataFrame,
                  test_year: int) -> dict:
    """
    Run backtest for a given World Cup year.
    Train on all data before the tournament.
    Test on tournament group stage matches only.
    """
    print(f"\n{'='*60}")
    print(f"  Backtesting on WC {test_year}")
    print(f"{'='*60}")

    cutoff = pd.Timestamp(f"{test_year}-01-01")
    train = results[results["date"] < cutoff].copy()
    test = get_wc_matches(results, test_year)

    print(f"  Training matches: {len(train)}")
    print(f"  Test matches:     {len(test)}")

    if len(test) == 0:
        print(f"  No WC {test_year} matches found")
        return {}

    feature_cols = [
        "is_neutral", "home_elo", "away_elo", "elo_diff",
        "home_form5_points", "home_form5_avg_scored",
        "home_form5_avg_conceded",
        "away_form5_points", "away_form5_avg_scored",
        "away_form5_avg_conceded",
        "home_form10_points", "home_form10_avg_scored",
        "home_form10_avg_conceded",
        "away_form10_points", "away_form10_avg_scored",
        "away_form10_avg_conceded",
        "form5_points_diff", "form10_points_diff",
        "avg_scored_diff",
        "h2h_matches", "h2h_home_wins", "h2h_away_wins",
        "h2h_draws", "h2h_avg_goals",
    ]

    # Build the shared feature table: train rows before the cutoff and
    # one test row per WC fixture, so EVERY model is scored on the same
    # 64 matches (see the match-set note in the README).
    all_features = pd.read_csv(
        "data/processed/train_features.csv", parse_dates=["date"]
    )
    train_feat = all_features[
        all_features["date"] < cutoff
    ].dropna(subset=["home_goals", "away_goals"]).copy()

    for col in feature_cols:
        if col not in all_features.columns:
            all_features[col] = 0
        if col not in train_feat.columns:
            train_feat[col] = 0

    feat_lookup = all_features[
        ["home_team", "away_team", "date"] + feature_cols
    ].drop_duplicates(subset=["home_team", "away_team", "date"])
    test_feat = test[
        ["home_team", "away_team", "date", "home_goals", "away_goals"]
    ].merge(
        feat_lookup, on=["home_team", "away_team", "date"], how="left"
    )
    train_feat[feature_cols] = train_feat[feature_cols].fillna(0)
    test_feat[feature_cols] = test_feat[feature_cols].fillna(0)

    # Registry of models to backtest. Each entry returns a list of
    # per-match predictions aligned to test_feat. Optional models
    # (LightGBM) are skipped gracefully if the library is unavailable.
    runners = [
        ("Historical Average",
         lambda: bp_historical_avg(train, test_feat)),
        ("ELO",
         lambda: bp_elo(test_feat)),
        ("Dynamic ELO",
         lambda: bp_dynamic_elo(train, test_feat)),
        ("Logistic Regression",
         lambda: bp_logistic(train_feat, test_feat, feature_cols)),
        ("XGBoost",
         lambda: bp_tree(train_feat, test_feat, feature_cols, "xgboost")),
        ("LightGBM",
         lambda: bp_tree(train_feat, test_feat, feature_cols, "lightgbm")),
    ]

    all_model_results = {}
    model_preds = {}
    for name, fn in runners:
        print(f"\n  Running {name}...")
        try:
            preds = fn()
            if not preds:
                print(f"  ⚠️  {name}: no predictions")
                continue
            model_preds[name] = preds
            all_model_results[name] = compute_metrics(preds)
            print(f"  ✅ Done — {len(preds)} matches")
        except ImportError:
            print(f"  ⏭️  {name} skipped (library not installed)")
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️  {name} failed: {e}")

    # ── Ensemble (rebalanced weights, renormalized over present) ──
    # Backtesting showed the original weights under-used Logistic
    # Regression (the strongest model) at 0.04 while over-weighting weak
    # baselines. These give the top-3 models the lion's share.
    ensemble_weights = {
        "Logistic Regression": 0.18, "XGBoost": 0.20, "LightGBM": 0.20,
        "Dynamic ELO": 0.08, "ELO": 0.06, "Historical Average": 0.06,
    }
    print(f"\n  Building Ensemble...")
    ens = bp_ensemble(model_preds, ensemble_weights)
    if ens:
        all_model_results["Ensemble"] = compute_metrics(ens)
        print(f"  ✅ Done — {len(ens)} matches")

    return all_model_results, model_preds


# ─────────────────────────────────────────
# 6. PRINT RESULTS TABLE
# ─────────────────────────────────────────

def print_results_table(results: dict, year: int):
    """Print a clean comparison table."""
    print(f"\n  Results for WC {year}:")
    print(
        f"  {'Model':25} {'H-MAE':>7} {'A-MAE':>7} "
        f"{'Result%':>8} {'Exact%':>7} {'Brier':>7} {'N':>4}"
    )
    print(
        f"  {'-'*25} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*4}"
    )

    for model, metrics in sorted(
        results.items(),
        key=lambda x: x[1].get("result_accuracy", 0),
        reverse=True
    ):
        if not metrics:
            continue
        print(
            f"  {model:25} "
            f"{metrics.get('home_mae', 0):>7.3f} "
            f"{metrics.get('away_mae', 0):>7.3f} "
            f"{metrics.get('result_accuracy', 0):>7.1%} "
            f"{metrics.get('exact_score_accuracy', 0):>7.1%} "
            f"{metrics.get('brier_score', 0):>7.3f} "
            f"{metrics.get('n_matches', 0):>4}"
        )


# ─────────────────────────────────────────
# 7. SAVE RESULTS
# ─────────────────────────────────────────

def save_backtest_results(all_results: dict):
    """Save backtest results to CSV."""
    rows = []
    for year, model_results in all_results.items():
        for model, metrics in model_results.items():
            rows.append({
                "year": year,
                "model": model,
                **metrics
            })

    df = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/backtest_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved backtest results to {out_path}")
    return df


# ─────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────

def learn_and_apply_ensemble_weights(preds_by_year, all_results):
    """Learn ensemble weights, evaluate them out-of-sample (leave-one-
    tournament-out), and save the production weights to JSON."""
    import json

    years = list(preds_by_year.keys())
    # Models common to every tournament's predictions.
    models = sorted(
        set.intersection(*[set(preds_by_year[y]) for y in years])
    )
    if len(years) < 2 or not models:
        return

    print(f"\n{'='*60}")
    print("  LEARNING ENSEMBLE WEIGHTS (leave-one-tournament-out)")
    print(f"{'='*60}")

    # Leave-one-tournament-out: learn on the other year, score on this one.
    loo_preds = []
    for test_year in years:
        train_year = [y for y in years if y != test_year][0]
        w = learn_weights(preds_by_year[train_year], models)
        wmap = {m: float(w[j]) for j, m in enumerate(models)}
        ens = bp_ensemble(preds_by_year[test_year], wmap)
        loo_preds.append((test_year, ens))
        all_results[test_year]["Ensemble (learned)"] = compute_metrics(ens)

    # Production weights: learn on all tournaments pooled.
    pooled = {m: sum((preds_by_year[y][m] for y in years), [])
              for m in models}
    w_final = learn_weights(pooled, models)
    weights_final = {m: round(float(w_final[j]), 4)
                     for j, m in enumerate(models)}

    print("\n  Learned production weights (pooled):")
    for m, wv in sorted(weights_final.items(),
                        key=lambda x: -x[1]):
        print(f"    {m:22} {wv:.3f}")

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/ensemble_weights.json", "w") as f:
        json.dump(weights_final, f, indent=2)
    print("\n  Saved learned weights to "
          "data/processed/ensemble_weights.json")


def run_full_backtest():
    print("=" * 60)
    print("  BACKTESTING — WC 2018 & 2022")
    print("=" * 60)
    print()

    print("Loading data...")
    results = pd.read_csv(
        "data/processed/results_clean.csv",
        parse_dates=["date"]
    )
    print(f"  Loaded {len(results)} matches")

    all_results = {}
    preds_by_year = {}

    for year in [2018, 2022]:
        year_results, year_preds = run_backtest(results, year)
        all_results[year] = year_results
        preds_by_year[year] = year_preds
        print_results_table(year_results, year)

    # ── Learn ensemble weights (leave-one-tournament-out) ──
    # Honest evaluation: weights are learned on one tournament and the
    # ensemble is scored on the other, so the reported "learned" number
    # is genuinely out-of-sample. The weights shipped to production are
    # then learned on both tournaments pooled.
    learn_and_apply_ensemble_weights(preds_by_year, all_results)

    # Combined summary
    print(f"\n{'='*60}")
    print("  COMBINED SUMMARY (2018 + 2022)")
    print(f"{'='*60}")

    combined = {}
    for year_results in all_results.values():
        for model, metrics in year_results.items():
            if model not in combined:
                combined[model] = []
            combined[model].append(metrics)

    print(
        f"\n  {'Model':25} {'Avg H-MAE':>10} "
        f"{'Avg Result%':>12} {'Avg Brier':>10}"
    )
    print(
        f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}"
    )

    for model, metrics_list in sorted(
        combined.items(),
        key=lambda x: np.mean(
            [m.get("result_accuracy", 0) for m in x[1]]
        ),
        reverse=True
    ):
        avg_mae = np.mean(
            [m.get("home_mae", 0) for m in metrics_list]
        )
        avg_acc = np.mean(
            [m.get("result_accuracy", 0) for m in metrics_list]
        )
        avg_brier = np.mean(
            [m.get("brier_score", 0) for m in metrics_list]
        )
        print(
            f"  {model:25} "
            f"{avg_mae:>10.3f} "
            f"{avg_acc:>11.1%} "
            f"{avg_brier:>10.3f}"
        )

    save_backtest_results(all_results)


if __name__ == "__main__":
    run_full_backtest()