"""
Smoke tests for the World Cup 2026 predictor.

These do not retrain models — they validate the data contracts and the
pure-Python prediction maths, so they run fast and catch the most common
breakages: malformed fixtures, probabilities that don't sum to 1,
out-of-range xG, and a broken Poisson predictor.

Run with:
    pytest tests/
or, without pytest installed:
    python3 tests/test_pipeline.py
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src", "evaluation"))

RAW = os.path.join(ROOT, "data", "raw")
PRED = os.path.join(ROOT, "data", "predictions")

EXPECTED_GROUPS = list("ABCDEFGHIJKL")  # 12 groups
PROB_COLS = ["home_win_prob", "draw_prob", "away_win_prob"]


# ── Fixtures / schema ───────────────────────────────────

def test_fixtures_count_and_groups():
    """All 72 group-stage fixtures across exactly 12 groups."""
    fx = pd.read_csv(os.path.join(RAW, "wc_2026_fixtures.csv"))
    assert len(fx) == 72, f"expected 72 fixtures, got {len(fx)}"
    assert sorted(fx["group"].unique()) == EXPECTED_GROUPS
    # 4 teams per group => 6 matches each
    counts = fx.groupby("group").size()
    assert (counts == 6).all(), f"groups not all 6 matches: {dict(counts)}"


def test_results_tracker_schema():
    """Live results tracker has the columns the updater relies on."""
    res = pd.read_csv(os.path.join(RAW, "wc_2026_results.csv"))
    required = {
        "match_id", "group", "home_team", "away_team",
        "matchday", "home_goals", "away_goals", "played",
    }
    assert required.issubset(res.columns), (
        f"missing columns: {required - set(res.columns)}"
    )
    assert len(res) == 72


# ── Prediction outputs ──────────────────────────────────

def _prediction_files():
    return sorted(glob.glob(os.path.join(PRED, "*.csv")))


def test_probabilities_valid():
    """Every prediction file with W/D/L columns has valid probabilities."""
    checked = 0
    for path in _prediction_files():
        df = pd.read_csv(path)
        if not set(PROB_COLS).issubset(df.columns):
            continue
        checked += 1
        name = os.path.basename(path)
        probs = df[PROB_COLS]
        assert (probs >= -1e-6).all().all(), f"{name}: negative prob"
        assert (probs <= 1 + 1e-6).all().all(), f"{name}: prob > 1"
        # Outcome probabilities are normalized at the source, so they
        # should sum to 1 within rounding. Small slack covers 4-dp
        # rounding in the saved CSVs.
        totals = probs.sum(axis=1)
        assert np.allclose(totals, 1.0, atol=0.005), (
            f"{name}: probabilities do not sum to 1 "
            f"(min={totals.min():.3f}, max={totals.max():.3f})"
        )
    assert checked > 0, "no prediction files with probability columns found"


def test_xg_in_sane_range():
    """Predicted expected goals stay in a plausible football range."""
    for path in _prediction_files():
        df = pd.read_csv(path)
        for col in ("home_xg", "away_xg"):
            if col in df.columns:
                vals = df[col].dropna()
                assert (vals >= 0).all(), f"{os.path.basename(path)}: negative xG"
                assert (vals <= 6).all(), f"{os.path.basename(path)}: xG > 6"


# ── Backtest results ────────────────────────────────────

def test_backtest_results_valid():
    """Backtest output (if present) has sane accuracies on 64 fixtures."""
    path = os.path.join(ROOT, "data", "processed", "backtest_results.csv")
    if not os.path.exists(path):
        return  # backtest hasn't been run; nothing to check
    df = pd.read_csv(path)
    assert {"model", "result_accuracy", "n_matches"}.issubset(df.columns)
    assert (df["result_accuracy"].between(0, 1)).all()
    # Every model must be scored on the same fixture count per tournament.
    assert (df["n_matches"] == 64).all(), (
        f"models scored on unequal match sets: "
        f"{df[['model', 'n_matches']].to_dict('records')}"
    )


# ── Poisson predictor maths ─────────────────────────────

def test_predict_poisson_sums_to_one():
    """The core Poisson predictor returns a valid probability distribution."""
    from backtest import predict_poisson
    for hxg, axg in [(1.5, 1.0), (0.3, 4.0), (2.2, 0.5)]:
        out = predict_poisson(hxg, axg)
        total = out["home_win_prob"] + out["draw_prob"] + out["away_win_prob"]
        assert abs(total - 1.0) < 0.05, f"probs sum to {total}"
        assert out["pred_home"] >= 0 and out["pred_away"] >= 0


# ── Plain-python runner (no pytest needed) ──────────────

if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
