"""
Calibration / reliability chart (docs/calibration.svg).

Across every played match, bin the model's predicted W/D/A probabilities
and compare each bin's average prediction to how often that outcome
actually happened. A perfectly calibrated model sits on the diagonal:
"when it says 70%, it happens 70% of the time." Points above the line =
under-confident; below = over-confident.

    python3 src/viz/build_calibration.py
"""

import os
import math
import pandas as pd

BINS = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5),
        (.5, .6), (.6, .7), (.7, 1.01)]


def collect():
    res = pd.read_csv("data/raw/wc_2026_results.csv")
    res = res[res["played"] == True]
    preds = {}
    for md in (1, 2, 3):
        p = f"data/predictions/ensemble_md{md}.csv"
        if os.path.exists(p):
            for _, r in pd.read_csv(p).iterrows():
                preds[int(r["match_id"])] = r
    pts = []
    for _, m in res.iterrows():
        pr = preds.get(int(m["match_id"]))
        if pr is None:
            continue
        h, a = int(m["home_goals"]), int(m["away_goals"])
        act = "H" if h > a else ("D" if h == a else "A")
        for k, col in [("H", "home_win_prob"), ("D", "draw_prob"),
                       ("A", "away_win_prob")]:
            pts.append((float(pr[col]), 1.0 if act == k else 0.0))
    rows = []
    for lo, hi in BINS:
        sel = [p for p in pts if lo <= p[0] < hi]
        if sel:
            ap = sum(p[0] for p in sel) / len(sel)
            af = sum(p[1] for p in sel) / len(sel)
            rows.append((ap, af, len(sel)))
    return rows


def x(v):  # 0..1 -> px
    return 70 + v * 400


def y(v):  # 0..1 -> px (inverted)
    return 490 - v * 400


def build_svg(rows):
    ticks = [0, .25, .5, .75, 1]
    grid = []
    for t in ticks:
        grid.append(f'<line x1="{x(t)}" y1="90" x2="{x(t)}" y2="490" '
                    f'stroke="#2a3446" stroke-width="1"/>')
        grid.append(f'<line x1="70" y1="{y(t)}" x2="470" y2="{y(t)}" '
                    f'stroke="#2a3446" stroke-width="1"/>')
        grid.append(f'<text x="{x(t)}" y="510" text-anchor="middle" '
                    f'font-size="11" fill="#9aa7bd">{int(t*100)}%</text>')
        grid.append(f'<text x="60" y="{y(t)+4}" text-anchor="end" '
                    f'font-size="11" fill="#9aa7bd">{int(t*100)}%</text>')
    poly = " ".join(f"{x(ap):.1f},{y(af):.1f}" for ap, af, _ in rows)
    dots = []
    for ap, af, n in rows:
        r = 4 + math.sqrt(n)
        dots.append(f'<circle cx="{x(ap):.1f}" cy="{y(af):.1f}" r="{r:.1f}" '
                    f'fill="#2ecc71" fill-opacity="0.85" stroke="#0e1117"/>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 560" \
font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
  <rect width="540" height="560" rx="16" fill="#161d2b"/>
  <text x="28" y="38" font-size="19" font-weight="800" fill="#f4f6fb">Probability calibration</text>
  <text x="28" y="58" font-size="12.5" fill="#9aa7bd">All played matches · predicted W/D/A probability vs actual frequency</text>
  {''.join(grid)}
  <line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#5f6b80" stroke-dasharray="5 5" stroke-width="1.5"/>
  <text x="{x(1)-6}" y="{y(1)+16}" text-anchor="end" font-size="11" fill="#5f6b80">perfect calibration</text>
  <polyline points="{poly}" fill="none" stroke="#2ecc71" stroke-width="2" stroke-opacity="0.5"/>
  {''.join(dots)}
  <text x="270" y="540" text-anchor="middle" font-size="12" fill="#9aa7bd">Model predicted probability  →  (dot size = sample size)</text>
  <text x="20" y="300" text-anchor="middle" font-size="12" fill="#9aa7bd" transform="rotate(-90 20 300)">Actual frequency</text>
</svg>"""


def main():
    rows = collect()
    os.makedirs("docs", exist_ok=True)
    with open("docs/calibration.svg", "w", encoding="utf-8") as f:
        f.write(build_svg(rows))
    print("✅ Wrote docs/calibration.svg")
    for ap, af, n in rows:
        print(f"  pred~{ap:.2f}  actual {af:.2f}  (n={n})")


if __name__ == "__main__":
    main()
