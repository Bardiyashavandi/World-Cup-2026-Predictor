"""
Title odds — Monte-Carlo over the real knockout bracket.

Instead of one deterministic champion, simulate the whole bracket many
times: every tie is decided probabilistically by the Elo-derived win
probability (strength = ELO + this tournament's group-stage form), and we
count how often each team lifts the trophy. Outputs docs/title_odds.svg
(a chart embedded in the README).

    python3 src/viz/build_title_odds.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
from build_bracket import REAL_R32, GROUP_PTS, FLAG  # noqa: E402

N_SIMS = 50000


def simulate_odds(elo, n=N_SIMS, seed=0):
    rng = np.random.default_rng(seed)

    def strength(t):
        return elo.get(t, 1500) + 5.0 * GROUP_PTS.get(t, 0)

    teams = [t for pair in REAL_R32 for t in pair]
    s = np.array([strength(t) for t in teams])
    titles = {t: 0 for t in teams}

    for _ in range(n):
        cur = list(range(len(teams)))  # indices into teams
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur), 2):
                a, b = cur[i], cur[i + 1]
                pa = 1.0 / (1.0 + 10 ** ((s[b] - s[a]) / 400.0))
                nxt.append(a if rng.random() < pa else b)
            cur = nxt
        titles[teams[cur[0]]] += 1

    odds = [(t, titles[t] / n * 100) for t in teams]
    odds.sort(key=lambda x: -x[1])
    return odds


def build_svg(odds, top=10):
    rows = odds[:top]
    maxv = rows[0][1]
    rh, top_pad = 30, 70
    h = top_pad + rh * len(rows) + 30
    x0, barmax = 230, 470
    bars = []
    for i, (team, pct) in enumerate(rows):
        y = top_pad + i * rh
        w = max(2, pct / maxv * barmax)
        col = "#2ecc71" if i == 0 else "#27a35a" if i < 3 else "#3a5a78"
        bars.append(
            f'<text x="222" y="{y+15}" text-anchor="end" font-size="13.5" '
            f'font-weight="600" fill="#cdd6e6">{FLAG(team)} {team}</text>'
            f'<rect x="{x0}" y="{y+3}" width="{w:.1f}" height="18" rx="4" '
            f'fill="{col}"/>'
            f'<text x="{x0+w+8:.1f}" y="{y+16}" font-size="12.5" '
            f'font-weight="700" fill="#9aa7bd">{pct:.1f}%</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {h}" \
font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
  <rect width="760" height="{h}" rx="16" fill="#161d2b"/>
  <text x="28" y="38" font-size="20" font-weight="800" fill="#f4f6fb">World Cup 2026 — title odds</text>
  <text x="28" y="59" font-size="13" fill="#9aa7bd">{N_SIMS:,} Monte-Carlo runs of the real knockout bracket · ELO + group form</text>
  {''.join(bars)}
</svg>"""


def main():
    elo = pd.read_csv("data/processed/elo_clean.csv").set_index(
        "country")["rating"].to_dict()
    odds = simulate_odds(elo)
    os.makedirs("docs", exist_ok=True)
    with open("docs/title_odds.svg", "w", encoding="utf-8") as f:
        f.write(build_svg(odds))
    print("✅ Wrote docs/title_odds.svg")
    for t, p in odds[:8]:
        print(f"  {t:14} {p:5.1f}%")


if __name__ == "__main__":
    main()
