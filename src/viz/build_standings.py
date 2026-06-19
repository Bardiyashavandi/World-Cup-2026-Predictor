"""
Build the predicted group-standings page (docs/standings.html).

Reuses the group-table and qualifier logic from the bracket builder, then
renders 12 group tables with qualification colour-coding:
  green  = top-2 (through)
  teal   = one of the 8 best third-placed teams (through)
  dim    = eliminated

    python3 src/viz/build_standings.py
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
from build_board import nav_html, TEAM_FLAGS  # noqa: E402
from build_bracket import group_tables, qualifiers  # noqa: E402

FLAG = lambda t: TEAM_FLAGS.get(t, "🏳️")

HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>World Cup 2026 — Predicted Standings</title>
<style>
  :root{--bg:#0e1117;--panel:#161d2b;--panel2:#1d2636;--line:#2a3446;
    --text:#f4f6fb;--muted:#9aa7bd;--green:#2ecc71;--teal:#27e0a0;}
  *{box-sizing:border-box;} body{margin:0;color:var(--text);
    background:radial-gradient(1200px 600px at 50% -10%,#16321f 0,transparent 60%),var(--bg);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    padding:30px 16px 60px;}
  header{text-align:center;margin-bottom:6px;}
  h1{font-size:28px;margin:0 0 6px;} .sub{color:var(--muted);font-size:14px;margin-bottom:4px;}
  .wrap{max-width:1080px;margin:0 auto;}
  .legend{display:flex;gap:16px;justify-content:center;font-size:12px;color:var(--muted);margin-bottom:22px;flex-wrap:wrap;}
  .dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:middle;}
  .groups{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px;}
  .group{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;}
  .group h2{margin:0;padding:11px 16px;font-size:13px;letter-spacing:1.5px;text-transform:uppercase;
    color:var(--muted);background:var(--panel2);border-bottom:1px solid var(--line);}
  table{width:100%;border-collapse:collapse;font-size:14px;}
  th{font-size:10.5px;letter-spacing:.5px;text-transform:uppercase;color:var(--muted);
    text-align:right;padding:8px 10px 6px;font-weight:600;}
  th.t{text-align:left;}
  td{padding:8px 10px;border-top:1px solid var(--line);text-align:right;}
  td.t{text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px;}
  td.pts{font-weight:800;}
  tr.q td{box-shadow:inset 3px 0 0 var(--green);} tr.q td.t{color:#eafff2;font-weight:700;}
  tr.q3 td{box-shadow:inset 3px 0 0 var(--teal);} tr.q3 td.t{color:#dffbef;font-weight:600;}
  tr.out td{opacity:.5;}
  .pos{color:var(--muted);font-size:12px;}
  footer{text-align:center;color:var(--muted);font-size:12px;margin-top:30px;line-height:1.6;}
</style></head><body>
<div class="wrap">
<header>
  <h1>📊 FIFA World Cup 2026 — Predicted Final Standings</h1>
  <div class="sub">Projected group tables from the ensemble's predicted scorelines</div>
</header>
__NAV__
<div class="legend">
  <span><span class="dot" style="background:var(--green)"></span>Top 2 — through</span>
  <span><span class="dot" style="background:var(--teal)"></span>Best third — through</span>
  <span><span class="dot" style="background:#3a4258"></span>Eliminated</span>
</div>
<div class="groups">
__BODY__
</div>
<footer>
  Points from predicted scorelines (3 win / 1 draw); tie-break GD, then goals, then ELO.<br>
  Eight best third-placed teams also advance. Built from <code>data/predictions/ensemble_md*.csv</code>.
</footer>
</div></body></html>
"""


def main():
    elo = pd.read_csv("data/processed/elo_clean.csv").set_index(
        "country")["rating"].to_dict()
    tables = group_tables(elo)
    qual = qualifiers(tables)
    qual_teams = {q["team"] for q in qual}
    # third-placed qualifiers (for the teal tint) = qualified but not top-2
    top2 = set()
    for teams in tables.values():
        for t in teams[:2]:
            top2.add(t["team"])
    third_q = qual_teams - top2

    blocks = []
    for g in sorted(tables):
        rows = []
        for i, t in enumerate(tables[g], 1):
            team = t["team"]
            cls = ("q" if team in top2 else
                   "q3" if team in third_q else "out")
            rows.append(
                f'<tr class="{cls}">'
                f'<td class="t"><span class="pos">{i}</span> '
                f'{FLAG(team)} {team}</td>'
                f'<td class="pts">{int(t["pts"])}</td>'
                f'<td>{int(t["gd"]):+d}</td>'
                f'<td>{int(t["gf"])}</td></tr>'
            )
        blocks.append(
            f'<div class="group"><h2>Group {g}</h2>'
            f'<table><tr><th class="t">Team</th><th>Pts</th>'
            f'<th>GD</th><th>GF</th></tr>' + "".join(rows) + "</table></div>"
        )

    html = (HTML
            .replace("__NAV__", nav_html("standings.html"))
            .replace("__BODY__", "\n".join(blocks)))
    os.makedirs("docs", exist_ok=True)
    with open("docs/standings.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("✅ Wrote docs/standings.html")


if __name__ == "__main__":
    main()
