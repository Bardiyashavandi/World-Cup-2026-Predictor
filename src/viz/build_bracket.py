"""
Build a predicted knockout bracket → champion (docs/bracket.html).

Pipeline:
  1. Turn the ensemble's predicted group-stage scorelines into 12 group
     tables (3 points a win, 1 a draw; tie-break on goal difference, then
     goals scored, then ELO).
  2. Qualify the top 2 of each group (24) plus the 8 best third-placed
     teams -> 32 teams.
  3. Seed the 32 by ELO into a balanced single-elimination bracket and
     simulate every tie (higher ELO advances; the win probability comes
     from the standard Elo formula) all the way to a predicted champion.
  4. Render a standalone, self-contained HTML bracket.

This is a *model-seeded* simulation, not the exact official FIFA slotting
(the real third-place assignment table is combinatorial) — but every team,
table and tie is driven by the project's own predictions and ratings.

    python3 src/viz/build_bracket.py
"""

import json
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
from build_board import TEAM_FLAGS, nav_html, with_head  # noqa: E402

FLAG = lambda t: TEAM_FLAGS.get(t, "🏳️")


# ── 1. Group tables from predicted scorelines ────────────

def group_tables(elo):
    frames = []
    for md in (1, 2, 3):
        p = f"data/predictions/ensemble_md{md}.csv"
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    matches = pd.concat(frames, ignore_index=True)

    stats = {}

    def touch(team, grp):
        stats.setdefault(team, {"group": grp, "pts": 0, "gf": 0,
                                "ga": 0, "gd": 0})

    for _, m in matches.iterrows():
        h, a, g = m["home_team"], m["away_team"], m["group"]
        hg, ag = int(m["predicted_home_goals"]), int(m["predicted_away_goals"])
        touch(h, g); touch(a, g)
        stats[h]["gf"] += hg; stats[h]["ga"] += ag
        stats[a]["gf"] += ag; stats[a]["ga"] += hg
        if hg > ag:
            stats[h]["pts"] += 3
        elif ag > hg:
            stats[a]["pts"] += 3
        else:
            stats[h]["pts"] += 1; stats[a]["pts"] += 1

    for t, s in stats.items():
        s["gd"] = s["gf"] - s["ga"]
        s["team"] = t
        s["elo"] = elo.get(t, 1500)

    tables = {}
    for g in sorted({s["group"] for s in stats.values()}):
        teams = [s for s in stats.values() if s["group"] == g]
        teams.sort(key=lambda s: (s["pts"], s["gd"], s["gf"], s["elo"]),
                   reverse=True)
        tables[g] = teams
    return tables


# ── 2. Qualifiers: 24 group + 8 best thirds ──────────────

def qualifiers(tables):
    winners, runners, thirds = [], [], []
    for g, teams in tables.items():
        winners.append(teams[0])
        runners.append(teams[1])
        if len(teams) > 2:
            thirds.append(teams[2])
    thirds.sort(key=lambda s: (s["pts"], s["gd"], s["gf"], s["elo"]),
                reverse=True)
    best_thirds = thirds[:8]
    return winners + runners + best_thirds


# ── 3. Seed + simulate ───────────────────────────────────

def seed_positions(n):
    """Standard single-elimination seed order for a power-of-two bracket."""
    order = [1, 2]
    while len(order) < n:
        m = len(order) * 2 + 1
        order = [x for s in order for x in (s, m - s)]
    return order


def win_prob(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def simulate(qual):
    # Seed the 32 qualifiers by ELO strength.
    qual = sorted(qual, key=lambda s: s["elo"], reverse=True)
    seeds = {i + 1: q for i, q in enumerate(qual)}
    order = seed_positions(len(qual))
    bracket = [seeds[s] for s in order]

    round_names = {32: "Round of 32", 16: "Round of 16", 8: "Quarter-finals",
                   4: "Semi-finals", 2: "Final"}
    rounds = []
    current = bracket
    while len(current) > 1:
        size = len(current)
        ties = []
        nxt = []
        for i in range(0, size, 2):
            a, b = current[i], current[i + 1]
            pa = win_prob(a["elo"], b["elo"])
            w, p = (a, pa) if a["elo"] >= b["elo"] else (b, 1 - pa)
            ties.append({
                "a": a["team"], "af": FLAG(a["team"]),
                "b": b["team"], "bf": FLAG(b["team"]),
                "w": w["team"], "conf": round(max(p, 1 - p) * 100),
            })
            nxt.append(w)
        rounds.append({"name": round_names.get(size, f"Round of {size}"),
                       "ties": ties})
        current = nxt
    champion = current[0]
    return rounds, champion


# ── 4. Render ────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>World Cup 2026 — Predicted Bracket</title>
<style>
  :root{--bg:#0e1117;--panel:#161d2b;--panel2:#1d2636;--line:#2a3446;
    --text:#f4f6fb;--muted:#9aa7bd;--green:#2ecc71;--gold:#f1c40f;}
  *{box-sizing:border-box;} body{margin:0;color:var(--text);
    background:radial-gradient(1200px 600px at 50% -10%,#16321f 0,transparent 60%),var(--bg);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    padding:30px 16px 60px;}
  header{text-align:center;margin-bottom:8px;}
  h1{font-size:28px;margin:0 0 6px;} .sub{color:var(--muted);font-size:14px;}
  .champ{margin:24px auto 30px;max-width:420px;text-align:center;
    background:linear-gradient(180deg,rgba(241,196,15,.16),rgba(241,196,15,.04));
    border:1px solid rgba(241,196,15,.45);border-radius:16px;padding:20px;}
  .champ .lbl{color:var(--gold);font-size:12px;letter-spacing:2px;text-transform:uppercase;}
  .champ .flag{font-size:58px;margin:6px 0;} .champ .name{font-size:26px;font-weight:800;}
  .scroll{overflow-x:auto;padding-bottom:12px;}
  .bracket{display:flex;gap:14px;min-width:max-content;margin:0 auto;justify-content:center;
    align-items:stretch;}
  .col{display:flex;flex-direction:column;min-width:188px;}
  .col h3{flex:none;text-align:center;font-size:11px;letter-spacing:1.4px;
    text-transform:uppercase;color:var(--muted);margin:0 0 10px;}
  .col-body{flex:1;display:flex;flex-direction:column;justify-content:space-around;}
  .tie{background:var(--panel);border:1px solid var(--line);border-radius:10px;
    overflow:hidden;margin:7px 0;}
  .side{display:flex;align-items:center;gap:7px;padding:7px 10px;font-size:13px;
    border-bottom:1px solid var(--line);}
  .side:last-child{border-bottom:none;}
  .side .flag{font-size:16px;} .side .nm{flex:1;white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis;}
  .side.win{background:rgba(46,204,113,.12);font-weight:700;}
  .side.win .nm{color:var(--green);}
  .side.lose{opacity:.5;}
  .conf{font-size:10px;color:var(--muted);padding:3px 10px 6px;text-align:right;}
  footer{text-align:center;color:var(--muted);font-size:12px;margin-top:30px;line-height:1.6;}
</style></head><body>
<header>
  <h1>🏆 FIFA World Cup 2026 — Predicted Bracket</h1>
  <div class="sub">Model-seeded knockout simulation from the ensemble's group-stage predictions</div>
</header>
__NAV__
<div class="champ">
  <div class="lbl">Predicted Champion</div>
  <div class="flag">__CF__</div>
  <div class="name">__CN__</div>
</div>
<div class="scroll"><div class="bracket" id="bracket"></div></div>
<footer>
  Higher-rated team advances each tie; the % is its Elo win probability.<br>
  A model-seeded simulation — not the official FIFA bracket slotting. Built from
  <code>data/predictions/ensemble_md*.csv</code>.
</footer>
<script>
const ROUNDS = __DATA__;
const wrap = document.getElementById("bracket");
ROUNDS.forEach(r=>{
  const col=document.createElement("div"); col.className="col";
  col.innerHTML=`<h3>${r.name}</h3><div class="col-body">`+r.ties.map(t=>`
    <div class="item">
      <div class="tie">
        <div class="side ${t.w===t.a?'win':'lose'}"><span class="flag">${t.af}</span><span class="nm">${t.a}</span></div>
        <div class="side ${t.w===t.b?'win':'lose'}"><span class="flag">${t.bf}</span><span class="nm">${t.b}</span></div>
      </div>
      <div class="conf">${t.conf}% ${t.w}</div>
    </div>`).join("")+`</div>`;
  wrap.appendChild(col);
});
// champion column
const cc=document.createElement("div"); cc.className="col";
cc.innerHTML=`<h3>Champion</h3><div class="col-body"><div class="item"><div class="tie"><div class="side win">
  <span class="flag">__CF__</span><span class="nm">__CN__</span></div></div></div></div>`;
wrap.appendChild(cc);
</script></body></html>
"""


def main():
    elo = pd.read_csv("data/processed/elo_clean.csv").set_index(
        "country")["rating"].to_dict()
    tables = group_tables(elo)
    qual = qualifiers(tables)
    rounds, champ = simulate(qual)

    html = with_head(HTML
                     .replace("__DATA__", json.dumps(rounds, ensure_ascii=False))
                     .replace("__NAV__", nav_html("bracket.html"))
                     .replace("__CF__", FLAG(champ["team"]))
                     .replace("__CN__", champ["team"]))
    os.makedirs("docs", exist_ok=True)
    with open("docs/bracket.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Wrote docs/bracket.html — predicted champion: "
          f"{champ['team']}")
    print("   Group winners:",
          ", ".join(t[0]["team"] for t in tables.values()))


if __name__ == "__main__":
    main()
