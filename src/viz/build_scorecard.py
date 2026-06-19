"""
Build the live accuracy scorecard (docs/scorecard.html).

Grades the model's pre-match ensemble predictions against the real results
entered in data/raw/wc_2026_results.csv — result hit-rate, exact-score
rate and Brier score — and compares them to the 56% backtest benchmark.
Regenerate it after each matchday to watch the model get graded live.

    python3 src/viz/build_scorecard.py
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
from build_board import nav_html, TEAM_FLAGS, with_head  # noqa: E402

FLAG = lambda t: TEAM_FLAGS.get(t, "🏳️")

# Backtest benchmarks (combined WC 2018 + 2022) for the ensemble.
BT_RESULT, BT_EXACT, BT_BRIER = 56.2, 11.0, 0.195


def outcome(h, a):
    return "H" if h > a else ("D" if h == a else "A")


def gather():
    res = pd.read_csv("data/raw/wc_2026_results.csv")
    res = res[res["played"].astype(str).str.lower() == "true"].dropna(
        subset=["home_goals", "away_goals"])
    preds = {}
    for md in (1, 2, 3):
        p = f"data/predictions/ensemble_md{md}.csv"
        if os.path.exists(p):
            for _, r in pd.read_csv(p).iterrows():
                preds[int(r["match_id"])] = r

    rows = []
    for _, m in res.iterrows():
        mid = int(m["match_id"])
        if mid not in preds:
            continue
        pr = preds[mid]
        ah, aa = int(m["home_goals"]), int(m["away_goals"])
        ph, pa = int(pr["predicted_home_goals"]), int(pr["predicted_away_goals"])
        probs = {"H": pr["home_win_prob"], "D": pr["draw_prob"],
                 "A": pr["away_win_prob"]}
        pred_res = max(probs, key=probs.get)
        act_res = outcome(ah, aa)
        # Brier (3-way)
        brier = sum((probs[k] - (1.0 if act_res == k else 0.0)) ** 2
                    for k in "HDA") / 3.0
        rows.append({
            "md": int(m["matchday"]), "group": m["group"],
            "home": m["home_team"], "away": m["away_team"],
            "pred": f"{ph}-{pa}", "act": f"{ah}-{aa}",
            "result_ok": pred_res == act_res,
            "exact_ok": (ph == ah and pa == aa),
            "brier": brier,
            "conf": int(round(max(probs.values()) * 100)),
        })
    return rows


HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>World Cup 2026 — Live Scorecard</title>
<style>
  :root{--bg:#0e1117;--panel:#161d2b;--panel2:#1d2636;--line:#2a3446;
    --text:#f4f6fb;--muted:#9aa7bd;--green:#2ecc71;--red:#e6584a;--amber:#f1b53d;}
  *{box-sizing:border-box;} body{margin:0;color:var(--text);
    background:radial-gradient(1200px 600px at 50% -10%,#16321f 0,transparent 60%),var(--bg);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    padding:30px 16px 60px;}
  .wrap{max-width:880px;margin:0 auto;} header{text-align:center;margin-bottom:6px;}
  h1{font-size:28px;margin:0 0 6px;} .sub{color:var(--muted);font-size:14px;}
  .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:8px 0 8px;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:18px;text-align:center;}
  .big{font-size:38px;font-weight:800;color:var(--green);}
  .lbl{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-top:4px;}
  .vs{font-size:11.5px;color:var(--muted);margin-top:8px;}
  .vs b{color:var(--text);}
  table{width:100%;border-collapse:collapse;font-size:14px;margin-top:18px;
    background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;}
  th{font-size:11px;letter-spacing:.5px;text-transform:uppercase;color:var(--muted);
    text-align:left;padding:10px 12px;background:var(--panel2);}
  td{padding:9px 12px;border-top:1px solid var(--line);}
  td.c{text-align:center;} .mdh{font-weight:700;color:var(--muted);background:var(--panel2);
    font-size:12px;letter-spacing:1px;text-transform:uppercase;}
  .ok{color:var(--green);font-weight:800;} .no{color:var(--red);font-weight:800;}
  .dim{color:var(--muted);} .ex{color:var(--amber);font-weight:800;}
  footer{text-align:center;color:var(--muted);font-size:12px;margin-top:26px;line-height:1.6;}
  .empty{text-align:center;color:var(--muted);padding:40px;background:var(--panel);
    border:1px solid var(--line);border-radius:14px;margin-top:18px;}
</style></head><body>
<div class="wrap">
<header>
  <h1>📈 FIFA World Cup 2026 — Live Scorecard</h1>
  <div class="sub">How the model's predictions are holding up against real results</div>
</header>
__NAV__
__BODY__
<footer>
  Graded on pre-match ensemble predictions vs actual results.
  ✓/✗ = correct outcome (H/D/A); ◆ = exact scoreline. Built from
  <code>wc_2026_results.csv</code> + <code>ensemble_md*.csv</code>.
</footer>
</div></body></html>
"""


def main():
    rows = gather()
    if not rows:
        body = ('<div class="empty">No results entered yet — fill in '
                '<code>data/raw/wc_2026_results.csv</code> and re-run.</div>')
        html = with_head(HTML.replace(
            "__NAV__", nav_html("scorecard.html")).replace("__BODY__", body))
        os.makedirs("docs", exist_ok=True)
        open("docs/scorecard.html", "w", encoding="utf-8").write(html)
        print("✅ Wrote docs/scorecard.html (no results yet)")
        return

    n = len(rows)
    res_acc = 100 * sum(r["result_ok"] for r in rows) / n
    exact = 100 * sum(r["exact_ok"] for r in rows) / n
    brier = sum(r["brier"] for r in rows) / n

    def delta(v, base, lower_better=False):
        d = v - base
        good = (d < 0) if lower_better else (d > 0)
        sign = "+" if d >= 0 else ""
        col = "var(--green)" if good else "var(--red)"
        return f'<span style="color:{col}">{sign}{d:.1f}</span>'

    cards = f"""<div class="cards">
      <div class="card"><div class="big">{res_acc:.0f}%</div>
        <div class="lbl">Result accuracy</div>
        <div class="vs">backtest <b>{BT_RESULT:.0f}%</b> · {delta(res_acc, BT_RESULT)}</div></div>
      <div class="card"><div class="big">{exact:.0f}%</div>
        <div class="lbl">Exact score</div>
        <div class="vs">backtest <b>{BT_EXACT:.0f}%</b> · {delta(exact, BT_EXACT)}</div></div>
      <div class="card"><div class="big">{brier:.3f}</div>
        <div class="lbl">Brier score</div>
        <div class="vs">backtest <b>{BT_BRIER:.3f}</b> · {delta(brier, BT_BRIER, True)}</div></div>
    </div>
    <div class="sub" style="text-align:center">Graded on {n} played match{'es' if n != 1 else ''} so far</div>"""

    # per-match table grouped by matchday
    trs = []
    for md in sorted({r["md"] for r in rows}):
        trs.append(f'<tr><td colspan="5" class="mdh">Matchday {md}</td></tr>')
        for r in [x for x in rows if x["md"] == md]:
            mark = ('<span class="ex">◆</span>' if r["exact_ok"]
                    else '<span class="ok">✓</span>' if r["result_ok"]
                    else '<span class="no">✗</span>')
            trs.append(
                f'<tr><td>{FLAG(r["home"])} {r["home"]} v {r["away"]} {FLAG(r["away"])}</td>'
                f'<td class="c dim">{r["pred"]}</td>'
                f'<td class="c">{r["act"]}</td>'
                f'<td class="c">{mark}</td>'
                f'<td class="c dim">{r["conf"]}%</td></tr>'
            )
    table = ('<table><tr><th>Match</th><th class="c">Pred</th>'
             '<th class="c">Actual</th><th class="c">Hit</th>'
             '<th class="c">Conf</th></tr>' + "".join(trs) + "</table>")

    html = with_head(HTML.replace("__NAV__", nav_html("scorecard.html"))
                     .replace("__BODY__", cards + table))
    os.makedirs("docs", exist_ok=True)
    open("docs/scorecard.html", "w", encoding="utf-8").write(html)
    print(f"✅ Wrote docs/scorecard.html — {res_acc:.0f}% result accuracy "
          f"on {n} matches (exact {exact:.0f}%, Brier {brier:.3f})")


if __name__ == "__main__":
    main()
