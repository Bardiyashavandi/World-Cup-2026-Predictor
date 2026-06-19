"""
Build a standalone, self-contained predictions board (docs/index.html).

Reads the ensemble predictions and bakes them into a single HTML file with
team flags, colour-coded probability bars and predicted scorelines — no
server or build step needed. Open the file directly, or host it free on
GitHub Pages (Settings → Pages → /docs).

    python3 src/viz/build_board.py
"""

import json
import os
import pandas as pd

TEAM_FLAGS = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺",
    "Austria": "🇦🇹", "Belgium": "🇧🇪", "Bosnia": "🇧🇦",
    "Brazil": "🇧🇷", "Canada": "🇨🇦", "Cape Verde": "🇨🇻",
    "Colombia": "🇨🇴", "Cote d'Ivoire": "🇨🇮", "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼", "Czech Republic": "🇨🇿", "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨", "Egypt": "🇪🇬", "England": "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "France": "🇫🇷", "Germany": "🇩🇪", "Ghana": "🇬🇭",
    "Haiti": "🇭🇹", "Iran": "🇮🇷", "Iraq": "🇮🇶", "Japan": "🇯🇵",
    "Jordan": "🇯🇴", "Mexico": "🇲🇽", "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱", "New Zealand": "🇳🇿", "Norway": "🇳🇴",
    "Panama": "🇵🇦", "Paraguay": "🇵🇾", "Portugal": "🇵🇹",
    "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦", "Scotland": "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "Senegal": "🇸🇳", "South Africa": "🇿🇦", "South Korea": "🇰🇷",
    "Spain": "🇪🇸", "Sweden": "🇸🇪", "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳", "Turkey": "🇹🇷", "USA": "🇺🇸",
    "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿",
}


NAV_LINKS = [("index.html", "Board"), ("standings.html", "Standings"),
             ("bracket.html", "Bracket")]


def nav_html(active):
    """Self-contained (inline-styled) nav bar shared across the pages."""
    items = []
    for href, label in NAV_LINKS:
        on = href == active
        style = ("background:#2ecc71;color:#06210f;" if on else
                 "background:#161d2b;color:#9aa7bd;border:1px solid #2a3446;")
        items.append(
            f'<a href="{href}" style="{style}text-decoration:none;'
            f'padding:8px 18px;border-radius:999px;font-size:14px;'
            f'font-weight:700;">{label}</a>'
        )
    return ('<div style="display:flex;gap:8px;justify-content:center;'
            'margin:4px 0 26px;flex-wrap:wrap;">' + "".join(items) + "</div>")


def load_matches():
    rows = []
    for md in (1, 2, 3):
        path = f"data/predictions/ensemble_md{md}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "md": int(r["matchday"]),
                "group": r["group"],
                "home": r["home_team"],
                "away": r["away_team"],
                "hf": TEAM_FLAGS.get(r["home_team"], "🏳️"),
                "af": TEAM_FLAGS.get(r["away_team"], "🏳️"),
                "hg": int(r["predicted_home_goals"]),
                "ag": int(r["predicted_away_goals"]),
                "hp": round(float(r["home_win_prob"]) * 100),
                "dp": round(float(r["draw_prob"]) * 100),
                "ap": round(float(r["away_win_prob"]) * 100),
                "conf": round(float(r.get("agreement_pct", 0))),
            })
    return rows


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>World Cup 2026 — Match Predictions</title>
<style>
  :root {
    --bg:#0e1117; --panel:#161d2b; --panel2:#1d2636; --line:#2a3446;
    --text:#f4f6fb; --muted:#9aa7bd; --green:#2ecc71; --amber:#f1b53d;
    --red:#e6584a;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:
      radial-gradient(1200px 600px at 50% -10%, #16321f 0%, transparent 60%),
      var(--bg);
    color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    padding:32px 18px 60px;
  }
  .wrap { max-width:1080px; margin:0 auto; }
  header { text-align:center; margin-bottom:8px; }
  h1 { font-size:30px; margin:0 0 6px; letter-spacing:.3px; }
  .sub { color:var(--muted); font-size:14px; }
  .pill { display:inline-block; background:rgba(46,204,113,.14);
    color:var(--green); border:1px solid rgba(46,204,113,.35);
    padding:3px 10px; border-radius:999px; font-size:12px; margin-top:10px; }
  .tabs { display:flex; gap:8px; justify-content:center; margin:22px 0 26px; flex-wrap:wrap; }
  .tab { background:var(--panel); color:var(--muted); border:1px solid var(--line);
    padding:9px 18px; border-radius:999px; cursor:pointer; font-size:14px;
    font-weight:600; transition:.15s; }
  .tab:hover { color:var(--text); }
  .tab.active { background:var(--green); color:#06210f; border-color:var(--green); }
  .groups { display:grid; grid-template-columns:repeat(auto-fill,minmax(345px,1fr));
    gap:16px; }
  .group { background:var(--panel); border:1px solid var(--line);
    border-radius:14px; overflow:hidden; }
  .group h2 { margin:0; padding:11px 16px; font-size:13px; letter-spacing:1.5px;
    text-transform:uppercase; color:var(--muted);
    background:var(--panel2); border-bottom:1px solid var(--line); }
  .match { padding:13px 16px; border-bottom:1px solid var(--line); }
  .match:last-child { border-bottom:none; }
  .row { display:flex; align-items:center; gap:10px; }
  .team { display:flex; align-items:center; gap:8px; font-size:14px; font-weight:600;
    flex:1 1 0; min-width:0; }
  .team .flag { font-size:18px; flex:none; }
  .team.away { flex-direction:row-reverse; text-align:right; }
  .name { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; min-width:0; }
  .score { flex:none; white-space:nowrap; font-size:19px; font-weight:800;
    padding:2px 12px; border-radius:8px; background:var(--panel2); text-align:center; }
  .score.h { color:var(--green); } .score.d { color:var(--amber); } .score.a { color:var(--red); }
  .bar { display:flex; height:9px; border-radius:6px; overflow:hidden; margin:11px 0 6px; }
  .bar i { display:block; }
  .bar .h { background:var(--green); } .bar .d { background:var(--amber); } .bar .a { background:var(--red); }
  .legend { display:flex; justify-content:space-between; font-size:11.5px; color:var(--muted); }
  .legend b { color:var(--text); font-weight:700; }
  .conf { float:right; font-size:11px; padding:1px 8px; border-radius:999px;
    margin-top:8px; font-weight:700; }
  .conf.hi { background:rgba(46,204,113,.16); color:var(--green); }
  .conf.md { background:rgba(241,181,61,.16); color:var(--amber); }
  .conf.lo { background:rgba(154,167,189,.16); color:var(--muted); }
  footer { text-align:center; color:var(--muted); font-size:12px; margin-top:32px; line-height:1.6; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>⚽ FIFA World Cup 2026 — Match Predictions</h1>
    <div class="sub">Ensemble of 9 models · ~56% backtested result accuracy · host USA · Canada · Mexico</div>
    <div class="pill">Green = home win · Amber = draw · Red = away win</div>
  </header>
  __NAV__
  <div class="tabs" id="tabs"></div>
  <div class="groups" id="board"></div>
  <footer>
    Predictions are model output, not certainties — the underlying expected goals drive each call.<br>
    Built from <code>data/predictions/ensemble_md*.csv</code>.
  </footer>
</div>
<script>
const MATCHES = __DATA__;
const GROUPS = [..."ABCDEFGHIJKL"];
function confClass(c){ return c>=80?"hi":c>=60?"md":"lo"; }
function confLabel(c){ return c>=80?"HIGH":c>=60?"MED":"LOW"; }
function resCls(m){ return m.hg>m.ag?"h":m.hg===m.ag?"d":"a"; }
function render(md){
  const board=document.getElementById("board"); board.innerHTML="";
  GROUPS.forEach(g=>{
    const ms=MATCHES.filter(m=>m.md===md && m.group===g);
    if(!ms.length) return;
    const el=document.createElement("div"); el.className="group";
    el.innerHTML=`<h2>Group ${g}</h2>`+ms.map(m=>`
      <div class="match">
        <div class="row">
          <div class="team home"><span class="flag">${m.hf}</span><span class="name">${m.home}</span></div>
          <div class="score ${resCls(m)}">${m.hg}–${m.ag}</div>
          <div class="team away"><span class="flag">${m.af}</span><span class="name">${m.away}</span></div>
        </div>
        <div class="bar">
          <i class="h" style="width:${m.hp}%"></i>
          <i class="d" style="width:${m.dp}%"></i>
          <i class="a" style="width:${m.ap}%"></i>
        </div>
        <div class="legend"><span><b>${m.hp}%</b> W</span><span><b>${m.dp}%</b> D</span><span><b>${m.ap}%</b> W</span></div>
        <span class="conf ${confClass(m.conf)}">${confLabel(m.conf)} ${m.conf}%</span>
        <div style="clear:both"></div>
      </div>`).join("");
    board.appendChild(el);
  });
}
const tabs=document.getElementById("tabs");
[1,2,3].forEach(md=>{
  const b=document.createElement("button");
  b.className="tab"+(md===1?" active":""); b.textContent="Matchday "+md;
  b.onclick=()=>{document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
    b.classList.add("active"); render(md);};
  tabs.appendChild(b);
});
render(1);
</script>
</body>
</html>
"""


def main():
    matches = load_matches()
    html = (HTML
            .replace("__DATA__", json.dumps(matches, ensure_ascii=False))
            .replace("__NAV__", nav_html("index.html")))
    os.makedirs("docs", exist_ok=True)
    out = "docs/index.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Wrote {out} ({len(matches)} matches)")
    print("   Open it directly, or host free on GitHub Pages (Settings → Pages → /docs).")


if __name__ == "__main__":
    main()
