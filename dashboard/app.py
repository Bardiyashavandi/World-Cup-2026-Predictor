import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

sys.path.append("src")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────

@st.cache_data
def load_predictions(matchday: int) -> pd.DataFrame:
    path = f"data/predictions/ensemble_md{matchday}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_all_model_predictions(matchday: int) -> dict:
    models = {
        "ELO":            "data/predictions/elo_all.csv",
        "Dixon-Coles":    "data/predictions/dixon_coles_all.csv",
        "Historical Avg": "data/predictions/historical_avg_all.csv",
        "Dynamic ELO":    "data/predictions/dynamic_elo_all.csv",
        "XGBoost":        "data/predictions/xgboost_all.csv",
        "LightGBM":       "data/predictions/lightgbm_all.csv",
        "Neural Network": "data/predictions/neural_net_all.csv",
        "Logistic Reg":   "data/predictions/logistic_all.csv",
    }
    result = {}
    for name, path in models.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "matchday" in df.columns:
                df = df[df["matchday"] == matchday]
            result[name] = df

    # Per-matchday files: the final ensemble and the market-blended
    # predictions (de-vigged bookmaker odds blended with the ensemble).
    extra = {
        "Ensemble": f"data/predictions/ensemble_md{matchday}.csv",
        "Market Blend": f"data/predictions/market_blended_md{matchday}.csv",
    }
    for name, path in extra.items():
        if os.path.exists(path):
            result[name] = pd.read_csv(path)
    return result


@st.cache_data
def load_results() -> pd.DataFrame:
    path = "data/raw/wc_2026_results.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df[df["played"] == True] if "played" in df.columns else pd.DataFrame()
    return pd.DataFrame()


@st.cache_data
def load_standings(matchday: int) -> pd.DataFrame:
    path = f"data/processed/standings_after_md{matchday}.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df if len(df) > 0 else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data
def load_backtest() -> pd.DataFrame:
    path = "data/processed/backtest_results.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ─────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────

def confidence_badge(pct: float) -> str:
    if pct >= 70:
        return "🔥 HIGH"
    elif pct >= 50:
        return "📊 MED"
    else:
        return "❓ LOW"


def result_color(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "#2ecc71"
    elif home_goals == away_goals:
        return "#f39c12"
    else:
        return "#e74c3c"


# Flag emoji per team (matches the naming used in the prediction CSVs).
TEAM_FLAGS = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺",
    "Austria": "🇦🇹", "Belgium": "🇧🇪", "Bosnia": "🇧🇦",
    "Brazil": "🇧🇷", "Canada": "🇨🇦", "Cape Verde": "🇨🇻",
    "Colombia": "🇨🇴", "Cote d'Ivoire": "🇨🇮", "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼", "Czech Republic": "🇨🇿", "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨", "Egypt": "🇪🇬", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "France": "🇫🇷", "Germany": "🇩🇪", "Ghana": "🇬🇭",
    "Haiti": "🇭🇹", "Iran": "🇮🇷", "Iraq": "🇮🇶", "Japan": "🇯🇵",
    "Jordan": "🇯🇴", "Mexico": "🇲🇽", "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱", "New Zealand": "🇳🇿", "Norway": "🇳🇴",
    "Panama": "🇵🇦", "Paraguay": "🇵🇾", "Portugal": "🇵🇹",
    "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Senegal": "🇸🇳", "South Africa": "🇿🇦", "South Korea": "🇰🇷",
    "Spain": "🇪🇸", "Sweden": "🇸🇪", "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳", "Turkey": "🇹🇷", "USA": "🇺🇸",
    "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿",
}


def flag(team: str) -> str:
    """Return the flag emoji for a team (empty string if unknown)."""
    return TEAM_FLAGS.get(team, "")


# ─────────────────────────────────────────
# COMPONENTS
# ─────────────────────────────────────────

def render_match_card(row: pd.Series, actual_result=None):
    """Render a single match prediction card."""
    home = row["home_team"]
    away = row["away_team"]
    h_goals = row["predicted_home_goals"]
    a_goals = row["predicted_away_goals"]
    h_xg = row.get("home_xg", 0)
    a_xg = row.get("away_xg", 0)
    h_prob = row.get("home_win_prob", 0)
    d_prob = row.get("draw_prob", 0)
    a_prob = row.get("away_win_prob", 0)
    agreement = row.get("agreement_pct", 0)

    with st.container():
        col1, col2, col3 = st.columns([3, 2, 3])

        with col1:
            st.markdown(f"### {flag(home)} {home}")
            st.markdown(f"xG: **{h_xg:.2f}**")
            st.progress(h_prob)
            st.caption(f"Win: {h_prob:.1%}")

        with col2:
            st.markdown(
                f"<h2 style='text-align:center;"
                f"color:{result_color(h_goals, a_goals)}'>"
                f"{h_goals} - {a_goals}</h2>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='text-align:center'>"
                f"Draw: {d_prob:.1%}</p>",
                unsafe_allow_html=True
            )
            if agreement:
                st.markdown(
                    f"<p style='text-align:center'>"
                    f"{confidence_badge(agreement)}</p>",
                    unsafe_allow_html=True
                )
            if actual_result is not None:
                st.markdown(
                    f"<p style='text-align:center;color:gray'>"
                    f"Actual: {int(actual_result['home_goals'])}"
                    f"-{int(actual_result['away_goals'])}</p>",
                    unsafe_allow_html=True
                )

        with col3:
            st.markdown(f"### {flag(away)} {away}")
            st.markdown(f"xG: **{a_xg:.2f}**")
            st.progress(a_prob)
            st.caption(f"Win: {a_prob:.1%}")

        # Stacked H / Draw / A probability bar spanning the full card.
        render_probability_bar(row)
        st.divider()


def render_probability_bar(row: pd.Series):
    """Render a horizontal probability bar for a match."""
    h_prob = row.get("home_win_prob", 0)
    d_prob = row.get("draw_prob", 0)
    a_prob = row.get("away_win_prob", 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[h_prob], y=[""],
        orientation="h",
        name=row["home_team"],
        marker_color="#2ecc71",
        text=f"{h_prob:.1%}",
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=[d_prob], y=[""],
        orientation="h",
        name="Draw",
        marker_color="#f39c12",
        text=f"{d_prob:.1%}",
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=[a_prob], y=[""],
        orientation="h",
        name=row["away_team"],
        marker_color="#e74c3c",
        text=f"{a_prob:.1%}",
        textposition="inside",
    ))
    fig.update_layout(
        barmode="stack",
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(showticklabels=False, range=[0, 1]),
        yaxis=dict(showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison(match_id: int,
                             home_team: str,
                             away_team: str,
                             all_preds: dict):
    """Show how different models predicted this match."""
    rows = []
    for model_name, df in all_preds.items():
        match = df[df["match_id"] == match_id]
        if len(match) == 0:
            continue
        r = match.iloc[0]
        if "home_xg" in r:
            rows.append({
                "Model": model_name,
                "Home xG": round(r["home_xg"], 2),
                "Away xG": round(r["away_xg"], 2),
                "H Win%": f"{r['home_win_prob']:.1%}",
                "Draw%": f"{r['draw_prob']:.1%}",
                "A Win%": f"{r['away_win_prob']:.1%}",
            })
        else:
            rows.append({
                "Model": model_name,
                "Home xG": "-",
                "Away xG": "-",
                "H Win%": f"{r['home_win_prob']:.1%}",
                "Draw%": f"{r['draw_prob']:.1%}",
                "A Win%": f"{r['away_win_prob']:.1%}",
            })
    if rows:
        st.dataframe(
            pd.DataFrame(rows).set_index("Model"),
            use_container_width=True
        )


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

def main():
    st.title("⚽ FIFA World Cup 2026 Predictor")
    st.markdown(
        "Multi-model prediction system combining ELO, "
        "Dixon-Coles, XGBoost, LightGBM, Neural Network "
        "and more into an ensemble prediction."
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select page",
        ["🏠 Overview", "📊 Predictions", "🏆 Standings",
         "🔍 Model Comparison", "📈 Results Tracker",
         "📉 Backtest Results"]
    )

    matchday = st.sidebar.selectbox(
        "Matchday", [1, 2, 3], index=0
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Models used:**")
    st.sidebar.markdown("""
    - Static ELO
    - Dixon-Coles Poisson
    - Historical Average
    - Dynamic ELO
    - XGBoost (time-weighted)
    - LightGBM (time-weighted)
    - Neural Network
    - Logistic Regression
    - Stakes Model (MD2/MD3)
    """)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "After entering results run:\n"
        "`python3 src/updater/update_predictions.py --matchday 1`"
    )

    # Load data
    ensemble_df = load_predictions(matchday)
    results_df = load_results()
    all_preds = load_all_model_predictions(matchday)

    # ── OVERVIEW PAGE ────────────────────
    if page == "🏠 Overview":
        st.header(f"MD{matchday} Overview")

        if len(ensemble_df) == 0:
            st.warning(
                f"No ensemble predictions found for MD{matchday}. "
                f"Run: python3 src/ensemble/ensemble.py {matchday}"
            )
            return

        col1, col2, col3, col4 = st.columns(4)
        home_wins = int((ensemble_df["predicted_home_goals"] > ensemble_df["predicted_away_goals"]).sum())
        draws = int((ensemble_df["predicted_home_goals"] == ensemble_df["predicted_away_goals"]).sum())
        away_wins = int((ensemble_df["predicted_home_goals"] < ensemble_df["predicted_away_goals"]).sum())
        avg_agreement = ensemble_df.get(
            "agreement_pct", pd.Series([0])
        ).mean()

        col1.metric("Home Wins", home_wins)
        col2.metric("Draws", draws)
        col3.metric("Away Wins", away_wins)
        col4.metric("Avg Confidence", f"{avg_agreement:.0f}%")

        st.markdown("---")
        st.subheader("All Matches")
        groups = sorted(ensemble_df["group"].unique())

        for group in groups:
            group_matches = ensemble_df[
                ensemble_df["group"] == group
            ]
            st.markdown(f"**Group {group}**")
            for _, row in group_matches.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 3, 2])
                col1.markdown(f"**{flag(row['home_team'])} {row['home_team']}**")
                col2.markdown(
                    f"**{row['predicted_home_goals']}-"
                    f"{row['predicted_away_goals']}**"
                )
                col3.markdown(f"**{flag(row['away_team'])} {row['away_team']}**")
                col4.markdown(
                    confidence_badge(row.get("agreement_pct", 0))
                )
                render_probability_bar(row)

    # ── PREDICTIONS PAGE ─────────────────
    elif page == "📊 Predictions":
        st.header(f"MD{matchday} Detailed Predictions")

        if len(ensemble_df) == 0:
            st.warning("No predictions found.")
            return

        group_filter = st.selectbox(
            "Filter by group",
            ["All"] + sorted(
                ensemble_df["group"].unique().tolist()
            )
        )

        filtered = (
            ensemble_df if group_filter == "All"
            else ensemble_df[
                ensemble_df["group"] == group_filter
            ]
        )

        for _, row in filtered.iterrows():
            actual = None
            if len(results_df) > 0 and "match_id" in results_df.columns:
                actual_rows = results_df[
                    results_df["match_id"] == row["match_id"]
                ]
                if len(actual_rows) > 0:
                    actual = actual_rows.iloc[0]

            st.markdown(
                f"**Group {row['group']} — "
                f"{row['home_team']} vs {row['away_team']}**"
            )
            render_match_card(row, actual)

    # ── STANDINGS PAGE ───────────────────
    elif page == "🏆 Standings":
        st.header("Group Standings")

        standings_md = st.selectbox(
            "Standings after matchday", [1, 2, 3], index=0
        )
        standings_df = load_standings(standings_md)

        if len(standings_df) == 0:
            st.info(
                f"No standings yet for MD{standings_md}. "
                "Fill in results and run group_standings.py"
            )
        else:
            groups = sorted(standings_df["group"].unique())
            cols = st.columns(2)
            for i, group in enumerate(groups):
                with cols[i % 2]:
                    st.markdown(f"**Group {group}**")
                    group_df = standings_df[
                        standings_df["group"] == group
                    ].sort_values("group_position")[
                        ["group_position", "team", "points",
                         "goal_difference", "goals_for",
                         "wins", "draws", "losses"]
                    ].rename(columns={
                        "group_position": "Pos",
                        "team": "Team",
                        "points": "Pts",
                        "goal_difference": "GD",
                        "goals_for": "GF",
                        "wins": "W",
                        "draws": "D",
                        "losses": "L"
                    })
                    st.dataframe(
                        group_df.set_index("Pos"),
                        use_container_width=True
                    )

    # ── MODEL COMPARISON PAGE ────────────
    elif page == "🔍 Model Comparison":
        st.header("Model Comparison")

        if len(ensemble_df) == 0:
            st.warning("No predictions found.")
            return

        match_options = {
            f"Group {r['group']}: {r['home_team']} vs {r['away_team']}": r["match_id"]
            for _, r in ensemble_df.iterrows()
        }

        selected = st.selectbox(
            "Select match", list(match_options.keys())
        )
        match_id = match_options[selected]
        match_row = ensemble_df[
            ensemble_df["match_id"] == match_id
        ].iloc[0]

        home = match_row["home_team"]
        away = match_row["away_team"]

        st.markdown(f"### {home} vs {away}")
        st.markdown(
            f"**Ensemble: "
            f"{match_row['predicted_home_goals']}-"
            f"{match_row['predicted_away_goals']}** | "
            f"xG: {match_row['home_xg']:.2f}-"
            f"{match_row['away_xg']:.2f}"
        )

        st.markdown("---")
        st.markdown("**Individual model predictions:**")
        render_model_comparison(match_id, home, away, all_preds)

        xg_rows = []
        for model_name, df in all_preds.items():
            match = df[df["match_id"] == match_id]
            if len(match) == 0 or "home_xg" not in match.columns:
                continue
            r = match.iloc[0]
            xg_rows.append({
                "Model": model_name,
                "Team": home,
                "xG": r["home_xg"]
            })
            xg_rows.append({
                "Model": model_name,
                "Team": away,
                "xG": r["away_xg"]
            })

        if xg_rows:
            xg_df = pd.DataFrame(xg_rows)
            fig = px.bar(
                xg_df, x="Model", y="xG",
                color="Team", barmode="group",
                title="Expected Goals by Model",
                color_discrete_sequence=["#2ecc71", "#e74c3c"]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ── RESULTS TRACKER PAGE ─────────────
    elif page == "📈 Results Tracker":
        st.header("Results Tracker")

        if len(results_df) == 0:
            st.info(
                "No results yet. "
                "Fill in data/raw/wc_2026_results.csv "
                "after matches are played."
            )
            return

        st.markdown(f"**{len(results_df)} matches played so far**")

        for _, row in results_df.iterrows():
            col1, col2, col3 = st.columns([3, 2, 3])
            col1.markdown(f"**{row['home_team']}**")
            col2.markdown(
                f"**{int(row['home_goals'])}-"
                f"{int(row['away_goals'])}**"
            )
            col3.markdown(f"**{row['away_team']}**")

    # ── BACKTEST PAGE ────────────────────
    elif page == "📉 Backtest Results":
        st.header("Model Backtesting — WC 2018 & 2022")
        st.markdown(
            "How well did our models predict real World Cup "
            "matches they had never seen before?"
        )

        bt = load_backtest()

        if len(bt) == 0:
            st.warning(
                "No backtest results found. "
                "Run: `python3 src/evaluation/backtest.py`"
            )
            return

        col1, col2 = st.columns(2)

        for col, year in zip([col1, col2], [2018, 2022]):
            with col:
                st.subheader(f"WC {year}")
                bt_year = bt[bt["year"] == year].sort_values(
                    "result_accuracy", ascending=False
                )[["model", "result_accuracy",
                   "exact_score_accuracy",
                   "brier_score", "home_mae",
                   "n_matches"]].rename(columns={
                    "model": "Model",
                    "result_accuracy": "Result %",
                    "exact_score_accuracy": "Exact %",
                    "brier_score": "Brier",
                    "home_mae": "H-MAE",
                    "n_matches": "N"
                })
                bt_year["Result %"] = bt_year[
                    "Result %"
                ].apply(lambda x: f"{x:.1%}")
                bt_year["Exact %"] = bt_year[
                    "Exact %"
                ].apply(lambda x: f"{x:.1%}")
                st.dataframe(
                    bt_year.set_index("Model"),
                    use_container_width=True
                )

        st.markdown("---")
        st.subheader("Combined Performance (2018 + 2022)")

        combined_rows = []
        for model in bt["model"].unique():
            model_data = bt[bt["model"] == model]
            combined_rows.append({
                "Model": model,
                "Avg Result %": f"{model_data['result_accuracy'].mean():.1%}",
                "Avg Exact %": f"{model_data['exact_score_accuracy'].mean():.1%}",
                "Avg Brier": f"{model_data['brier_score'].mean():.3f}",
                "Avg H-MAE": f"{model_data['home_mae'].mean():.3f}",
            })

        combined_df = pd.DataFrame(combined_rows).sort_values(
            "Avg Result %", ascending=False
        )
        st.dataframe(
            combined_df.set_index("Model"),
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("Key Takeaways")
        st.markdown("""
        - The **Ensemble** is the best all-rounder: ~56% result accuracy
          (tied best) with the best scoreline error and Brier score
        - **Logistic Regression** ties it on result accuracy; **LightGBM /
          XGBoost** are best for scorelines
        - **Betting markets** predict ~55% — the top models are right there
        - Rebalancing the ensemble weights toward the strongest models lifted
          it from ~54% to ~56% (it had badly under-weighted Logistic)
        - **Brier ~0.19** shows well-calibrated probabilities
        - **Historical Average** is the weakest baseline, as expected
        - Every model is scored on the identical 64 fixtures per tournament
        """)


if __name__ == "__main__":
    main()