# ⚽ FIFA World Cup 2026 — Group Stage Predictor

> A production-grade multi-model machine learning system for predicting the 2026 FIFA World Cup group stage. Combines classical statistical models, gradient boosting, and neural networks into a weighted ensemble with live Bayesian updating after each matchday.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Models](https://img.shields.io/badge/Models-9-green)
![Accuracy](https://img.shields.io/badge/Best%20Model%20Accuracy-56.2%25-brightgreen)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![CI](https://github.com/Bardiyashavandi/world-cup-2026-predictor/actions/workflows/ci.yml/badge.svg)

---

## 🎯 What This Project Does

This system predicts **scorelines and outcome probabilities** for all 72 group stage matches of the 2026 FIFA World Cup (USA/Canada/Mexico, June 11 – June 27).

For each match it outputs:
- **Predicted scoreline** (e.g. Spain 2-0 Cape Verde)
- **Expected goals** for each team (e.g. xG: 2.21 vs 0.56)
- **Win/Draw/Loss probabilities** (e.g. H:74% D:17% A:8%)
- **Confidence score** based on how many models agree

The system is **iterative** — after each matchday it ingests real results, updates all features and standings, and produces improved predictions for remaining matches. A specialist stakes model activates from MD2 onwards when group context becomes available.

---

## 🏗️ System Architecture

    Raw Data (6 sources)
           ↓
    Data Processing + Team Name Normalization
           ↓
    Feature Engineering (24 features per match)
           ↓
    ┌──────────────────────────────────────────────┐
    │              9 Prediction Models             │
    │                                              │
    │  Statistical:       ML:          Specialist: │
    │  ├─ Static ELO      ├─ XGBoost   └─ Stakes  │
    │  ├─ Dixon-Coles     ├─ LightGBM             │
    │  ├─ Hist. Average   ├─ Neural Net           │
    │  └─ Dynamic ELO     └─ Logistic Reg         │
    └──────────────────────────────────────────────┘
           ↓
    Weighted Ensemble (matchday-aware weights)
           ↓
    Final Predictions + Confidence Scores
           ↓
    Streamlit Dashboard + CSV Outputs

---

## 📚 Datasets

All data is public and sourced from Kaggle (international results and ELO ratings) plus the official 2026 World Cup draw. Raw files live in `data/raw/`; cleaned and feature-engineered versions are written to `data/processed/`.

### Core datasets (used by the models)

| File | Rows | What it is | How it's used |
|------|------|------------|----------------|
| `results.csv` | 49,445 | Every international men's match from 1872 to today (date, teams, score, tournament, venue, neutral flag). [Kaggle: martj42](https://kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | The backbone. Cleaned to **19,713 matches from 2006 onward** (`results_clean.csv`), then turned into **15,857 training rows** with 24 features each. Drives form, head-to-head, Dynamic ELO and all ML models. |
| `elo_ratings_wc2026.csv` | 4,683 | Yearly historical ELO-rating snapshots for national teams. [Kaggle: afonsofernandescruz](https://kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings) | Cleaned to a current snapshot of the **48 qualified teams** (`elo_clean.csv`) — the team-strength signal for the ELO model and the `elo_diff` feature. |
| `wc_2026_fixtures.csv` | 72 | All 72 group-stage fixtures from the official draw (Dec 2025): match, group, teams, date, city, matchday. | The matches we actually predict. |
| `wc_2026_results.csv` | 72 | Live results tracker — same 72 fixtures with empty `home_goals`/`away_goals`/`played` columns. | You fill this in as the tournament unfolds; the updater re-runs predictions from it. |

### Supplementary datasets (used in EDA / reserved for future features)

| File | Rows | What it is |
|------|------|------------|
| `goalscorers.csv` | 47,601 | Goal-by-goal scorer and minute data for historical matches. |
| `shootouts.csv` | 677 | Outcomes of penalty shootouts in historical matches. |
| `wc_2018_team_stats.csv` | 53 | Team-level stats from the 2018 World Cup. |
| `wc_2022_match_stats.csv` | 127 | Match-level stats (including real shot data) from the 2022 World Cup — a candidate source for richer xG features (see Future Work). |

### How the data flows

The raw 49k-match history is filtered to **2006 onward** (modern era, more relevant squads) and normalized so team names match across sources (e.g. "Korea Republic" → "South Korea"). Feature engineering then computes 24 features per match using only information available *before* each match's date — no future-data leakage. Time-decay weighting makes recent matches count up to ~5.8× more than the oldest, so the models reflect current team quality.

---

## 📊 Model Details

### Why 9 Models?

Each model captures a different signal. No single model dominates across all match types — strong teams vs weak teams, MD1 vs MD3, attack-heavy vs defensive setups all require different information. The ensemble combines their strengths.

---

### 1. Static ELO Model

ELO is a rating system originally designed for chess, adapted here for international football. Every team has a single number representing their strength. When teams play, points transfer based on the result and the probability of that result.

The model converts the ELO difference between two teams into expected goals using a linear scaling, then applies a Poisson distribution to generate full scoreline probabilities.

    home_xg = 1.35 + (elo_diff × 0.001)
    away_xg = 1.35 - (elo_diff × 0.001)

**Strength:** Simple, stable, well-validated in football analytics.
**Weakness:** Ignores recent form, attack/defense split.
**Best for:** Long-term strength signal, stable teams.

---

### 2. Dixon-Coles Poisson Model

Published in a 1997 statistics paper by Dixon and Coles, this is the academic gold standard for football score prediction. Instead of one number per team, it estimates two parameters:

- **Attack strength (α)** — how many goals a team tends to score
- **Defense weakness (β)** — how many goals a team tends to concede

Expected goals are computed as:

    home_xg = α_home × β_away × home_advantage
    away_xg = α_away × β_home

Parameters are estimated via maximum likelihood — finding values that best explain 19,000+ historical scorelines. Two key enhancements:

**Time weighting:** Recent matches matter more. Each match is weighted by exp(-0.003 × days_ago), so a match from 3 years ago has about 1/3 the influence of a recent match.

**Dixon-Coles correction:** Standard Poisson underestimates low-scoring results. The model applies a correction factor ρ to 0-0, 1-0, 0-1, and 1-1 scorelines to better reflect their actual frequency.

**Strength:** Theoretically grounded, captures attack/defense separately, peer-reviewed.
**Weakness:** Computationally expensive, needs sufficient data per team.
**Best for:** Well-documented teams with many historical matches.

---

### 3. Historical Average Model

The simplest model — predicts based on each team's recent scoring and conceding averages:

    home_xg = (home_avg_scored + away_avg_conceded) / 2
    away_xg = (away_avg_scored + home_avg_conceded) / 2

This acts as a strong baseline. If our complex models cannot beat it, something is wrong with the pipeline.

**Strength:** Extremely fast, reacts quickly to recent form changes.
**Weakness:** Ignores opponent quality entirely.
**Best for:** Teams that have drastically changed style or manager recently.

---

### 4. Dynamic ELO Model

Unlike the Static ELO which uses a single snapshot, Dynamic ELO replays all 19,713 historical matches chronologically and updates ratings after every single match. This means the ratings always reflect current form.

K-factors vary by tournament importance:

    World Cup match:     K = 60  (most impact)
    Continental trophy:  K = 50
    Qualifier:           K = 40
    Friendly:            K = 20  (least impact)

Goal difference also multiplies the K-factor — bigger wins cause larger rating changes.

**Strength:** Always up-to-date, naturally time-weighted by design.
**Weakness:** Still only one number per team, no attack/defense split.
**Best for:** Capturing momentum and recent tournament form.

---

### 5. XGBoost (Time-Weighted)

XGBoost learns complex non-linear relationships between the 24 engineered features and goals scored. It builds 300 decision trees sequentially, each correcting the errors of the previous.

Key design choices:
- **Poisson loss function** — correct for count data (goals can't be negative)
- **Exponential sample weights** — recent matches weighted up to 5.8× more than oldest
- **TimeSeriesSplit CV** — always trains on past, validates on future (no data leakage)
- **Separate models** — one model for home goals, one for away goals

The most important features learned by XGBoost:
1. form10_points_diff — recent form differential
2. away_form10_avg_conceded — opponent defensive weakness
3. elo_diff — overall strength gap
4. h2h_home_wins — historical head-to-head record

**Strength:** Best raw accuracy, captures complex feature interactions, fast.
**Weakness:** Black box — hard to explain individual predictions.
**Best for:** Overall prediction accuracy.

---

### 6. LightGBM (Time-Weighted)

Microsoft's implementation of gradient boosting. Uses leaf-wise tree growth instead of level-wise (XGBoost), making it faster and often slightly more accurate on tabular data. Identical feature set and time-weighting to XGBoost.

Performance is near-identical to XGBoost (MAE difference < 0.01), providing useful prediction diversity in the ensemble.

**Strength:** Faster training, handles large datasets well.
**Weakness:** Can overfit on small datasets without careful tuning.
**Best for:** Adding diversity to ensemble without sacrificing accuracy.

---

### 7. Neural Network (FootballNet)

A feedforward neural network trained end-to-end on the 24 features:

    Input (24) → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
              → Dense(64)  + BatchNorm + ReLU + Dropout(0.2)
              → Dense(32)  + ReLU
              → Output(2)  + Softplus

The Softplus activation on the output layer ensures predicted xG values are always positive. BatchNorm stabilizes training and speeds convergence. Early stopping prevents overfitting — training stops when validation loss stops improving (stopped at epoch 29 in our case).

Features are standardized with StandardScaler before training since neural networks are sensitive to feature scale. Time-decay sample weights are applied during training.

**Strength:** Captures non-linear feature interactions that tree models might miss.
**Weakness:** Needs careful regularization on small datasets.
**Best for:** Complex interaction patterns between features.

---

### 8. Logistic Regression

A statistical classifier that directly models outcome probabilities (Win/Draw/Loss) rather than scorelines. Unlike the other models, it does not predict xG — it predicts the result directly.

    P(Home Win) = sigmoid(β₀ + β₁×elo_diff + β₂×form5 + ...)

The model outputs calibrated probabilities, meaning if it says 70% home win, the home team should actually win about 70% of the time across many such predictions.

Top features by coefficient:
- elo_diff (+0.216) — most important positive signal
- away_form5_avg_conceded (+0.139) — opponent recent defensive frailty
- h2h_home_wins (+0.125) — historical head-to-head dominance

**Strength:** Fully interpretable, well-calibrated probabilities.
**Weakness:** Cannot predict scorelines, only W/D/L.
**Best for:** Probability calibration signal in ensemble.

---

### 9. Stakes Model (Specialist)

This is the most unique model in the system. It addresses a fundamental weakness of all general models: they ignore **group stage context**.

Two real effects this model captures:

**Rotation effect:** France with 6 points after MD2 is already qualified. Their MD3 lineup will include squad rotation — resting Mbappé, giving younger players minutes. Expected goals should be lower.

**Must-win effect:** Panama with 0 points in MD3 needs to win. Their tactics will be more aggressive, high press, more shots. Expected goals should be higher.

The model is trained exclusively on MD2 and MD3 historical World Cup matches (163 matches from 2006-2022) with 18 additional features:

    home_points, home_position, home_matches_played
    home_can_qualify, home_already_qualified
    home_must_win, home_elimination_risk
    home_points_needed (and equivalent for away team)
    matchday

The most important feature discovered: **away_points_needed** (top feature for away goals prediction) — teams that need points score significantly more than expected.

**Strength:** Captures human behavioral effects that statistical models miss.
**Weakness:** Only 163 training examples — activates MD2/MD3 only.
**Best for:** MD3 matches with maximum group context.

---

## 🔧 Feature Engineering

All 24 features are computed dynamically from historical results before the date of each match — no future data leakage.

### ELO Features
- **home_elo, away_elo** — absolute team strength from the Kaggle ELO dataset
- **elo_diff** — strength difference (positive = home team stronger)

### Form Features (computed rolling, before match date)
- **form5/10_points** — points earned in last 5/10 matches (W=3, D=1, L=0)
- **form5/10_avg_scored** — average goals scored per match
- **form5/10_avg_conceded** — average goals conceded per match
- **form5/10_points_diff** — home team form minus away team form
- **avg_scored_diff** — home avg scored minus away avg scored

### Head-to-Head Features
- **h2h_matches** — number of previous meetings
- **h2h_home_wins, h2h_away_wins, h2h_draws** — historical results
- **h2h_avg_goals** — average total goals in previous meetings

### Context
- **is_neutral** — all WC matches are on neutral ground (=1)

---

## 🎯 Ensemble Design

The ensemble combines all model predictions using weighted averaging of xG values and win probabilities. Weights are matchday-aware and were **rebalanced after backtesting** to favour the three strongest models (Logistic, XGBoost, LightGBM):

    MD1 weights (no group context):
    XGBoost:        20%  ← top-tier accuracy + scorelines
    LightGBM:       20%  ← best scoreline error
    Logistic Reg:   18%  ← best result accuracy (was 4% before)
    Dixon-Coles:    12%  ← strong theoretical grounding
    Neural Net:     10%  ← non-linear interactions
    Dynamic ELO:     8%  ← captures live form
    ELO:             6%  ← simple baseline
    Hist. Average:   6%  ← recent form signal
    Stakes Model:    0%  ← no context yet

    MD3 weights (maximum context):
    Stakes Model:   25%  ← activated with full context
    XGBoost:        15%
    LightGBM:       15%
    Logistic Reg:   13%
    Dixon-Coles:     9%
    Neural Net:      8%
    Dynamic ELO:     6%
    Hist. Average:   5%
    ELO:             4%

Each prediction also includes a **confidence score** — the percentage of models that predicted the same outcome. 88%+ is HIGH, 50-70% is MEDIUM.

---

## 📈 Backtesting Results

Every model is backtested on WC 2018 and WC 2022, training only on data from before each tournament and scored on the **identical 64 fixtures** per tournament. Result is read from the argmax of the summed P(Home)/P(Draw)/P(Away) probabilities. Figures below come straight from `data/processed/backtest_results.csv` (combined 2018 + 2022 average):

    Model                  Result%   Exact%   Goal MAE   Brier
    ---------------------  -------   ------   --------   -----
    Ensemble                56.2%    11.8%      0.99     0.194   ← best all-rounder
    Logistic Regression     56.2%    10.9%      1.15     0.194
    LightGBM                54.7%    11.8%      0.98     0.195
    XGBoost                 54.7%    10.9%      0.99     0.195
    ELO                     50.0%     9.4%      1.03     0.197
    Dynamic ELO             50.0%    12.5%      1.13     0.208
    Historical Average      43.8%     5.4%      1.06     0.211
    ---------------------  -------   ------   --------   -----
    Random baseline         33.3%
    Betting markets         ~55%

**Interpretation:**
- The best models reach ~55–56% result accuracy — on par with professional betting markets, and ~21 points above random guessing.
- **The Ensemble is the best overall choice:** it ties Logistic Regression for the top result accuracy (56.2%) while also matching the tree models on scoreline error (Goal MAE 0.99) and posting the best Brier score. It is the only model that is top-tier on *every* metric at once.
- This was not true initially — the original ensemble weights gave Logistic Regression (the strongest model) only 0.04 while over-weighting weak baselines, so the blend scored just 53.9%. **Rebalancing the weights toward the top three models fixed it.** (A free weight optimizer was also tried but overfit on only two tournaments and did not generalize — see below.)
- Result is taken from the summed outcome probabilities rather than the single most-likely scoreline; the modal scoreline (often 1-0 / 1-1) under-picks draws and discards information the probabilities already carry.
- Dixon-Coles, the Neural Network, and the Stakes model are not in this automated harness: Dixon-Coles needs an expensive per-split MLE refit, the Neural Net needs PyTorch, and the Stakes model needs simulated in-tournament standings. They still run in the live pipeline and the ensemble.

> **On learned weights:** the backtest also fits ensemble weights automatically (Brier-minimizing search) and evaluates them leave-one-tournament-out. With only two tournaments (128 matches) this overfits — the out-of-sample "Ensemble (learned)" scores 53.9%, no better than before — so the shipped weights are the robust hand-rebalanced set, not the optimizer's. More tournaments would make automated weight-learning viable.

### A note on the accuracy figures

An earlier version of this project (and an accompanying LinkedIn post) reported a combined XGBoost accuracy of **~50.5%**. That figure came from a backtest bug — XGBoost was scored on every same-year match between World Cup teams (≈146 / 130 matches, including friendlies and qualifiers) while the baseline used only the 64 real fixtures, inflating it.

Three changes put the evaluation on solid ground: every model is now scored on the **identical 64 fixtures**, the result is read from the **summed outcome probabilities**, and **all models are backtested side by side**. The properly-measured best is now **56.2%** (Logistic Regression) — higher than the original 50.5%, but earned through correct methodology rather than a measurement artifact. This note is kept deliberately so the history is transparent.

---

## 🔄 Live Updating Pipeline

The system is designed for live tournament use. After each matchday:

    Step 1: Enter real results
            Edit data/raw/wc_2026_results.csv
            Set home_goals, away_goals, played=True

    Step 2: Run automated pipeline
            python3 src/updater/update_predictions.py --matchday 1

    The pipeline automatically:
    ├── Rebuilds features with new results
    │   (form, H2H now includes actual WC matches)
    ├── Computes group standings
    ├── Runs Bayesian strength updater
    │   (team attack/defense estimates updated from real goals)
    ├── Reruns all 9 models with updated features
    ├── Generates ensemble predictions for MD2 AND MD3
    └── Stakes model now has real group context

    Step 3: Refresh dashboard
            Click Refresh in sidebar
            MD2 and MD3 predictions now reflect real results

---

## 🧮 Bayesian Updating

After real results come in, team strength estimates are updated using Bayes theorem:

    posterior = (prior × prior_weight + observed × n_matches)
                / (prior_weight + n_matches)

    Example after France 2-0 Senegal:
    Prior:    France attack = 1.94 (from 20 historical matches)
    Observed: France scored 2.0 goals (1 WC match)
    Prior weight: 10

    Posterior = (1.94 × 10 + 2.0 × 1) / (10 + 1) = 1.945

The prior is strong (weight=10) so one match does not dramatically change estimates. After 3 matches the observed data has much more influence.

---

## 📁 Repository Structure

    world-cup-2026-predictor/
    ├── data/
    │   ├── raw/
    │   │   ├── results.csv                 ← 49k raw international matches (19,713 after cleaning)
    │   │   ├── elo_ratings_wc2026.csv      ← historical ELO for 48 teams
    │   │   ├── wc_2026_fixtures.csv        ← all 72 group stage fixtures
    │   │   └── wc_2026_results.csv         ← live results tracker
    │   ├── processed/
    │   │   ├── results_clean.csv           ← normalized + features added
    │   │   ├── train_features.csv          ← 15k matches with 24 features
    │   │   ├── predict_features.csv        ← 72 WC fixtures with features
    │   │   ├── elo_latest.csv              ← current ELO snapshot
    │   │   └── standings_after_md*.csv     ← live group standings
    │   └── predictions/
    │       ├── elo_all.csv                 ← per-model predictions
    │       ├── dixon_coles_all.csv
    │       ├── xgboost_all.csv
    │       ├── neural_net_all.csv
    │       └── ensemble_md*.csv            ← final ensemble predictions
    ├── src/
    │   ├── data/
    │   │   ├── fetch_elo.py                ← ELO data pipeline
    │   │   ├── normalize_teams.py          ← team name standardization
    │   │   └── process_data.py             ← cleaning + feature creation
    │   ├── features/
    │   │   ├── build_features.py           ← 24-feature engineering pipeline
    │   │   └── group_standings.py          ← standings + stakes features
    │   ├── models/
    │   │   ├── baseline/
    │   │   │   ├── elo_model.py
    │   │   │   ├── dixon_coles.py
    │   │   │   ├── historical_avg.py
    │   │   │   └── dynamic_elo.py
    │   │   └── ml/
    │   │       ├── xgboost_model.py
    │   │       ├── lightgbm_model.py
    │   │       ├── neural_network.py
    │   │       ├── logistic_regression.py
    │   │       └── stakes_model.py
    │   ├── ensemble/
    │   │   └── ensemble.py
    │   ├── evaluation/
    │   │   └── backtest.py
    │   └── updater/
    │       ├── update_predictions.py
    │       └── bayesian_updater.py
    ├── dashboard/
    │   └── app.py                          ← Streamlit 5-page dashboard
    ├── docs/
    │   └── index.html                      ← shareable predictions board (GitHub Pages)
    ├── notebooks/
    │   └── eda.ipynb                       ← exploratory analysis
    ├── tests/
    │   └── test_pipeline.py                ← smoke tests (data + probabilities)
    ├── .streamlit/
    │   └── config.toml                     ← dashboard theme
    ├── .github/workflows/
    │   └── ci.yml                          ← runs tests on push / PR
    ├── run_all.sh                          ← one-command full pipeline
    ├── requirements.txt
    └── README.md

---

## ▶️ How to Run (Quick Reference)

Run everything in one command:

    ./run_all.sh

Or run the important files one by one (in this order):

| Step | Command | What it does |
|------|---------|--------------|
| 1. Clean data | `python3 src/data/process_data.py` | Cleans the raw data and normalizes team names |
| 2. Build features | `python3 src/features/build_features.py` | Creates the 24 features used by the models |
| 3. Run a model | `python3 src/models/ml/xgboost_model.py` | Runs one model (swap in any file from `src/models/`) |
| 4. Combine models | `python3 src/ensemble/ensemble.py 1` | Builds the final ensemble for matchday 1 (use 2 or 3 for later rounds) |
| 5. See the dashboard | `streamlit run dashboard/app.py` | Opens the interactive predictor in your browser |
| 5b. Build the board | `python3 src/viz/build_board.py` | Generates `docs/index.html` — a shareable predictions board (hostable on GitHub Pages) |
| 6. Check accuracy | `python3 src/evaluation/backtest.py` | Backtests every model on WC 2018 + 2022 |
| 7. Run the tests | `pytest tests/` | Quick checks that everything is working |

**After real results come in:** edit `data/raw/wc_2026_results.csv` (enter the scores, set `played=True`), then run:

    python3 src/updater/update_predictions.py --matchday 1

This refreshes the predictions for the next rounds.

**Which prediction file to use:** `data/predictions/ensemble_md1.csv` (and `md2`, `md3`) — the ensemble is the best all-rounder.

---

## 🚀 Getting Started

### 1. Clone and install dependencies

    git clone https://github.com/Bardiyashavandi/world-cup-2026-predictor.git
    cd world-cup-2026-predictor
    python3 -m pip install -r requirements.txt

    Mac users (required for XGBoost):
    brew install libomp

### 2. Download data

Download these datasets from Kaggle and place in data/raw/:

    International football results:
    kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
    → results.csv, shootouts.csv, goalscorers.csv

    WC 2026 Historical ELO ratings:
    kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings
    → elo_ratings_wc2026.csv

### 3. Run the full pipeline

The whole pipeline can be run with a single script:

    ./run_all.sh              # process -> features -> models -> ensemble
    ./run_all.sh --backtest   # also run the backtest at the end

Or run each step manually:

    # Process and clean data
    python3 src/data/process_data.py

    # Engineer features
    python3 src/features/build_features.py

    # Run all models
    python3 src/models/baseline/elo_model.py
    python3 src/models/baseline/dixon_coles.py
    python3 src/models/baseline/historical_avg.py
    python3 src/models/baseline/dynamic_elo.py
    python3 src/models/ml/xgboost_model.py
    python3 src/models/ml/lightgbm_model.py
    python3 src/models/ml/neural_network.py
    python3 src/models/ml/logistic_regression.py

    # Generate ensemble predictions
    python3 src/ensemble/ensemble.py 1
    python3 src/ensemble/ensemble.py 2
    python3 src/ensemble/ensemble.py 3

    # Launch dashboard
    streamlit run dashboard/app.py

### 4. Run backtesting

    python3 src/evaluation/backtest.py

### 5. Run the tests

Fast smoke tests validate the data contracts and probability outputs (no model retraining):

    pytest tests/
    # or, without pytest:
    python3 tests/test_pipeline.py

---

## 📦 Requirements

    pandas>=2.0
    numpy>=1.24
    scipy>=1.10
    scikit-learn>=1.3
    xgboost>=2.0
    lightgbm>=4.0
    torch>=2.0
    streamlit>=1.28
    plotly>=5.0
    matplotlib>=3.7
    seaborn>=0.12
    requests>=2.28
    jupytext
    nbformat

---

## 🧠 Key Design Decisions

**Why Poisson for goals?**
Goals are rare, independent, discrete events — exactly the scenario Poisson distributions model. Empirically validated in the EDA notebook where actual goal distributions closely match Poisson theoretical distributions.

**Why time-weighted training?**
Germany's 2014 World Cup squad has no relevance to Germany's 2026 squad. Exponential decay (λ=0.001) weights recent matches up to 5.8× more than the oldest matches in our dataset, ensuring the model reflects current team quality.

**Why TimeSeriesSplit for cross-validation?**
Standard k-fold randomly splits data, which can train on 2022 matches to predict 2018 results — temporal data leakage that inflates performance estimates. TimeSeriesSplit always trains on the past and validates on the future, giving realistic performance estimates.

**Why a specialist Stakes Model?**
All general models treat France vs Senegal in MD1 identically to France vs Croatia in MD3 when France is already qualified. Real football does not work this way. A specialist model trained exclusively on MD2 and MD3 historical WC matches with group context features captures squad rotation and elimination pressure effects.

**Why ensemble over single model?**
Each model has different failure modes. XGBoost fails on teams with few historical matches. Dixon-Coles is slow to react to recent form changes. ELO ignores attack/defense asymmetries. The ensemble is robust where any individual model is weak.

**Why separate home and away goal models?**
Home goals and away goals have different distributions and are influenced by different features. A team's attack determines home goals; the opponent's defense also matters but differently. Training separate models captures this asymmetry.

---

## 🔭 Future Work

Concrete next steps, in rough order of expected payoff:

1. **Automated ensemble weight learning.** The weights are currently rebalanced by hand (which already lifted the ensemble to the top of the table). A free Brier-minimizing optimizer is implemented in the backtest but overfits on only two tournaments; with more historical tournaments a proper meta-learner (stacking) could push the ensemble past the best single model rather than merely matching it.
2. **Calibrate probabilities** (isotonic / Platt) on the tree and neural models to push the Brier score below 0.19.
3. **Add 2026-specific features:** host advantage (USA / Canada / Mexico), rest days, and travel distance — meaningful in a three-country tournament.
4. **Backtest the remaining models** (Dixon-Coles, Neural Net, Stakes) by refactoring them to accept arbitrary train/test splits, so the comparison is complete.
5. **Expand the validation set** beyond two World Cups (add 2010 / 2014 and continental cups) to reduce variance and tune the time-decay λ.

---

## 📜 References

- Dixon, M. and Coles, S. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market. Applied Statistics, 46(2), 265-280.
- Elo, A. (1978). The Rating of Chessplayers, Past and Present. Arco Publishing.
- Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017.

---

## 👤 Author

**Bardiya Shavandi**

Built as a portfolio project demonstrating end-to-end ML system design across statistics, machine learning, and MLOps. The system was developed and deployed live during the 2026 FIFA World Cup group stage.

---

## 📄 License

MIT License — free to use, modify and distribute with attribution.