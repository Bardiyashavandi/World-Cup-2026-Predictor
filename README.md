---

## 🧠 Key Design Decisions

**Why Poisson for goals?**
Goals are rare independent events — exactly what Poisson models. Validated empirically in the EDA notebook.

**Why time-weighted training?**
Germany 2014 is irrelevant to Germany 2026. Exponential decay (λ=0.001) weights recent matches up to 5.8× more than oldest.

**Why TimeSeriesSplit for CV?**
Standard k-fold would train on future data to predict the past — data leakage. TimeSeriesSplit always trains on past, validates on future.

**Why a specialist Stakes Model?**
General models ignore group context. France with 6 points in MD3 rotates their squad. Panama needing a win plays completely differently. A specialist model trained only on MD2/MD3 historical WC data captures this.

**Why ensemble over single model?**
Each model captures something different — ELO captures strength, Dixon-Coles captures attack/defense, XGBoost captures non-linear feature interactions. The ensemble is more robust than any individual model.

---

## 📜 References

- Dixon, M. & Coles, S. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*
- Elo, A. (1978). *The Rating of Chessplayers, Past and Present*
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*

---

## 👤 Author

**Bardiya Shavandi**
Built as a portfolio project demonstrating end-to-end ML system design across statistics, machine learning, and MLOps.