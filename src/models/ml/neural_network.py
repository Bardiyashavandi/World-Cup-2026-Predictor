import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import poisson
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append("src")


# ─────────────────────────────────────────
# 1. FEATURE COLUMNS
# ─────────────────────────────────────────

FEATURE_COLS = [
    "is_neutral",
    "home_elo", "away_elo", "elo_diff",
    "home_form5_points", "home_form5_avg_scored", "home_form5_avg_conceded",
    "away_form5_points", "away_form5_avg_scored", "away_form5_avg_conceded",
    "home_form10_points", "home_form10_avg_scored", "home_form10_avg_conceded",
    "away_form10_points", "away_form10_avg_scored", "away_form10_avg_conceded",
    "form5_points_diff", "form10_points_diff", "avg_scored_diff",
    "h2h_matches", "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    "h2h_avg_goals",
]

REFERENCE_DATE = pd.Timestamp("2026-06-11")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────
# 2. NETWORK ARCHITECTURE
# ─────────────────────────────────────────

class FootballNet(nn.Module):
    """
    Feedforward neural network for scoreline prediction.

    Input:  24 match features
    Output: 2 values (home_xg, away_xg)

    Architecture:
        24 → 128 → 64 → 32 → 2

    Key design choices:
    - BatchNorm: stabilizes training, speeds convergence
    - Dropout: prevents overfitting on small dataset
    - Softplus output: ensures xG values are always positive
    """

    def __init__(self, input_dim: int = 24):
        super(FootballNet, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output — 2 values: home_xg, away_xg
            nn.Linear(32, 2),
            nn.Softplus()  # ensures output always positive
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────
# 3. LOAD DATA
# ─────────────────────────────────────────

def load_data():
    train = pd.read_csv("data/processed/train_features.csv")
    predict = pd.read_csv("data/processed/predict_features.csv")

    train = train.dropna(subset=["home_goals", "away_goals"])
    train[FEATURE_COLS] = train[FEATURE_COLS].fillna(0)
    predict[FEATURE_COLS] = predict[FEATURE_COLS].fillna(0)

    return train, predict


# ─────────────────────────────────────────
# 4. TIME WEIGHTS
# ─────────────────────────────────────────

def compute_sample_weights(train: pd.DataFrame) -> np.ndarray:
    """Exponential time decay weights — recent matches matter more."""
    train = train.copy()
    train["date"] = pd.to_datetime(train["date"])
    weights = train["date"].apply(
        lambda d: np.exp(-0.001 * (REFERENCE_DATE - d).days)
    )
    weights = weights / weights.sum() * len(train)
    return weights.values.astype(np.float32)


# ─────────────────────────────────────────
# 5. TRAIN MODEL
# ─────────────────────────────────────────

def train_neural_network(train: pd.DataFrame,
                          epochs: int = 100,
                          batch_size: int = 256,
                          lr: float = 0.001) -> tuple:
    """
    Train FootballNet with:
    - Poisson loss (correct for count data)
    - Time-based sample weights
    - Early stopping to prevent overfitting
    - Learning rate scheduling
    """
    print(f"  Device: {DEVICE}")

    X = train[FEATURE_COLS].values.astype(np.float32)
    y = train[["home_goals", "away_goals"]].values.astype(np.float32)
    weights = compute_sample_weights(train)

    # Scale features — critical for neural networks
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Sample weight range: "
          f"{weights.min():.3f} → {weights.max():.3f}")
    print()

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
    y_tensor = torch.FloatTensor(y).to(DEVICE)
    w_tensor = torch.FloatTensor(weights).to(DEVICE)

    # Train/val split — last 20% for validation
    split = int(len(X_tensor) * 0.8)
    X_train = X_tensor[:split]
    y_train = y_tensor[:split]
    w_train = w_tensor[:split]
    X_val = X_tensor[split:]
    y_val = y_tensor[split:]

    dataset = TensorDataset(X_train, y_train, w_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, optimizer, scheduler
    model = FootballNet(input_dim=len(FEATURE_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5
    )

    # Poisson loss function
    def poisson_loss(pred, target, weight):
        """
        Poisson negative log likelihood loss.
        pred = lambda (expected goals)
        target = actual goals
        loss = lambda - target * log(lambda)
        """
        loss = pred - target * torch.log(pred + 1e-8)
        return (loss * weight.unsqueeze(1)).mean()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15
    best_state = None

    print(f"  Training for up to {epochs} epochs "
          f"(early stopping patience={patience})...")
    print()
    print(f"  {'Epoch':>6} | {'Train Loss':>10} | "
          f"{'Val Loss':>10} | {'LR':>8}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch, w_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = poisson_loss(pred, y_batch, w_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = (val_pred - y_val *
                        torch.log(val_pred + 1e-8)).mean().item()

        scheduler.step(val_loss)

        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            lr_current = optimizer.param_groups[0]["lr"]
            marker = " ✓ best" if val_loss < best_val_loss else ""
            print(f"  {epoch+1:>6} | {train_loss:>10.4f} | "
                  f"{val_loss:>10.4f} | {lr_current:>8.6f}{marker}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in
                          model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    print(f"\n  Best validation loss: {best_val_loss:.4f}")

    # Evaluate MAE on validation set
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).cpu().numpy()
        val_true = y_val.cpu().numpy()
        home_mae = np.mean(np.abs(val_pred[:, 0] - val_true[:, 0]))
        away_mae = np.mean(np.abs(val_pred[:, 1] - val_true[:, 1]))

    print(f"  Home goals MAE: {home_mae:.3f}")
    print(f"  Away goals MAE: {away_mae:.3f}")

    return model, scaler


# ─────────────────────────────────────────
# 6. PREDICT MATCHES
# ─────────────────────────────────────────

def predict_scorelines(model, scaler,
                        predict_df: pd.DataFrame) -> pd.DataFrame:
    """Generate scoreline predictions for all WC 2026 fixtures."""
    X = predict_df[FEATURE_COLS].fillna(0).values.astype(np.float32)
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    home_xg = np.clip(predictions[:, 0], 0.3, 4.0)
    away_xg = np.clip(predictions[:, 1], 0.3, 4.0)

    results = []
    for i, (_, row) in enumerate(predict_df.iterrows()):
        hxg = home_xg[i]
        axg = away_xg[i]

        scorelines = []
        for h in range(7):
            for a in range(7):
                p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
                scorelines.append({"home_goals": h,
                                   "away_goals": a,
                                   "probability": p})

        score_df = pd.DataFrame(scorelines).sort_values(
            "probability", ascending=False
        )

        home_win = score_df[
            score_df["home_goals"] > score_df["away_goals"]
        ]["probability"].sum()
        draw = score_df[
            score_df["home_goals"] == score_df["away_goals"]
        ]["probability"].sum()
        away_win = score_df[
            score_df["home_goals"] < score_df["away_goals"]
        ]["probability"].sum()

        top = score_df.iloc[0]

        results.append({
            "match_id": row.get("match_id", i),
            "group": row.get("group", ""),
            "matchday": row.get("matchday", 1),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_xg": round(float(hxg), 3),
            "away_xg": round(float(axg), 3),
            "predicted_home_goals": int(top["home_goals"]),
            "predicted_away_goals": int(top["away_goals"]),
            "home_win_prob": round(home_win, 3),
            "draw_prob": round(draw, 3),
            "away_win_prob": round(away_win, 3),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────
# 7. RUN FULL PIPELINE
# ─────────────────────────────────────────

def run_neural_network():
    print("=" * 60)
    print("  NEURAL NETWORK MODEL — WC 2026 Predictor")
    print("=" * 60)
    print()

    print("Loading data...")
    train, predict = load_data()
    print(f"  Training set: {len(train)} matches")
    print(f"  Prediction set: {len(predict)} fixtures")
    print()

    print("Training FootballNet...")
    model, scaler = train_neural_network(train)

    print("\nGenerating WC 2026 predictions...")
    out_df = predict_scorelines(model, scaler, predict)

    os.makedirs("data/predictions", exist_ok=True)
    out_df.to_csv("data/predictions/neural_net_all.csv", index=False)
    out_df[out_df["matchday"] == 1].to_csv(
        "data/predictions/neural_net_md1.csv", index=False
    )

    print()
    print("Neural Network Predictions — WC 2026 Group Stage")
    print("=" * 60)
    for _, row in out_df.iterrows():
        print(
            f"Group {row['group']} MD{row['matchday']} | "
            f"{row['home_team']:20} {row['predicted_home_goals']}-"
            f"{row['predicted_away_goals']} {row['away_team']:20} | "
            f"xG: {row['home_xg']:.2f}-{row['away_xg']:.2f} | "
            f"H:{row['home_win_prob']} "
            f"D:{row['draw_prob']} "
            f"A:{row['away_win_prob']}"
        )

    print()
    print("Saved to data/predictions/neural_net_all.csv")
    print("Saved to data/predictions/neural_net_md1.csv")


if __name__ == "__main__":
    run_neural_network()