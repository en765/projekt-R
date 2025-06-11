import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ==== 1. UČITAVANJE I PRIPREMA PODATAKA ====

df = pd.read_csv("output/spojeni_podaci.csv")
df = df.dropna(subset=["PV Yield (kWh)"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Dodavanje vremenskih značajki
df["dan_u_tjednu"] = df["Date"].dt.dayofweek
df["mjesec"] = df["Date"].dt.month
df["dan_u_godini"] = df["Date"].dt.dayofyear
df["temp_naoblaka_ratio"] = df["temperatura"] / (df["naoblaka"] + 1e-5)

# Odabrane značajke i cilj
features = [
    "insolacija", "temperatura", "Peak Power (kW)",
    "Specific Energy (kWh/kWp)", "dan_u_tjednu", "mjesec",
    "dan_u_godini", "temp_naoblaka_ratio"
]
target = "PV Yield (kWh)"
df = df[features + [target]].dropna()

# ==== 2. SKALIRANJE I SEKVENCIJE ====

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[[target]]).flatten()

def create_sequences(X, y, seq_length=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled)

# Train/test split
split_idx = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# Tenzori
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# ==== 3. BOLJI LSTM MODEL ====

class BetterLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=False):
        super(BetterLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size * direction_factor),
            nn.Linear(hidden_size * direction_factor, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # samo zadnji korak
        out = self.fc(out)
        return out

model = BetterLSTMModel(input_size=X_train.shape[2])

# ==== 4. TRENING ====

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 150

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_dl:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(train_dl):.4f}")

# ==== 5. EVALUACIJA ====

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()
    y_true_scaled = y_test_tensor.numpy()

# Inverzija skaliranja
y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)
y_true_inv = scaler_y.inverse_transform(y_true_scaled)

# Metrike
mae = mean_absolute_error(y_true_inv, y_pred_inv)
mse = mean_squared_error(y_true_inv, y_pred_inv)
r2 = r2_score(y_true_inv, y_pred_inv)
corr, _ = pearsonr(y_true_inv.flatten(), y_pred_inv.flatten())

print("\n📊 Evaluacija poboljšanog LSTM modela:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
print(f"Pearsonova korelacija: {corr:.2f}")

# ==== 6. GRAF ====

plt.figure(figsize=(8,6))
plt.scatter(y_true_inv, y_pred_inv, color='orange', alpha=0.6, label='Predikcija vs Stvarna')
lims = [min(y_true_inv.min(), y_pred_inv.min()), max(y_true_inv.max(), y_pred_inv.max())]
plt.plot(lims, lims, 'r--', label='Savršena predikcija')
plt.xlabel("Stvarne vrijednosti (PV Yield)")
plt.ylabel("Predviđene vrijednosti (PV Yield)")
plt.title("LSTM: Stvarne vs Predviđene vrijednosti")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafovi/lstm_bolji_predikcija.png")