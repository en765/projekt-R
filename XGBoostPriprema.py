import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Učitavanje podataka
df = pd.read_csv("output/spojeni_podaci.csv")

# Čišćenje nepotrebnih stupaca
drop_columns = [
    "Unnamed: 0", "SE11_10-2024.xlsx", "Date",
    "Global Irradiation (kWh/㎡)", "Theoretical Yield (kWh)",
    "Loss Due to Export Limitation (kWh)", "Loss Due to Export Limitation(€)",
    "Performance Ratio(%)", "Charge (kWh)", "Discharge (kWh)"
]
df = df.drop(columns=drop_columns, errors='ignore')
df.columns = df.columns.str.strip()

# Pretvori sve stupce u numeričke vrijednosti
df = df.apply(pd.to_numeric, errors='coerce')

# Popuni preostale NaN vrijednosti medijanom
df = df.fillna(df.median(numeric_only=True))

# Korelacija s ciljanom varijablom
correlation_with_target = df.corr()["PV Yield (kWh)"].sort_values(ascending=False)
top_features = correlation_with_target[abs(correlation_with_target) > 0.3].index.tolist()

# Prikaz toplinske mape najrelevantnijih značajki
plt.figure(figsize=(12, 10))
sns.heatmap(df[top_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Korelacija najrelevantnijih značajki s PV Yield (kWh)")
plt.tight_layout()
plt.savefig("grafovi/korelacija_top_znacajke.png")
plt.close()

# Standardizacija značajki
scaler = StandardScaler()
X = df[top_features].drop(columns=["PV Yield (kWh)"])
X_scaled = scaler.fit_transform(X)
y = df["PV Yield (kWh)"]

print("Podaci su očišćeni, standardizirani i spremni za treniranje XGBoost modela.")
