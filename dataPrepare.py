import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Učitavanje podataka (zamijeni putanju svojim .csv/.xlsx fajlom ako je potrebno)
df = pd.read_csv("output/spojeni_podaci.csv")  # zamijeni sa stvarnim nazivom datoteke

# Prikaz nedostajućih vrijednosti po stupcima i uklanjanje redaka s nedostajućim ključnim vrijednostima
print("\nPregled nedostajućih vrijednosti po stupcima:")
print(df[["temperatura", "naoblaka", "insolacija", "PV Yield (kWh)"]].isnull().sum())
df = df.dropna(subset=["temperatura", "naoblaka", "insolacija", "PV Yield (kWh)"])
print(f"\nBroj preostalih redaka nakon čišćenja: {len(df)}")


# Pretvaranje datuma
df["Date"] = pd.to_datetime(df["Date"])

# Odabir značajki za analizu i model
features = ["temperatura", "naoblaka", "insolacija", "PV Yield (kWh)"]
df_selected = df[features]

# Normalizacija (Min-Max Scaling)
minmax_scaler = MinMaxScaler()
df_norm = pd.DataFrame(minmax_scaler.fit_transform(df_selected), columns=[f"{col}_norm" for col in features])

# Standardizacija (Z-score)
standard_scaler = StandardScaler()
df_std = pd.DataFrame(standard_scaler.fit_transform(df_selected), columns=[f"{col}_std" for col in features])

# Korelacijska matrica
correlation_matrix = df_selected.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Korelacija između vremenskih uvjeta i PV proizvodnje")
plt.show()

# Detekcija outliera (IQR metoda)
Q1 = df_selected.quantile(0.25)
Q3 = df_selected.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df_selected < (Q1 - 1.5 * IQR)) | (df_selected > (Q3 + 1.5 * IQR)))
print("Broj outliera po stupcu:\n", outliers.sum())

# Uklanjanje outliera (opcionalno)
df_no_outliers = df_selected[~outliers.any(axis=1)]

# Spajanje svih priprema u jedan DataFrame (za trening modela)
df_final = pd.concat([df, df_norm, df_std], axis=1)

# Spremno za XGBoost trening
X = df_no_outliers[["temperatura", "naoblaka", "insolacija"]]
y = df_no_outliers["PV Yield (kWh)"]