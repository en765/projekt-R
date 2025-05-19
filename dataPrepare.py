# dataPrepare.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def pripremi_podatke():
    # Učitavanje i čišćenje
    df = pd.read_csv("output/spojeni_podaci.csv")
    drop_columns = [
        "Unnamed: 0", "SE11_10-2024.xlsx", "Date",
        "Global Irradiation (kWh/㎡)", "Theoretical Yield (kWh)",
        "Loss Due to Export Limitation (kWh)", "Loss Due to Export Limitation(€)",
        "Performance Ratio(%)", "Charge (kWh)", "Discharge (kWh)"
    ]
    df = df.drop(columns=drop_columns, errors='ignore')
    df.columns = df.columns.str.strip()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=["temperatura", "naoblaka", "insolacija", "PV Yield (kWh)"])
    df = df.fillna(df.median(numeric_only=True))

    # Korelacija i odabir značajki
    correlation_with_target = df.corr()["PV Yield (kWh)"].sort_values(ascending=False)
    top_features = correlation_with_target[abs(correlation_with_target) > 0.3].index.tolist()
    df_selected = df[top_features]

    # Uklanjanje outliera (IQR)
    Q1 = df_selected.quantile(0.25)
    Q3 = df_selected.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_selected < (Q1 - 1.5 * IQR)) | (df_selected > (Q3 + 1.5 * IQR)))
    df_no_outliers = df_selected[~outliers.any(axis=1)]

    # Skaliranje (standardizacija)
    X = df_no_outliers.drop(columns=["PV Yield (kWh)"])
    y = df_no_outliers["PV Yield (kWh)"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y