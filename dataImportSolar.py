import pandas as pd
import os
from glob import glob
import sys

print("Python path:", sys.executable)

# Funkcija za učitavanje svih Excel datoteka iz direktorija
# Funkcija za učitavanje svih Excel datoteka iz direktorija - samo solarni podaci
def ucitaj_solarne_podatke(putanja_foldera):
    excel_fajlovi = glob(os.path.join(putanja_foldera, "*.xlsx"))
    svi_podaci = []

    for fajl in excel_fajlovi:
        try:
            df = pd.read_excel(fajl)  # Učitavanje Excel fajla     
            df['Izvor_datoteke'] = os.path.basename(fajl)  # Dodavanje info o izvoru
            kolone_za_ocistiti = [
                "Global Irradiation (kWh/㎡)",
                "Theoretical Yield (kWh)",
                "Loss Due to Export Limitation (kWh)",
                "Loss Due to Export Limitation(€)",
                "Performance Ratio(%)",
                "Charge (kWh)",
                "Discharge (kWh)"
            ]
            for kolona in kolone_za_ocistiti:
                if kolona in df.columns:
                    df[kolona] = pd.NA
            svi_podaci.append(df)
        except Exception as e:
            print(f"Greška pri učitavanju {fajl}: {e}")

    if svi_podaci:
        return pd.concat(svi_podaci, ignore_index=True)
    else:
        print("Nema dostupnih podataka za učitavanje.")
        return None

# Definiranje putanja
putanja_solarni = "/Users/enesek/Downloads/zavrsni/projekt-R/SE11"

# Učitavanje i spremanje solarnih podataka
solar_podaci = ucitaj_solarne_podatke(putanja_solarni)
# Spremanje u CSV
if solar_podaci is not None:
    solar_podaci.to_csv("output/solar_podaci.csv", index=False)
    print("Solarni podaci su uspješno spremljeni.")

