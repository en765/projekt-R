import pandas as pd
import os
from glob import glob
import sys
print("Python path:", sys.executable)

# Funkcija za učitavanje svih Excel datoteka iz direktorija
def ucitaj_podatke(putanja_foldera):
    excel_fajlovi = glob(os.path.join("/Users/enesek/Downloads/zavrsni/projekt-R/SE11", "*.xlsx"))  # Pronalazi sve .xlsx fajlove
    svi_podaci = []  # Lista za spremanje DataFrame-ova
    
    for fajl in excel_fajlovi:
        try:
            df = pd.read_excel(fajl)  # Učitavanje Excel fajla
            df['Izvor_datoteke'] = os.path.basename(fajl)  # Dodavanje info o izvoru
            svi_podaci.append(df)
        except Exception as e:
            print(f"Greška pri učitavanju {fajl}: {e}")
    
    if svi_podaci:
        kombinirani_podaci = pd.concat(svi_podaci, ignore_index=True)  # Spajanje svih podataka
        return kombinirani_podaci
    else:
        print("Nema dostupnih podataka za učitavanje.")
        return None

# Primjer korištenja funkcije
putanja_do_podataka = "/mnt/data/"  # Prilagodite ovu putanju prema vašim datotekama
svi_podaci = ucitaj_podatke(putanja_do_podataka)

# Pohrana podataka u CSV/Parquet
if svi_podaci is not None:
    svi_podaci.to_csv("output/solar_podaci.csv", index=False)
    print("Podaci su uspješno spremljeni u CSV format.")