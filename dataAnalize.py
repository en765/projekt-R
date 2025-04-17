import pandas as pd

# Učitavanje podataka iz CSV datoteke
data = pd.read_csv("/Users/enesek/Downloads/zavrsni/projekt-R/solar_podaci.csv")

# Prikaz prvih nekoliko redova podataka
print(data.head())

# Informacije o podacima (tipovi podataka, broj ne-praznih vrijednosti)
print(data.info())