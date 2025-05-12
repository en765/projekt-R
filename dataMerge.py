import pandas as pd

# Učitaj vremenske podatke
vremenski_df = pd.read_csv("output/vremenski_podaci.csv")

# Učitaj solarne podatke, preskoči prvi red koji sadrži nazive kolona
solarni_df = pd.read_csv("output/solar_podaci.csv", skiprows=1)

# Preimenuj kolonu "Statistical Period" u "Date" da se može spojiti po njoj
solarni_df.rename(columns={"Statistical Period": "Date"}, inplace=True)

# Provjeri format datuma i konvertiraj obje kolone u datetime
vremenski_df["Date"] = pd.to_datetime(vremenski_df["Date"])
solarni_df["Date"] = pd.to_datetime(solarni_df["Date"], errors='coerce')

# Spoji podatke po datumu (unutarnji join - samo datumi koji postoje u obje tablice)
spojeni_df = pd.merge(vremenski_df, solarni_df, on="Date", how="inner")

# Spremi rezultat u novi CSV
spojeni_df.to_csv("output/spojeni_podaci.csv", index=False)

print("Spojeni podaci uspješno spremljeni u 'spojeni_podaci.csv'")