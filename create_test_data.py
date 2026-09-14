"""Erstellt eine Beispiel-Eingabedatei zum Testen."""
import pandas as pd

data = [
    ("MA001", "2024-06-03", 12.5),
    ("MA001", "2024-06-04", 6.0),
    ("MA001", "2024-06-05", 8.0),
    ("MA001", "2024-06-06", 13.0),
    ("MA001", "2024-06-07", 4.0),
    ("MA002", "2024-06-03", 9.5),
    ("MA002", "2024-06-04", 11.0),
    ("MA002", "2024-06-05", 10.0),
    ("MA002", "2024-06-06", 7.0),
    ("MA003", "2024-06-03", 15.0),
    ("MA003", "2024-06-04", 3.0),
]

df = pd.DataFrame(data, columns=["MitarbeiterID", "Datum", "Arbeitszeit_Stunden"])
df.to_excel("testdaten.xlsx", index=False)
print("testdaten.xlsx erstellt.")
