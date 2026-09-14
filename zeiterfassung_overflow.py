"""
Zeiterfassung Overflow-Verteilung
==================================
Liest eine Excel-Tabelle mit Spalten: MitarbeiterID, Datum, Arbeitszeit_Stunden
- Begrenzt Arbeitszeiten auf max. 10h/Tag
- Verteilt Überstunden auf unterlastete Tage im Abrechnungszeitraum
- Erstellt neue Schicht-Einträge für verbleibende Overflow-Stunden
- Speichert das Ergebnis als neue Excel-Datei
"""

import sys
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import date
import argparse

MAX_HOURS = 10.0


def read_input(filepath: str) -> pd.DataFrame:
    """Liest die Excel-Datei und normalisiert die Spalten."""
    try:
        df = pd.read_excel(filepath, dtype={"MitarbeiterID": str})
    except FileNotFoundError:
        sys.exit(f"Fehler: Datei '{filepath}' nicht gefunden.")
    except Exception as e:
        sys.exit(f"Fehler beim Lesen der Datei: {e}")

    # Flexibles Spalten-Mapping (case-insensitive, Leerzeichen ignorieren)
    col_map = {}
    for col in df.columns:
        normalized = col.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in ("mitarbeiterid", "mitarbeiter_id", "ma_id", "id"):
            col_map[col] = "MitarbeiterID"
        elif normalized in ("datum", "date", "tag"):
            col_map[col] = "Datum"
        elif normalized in (
            "arbeitszeit_stunden", "stunden", "hours", "arbeitszeit",
            "arbeitszeit_h", "arbeitsstunden",
        ):
            col_map[col] = "Arbeitszeit_Stunden"

    df = df.rename(columns=col_map)

    required = {"MitarbeiterID", "Datum", "Arbeitszeit_Stunden"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"Fehler: Folgende Spalten fehlen oder konnten nicht erkannt werden: {missing}\n"
            f"Gefundene Spalten: {list(df.columns)}"
        )

    # Datum parsen
    df["Datum"] = pd.to_datetime(df["Datum"], dayfirst=True, errors="coerce")
    invalid_dates = df["Datum"].isna()
    if invalid_dates.any():
        print(
            f"Warnung: {invalid_dates.sum()} Zeilen mit ungültigem Datum werden übersprungen."
        )
        df = df[~invalid_dates].copy()

    # Stunden zu float
    df["Arbeitszeit_Stunden"] = pd.to_numeric(
        df["Arbeitszeit_Stunden"], errors="coerce"
    )
    invalid_hours = df["Arbeitszeit_Stunden"].isna()
    if invalid_hours.any():
        print(
            f"Warnung: {invalid_hours.sum()} Zeilen mit ungültigem Stundenwert werden übersprungen."
        )
        df = df[~invalid_hours].copy()

    if df.empty:
        sys.exit("Fehler: Keine gültigen Daten nach der Bereinigung.")

    return df[["MitarbeiterID", "Datum", "Arbeitszeit_Stunden"]].copy()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert Stunden pro Mitarbeiter und Tag."""
    return (
        df.groupby(["MitarbeiterID", "Datum"], as_index=False)["Arbeitszeit_Stunden"]
        .sum()
        .sort_values(["MitarbeiterID", "Datum"])
        .reset_index(drop=True)
    )


def distribute_overflow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kern-Algorithmus:
    1. Berechne Overflow (> 10h) und freie Kapazität (< 10h) pro Mitarbeiter.
    2. Verteile Overflow auf freie Kapazitäten (chronologisch).
    3. Reste werden als neue Schicht-Einträge angehängt.
    """
    results = []
    new_shifts = []

    for ma_id, group in df.groupby("MitarbeiterID"):
        group = group.copy().sort_values("Datum").reset_index(drop=True)

        hours = group["Arbeitszeit_Stunden"].tolist()
        dates = group["Datum"].tolist()

        # Overflow pro Tag berechnen und auf 10h kappen
        overflow_pool = 0.0
        for i, h in enumerate(hours):
            if h > MAX_HOURS:
                overflow_pool += h - MAX_HOURS
                hours[i] = MAX_HOURS

        # Freie Kapazitäten auffüllen
        if overflow_pool > 0:
            for i, h in enumerate(hours):
                if overflow_pool <= 0:
                    break
                free = MAX_HOURS - h
                if free > 0:
                    fill = min(free, overflow_pool)
                    hours[i] += fill
                    overflow_pool -= fill
                    overflow_pool = round(overflow_pool, 10)  # Floating-Point-Schutz

        # Ergebniszeilen für diesen Mitarbeiter
        for d, h in zip(dates, hours):
            results.append(
                {
                    "MitarbeiterID": ma_id,
                    "Datum": d,
                    "Arbeitszeit_Stunden": round(h, 4),
                    "Hinweis": "",
                }
            )

        # Restlicher Overflow → neue Schicht
        if overflow_pool > 0.001:
            last_date = dates[-1]
            # Neues Datum: nächster Werktag nach dem letzten Tag
            next_date = last_date + pd.offsets.BusinessDay(1)
            new_shifts.append(
                {
                    "MitarbeiterID": ma_id,
                    "Datum": next_date,
                    "Arbeitszeit_Stunden": round(overflow_pool, 4),
                    "Hinweis": "Neue Schicht (Overflow-Rest)",
                }
            )
            print(
                f"  → Mitarbeiter {ma_id}: {overflow_pool:.2f}h Overflow-Rest "
                f"als neue Schicht am {next_date.date()}"
            )

    result_df = pd.DataFrame(results + new_shifts)
    return result_df.sort_values(["MitarbeiterID", "Datum"]).reset_index(drop=True)


def write_output(df: pd.DataFrame, filepath: str) -> None:
    """Schreibt das Ergebnis als formatierte Excel-Datei."""
    df_export = df.copy()
    df_export["Datum"] = df_export["Datum"].dt.date

    df_export.to_excel(filepath, index=False, sheet_name="Optimierte_Zeiten")

    wb = load_workbook(filepath)
    ws = wb.active

    # Stile
    header_font = Font(name="Arial", bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", start_color="1F4E79")
    new_shift_fill = PatternFill("solid", start_color="FFF2CC")
    over_fill = PatternFill("solid", start_color="FCE4D6")
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    col_widths = {"A": 18, "B": 16, "C": 22, "D": 35}

    # Header formatieren
    for col_letter, width in col_widths.items():
        ws.column_dimensions[col_letter].width = width

    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border

    # Datenzeilen formatieren
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        hinweis = row[3].value or ""
        for cell in row:
            cell.border = border
            cell.font = Font(name="Arial", size=10)
            if hinweis:
                cell.fill = new_shift_fill
            elif row[2].value is not None and float(row[2].value) >= MAX_HOURS:
                cell.fill = over_fill
            cell.alignment = center if cell.column > 1 else left

    # Autofilter
    ws.auto_filter.ref = ws.dimensions

    # Legende
    ws_legend = wb.create_sheet("Legende")
    legend_data = [
        ("Farbe", "Bedeutung"),
        ("Orange Hintergrund", f"Tag wurde auf {MAX_HOURS}h begrenzt"),
        ("Gelber Hintergrund", "Neue Schicht für verbleibende Overflow-Stunden"),
        ("Kein Hintergrund", "Unveränderter oder aufgefüllter Tag"),
    ]
    for i, (a, b) in enumerate(legend_data, 1):
        ws_legend[f"A{i}"] = a
        ws_legend[f"B{i}"] = b
        if i == 1:
            ws_legend[f"A{i}"].font = Font(bold=True, name="Arial")
            ws_legend[f"B{i}"].font = Font(bold=True, name="Arial")
    ws_legend.column_dimensions["A"].width = 25
    ws_legend.column_dimensions["B"].width = 45

    wb.save(filepath)
    print(f"\nErgebnis gespeichert: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Zeiterfassung Overflow-Verteilung (Excel → Excel)"
    )
    parser.add_argument("input", help="Pfad zur Eingabe-Excel-Datei (.xlsx)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Pfad zur Ausgabe-Excel-Datei (Standard: <input>_optimiert.xlsx)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=10.0,
        help="Maximale Stunden pro Tag (Standard: 10.0)",
    )
    args = parser.parse_args()

    global MAX_HOURS
    MAX_HOURS = args.max_hours

    output_path = args.output
    if not output_path:
        base = args.input.rsplit(".", 1)[0]
        output_path = f"{base}_optimiert.xlsx"

    print(f"Lese Daten aus: {args.input}")
    df_raw = read_input(args.input)
    print(f"  {len(df_raw)} gültige Einträge geladen.")

    print("Aggregiere nach Mitarbeiter und Datum...")
    df_agg = aggregate(df_raw)
    print(f"  {len(df_agg)} Tageseinträge nach Aggregation.")

    print(f"Verteile Overflow (Max {MAX_HOURS}h/Tag)...")
    df_result = distribute_overflow(df_agg)

    write_output(df_result, output_path)


if __name__ == "__main__":
    main()
