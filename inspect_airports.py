import csv
import os

FOLDER = "kg_airports"

def inspect(filename, max_rows=3):
    path = os.path.join(FOLDER, filename)
    print(f"\n{'='*60}")
    print(f"FILE: {filename}")
    print(f"{'='*60}")
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"Total rows : {len(rows)}")
        print(f"Columns    : {list(rows[0].keys())}")
        print(f"\nSample rows:")
        for row in rows[:max_rows]:
            for k, v in row.items():
                print(f"  {k:30} → {v}")
            print()

# ── INSPECT ALL FOUR FILES ────────────────────────────────
inspect("countries.csv")
inspect("regions.csv")
inspect("airports.csv")
inspect("runways.csv")

# ── CHECK OVERLAP WITH YOUR FLIGHT KG ────────────────────
# These are the airports that appear in your existing 369 flights
# We check if OurAirports contains them so we know the join is possible

KNOWN_AIRPORTS_FROM_FLIGHT_KG = [
    "VIE", "FRA", "MUC", "LHR", "CDG", "AMS", "IST",
    "ZRH", "BRU", "WAW", "HEL", "LUX", "FCO", "BCN",
    "ATH", "DBV", "SKG", "TIA", "LCA", "GRZ", "SZG",
    "INN", "KLU", "LNZ", "SDV", "EVN", "TPE", "ICN",
    "BOM", "RIX"
]

print(f"\n{'='*60}")
print("OVERLAP CHECK — Flight KG airports vs OurAirports")
print(f"{'='*60}")

path = os.path.join(FOLDER, "airports.csv")
with open(path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    all_airports = list(reader)

# Build index by IATA code
iata_index = {
    row["iata_code"].strip(): row
    for row in all_airports
    if row.get("iata_code", "").strip()
}

found    = []
notfound = []

for code in KNOWN_AIRPORTS_FROM_FLIGHT_KG:
    if code in iata_index:
        airport = iata_index[code]
        found.append((code, airport.get("name", "?"), airport.get("municipality", "?"), airport.get("iso_country", "?")))
    else:
        notfound.append(code)

print(f"\nFound in OurAirports : {len(found)}/{len(KNOWN_AIRPORTS_FROM_FLIGHT_KG)}")
for code, name, city, country in found:
    print(f"  {code:6} → {name} | {city} | {country}")

if notfound:
    print(f"\nNOT found : {notfound}")
else:
    print(f"\nAll airports matched perfectly.")