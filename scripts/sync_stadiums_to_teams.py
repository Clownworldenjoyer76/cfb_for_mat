#!/usr/bin/env python3
import os, sys, csv, datetime as dt

STADIUMS_PATH = "data/reference/stadiums.csv"
SUMMARY_PATH = "summaries/sync_stadiums_summary.txt"

REQUIRED = ["team","venue","city","state","country","lat","lon","timezone","altitude_m","is_neutral_site","notes"]
DEFAULTS = {"country":"USA","is_neutral_site":"0"}

def read_csv(p):
    with open(p, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f); rows = [dict(x) for x in r]; headers = r.fieldnames or []
    return headers, rows

def ensure_columns(headers, rows):
    s = set(headers)
    for c in REQUIRED:
        if c not in s:
            headers.append(c); s.add(c)
            for r in rows: r[c] = ""
    for r in rows:
        for k, v in DEFAULTS.items():
            if k in r and (r[k] is None or str(r[k]).strip() == ""): r[k] = v
    return headers, rows

def dedupe(rows):
    seen = set(); out = []
    for r in rows:
        t = str(r.get("team","")).strip().lower()
        if t and t not in seen:
            seen.add(t); out.append(r)
    return out

def reorder(headers):
    return REQUIRED + [h for h in headers if h not in REQUIRED]

def write_csv(p, headers, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            for h in headers:
                if h not in r: r[h] = ""
            w.writerow(r)

def summary(before, after, rows):
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    miss = {"lat":0,"lon":0,"timezone":0,"altitude_m":0}
    for r in rows:
        for k in miss.keys():
            if k not in r or str(r[k]).strip() == "": miss[k]+=1
    lines = [
        "Sync Diagnostics",
        "Timestamp (UTC): " + dt.datetime.utcnow().isoformat() + "Z",
        "File: " + STADIUMS_PATH,
        "Rows before: %d" % before,
        "Rows after: %d" % after,
        "Missing counts:",
        "  lat: %d" % miss["lat"],
        "  lon: %d" % miss["lon"],
        "  timezone: %d" % miss["timezone"],
        "  altitude_m: %d" % miss["altitude_m"],
    ]
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main():
    if not os.path.exists(STADIUMS_PATH):
        print("[sync] ERROR: file not found: %s" % STADIUMS_PATH, file=sys.stderr); sys.exit(1)
    headers, rows = read_csv(STADIUMS_PATH); before = len(rows)
    headers, rows = ensure_columns(headers, rows)
    rows = dedupe(rows); headers = reorder(headers)
    write_csv(STADIUMS_PATH, headers, rows)
    summary(before, len(rows), rows)
    print("[sync] OK (rows before: %d, after: %d)" % (before, len(rows)))

if __name__ == "__main__": main()
