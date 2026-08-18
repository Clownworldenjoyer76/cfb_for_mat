#!/usr/bin/env python3
"""Direct finished-event movement probe for DraftKings/FanDuel ML history.

Used only after /historical/events has resolved a 2025 NFL event. ML is chosen
because /odds/movements does not require marketLine for ML, so this cleanly tests
whether finished-event time-series history exists independent of closing-line
availability.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://api.odds-api.io/v3"
EVENT_ID = "60525453"  # 2025 DAL @ PHI, resolved by /historical/events probe.
BOOKMAKERS = ("DraftKings", "FanDuel")
OUTPUT = Path("docs/win/football/nfl/training/market_timing_direct_movement_probe.json")


def request(bookmaker, api_key):
    query = urlencode({"apiKey": api_key, "eventId": EVENT_ID, "bookmaker": bookmaker, "market": "ML"})
    req = Request(f"{BASE}/odds/movements?{query}", headers={"User-Agent": "nfl-market-timing-direct-probe/1.0"})
    try:
        with urlopen(req, timeout=45) as response:
            status = response.status
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except URLError as exc:
        return {"bookmaker": bookmaker, "http_status": None, "error": str(exc.reason)}

    try:
        payload = json.loads(body)
    except Exception:
        payload = {"body_prefix": body[:300]}

    result = {
        "bookmaker": bookmaker,
        "http_status": status,
        "opening_present": False,
        "latest_present": False,
        "movement_count": 0,
        "opening_timestamp": None,
        "latest_timestamp": None,
    }
    if isinstance(payload, dict):
        opening = payload.get("opening")
        latest = payload.get("latest")
        movements = payload.get("movements")
        if isinstance(opening, dict) and opening:
            result["opening_present"] = True
            result["opening_timestamp"] = opening.get("timestamp")
        if isinstance(latest, dict) and latest:
            result["latest_present"] = True
            result["latest_timestamp"] = latest.get("timestamp")
        if isinstance(movements, list):
            result["movement_count"] = len(movements)
        if status >= 400:
            result["provider_error"] = payload.get("message") or payload.get("error") or payload.get("detail")
    return result


def main():
    api_key = os.getenv("API_ODDS", "").strip()
    if not api_key:
        raise RuntimeError("Missing environment variable: API_ODDS")
    results = [request(bookmaker, api_key) for bookmaker in BOOKMAKERS]
    usable = [r for r in results if r.get("http_status") == 200 and r.get("opening_present") and r.get("movement_count", 0) >= 1]
    report = {
        "probe": "finished_2025_nfl_direct_ml_movements",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "event_id": EVENT_ID,
        "market": "ML",
        "results": results,
        "asof_backfill_capability_for_live_books": bool(usable),
        "reason": "finished_event_ml_movements_available" if usable else "no_finished_event_ml_movements_for_draftkings_or_fanduel",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
