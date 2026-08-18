#!/usr/bin/env python3
"""Minimal provider probe for historical NFL market-timing availability.

This intentionally makes only a handful of requests. It does not bulk-download
historical odds. The output is a sanitized capability report; the API key and
request URLs are never written.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_DIR = Path("docs/win/football/nfl")
OUTPUT = BASE_DIR / "training" / "market_timing_provider_probe.json"
API_BASE = "https://api.odds-api.io/v3"
API_KEY_ENV = "API_ODDS"
SPORT = "american-football"
LEAGUE = "usa-nfl"
BOOKMAKERS = ["DraftKings", "FanDuel"]
FROM = "2025-09-04T00:00:00Z"
TO = "2025-09-09T23:59:59Z"
TARGET_HOME = "Philadelphia Eagles"
TARGET_AWAY = "Dallas Cowboys"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_url(path: str, params: dict[str, str]) -> str:
    return f"{API_BASE}{path}?{urlencode(params)}"


def request_json(path: str, params: dict[str, str]) -> tuple[int | None, object, str]:
    url = build_url(path, params)
    req = Request(url, headers={"User-Agent": "nfl-market-timing-probe/1.0"})
    try:
        with urlopen(req, timeout=45) as response:
            status = response.status
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, safe_json(body), "http_error"
    except URLError as exc:
        return None, {"error": str(exc.reason)}, "network_error"
    except Exception as exc:
        return None, {"error": str(exc)}, "request_error"

    try:
        return status, json.loads(body), "ok"
    except Exception:
        return status, {"unparsed_body_prefix": body[:500]}, "json_error"


def safe_json(text: str):
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            # Keep provider error fields but never echo request URLs/query strings.
            return {k: v for k, v in value.items() if k.lower() not in {"url", "request", "requesturl"}}
        return value
    except Exception:
        return {"body_prefix": text[:500]}


def find_target(events: object) -> dict | None:
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("home", "")).strip() == TARGET_HOME and str(event.get("away", "")).strip() == TARGET_AWAY:
            return event
    return None


def market_first_line(historical_odds: dict, bookmaker: str, market_name: str):
    bookmakers = historical_odds.get("bookmakers")
    if not isinstance(bookmakers, dict):
        return None
    markets = bookmakers.get(bookmaker)
    if not isinstance(markets, list):
        return None
    for market in markets:
        if not isinstance(market, dict) or str(market.get("name", "")).lower() != market_name.lower():
            continue
        odds = market.get("odds")
        if isinstance(odds, list) and odds and isinstance(odds[0], dict):
            return odds[0].get("hdp")
    return None


def market_present(historical_odds: dict, bookmaker: str, market_name: str) -> bool:
    bookmakers = historical_odds.get("bookmakers")
    if not isinstance(bookmakers, dict):
        return False
    markets = bookmakers.get(bookmaker)
    if not isinstance(markets, list):
        return False
    return any(isinstance(m, dict) and str(m.get("name", "")).lower() == market_name.lower() for m in markets)


def summarize_movement(status, payload, result_kind, bookmaker, market, market_line):
    summary = {
        "bookmaker": bookmaker,
        "market": market,
        "market_line": market_line,
        "http_status": status,
        "result_kind": result_kind,
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
            summary["opening_present"] = True
            summary["opening_timestamp"] = opening.get("timestamp")
            summary["opening_hdp"] = opening.get("hdp")
        if isinstance(latest, dict) and latest:
            summary["latest_present"] = True
            summary["latest_timestamp"] = latest.get("timestamp")
            summary["latest_hdp"] = latest.get("hdp")
        if isinstance(movements, list):
            summary["movement_count"] = len(movements)
        if status and status >= 400:
            for key in ("error", "message", "detail"):
                if key in payload:
                    summary["provider_error"] = payload.get(key)
                    break
    return summary


def main() -> int:
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing environment variable: {API_KEY_ENV}")

    report = {
        "probe": "nfl_market_timing_historical_capability",
        "created_utc": now_iso(),
        "request_budget": "minimal probe only",
        "target_window": {"from": FROM, "to": TO},
        "target_game": {"away": TARGET_AWAY, "home": TARGET_HOME},
        "historical_events": {},
        "historical_odds": {},
        "movements": [],
        "asof_backfill_capability": False,
        "asof_backfill_reason": "not_evaluated",
    }

    events_status, events_payload, events_kind = request_json(
        "/historical/events",
        {"apiKey": api_key, "sport": SPORT, "league": LEAGUE, "from": FROM, "to": TO},
    )
    event_count = len(events_payload) if isinstance(events_payload, list) else None
    target = find_target(events_payload)
    report["historical_events"] = {
        "http_status": events_status,
        "result_kind": events_kind,
        "event_count": event_count,
        "target_found": bool(target),
    }
    if events_status and events_status >= 400 and isinstance(events_payload, dict):
        report["historical_events"]["provider_error"] = events_payload.get("message") or events_payload.get("error")

    if not target:
        report["asof_backfill_reason"] = "historical_event_not_available_or_endpoint_not_authorized"
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    event_id = str(target.get("id", "")).strip()
    report["historical_events"]["target_event_id"] = event_id
    report["historical_events"]["target_event_date"] = target.get("date")

    odds_status, odds_payload, odds_kind = request_json(
        "/historical/odds",
        {
            "apiKey": api_key,
            "eventId": event_id,
            "bookmakers": ",".join(BOOKMAKERS),
            "markets": "ML,Spread,Totals",
        },
    )
    odds_dict = odds_payload if isinstance(odds_payload, dict) else {}
    report["historical_odds"] = {
        "http_status": odds_status,
        "result_kind": odds_kind,
        "bookmakers_returned": sorted(list(odds_dict.get("bookmakers", {}).keys())) if isinstance(odds_dict.get("bookmakers"), dict) else [],
    }
    if odds_status and odds_status >= 400:
        report["historical_odds"]["provider_error"] = odds_dict.get("message") or odds_dict.get("error")

    for bookmaker in BOOKMAKERS:
        if not isinstance(odds_dict.get("bookmakers"), dict) or bookmaker not in odds_dict.get("bookmakers", {}):
            continue
        for market in ("ML", "Spread", "Totals"):
            if not market_present(odds_dict, bookmaker, market):
                continue
            market_line = None if market == "ML" else market_first_line(odds_dict, bookmaker, market)
            params = {
                "apiKey": api_key,
                "eventId": event_id,
                "bookmaker": bookmaker,
                "market": market,
            }
            if market_line not in (None, ""):
                params["marketLine"] = str(market_line)
            status, payload, kind = request_json("/odds/movements", params)
            report["movements"].append(summarize_movement(status, payload, kind, bookmaker, market, market_line))

    usable = [
        item for item in report["movements"]
        if item.get("http_status") == 200
        and item.get("opening_present")
        and item.get("latest_present")
        and item.get("movement_count", 0) >= 1
        and item.get("opening_timestamp") is not None
    ]
    if usable:
        report["asof_backfill_capability"] = True
        report["asof_backfill_reason"] = "finished_event_movement_history_available"
    elif report["movements"]:
        report["asof_backfill_reason"] = "finished_event_found_but_movement_history_unavailable"
    else:
        report["asof_backfill_reason"] = "historical_odds_or_bookmaker_data_unavailable"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
