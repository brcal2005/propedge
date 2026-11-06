"""Minimal client for The Odds API (https://the-odds-api.com) — player props.

Set ODDS_API_KEY in your environment. Player props are fetched per event, and each call
costs (number of markets x number of regions) credits, so be selective with markets.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

BASE = "https://api.the-odds-api.com/v4"
LAST_QUOTA: dict = {}


def _config_key() -> str | None:
    from .paths import home
    path = os.path.join(home(), "config.json")
    try:
        with open(path) as f:
            return json.load(f).get("odds_api_key")
    except (OSError, ValueError):
        return None


def _get(path: str, **params) -> list | dict:
    key = os.environ.get("ODDS_API_KEY") or _config_key()
    if not key:
        raise SystemExit("ODDS_API_KEY is not set. Get a key at https://the-odds-api.com")
    params["apiKey"] = key
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        remaining = r.headers.get("x-requests-remaining")
        used = r.headers.get("x-requests-used")
        if remaining is not None:
            LAST_QUOTA.update(remaining=remaining, used=used)
            print(f"[odds-api] credits used={used} remaining={remaining}", file=sys.stderr)
        return json.loads(r.read())


def list_events(sport: str) -> list[dict]:
    return _get(f"/sports/{sport}/events")


def event_odds(sport: str, event_id: str, markets: list[str], regions: str = "us,us2,eu",
               bookmakers: str | None = None) -> dict:
    params = {"markets": ",".join(markets), "oddsFormat": "decimal"}
    if bookmakers:
        params["bookmakers"] = bookmakers
    else:
        params["regions"] = regions
    return _get(f"/sports/{sport}/events/{event_id}/odds", **params)


def fetch_props(sport: str, markets: list[str], regions: str = "us,us2,eu",
                bookmakers: str | None = None, max_events: int | None = None,
                hours_ahead: float = 36.0) -> list[dict]:
    import datetime as dt
    events = list_events(sport)
    now = dt.datetime.now(dt.timezone.utc)
    soon = []
    for e in events:
        t = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if now <= t <= now + dt.timedelta(hours=hours_ahead):
            soon.append(e)
    if max_events:
        soon = soon[:max_events]
    out = []
    for e in soon:
        try:
            out.append(event_odds(sport, e["id"], markets, regions, bookmakers))
        except Exception as ex:  # keep going on per-event failures
            print(f"[odds-api] {e.get('away_team')} @ {e.get('home_team')}: {ex}", file=sys.stderr)
    return out
