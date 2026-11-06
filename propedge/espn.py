"""Download player game logs from ESPN's public scoreboard / box-score API.

Works from home and from cloud servers (NBA.com's own stats API blocks most cloud IPs).
Team names are written in full ("Boston Celtics") to match The Odds API, so `scan --logs`
can work out each player's opponent automatically.

Each game's box score is cached on disk, so re-running only downloads new games.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
import time
import urllib.request

BASE = "https://site.api.espn.com/apis/site/v2/sports"
LEAGUES = {"nba": "basketball/nba", "wnba": "basketball/wnba"}

# ESPN display name -> The Odds API name, where they differ
TEAM_FIX = {"LA Clippers": "Los Angeles Clippers"}

# ESPN box-score stat key -> propedge column
KEYMAP = {
    "minutes": "minutes", "points": "pts", "rebounds": "reb", "assists": "ast",
    "steals": "stl", "blocks": "blk", "turnovers": "tov",
    "offensiveRebounds": "oreb", "defensiveRebounds": "dreb",
}
FIELDS = ["player", "date", "team", "opponent", "minutes", "pts", "reb", "ast", "fg3m", "fg3a",
          "fgm", "fga", "ftm", "fta", "stl", "blk", "tov", "oreb", "dreb"]


def _get(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "propedge/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def _team(name: str) -> str:
    return TEAM_FIX.get(name, name)


def parse_boxscore(summary: dict, date: str) -> list[dict]:
    teams = summary.get("boxscore", {}).get("players", [])
    if len(teams) != 2:
        return []
    names = [_team(t["team"]["displayName"]) for t in teams]
    rows = []
    for i, t in enumerate(teams):
        for block in t.get("statistics", []):
            keys = block.get("keys", [])
            for a in block.get("athletes", []):
                stats = a.get("stats") or []
                if a.get("didNotPlay") or len(stats) != len(keys):
                    continue
                row = {"player": a["athlete"]["displayName"], "date": date,
                       "team": names[i], "opponent": names[1 - i]}
                for k, v in zip(keys, stats):
                    if k in KEYMAP:
                        try:
                            row[KEYMAP[k]] = float(v)
                        except ValueError:
                            row[KEYMAP[k]] = None
                    elif "-" in k and "-" in str(v):  # e.g. "7-13" made-attempted
                        made, att = (int(x) for x in str(v).split("-")[:2])
                        prefix = {"fieldGoalsMade": "fg", "threePointFieldGoalsMade": "fg3",
                                  "freeThrowsMade": "ft"}.get(k.split("-")[0])
                        if prefix:
                            row[prefix + "m"], row[prefix + "a"] = made, att
                if not row.get("minutes"):
                    continue
                rows.append(row)
    return rows


def fetch_logs(start: dt.date, end: dt.date, league: str = "nba",
               cache_dir: str | None = None, verbose: bool = True) -> list[dict]:
    path = LEAGUES[league]
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".propedge", "espn_cache", league)
    os.makedirs(cache_dir, exist_ok=True)
    rows, day, n_games, n_new = [], start, 0, 0
    while day <= end:
        ds = day.strftime("%Y%m%d")
        sb = _get(f"{BASE}/{path}/scoreboard?dates={ds}")
        for ev in sb.get("events", []):
            if not ev.get("status", {}).get("type", {}).get("completed"):
                continue
            # skip preseason (season type 1) and All-Star style exhibitions
            if ev.get("season", {}).get("type") == 1:
                continue
            f = os.path.join(cache_dir, f"{ev['id']}.json")
            if os.path.exists(f):
                with open(f) as fh:
                    summary = json.load(fh)
            else:
                summary = _get(f"{BASE}/{path}/summary?event={ev['id']}")
                with open(f, "w") as fh:
                    json.dump({"boxscore": summary.get("boxscore", {})}, fh)
                n_new += 1
                time.sleep(0.15)
            rows += parse_boxscore(summary, day.isoformat())
            n_games += 1
        if verbose:
            print(f"\r[espn] {day.isoformat()}  games={n_games} (new {n_new})  player-rows={len(rows)}",
                  end="", file=sys.stderr)
        day += dt.timedelta(days=1)
    if verbose:
        print(file=sys.stderr)
    return rows


def write_csv(rows: list[dict], path: str):
    rows = sorted(rows, key=lambda r: (r["player"], r["date"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
