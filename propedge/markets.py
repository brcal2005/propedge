"""Market definitions: which game-log columns make up each prop, and dispersion priors.

Keys match The Odds API market keys. phi = variance/mean prior for the stat.
"""
from __future__ import annotations

import re

MARKETS: dict[str, tuple[list[str], float]] = {
    # Basketball (NBA / WNBA / NCAAB)
    "player_points": (["pts"], 2.0),
    "player_rebounds": (["reb"], 1.3),
    "player_assists": (["ast"], 1.3),
    "player_threes": (["fg3m"], 1.1),
    "player_blocks": (["blk"], 1.2),
    "player_steals": (["stl"], 1.1),
    "player_turnovers": (["tov"], 1.1),
    "player_points_rebounds_assists": (["pts", "reb", "ast"], 2.0),
    "player_points_rebounds": (["pts", "reb"], 2.0),
    "player_points_assists": (["pts", "ast"], 2.0),
    "player_rebounds_assists": (["reb", "ast"], 1.4),
    # Football (NFL / NCAAF)
    "player_pass_yds": (["pass_yds"], 15.0),
    "player_pass_completions": (["pass_cmp"], 1.3),
    "player_pass_attempts": (["pass_att"], 1.4),
    "player_pass_tds": (["pass_td"], 1.0),
    "player_pass_interceptions": (["pass_int"], 1.0),
    "player_rush_yds": (["rush_yds"], 10.0),
    "player_rush_attempts": (["rush_att"], 1.5),
    "player_receptions": (["rec"], 1.1),
    "player_reception_yds": (["rec_yds"], 12.0),
    # Baseball (MLB)
    "pitcher_strikeouts": (["so"], 1.1),
    "pitcher_outs": (["outs"], 1.5),
    "pitcher_hits_allowed": (["h_allowed"], 1.2),
    "batter_hits": (["h"], 1.0),
    "batter_total_bases": (["tb"], 1.6),
    "batter_rbis": (["rbi"], 1.5),
    "batter_runs_scored": (["r"], 1.2),
    # Hockey (NHL)
    "player_shots_on_goal": (["sog"], 1.1),
    "player_goals": (["g"], 1.0),
}

# Sport-specific overrides where market keys collide across sports.
SPORT_OVERRIDES: dict[str, dict[str, tuple[list[str], float]]] = {
    "icehockey": {
        "player_points": (["pts"], 1.05),
        "player_assists": (["ast"], 1.0),
    },
}

ALIASES = {
    "pts": "player_points", "points": "player_points",
    "reb": "player_rebounds", "rebounds": "player_rebounds",
    "ast": "player_assists", "assists": "player_assists",
    "threes": "player_threes", "3pm": "player_threes", "fg3m": "player_threes",
    "pra": "player_points_rebounds_assists",
    "pr": "player_points_rebounds", "pa": "player_points_assists", "ra": "player_rebounds_assists",
    "blk": "player_blocks", "stl": "player_steals", "tov": "player_turnovers",
    "pass_yds": "player_pass_yds", "rush_yds": "player_rush_yds",
    "rec": "player_receptions", "receptions": "player_receptions", "rec_yds": "player_reception_yds",
    "ks": "pitcher_strikeouts", "strikeouts": "pitcher_strikeouts",
    "hits": "batter_hits", "tb": "batter_total_bases", "sog": "player_shots_on_goal",
}


def resolve(market: str, sport: str | None = None) -> tuple[str, list[str], float]:
    """Return (market_key, columns, phi_prior). Also accepts 'col:a+b' for custom combos."""
    m = market.strip()
    if m.startswith("col:"):
        cols = m[4:].split("+")
        return m, cols, 1.3
    key = ALIASES.get(m.lower(), m.lower())
    if sport:
        group = sport.split("_")[0]
        if key in SPORT_OVERRIDES.get(group, {}):
            cols, phi = SPORT_OVERRIDES[group][key]
            return key, cols, phi
    if key in MARKETS:
        cols, phi = MARKETS[key]
        return key, cols, phi
    raise KeyError(f"unknown market '{market}'. Known: {', '.join(sorted(MARKETS))} (or use col:a+b)")


def phi_prior(market: str, sport: str | None = None) -> float:
    try:
        return resolve(market, sport)[2]
    except KeyError:
        return 1.3


_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str) -> str:
    s = name.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return " ".join(s.split())
