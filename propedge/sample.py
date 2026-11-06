"""Synthetic data generator so you can try every command without API keys.

The fake 'soft books' set lines partly from naive season averages (ignoring role changes
and opponent strength) while the 'sharp book' prices close to the truth. This only
demonstrates the mechanics — real books are far sharper than this.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import os
import random

from .dist import CountDist
from .odds import decimal_to_american

FIRST = ["Marcus", "Jalen", "Devin", "Tyrese", "Luka", "Andre", "Malik", "Cole", "Isaiah", "Darius",
         "Trey", "Jordan", "Keon", "Miles", "Evan", "Zion", "Bryce", "Caleb", "Dante", "Owen"]
LAST = ["Carter", "Hayes", "Brooks", "Reed", "Foster", "Graves", "Nash", "Price", "Wells", "Boone",
        "Cross", "Dixon", "Frye", "Hale", "Knox", "Lowe", "Mercer", "Pryor", "Rowe", "Stone"]
TEAMS = ["Harbor Hawks", "Desert Coyotes", "Metro Knights", "Bay Sharks", "Summit Rams",
         "Prairie Wolves", "Iron Titans", "Coastal Waves", "River Bandits", "Northern Lights"]

STATS = {"pts": 1.8, "reb": 1.3, "ast": 1.3, "fg3m": 1.1}
MARKET_COLS = {"player_points": ["pts"], "player_rebounds": ["reb"],
               "player_assists": ["ast"], "player_points_rebounds_assists": ["pts", "reb", "ast"]}
MARKET_PHI = {"player_points": 2.0, "player_rebounds": 1.3, "player_assists": 1.3,
              "player_points_rebounds_assists": 2.0}


def _poisson(rng, lam):
    if lam > 60:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _nb(rng, mean, phi):
    if mean <= 0:
        return 0
    if phi <= 1.0001:
        return _poisson(rng, mean)
    r = mean / (phi - 1)
    return _poisson(rng, rng.gammavariate(r, mean / r))


def _half(x):
    return math.floor(x) + 0.5


def _prices(dist, line, margin):
    o, u, _ = dist.probs(line)
    s = o + u
    o, u = o / s * (1 + margin), u / s * (1 + margin)
    clamp = lambda p: min(max(p, 0.05), 0.95)
    return decimal_to_american(1 / clamp(o)), decimal_to_american(1 / clamp(u))


def generate(out_dir: str, seed: int = 7, n_days: int = 60) -> dict[str, str]:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    defense = {t: rng.gauss(1.0, 0.07) for t in TEAMS}
    names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 30)
    players = []
    for i, name in enumerate(names):
        players.append({
            "name": name, "team": TEAMS[i // 3],
            "min": rng.uniform(24, 36),
            "rate": {"pts": rng.uniform(0.38, 0.85), "reb": rng.uniform(0.1, 0.35),
                     "ast": rng.uniform(0.06, 0.28), "fg3m": rng.uniform(0.02, 0.12)},
            "role_change": rng.choice([0, 0, 0, 0, 6, -6]),
        })
    start = dt.date(2026, 10, 1)
    logs, history = [], []
    games_so_far = {p["name"]: [] for p in players}

    def schedule():
        t = TEAMS[:]
        rng.shuffle(t)
        return [(t[i], t[i + 1]) for i in range(0, len(t), 2)]

    def cur_min(p, day):
        return p["min"] + (p["role_change"] if day >= 35 else 0)

    for day in range(n_days):
        date = (start + dt.timedelta(days=day)).isoformat()
        for home, away in schedule():
            for p in players:
                if p["team"] not in (home, away):
                    continue
                opp = away if p["team"] == home else home
                mins = max(10.0, rng.gauss(cur_min(p, day), 3.0))
                row = {"player": p["name"], "date": date, "team": p["team"], "opponent": opp,
                       "minutes": round(mins, 1)}
                for s, phi in STATS.items():
                    row[s] = _nb(rng, p["rate"][s] * mins * defense[opp], phi)
                prior = games_so_far[p["name"]]
                if len(prior) >= 10:
                    for mk, cols in MARKET_COLS.items():
                        naive = sum(sum(g[c] for c in cols) for g in prior) / len(prior)
                        line = _half(naive)
                        history.append({"date": date, "player": p["name"], "market": mk, "line": line,
                                        "over_odds": -110, "under_odds": -110, "opponent": opp,
                                        "actual": sum(row[c] for c in cols)})
                prior.append(row)
                logs.append(row)

    # Upcoming slate: one sharp book near the truth, four soft books anchored to naive averages.
    date = (start + dt.timedelta(days=n_days)).isoformat()
    quotes = []
    soft = ["draftkings", "fanduel", "betmgm", "caesars"]
    for home, away in schedule():
        event = f"{away} @ {home}"
        for p in players:
            if p["team"] not in (home, away):
                continue
            opp = away if p["team"] == home else home
            prior = games_so_far[p["name"]]
            for mk, cols in MARKET_COLS.items():
                phi = MARKET_PHI[mk]
                true_mean = sum(p["rate"][c] for c in cols) * cur_min(p, n_days) * defense[opp]
                naive = sum(sum(g[c] for c in cols) for g in prior) / len(prior)
                d_true = CountDist(true_mean, phi)
                line = _half(d_true.median() - 0.5)
                o, u = _prices(d_true, line, 0.025)
                quotes.append([event, home, away, p["name"], mk, "pinnacle", line, "over", o])
                quotes.append([event, home, away, p["name"], mk, "pinnacle", line, "under", u])
                for b in soft:
                    belief = 0.15 * naive + 0.85 * true_mean + rng.gauss(0, 0.03 * true_mean)
                    d_b = CountDist(max(belief, 0.3), phi)
                    bl = _half(d_b.median() - 0.5 + rng.choice([-1, 0, 0, 0, 0, 0, 1]))
                    o, u = _prices(d_b, bl, rng.choice([0.045, 0.05, 0.06]))
                    quotes.append([event, home, away, p["name"], mk, b, bl, "over", o])
                    quotes.append([event, home, away, p["name"], mk, b, bl, "under", u])

    paths = {"logs": os.path.join(out_dir, "gamelogs.csv"),
             "history": os.path.join(out_dir, "prop_history.csv"),
             "quotes": os.path.join(out_dir, "quotes_today.csv")}
    with open(paths["logs"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["player", "date", "team", "opponent", "minutes", *STATS])
        w.writeheader(); w.writerows(logs)
    with open(paths["history"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader(); w.writerows(history)
    with open(paths["quotes"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "home", "away", "player", "market", "book", "line", "side", "price"])
        w.writerows(quotes)
    return paths
