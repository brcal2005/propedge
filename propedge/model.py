"""Projection model from player game logs.

Approach (deliberately simple and transparent):
  * recency-weighted per-minute rate (exponential decay, half-life in games)
  * x projected minutes (recency-weighted average, or your override for role changes / injuries)
  * x opponent factor (how much players produce vs that opponent relative to their own average,
    shrunk heavily toward 1.0)
  * dispersion (variance/mean) learned from the player's own history, shrunk toward a market prior
  * Poisson / negative-binomial distribution -> P(over), P(under), P(push)

Game log CSV: one row per player-game. Required columns: player, date (YYYY-MM-DD).
Optional: team, opponent, minutes. Stat columns: pts, reb, ast, fg3m, ... (see markets.py).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

from .dist import CountDist
from .markets import norm_name, resolve


@dataclass
class Projection:
    player: str
    market: str
    mean: float
    phi: float
    n_games: int
    minutes: float | None
    opp_factor: float
    recent: list[float] = field(default_factory=list)

    @property
    def dist(self) -> CountDist:
        return CountDist(self.mean, self.phi)

    def probs(self, line: float) -> tuple[float, float, float]:
        return self.dist.probs(line)


class GameLogs:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.by_player: dict[str, list[dict]] = {}
        self.display: dict[str, str] = {}
        for r in rows:
            k = r["_pkey"]
            self.by_player.setdefault(k, []).append(r)
            self.display[k] = r["player"]
        for g in self.by_player.values():
            g.sort(key=lambda r: r["date"])
        self._opp_cache: dict = {}

    @classmethod
    def from_csv(cls, path: str) -> "GameLogs":
        rows = []
        with open(path, newline="") as f:
            for raw in csv.DictReader(f):
                r: dict = {}
                for k, v in raw.items():
                    k = k.strip().lower()
                    if k in ("player", "date", "team", "opponent", "opp", "athlete_id", "pos"):
                        r["opponent" if k == "opp" else k] = (v or "").strip()
                    else:
                        try:
                            r[k] = float(v) if v not in (None, "") else None
                        except ValueError:
                            r[k] = v
                if not r.get("player") or not r.get("date"):
                    continue
                r["_pkey"] = norm_name(r["player"])
                rows.append(r)
        return cls(rows)

    def players(self) -> list[str]:
        return sorted(self.display.values())

    def team_of(self, player: str) -> str | None:
        g = self.by_player.get(norm_name(player))
        return g[-1].get("team") if g else None

    @staticmethod
    def _stat(r: dict, cols: list[str]) -> float | None:
        vals = [r.get(c) for c in cols]
        if any(v is None or isinstance(v, str) for v in vals):
            return None
        return float(sum(vals))

    def opponent_factors(self, cols: list[str], before: str | None = None,
                         shrink: float = 30.0, min_games: int = 8) -> dict[str, float]:
        key = (tuple(cols), before)
        if key in self._opp_cache:
            return self._opp_cache[key]
        sums: dict[str, list[float]] = {}
        for games in self.by_player.values():
            gs = [(g, self._stat(g, cols)) for g in games if before is None or g["date"] < before]
            gs = [(g, y) for g, y in gs if y is not None]
            if len(gs) < min_games:
                continue
            avg = sum(y for _, y in gs) / len(gs)
            if avg <= 0:
                continue
            for g, y in gs:
                opp = (g.get("opponent") or "").lower()
                if opp:
                    sums.setdefault(opp, []).append(y / avg)
        out = {o: (sum(v) + shrink) / (len(v) + shrink) for o, v in sums.items()}
        self._opp_cache[key] = out
        return out

    def project(self, player: str, market: str, before: str | None = None,
                opponent: str | None = None, minutes: float | None = None,
                half_life: float = 10.0, min_games: int = 5, sport: str | None = None,
                use_opponent: bool = True) -> Projection | None:
        key, cols, phi_prior = resolve(market, sport)
        games = self.by_player.get(norm_name(player), [])
        games = [g for g in games if before is None or g["date"] < before]
        data = [(g, self._stat(g, cols)) for g in games]
        data = [(g, y) for g, y in data if y is not None]
        n = len(data)
        if n < min_games:
            return None
        ys = [y for _, y in data]
        ws = [0.5 ** ((n - 1 - i) / half_life) for i in range(n)]
        sw = sum(ws)
        mins = [g.get("minutes") for g, _ in data]
        has_min = all(isinstance(m, float) and m > 0 for m in mins)
        if has_min:
            rate = sum(w * y for w, y in zip(ws, ys)) / sum(w * m for w, m in zip(ws, mins))
            proj_min = minutes if minutes else sum(w * m for w, m in zip(ws, mins)) / sw
            base = rate * proj_min
        else:
            proj_min = None
            base = sum(w * y for w, y in zip(ws, ys)) / sw

        opp_f = 1.0
        if use_opponent and opponent:
            opp_f = self.opponent_factors(cols, before).get(opponent.lower(), 1.0)
        mean = base * opp_f

        wm = sum(w * y for w, y in zip(ws, ys)) / sw
        wvar = sum(w * (y - wm) ** 2 for w, y in zip(ws, ys)) / sw * n / max(n - 1, 1)
        phi_obs = wvar / wm if wm > 0 else phi_prior
        k = 10.0
        phi = max(1.0, (n * phi_obs + k * phi_prior) / (n + k))

        return Projection(self.display[norm_name(player)], key, mean, phi, n,
                          proj_min, opp_f, recent=ys[-10:])
