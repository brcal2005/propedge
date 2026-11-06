"""Market scanner: find +EV prop prices by comparing each book against a no-vig
fair price derived from the whole market (sharp books weighted more).

Key idea: for every (game, player, market) we devig every two-sided book quote,
then fit one distribution whose mean best matches all of them. That lets us price
*any* line — so a soft book hanging 24.5 when the sharp market is at 26.5 is
evaluated correctly, not just exact-line matches.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass

from .dist import CountDist, fit_mean
from .markets import norm_name, phi_prior
from .odds import devig, expected_value, kelly_fraction, parse_odds

DEFAULT_SHARP = {"pinnacle", "circasports", "bookmaker", "betcris", "lowvig", "betonlineag"}


@dataclass
class Quote:
    event: str
    player: str
    market: str
    book: str
    line: float
    side: str          # "over" | "under"
    price: float       # decimal odds
    home: str = ""
    away: str = ""
    commence: str = ""


@dataclass
class Opportunity:
    quote: Quote
    fair_p: float
    push_p: float
    ev: float
    kelly: float
    market_mean: float
    n_books: int
    sharp_used: bool
    model_p: float | None = None
    model_ev: float | None = None
    model_mean: float | None = None


def quotes_from_odds_api(events: list[dict]) -> list[Quote]:
    out = []
    for ev in events:
        name = f"{ev.get('away_team', '')} @ {ev.get('home_team', '')}"
        for bm in ev.get("bookmakers", []):
            for m in bm.get("markets", []):
                for o in m.get("outcomes", []):
                    side = (o.get("name") or "").lower()
                    if side not in ("over", "under") or o.get("point") is None:
                        continue
                    out.append(Quote(
                        event=name, player=o.get("description", ""), market=m["key"],
                        book=bm["key"], line=float(o["point"]), side=side,
                        price=parse_odds(o["price"]), home=ev.get("home_team", ""),
                        away=ev.get("away_team", ""), commence=ev.get("commence_time", "")))
    return out


def quotes_from_csv(path: str) -> list[Quote]:
    """CSV columns: event, player, market, book, line, side, price  (price American or decimal)
    optional: home, away."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items()}
            out.append(Quote(r.get("event", ""), r["player"], r["market"], r["book"].lower(),
                             float(r["line"]), r["side"].lower(), parse_odds(r["price"]),
                             r.get("home", ""), r.get("away", "")))
    return out


def load_quotes(path: str) -> list[Quote]:
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return quotes_from_odds_api(data if isinstance(data, list) else [data])
    return quotes_from_csv(path)


def analyze(quotes: list[Quote], sport: str | None = None, sharp_books=DEFAULT_SHARP,
            sharp_weight: float = 3.0, min_books: int = 2, devig_method: str = "power",
            sharp_only: bool = False, logs=None, consensus: bool = False) -> list[Opportunity]:
    groups: dict[tuple, list[Quote]] = {}
    for q in quotes:
        groups.setdefault((q.event, norm_name(q.player), q.market), []).append(q)

    results: list[Opportunity] = []
    for (_, _, market), qs in groups.items():
        pairs: dict[tuple[str, float], dict[str, float]] = {}
        for q in qs:
            pairs.setdefault((q.book, q.line), {})[q.side] = q.price
        obs_all = []  # (book, line, p_over_fair, weight)
        for (book, line), sides in pairs.items():
            if "over" in sides and "under" in sides:
                p_over = devig([1 / sides["over"], 1 / sides["under"]], devig_method)[0]
                w = sharp_weight if book in sharp_books else 1.0
                obs_all.append((book, line, p_over, w))
        sharp_obs = [o for o in obs_all if o[0] in sharp_books]
        has_sharp = bool(sharp_obs)
        # If a sharp book is pricing this prop, it alone defines "fair" (soft books are what
        # we're trying to beat, so letting them vote dilutes the signal). Otherwise fall back
        # to a consensus of every book.
        basis = sharp_obs if (has_sharp and not consensus) else obs_all
        if sharp_only and not has_sharp:
            continue
        books = {o[0] for o in basis}
        if not has_sharp and len(books) < min_books:
            continue
        phi = phi_prior(market, sport)

        fair_cache: dict[str, float | None] = {}

        def fair_mean_for(book: str) -> float | None:
            # Leave-one-out: a book's own price never defines "fair" for itself.
            if book in fair_cache:
                return fair_cache[book]
            obs = [(l, p, w) for b, l, p, w in basis if b != book]
            enough = obs and (has_sharp or len({b for b, *_ in basis if b != book}) >= min_books - 1)
            m = fit_mean(obs, phi) if enough else None
            fair_cache[book] = m
            return m

        proj = None
        if logs is not None:
            q0 = qs[0]
            team = logs.team_of(q0.player)
            opp = None
            if team and q0.home and q0.away:
                if team.lower() == q0.home.lower():
                    opp = q0.away
                elif team.lower() == q0.away.lower():
                    opp = q0.home
            try:
                proj = logs.project(q0.player, market, opponent=opp, sport=sport)
            except KeyError:
                proj = None

        for q in qs:
            m = fair_mean_for(q.book)
            if m is None:
                continue
            d = CountDist(m, phi)
            over, under, push = d.probs(q.line)
            p = over if q.side == "over" else under
            ev = expected_value(p, q.price, push)
            opp_ = Opportunity(q, p, push, ev, kelly_fraction(p, q.price, push), m,
                               len(books - {q.book}), has_sharp and not consensus)
            if proj is not None:
                mo, mu, mp = proj.probs(q.line)
                pm = mo if q.side == "over" else mu
                opp_.model_p, opp_.model_ev, opp_.model_mean = pm, expected_value(pm, q.price, mp), proj.mean
            results.append(opp_)
    results.sort(key=lambda o: o.ev, reverse=True)
    return results


def best_prices_only(opps: list[Opportunity]) -> list[Opportunity]:
    """Keep only the best-EV book for each (event, player, market, side)."""
    seen, out = set(), []
    for o in opps:  # already sorted by ev desc
        k = (o.quote.event, norm_name(o.quote.player), o.quote.market, o.quote.side)
        if k not in seen:
            seen.add(k)
            out.append(o)
    return out
