"""Walk-forward backtest of the projection model against historical prop lines.

History CSV columns: date, player, market, line, over_odds, under_odds, actual
optional: opponent, book

For each prop the model only sees games *before* that date. We report:
  * Brier score / log loss of the model vs the de-vigged market — if the model can't beat
    the market's own probabilities, any "edge" it shows is probably noise.
  * Flat-stake results betting only when model EV >= threshold.
  * Calibration table.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

from .odds import devig, expected_value, parse_odds


@dataclass
class BacktestResult:
    n_props: int = 0
    brier_model: float = 0.0
    brier_market: float = 0.0
    logloss_model: float = 0.0
    logloss_market: float = 0.0
    bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    staked: float = 0.0
    profit: float = 0.0
    by_market: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    bet_log: list = field(default_factory=list)


def run(logs, history_path: str, min_ev: float = 0.03, half_life: float = 10.0,
        use_opponent: bool = True, sport: str | None = None, max_ev: float = 0.25) -> BacktestResult:
    res = BacktestResult()
    n_scored = 0
    with open(history_path, newline="") as f:
        rows = [{k.strip().lower(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    rows.sort(key=lambda r: r["date"])
    for r in rows:
        try:
            proj = logs.project(r["player"], r["market"], before=r["date"],
                                opponent=r.get("opponent") or None, half_life=half_life,
                                sport=sport, use_opponent=use_opponent)
        except KeyError:
            continue
        if proj is None or not r.get("actual"):
            continue
        line, actual = float(r["line"]), float(r["actual"])
        d_over, d_under = parse_odds(r["over_odds"]), parse_odds(r["under_odds"])
        po, pu, pp = proj.probs(line)
        mkt_over = devig([1 / d_over, 1 / d_under], "power")[0]
        model_over = po / (po + pu) if po + pu else 0.5
        res.n_props += 1

        if actual != line:
            y = 1.0 if actual > line else 0.0
            n_scored += 1
            res.brier_model += (model_over - y) ** 2
            res.brier_market += (mkt_over - y) ** 2
            eps = 1e-6
            res.logloss_model -= y * math.log(max(model_over, eps)) + (1 - y) * math.log(max(1 - model_over, eps))
            res.logloss_market -= y * math.log(max(mkt_over, eps)) + (1 - y) * math.log(max(1 - mkt_over, eps))
            b = min(int(model_over * 10), 9)
            c = res.calibration.setdefault(b, [0, 0.0, 0.0])
            c[0] += 1; c[1] += model_over; c[2] += y

        ev_o, ev_u = expected_value(po, d_over, pp), expected_value(pu, d_under, pp)
        side, ev, dec = ("over", ev_o, d_over) if ev_o >= ev_u else ("under", ev_u, d_under)
        if ev < min_ev or ev > max_ev:  # very large "edges" are usually model errors (injury, role)
            continue
        res.bets += 1
        res.staked += 1
        if actual == line:
            res.pushes += 1; pnl = 0.0
        elif (actual > line) == (side == "over"):
            res.wins += 1; pnl = dec - 1
        else:
            res.losses += 1; pnl = -1.0
        res.profit += pnl
        bm = res.by_market.setdefault(r["market"], [0, 0.0])
        bm[0] += 1; bm[1] += pnl
        res.bet_log.append((r["date"], r["player"], r["market"], side, line, dec, ev, actual, pnl))

    if n_scored:
        res.brier_model /= n_scored; res.brier_market /= n_scored
        res.logloss_model /= n_scored; res.logloss_market /= n_scored
    return res
