"""propedge — find +EV player props.

  propedge scan      live (The Odds API) or offline market scan for +EV prices
  propedge project   model projection & fair odds for one player prop
  propedge backtest  walk-forward test of the model vs historical lines
  propedge fair      de-vig a two-way price
  propedge fetch-logs download NBA/WNBA game logs from ESPN
  propedge bet       track bets, closing line value and ROI
  propedge demo      generate synthetic data and run everything
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from . import backtest as bt
from . import tracker
from .markets import MARKETS
from .model import GameLogs
from .odds import (devig, expected_value, fmt_american, kelly_fraction, parse_odds,
                   prob_to_american)
from .scanner import DEFAULT_SHARP, analyze, best_prices_only, load_quotes, quotes_from_odds_api

SHORT = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST",
         "player_threes": "3PM", "player_points_rebounds_assists": "PRA",
         "player_points_rebounds": "P+R", "player_points_assists": "P+A",
         "player_rebounds_assists": "R+A"}


def table(rows: list[list], headers: list[str]) -> str:
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def pct(x, d=1):
    return "" if x is None else f"{x * 100:+.{d}f}%"


# ---------------------------------------------------------------- scan
def cmd_scan(a):
    if a.file:
        quotes = load_quotes(a.file)
    else:
        from .odds_api import fetch_props
        markets = a.markets.split(",")
        events = fetch_props(a.sport, markets, a.regions, a.bookmakers, a.max_events, a.hours)
        if a.save:
            with open(a.save, "w") as f:
                json.dump(events, f)
            print(f"saved raw odds to {a.save}", file=sys.stderr)
        quotes = quotes_from_odds_api(events)
    if not quotes:
        print("no prop quotes found"); return
    logs = GameLogs.from_csv(a.logs) if a.logs else None
    sharp = set(a.sharp.split(",")) if a.sharp else DEFAULT_SHARP
    opps = analyze(quotes, sport=a.sport, sharp_books=sharp, min_books=a.min_books,
                   devig_method=a.devig, sharp_only=a.sharp_only, logs=logs,
                   consensus=getattr(a, 'consensus', False))
    if a.books:
        allowed = set(a.books.lower().split(","))
        opps = [o for o in opps if o.quote.book in allowed]
    if not a.all_prices:
        opps = best_prices_only(opps)
    opps = [o for o in opps if o.ev >= a.min_ev and o.ev <= a.max_ev]
    if a.exact_lines:
        opps = [o for o in opps if o.exact_line]
    def news_gap(o):
        # Model and market far apart = the market likely knows something the box scores don't
        # (injury, lineup change, minutes restriction). That's a warning, not an edge.
        return o.model_mean is not None and abs(o.model_mean - o.market_mean) > a.max_gap * max(o.market_mean, 1.0)
    if a.require_model:
        opps = [o for o in opps if o.model_ev is not None and o.model_ev > 0 and not news_gap(o)]
    opps = opps[: a.limit]

    rows = []
    for o in opps:
        q = o.quote
        stake = min(o.kelly * a.kelly, a.max_stake) * a.bankroll
        fair_cond = o.fair_p / (1 - o.push_p) if o.push_p < 1 else o.fair_p
        rows.append([q.player, SHORT.get(q.market, q.market), q.side[0].upper() + " " + f"{q.line:g}",
                     q.book, fmt_american(q.price), f"{prob_to_american(fair_cond):+d}",
                     f"{o.fair_p * 100:.1f}%", pct(o.ev), f"${stake:,.0f}",
                     f"{o.market_mean:.1f}", "" if o.model_mean is None else f"{o.model_mean:.1f}",
                     pct(o.model_ev), "Y" if o.sharp_used else "", "" if o.exact_line else "~",
                     "CHECK NEWS" if news_gap(o) else "", q.event[:28]])
    hdr = ["Player", "Mkt", "Bet", "Book", "Price", "Fair", "WinP", "EV", "Stake",
           "MktProj", "ModelProj", "ModelEV", "Sharp", "Alt", "Note", "Event"]
    print(table(rows, hdr) if rows else f"No bets with EV >= {a.min_ev:.1%}.")
    print(f"\n{len(quotes)} quotes scanned. Stake = {a.kelly:g} Kelly on ${a.bankroll:,.0f}, "
          f"capped at {a.max_stake:.0%}. Fair = no-vig sharp price when available (Sharp=Y), else consensus.\n"
          f"Alt=~ means the sharp book posted a different line and the fair price was translated — less reliable."
          + (f"\nCHECK NEWS = model and market projections differ by >{a.max_gap:.0%}: usually injury/lineup news "
             "the box scores can't see. Not an edge until you know why." if a.logs else ""))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
        print(f"wrote {a.csv}")


# ---------------------------------------------------------------- project
def cmd_project(a):
    logs = GameLogs.from_csv(a.logs)
    p = logs.project(a.player, a.market, opponent=a.opp, minutes=a.minutes,
                     half_life=a.half_life, sport=a.sport, before=a.before)
    if p is None:
        sys.exit(f"not enough games for {a.player} (need 5)")
    print(f"{p.player} — {p.market}")
    print(f"  projection: {p.mean:.2f}   sd {p.dist.sd:.2f}   dispersion {p.phi:.2f}   games {p.n_games}")
    if p.minutes:
        print(f"  minutes: {p.minutes:.1f}")
    print(f"  opponent factor: {p.opp_factor:.3f}")
    print(f"  last {len(p.recent)}: {', '.join(f'{x:g}' for x in p.recent)}")
    lines = [a.line] if a.line is not None else [round(p.mean) + d + 0.5 for d in (-3, -2, -1, 0, 1, 2)]
    rows = []
    for L in lines:
        o, u, push = p.probs(L)
        rows.append([f"{L:g}", f"{o * 100:.1f}%", f"{u * 100:.1f}%", f"{push * 100:.1f}%" if push else "",
                     f"{prob_to_american(o / (o + u)):+d}", f"{prob_to_american(u / (o + u)):+d}"])
    print(table(rows, ["Line", "P(over)", "P(under)", "P(push)", "Fair O", "Fair U"]))
    if a.line is not None and (a.over or a.under):
        o, u, push = p.probs(a.line)
        for side, odds, prob in (("over", a.over, o), ("under", a.under, u)):
            if odds:
                d = parse_odds(odds)
                ev = expected_value(prob, d, push)
                print(f"  {side} {a.line:g} @ {fmt_american(d)}: EV {pct(ev)}  "
                      f"Kelly {kelly_fraction(prob, d, push) * 100:.1f}%")


# ---------------------------------------------------------------- backtest
def cmd_backtest(a):
    logs = GameLogs.from_csv(a.logs)
    r = bt.run(logs, a.history, min_ev=a.min_ev, half_life=a.half_life,
               use_opponent=not a.no_opp, sport=a.sport)
    print(f"Props evaluated: {r.n_props}")
    print(f"  Brier   model {r.brier_model:.4f}   market {r.brier_market:.4f}   "
          f"({'model better' if r.brier_model < r.brier_market else 'MARKET better'})")
    print(f"  LogLoss model {r.logloss_model:.4f}   market {r.logloss_market:.4f}")
    print(f"\nBets (model EV >= {a.min_ev:.0%}, flat 1u): {r.bets}   W {r.wins}  L {r.losses}  P {r.pushes}")
    if r.staked:
        print(f"  profit {r.profit:+.2f}u   ROI {r.profit / r.staked:+.2%}")
        z = 0.0
        n = r.wins + r.losses
        if n:
            # rough z-score of win rate vs 52.38% breakeven at -110
            import math
            z = (r.wins / n - 0.5238) / math.sqrt(0.5238 * 0.4762 / n)
        print(f"  win rate {r.wins / max(n, 1):.1%}  (z vs -110 breakeven: {z:+.2f})")
    if r.by_market:
        print("\n" + table([[m, n, f"{p:+.2f}", f"{p / n:+.1%}"] for m, (n, p) in sorted(r.by_market.items())],
                           ["Market", "Bets", "Units", "ROI"]))
    if r.calibration:
        print("\nCalibration (model P(over) bucket -> actual over rate):")
        print(table([[f"{b * 10}-{b * 10 + 10}%", c[0], f"{c[1] / c[0]:.1%}", f"{c[2] / c[0]:.1%}"]
                     for b, c in sorted(r.calibration.items())], ["Bucket", "N", "Predicted", "Actual"]))
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "player", "market", "side", "line", "odds", "ev", "actual", "pnl"])
            w.writerows(r.bet_log)
        print(f"\nwrote bet log to {a.out}")


# ---------------------------------------------------------------- fair
def cmd_fair(a):
    d1, d2 = parse_odds(a.side1), parse_odds(a.side2)
    imp = [1 / d1, 1 / d2]
    print(f"Hold: {(sum(imp) - 1) * 100:.2f}%")
    rows = []
    for m in ("multiplicative", "additive", "power"):
        p1, p2 = devig(imp, m)
        rows.append([m, f"{p1 * 100:.2f}%", f"{prob_to_american(p1):+d}", f"{p2 * 100:.2f}%", f"{prob_to_american(p2):+d}"])
    print(table(rows, ["Method", "P1", "Fair1", "P2", "Fair2"]))
    if a.target:
        t = parse_odds(a.target)
        p1 = devig(imp, "power")[0]
        print(f"\nSide 1 at {fmt_american(t)} elsewhere: EV {pct(expected_value(p1, t))}, "
              f"Kelly {kelly_fraction(p1, t) * 100:.2f}%")


# ---------------------------------------------------------------- bet
def cmd_bet(a):
    con = tracker.connect(a.db)
    if a.action == "add":
        fair = a.fair
        bid = tracker.add(con, a.player, a.market, a.line, a.side, parse_odds(a.odds), a.stake,
                          a.book or "", a.event or "", fair, a.notes or "")
        print(f"added bet #{bid}")
    elif a.action == "close":
        tracker.close(con, a.id, parse_odds(a.over), parse_odds(a.under)); print("closing line recorded")
    elif a.action == "settle":
        tracker.settle(con, a.id, a.result); print("settled")
    elif a.action == "list":
        rows = tracker.list_bets(con, a.open)
        print(table([[r["id"], r["placed_at"][:10], r["player"], r["market"], f"{r['side']} {r['line']:g}",
                      fmt_american(r["odds"]), r["stake"], r["book"],
                      pct(r["close_fair_p"] * r["odds"] - 1) if r["close_fair_p"] else "",
                      r["result"] or "open", "" if r["profit"] is None else f"{r['profit']:+.2f}"]
                     for r in rows],
                    ["#", "Date", "Player", "Market", "Bet", "Odds", "Stake", "Book", "CLV", "Result", "P/L"]))
    elif a.action == "report":
        s = tracker.report(con)
        print(f"Bets {s['bets']} (settled {s['settled']}: {s['wins']}W-{s['losses']}L-{s['pushes']}P)")
        print(f"Staked {s['staked']:.2f}   Profit {s['profit']:+.2f}   ROI {s['roi']:+.2%}")
        if s["avg_ev_at_bet"] is not None:
            print(f"Avg EV at placement: {pct(s['avg_ev_at_bet'], 2)}")
        if s["avg_clv"] is not None:
            print(f"Avg CLV: {pct(s['avg_clv'], 2)}   beat close {s['beat_close_pct']:.0%} of {s['n_clv']} bets")
            print("  (Positive average CLV over a few hundred bets is the best evidence of a real edge.)")


# ---------------------------------------------------------------- demo
def cmd_demo(a):
    from .sample import generate
    paths = generate(a.dir)
    print(f"Generated synthetic data in {a.dir}/\n")
    print("=" * 78 + "\n1) BACKTEST — model vs historical (synthetic) lines\n" + "=" * 78)
    cmd_backtest(argparse.Namespace(logs=paths["logs"], history=paths["history"], min_ev=0.03,
                                    half_life=10.0, no_opp=False, sport=None, out=None))
    print("\n" + "=" * 78 + "\n2) SCAN — today's slate: soft books vs no-vig sharp price\n" + "=" * 78)
    cmd_scan(argparse.Namespace(file=paths["quotes"], logs=paths["logs"], sport="basketball_nba",
                                sharp=None, min_books=2, devig="power", sharp_only=False, books=None,
                                all_prices=False, exact_lines=False, max_gap=0.25, min_ev=0.03, max_ev=0.3, require_model=False,
                                limit=15, kelly=0.25, max_stake=0.03, bankroll=1000, csv=None))
    print("\nTry next:")
    print(f"  python -m propedge project --logs {paths['logs']} --player \"<name from above>\" --market pra")
    print(f"  python -m propedge scan --file {paths['quotes']} --logs {paths['logs']} --require-model")


# ---------------------------------------------------------------- fetch-logs
def cmd_fetch_logs(a):
    import datetime as dt
    from .espn import fetch_logs, write_csv
    today = dt.date.today()
    if a.start:
        start = dt.date.fromisoformat(a.start)
    else:  # NBA season starts in October; WNBA in May
        y = today.year if today.month >= (10 if a.league == "nba" else 5) else today.year - 1
        start = dt.date(y, 10 if a.league == "nba" else 5, 1)
    end = dt.date.fromisoformat(a.end) if a.end else today - dt.timedelta(days=1)
    rows = fetch_logs(start, end, a.league)
    write_csv(rows, a.out)
    print(f"wrote {len(rows)} player-games ({len({r['player'] for r in rows})} players) to {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="propedge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan prop markets for +EV prices")
    s.add_argument("--sport", default="basketball_nba", help="Odds API sport key, e.g. basketball_nba, americanfootball_nfl, baseball_mlb, icehockey_nhl")
    s.add_argument("--markets", default="player_points,player_rebounds,player_assists,player_threes")
    s.add_argument("--regions", default="us,us2,eu")
    s.add_argument("--bookmakers", help="comma list of Odds API book keys (overrides regions)")
    s.add_argument("--max-events", type=int)
    s.add_argument("--hours", type=float, default=36, help="only games starting within N hours")
    s.add_argument("--file", help="offline: quotes CSV or saved Odds API JSON")
    s.add_argument("--save", help="save fetched raw JSON for re-use")
    s.add_argument("--logs", help="game log CSV to add model projections")
    s.add_argument("--require-model", action="store_true", help="only show bets the model also likes")
    s.add_argument("--max-gap", type=float, default=0.25, help="flag model/market projection gaps above this fraction")
    s.add_argument("--exact-lines", action="store_true", help="skip bets whose fair price was translated from a different line")
    s.add_argument("--books", help="only show bets at these books (the ones you have accounts at)")
    s.add_argument("--sharp", help=f"comma list of sharp books (default {','.join(sorted(DEFAULT_SHARP))})")
    s.add_argument("--sharp-only", action="store_true", help="fair price from sharp books only")
    s.add_argument("--consensus", action="store_true", help="fair price from all books, even when a sharp book is available")
    s.add_argument("--min-books", type=int, default=2)
    s.add_argument("--devig", default="power", choices=["power", "multiplicative", "additive"])
    s.add_argument("--min-ev", type=float, default=0.02)
    s.add_argument("--max-ev", type=float, default=0.30, help="hide absurd edges (usually stale/bad data)")
    s.add_argument("--all-prices", action="store_true", help="show every +EV book, not just the best")
    s.add_argument("--limit", type=int, default=40)
    s.add_argument("--bankroll", type=float, default=1000)
    s.add_argument("--kelly", type=float, default=0.25, help="Kelly multiplier (0.25 = quarter Kelly)")
    s.add_argument("--max-stake", type=float, default=0.03, help="max fraction of bankroll per bet")
    s.add_argument("--csv", help="write results to CSV")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("project", help="model projection for one player prop")
    s.add_argument("--logs", required=True)
    s.add_argument("--player", required=True)
    s.add_argument("--market", required=True, help=f"e.g. pts, reb, ast, pra, {', '.join(list(MARKETS)[:3])}…")
    s.add_argument("--line", type=float)
    s.add_argument("--over", help="over odds, e.g. -115")
    s.add_argument("--under", help="under odds")
    s.add_argument("--opp", help="opponent (as written in logs)")
    s.add_argument("--minutes", type=float, help="override projected minutes")
    s.add_argument("--half-life", type=float, default=10.0)
    s.add_argument("--before", help="only use games before YYYY-MM-DD")
    s.add_argument("--sport")
    s.set_defaults(func=cmd_project)

    s = sub.add_parser("backtest", help="walk-forward backtest vs historical lines")
    s.add_argument("--logs", required=True)
    s.add_argument("--history", required=True)
    s.add_argument("--min-ev", type=float, default=0.03)
    s.add_argument("--half-life", type=float, default=10.0)
    s.add_argument("--no-opp", action="store_true")
    s.add_argument("--sport")
    s.add_argument("--out", help="write bet log CSV")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("fair", help="de-vig a two-way price, e.g. fair -125 +105")
    s.add_argument("side1"); s.add_argument("side2")
    s.add_argument("--target", help="price for side 1 at another book, to get EV")
    s.set_defaults(func=cmd_fair)

    s = sub.add_parser("bet", help="track bets / CLV")
    s.add_argument("--db", help="sqlite path (default ~/.propedge/bets.db)")
    bs = s.add_subparsers(dest="action", required=True)
    x = bs.add_parser("add")
    x.add_argument("--player", required=True); x.add_argument("--market", required=True)
    x.add_argument("--line", type=float, required=True); x.add_argument("--side", required=True, choices=["over", "under"])
    x.add_argument("--odds", required=True); x.add_argument("--stake", type=float, required=True)
    x.add_argument("--book"); x.add_argument("--event"); x.add_argument("--notes")
    x.add_argument("--fair", type=float, help="your fair win probability (from scan WinP)")
    x = bs.add_parser("close"); x.add_argument("id", type=int)
    x.add_argument("--over", required=True); x.add_argument("--under", required=True)
    x = bs.add_parser("settle"); x.add_argument("id", type=int)
    x.add_argument("result", choices=["win", "loss", "push", "void"])
    x = bs.add_parser("list"); x.add_argument("--open", action="store_true")
    bs.add_parser("report")
    s.set_defaults(func=cmd_bet)

    s = sub.add_parser("fetch-logs", help="download player game logs from ESPN")
    s.add_argument("--league", default="nba", choices=["nba", "wnba"])
    s.add_argument("--start", help="YYYY-MM-DD (default: start of the current season)")
    s.add_argument("--end", help="YYYY-MM-DD (default: yesterday)")
    s.add_argument("--out", default="nba_logs.csv")
    s.set_defaults(func=cmd_fetch_logs)

    s = sub.add_parser("demo", help="generate synthetic data and run backtest + scan")
    s.add_argument("--dir", default="demo_data")
    s.set_defaults(func=cmd_demo)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
