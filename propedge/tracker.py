"""Bet tracker with closing-line-value (CLV).

CLV is the single best early signal of whether you actually have an edge: results take
thousands of bets to converge, but consistently beating the closing no-vig price shows up
in a few hundred.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3

from .odds import devig
from .paths import home

SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  placed_at TEXT, event TEXT, player TEXT, market TEXT, line REAL, side TEXT,
  odds REAL, stake REAL, book TEXT, fair_p REAL,
  close_fair_p REAL, result TEXT, profit REAL, notes TEXT
);
"""


def db_path() -> str:
    return os.environ.get("PROPEDGE_DB", os.path.join(home(), "bets.db"))


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def add(con, player, market, line, side, odds, stake, book="", event="", fair_p=None, notes="") -> int:
    cur = con.execute(
        "INSERT INTO bets (placed_at,event,player,market,line,side,odds,stake,book,fair_p,notes)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (dt.datetime.now().isoformat(timespec="seconds"), event, player, market, line, side.lower(),
         odds, stake, book, fair_p, notes))
    con.commit()
    return cur.lastrowid


def close(con, bet_id: int, close_over: float, close_under: float):
    """Record the closing two-way prices (decimal) to compute CLV."""
    row = con.execute("SELECT side FROM bets WHERE id=?", (bet_id,)).fetchone()
    if not row:
        raise SystemExit(f"no bet {bet_id}")
    po, pu = devig([1 / close_over, 1 / close_under], "power")
    con.execute("UPDATE bets SET close_fair_p=? WHERE id=?", (po if row["side"] == "over" else pu, bet_id))
    con.commit()


def settle(con, bet_id: int, result: str):
    row = con.execute("SELECT odds, stake FROM bets WHERE id=?", (bet_id,)).fetchone()
    if not row:
        raise SystemExit(f"no bet {bet_id}")
    result = result.lower()
    profit = {"win": row["stake"] * (row["odds"] - 1), "loss": -row["stake"], "push": 0.0, "void": 0.0}[result]
    con.execute("UPDATE bets SET result=?, profit=? WHERE id=?", (result, profit, bet_id))
    con.commit()


def report(con) -> dict:
    rows = con.execute("SELECT * FROM bets").fetchall()
    settled = [r for r in rows if r["result"] in ("win", "loss", "push")]
    staked = sum(r["stake"] for r in settled)
    profit = sum(r["profit"] for r in settled)
    clv = [r["close_fair_p"] * r["odds"] - 1 for r in rows if r["close_fair_p"]]
    ev = [r["fair_p"] * r["odds"] - 1 for r in rows if r["fair_p"]]
    return {
        "bets": len(rows), "settled": len(settled),
        "wins": sum(r["result"] == "win" for r in settled),
        "losses": sum(r["result"] == "loss" for r in settled),
        "pushes": sum(r["result"] == "push" for r in settled),
        "staked": staked, "profit": profit, "roi": profit / staked if staked else 0.0,
        "avg_ev_at_bet": sum(ev) / len(ev) if ev else None,
        "avg_clv": sum(clv) / len(clv) if clv else None,
        "beat_close_pct": sum(c > 0 for c in clv) / len(clv) if clv else None,
        "n_clv": len(clv),
    }


def list_bets(con, open_only=False):
    q = "SELECT * FROM bets" + (" WHERE result IS NULL" if open_only else "") + " ORDER BY id DESC"
    return con.execute(q).fetchall()
