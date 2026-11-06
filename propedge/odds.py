"""Odds math: conversions, de-vigging, expected value, Kelly sizing."""
from __future__ import annotations


def parse_odds(x) -> float:
    """Parse American (+120 / -110) or decimal (2.2) odds into decimal odds."""
    v = float(str(x).strip().replace("+", ""))
    if abs(v) >= 100:
        return american_to_decimal(v)
    if v <= 1.0:
        raise ValueError(f"invalid decimal odds: {x}")
    return v


def american_to_decimal(a: float) -> float:
    a = float(a)
    if -100 < a < 100:
        raise ValueError(f"invalid American odds: {a}")
    return 1 + (a / 100 if a > 0 else 100 / -a)


def decimal_to_american(d: float) -> int:
    if d <= 1:
        raise ValueError(f"invalid decimal odds: {d}")
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def prob_to_american(p: float) -> int:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return decimal_to_american(1 / p)


def fmt_american(d: float) -> str:
    a = decimal_to_american(d)
    return f"+{a}" if a > 0 else str(a)


def devig(implied: list[float], method: str = "power") -> list[float]:
    """Remove the bookmaker margin from a set of implied probabilities.

    multiplicative: scale proportionally (simple, standard)
    additive:       subtract equal share of the overround
    power:          p_i ** k with k chosen so they sum to 1 — accounts for
                    favourite/longshot bias; usually the best of the three for props.
    """
    total = sum(implied)
    if method == "multiplicative":
        return [p / total for p in implied]
    if method == "additive":
        over = (total - 1) / len(implied)
        return [min(max(p - over, 1e-6), 1 - 1e-6) for p in implied]
    if method == "power":
        lo, hi = 0.01, 100.0
        for _ in range(100):
            k = (lo + hi) / 2
            s = sum(p ** k for p in implied)
            if s > 1:
                lo = k
            else:
                hi = k
        k = (lo + hi) / 2
        out = [p ** k for p in implied]
        s = sum(out)
        return [p / s for p in out]
    raise ValueError(f"unknown devig method {method}")


def expected_value(p_win: float, dec: float, p_push: float = 0.0) -> float:
    """EV per 1 unit staked."""
    p_lose = max(0.0, 1 - p_win - p_push)
    return p_win * (dec - 1) - p_lose


def kelly_fraction(p_win: float, dec: float, p_push: float = 0.0) -> float:
    """Full-Kelly fraction of bankroll (push-aware). 0 if no edge."""
    b = dec - 1
    q = max(0.0, 1 - p_win - p_push)
    if b <= 0 or p_win + q <= 0:
        return 0.0
    f = (b * p_win - q) / (b * (p_win + q))
    return max(0.0, f)
