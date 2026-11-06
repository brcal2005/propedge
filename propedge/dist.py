"""Count distributions for player stats (Poisson / negative binomial)."""
from __future__ import annotations

import math


class CountDist:
    """Distribution of a non-negative integer stat with given mean and
    dispersion phi = variance / mean.  phi<=1 -> Poisson, phi>1 -> negative binomial."""

    def __init__(self, mean: float, phi: float = 1.0):
        self.mean = max(float(mean), 1e-6)
        self.phi = max(float(phi), 1.0)
        self._cum: list[float] = []
        self._last = 0.0
        if self.phi > 1.0001:
            self.r = self.mean / (self.phi - 1)
            self.p = 1 / self.phi
        else:
            self.r = None

    @property
    def sd(self) -> float:
        return math.sqrt(self.mean * self.phi)

    def pmf(self, k: int) -> float:
        if k < 0:
            return 0.0
        if self.r is None:
            lp = k * math.log(self.mean) - self.mean - math.lgamma(k + 1)
        else:
            r, p = self.r, self.p
            lp = (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                  + r * math.log(p) + k * math.log1p(-p))
        return math.exp(lp)

    def cdf(self, k: int) -> float:
        if k < 0:
            return 0.0
        # cumulative table built with the pmf recurrence (much faster than lgamma per term)
        c = self._cum
        if not c:
            self._last = self.pmf(0)
            c.append(self._last)
        while len(c) <= k:
            i = len(c) - 1
            if self.r is None:
                self._last *= self.mean / (i + 1)
            else:
                self._last *= (i + self.r) / (i + 1) * (1 - self.p)
            c.append(min(c[-1] + self._last, 1.0))
        return c[k]

    def probs(self, line: float) -> tuple[float, float, float]:
        """(P(over), P(under), P(push)) for a prop line."""
        if float(line).is_integer():
            L = int(line)
            push = self.pmf(L)
            under = self.cdf(L - 1)
            return max(0.0, 1 - under - push), under, push
        k = math.floor(line)
        under = self.cdf(k)
        return max(0.0, 1 - under), under, 0.0

    def p_over_no_push(self, line: float) -> float:
        o, u, _ = self.probs(line)
        return o / (o + u) if o + u > 0 else 0.5

    def median(self) -> float:
        k = 0
        while self.cdf(k) < 0.5 and k < 10000:
            k += 1
        return k


def fit_mean(observations: list[tuple[float, float, float]], phi: float) -> float:
    """Find the distribution mean that best reproduces observed (line, P(over|no push), weight)
    points — i.e. the market-implied projection. Golden-section search on weighted SSE."""
    hi = max(l for l, _, _ in observations) * 3 + 10

    def loss(m: float) -> float:
        d = CountDist(m, phi)
        return sum(w * (d.p_over_no_push(l) - p) ** 2 for l, p, w in observations)

    a, b = 0.01, hi
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = loss(d)
        if b - a < 1e-3:
            break
    return (a + b) / 2


def fit_mean_phi(observations: list[tuple[float, float, float]], phi_prior: float,
                 prior_strength: float = 0.002) -> float:
    """Learn the dispersion from the market's own ladder of lines (needs >= 2 distinct lines).
    A light penalty pulls toward the prior so two nearly-identical points can't run away."""
    if len({l for l, _, _ in observations}) < 2:
        return phi_prior
    import math as _m

    def loss(phi: float) -> float:
        m = fit_mean(observations, phi)
        d = CountDist(m, phi)
        sse = sum(w * (d.p_over_no_push(l) - p) ** 2 for l, p, w in observations)
        return sse + prior_strength * _m.log(phi / phi_prior) ** 2

    a, b = 1.0, max(3.0, phi_prior * 2.5)
    g = (_m.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(25):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = loss(d)
    return (a + b) / 2
