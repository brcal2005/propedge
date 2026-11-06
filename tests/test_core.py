import os
import tempfile
import unittest

from propedge.dist import CountDist, fit_mean
from propedge.odds import (american_to_decimal, decimal_to_american, devig, expected_value,
                           kelly_fraction, parse_odds)
from propedge.scanner import Quote, analyze
from propedge import tracker


class TestOdds(unittest.TestCase):
    def test_conversions(self):
        self.assertAlmostEqual(american_to_decimal(-110), 1.90909, places=4)
        self.assertAlmostEqual(american_to_decimal(150), 2.5)
        self.assertEqual(decimal_to_american(2.5), 150)
        self.assertEqual(decimal_to_american(1.5), -200)
        self.assertAlmostEqual(parse_odds("+120"), 2.2)
        self.assertAlmostEqual(parse_odds("1.95"), 1.95)

    def test_devig_sums_to_one(self):
        imp = [1 / american_to_decimal(-140), 1 / american_to_decimal(115)]
        for m in ("multiplicative", "additive", "power"):
            p = devig(imp, m)
            self.assertAlmostEqual(sum(p), 1.0, places=6)
            self.assertGreater(p[0], p[1])

    def test_symmetric_devig(self):
        p = devig([1 / 1.909, 1 / 1.909], "power")
        self.assertAlmostEqual(p[0], 0.5, places=6)

    def test_ev_kelly(self):
        self.assertAlmostEqual(expected_value(0.5, 2.0), 0.0)
        self.assertAlmostEqual(expected_value(0.55, 2.0), 0.10)
        self.assertAlmostEqual(kelly_fraction(0.55, 2.0), 0.10)
        self.assertEqual(kelly_fraction(0.45, 2.0), 0.0)
        # push-aware: 10% push, 50% win at even money -> positive edge
        self.assertGreater(kelly_fraction(0.5, 2.0, 0.1), 0)


class TestDist(unittest.TestCase):
    def test_probs_sum(self):
        for phi in (1.0, 2.0):
            d = CountDist(20, phi)
            o, u, p = d.probs(19.5)
            self.assertAlmostEqual(o + u + p, 1.0, places=6)
            o, u, p = d.probs(20)
            self.assertAlmostEqual(o + u + p, 1.0, places=6)
            self.assertGreater(p, 0)

    def test_variance(self):
        d = CountDist(10, 2.0)
        mean = sum(k * d.pmf(k) for k in range(200))
        var = sum((k - mean) ** 2 * d.pmf(k) for k in range(200))
        self.assertAlmostEqual(mean, 10, places=3)
        self.assertAlmostEqual(var, 20, places=2)

    def test_fit_mean_recovers(self):
        d = CountDist(24.3, 2.0)
        obs = [(L, d.p_over_no_push(L), 1.0) for L in (22.5, 24.5, 26.5)]
        self.assertAlmostEqual(fit_mean(obs, 2.0), 24.3, places=2)


class TestScanner(unittest.TestCase):
    def test_finds_soft_line(self):
        # Sharp book: 25.5 at -110/-110 (fair ~50/50). Soft book hangs 23.5 over at -110 -> +EV over.
        q = lambda b, l, s, a: Quote("A @ B", "Some Player", "player_points", b, l, s, american_to_decimal(a))
        quotes = [q("pinnacle", 25.5, "over", -110), q("pinnacle", 25.5, "under", -110),
                  q("draftkings", 23.5, "over", -110), q("draftkings", 23.5, "under", -110)]
        opps = analyze(quotes)
        best = opps[0]
        self.assertEqual((best.quote.book, best.quote.side, best.quote.line), ("draftkings", "over", 23.5))
        self.assertGreater(best.ev, 0.03)
        # The sharp book must never be graded against itself / soft books
        self.assertFalse(any(o.quote.book == "pinnacle" for o in opps))


class TestTracker(unittest.TestCase):
    def test_flow(self):
        with tempfile.TemporaryDirectory() as d:
            con = tracker.connect(os.path.join(d, "b.db"))
            bid = tracker.add(con, "X", "player_points", 20.5, "over", 2.0, 10, fair_p=0.53)
            tracker.close(con, bid, 1.8, 2.05)
            tracker.settle(con, bid, "win")
            r = tracker.report(con)
            self.assertEqual(r["wins"], 1)
            self.assertAlmostEqual(r["profit"], 10)
            self.assertGreater(r["avg_clv"], 0)


if __name__ == "__main__":
    unittest.main()
