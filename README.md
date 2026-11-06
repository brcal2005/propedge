# propedge — find +EV player props

A dependency-free Python command-line tool for player-prop betting. It finds edges in two ways:

1. **Market edge (the main one).** It removes the vig from sharp books such as Pinnacle and Circa
   to get a fair price, then flags soft-book prices (DraftKings, FanDuel, MGM, …) that beat it.
   It fits one distribution per player-market, so it handles **different lines across books**:
   for example, an Over 23.5 at DraftKings is priced correctly against Pinnacle's 25.5.
2. **Model edge.** It builds a projection from game logs (recency-weighted per-minute rate ×
   minutes × opponent factor, using a Poisson or negative-binomial distribution). Use it to
   confirm market edges, or to find spots the market hasn't adjusted to yet, such as role
   changes or injuries (pass `--minutes`).

It also includes a walk-forward **backtester** and a **bet tracker with CLV**, so you can find out
whether an edge is real rather than just believing it is.

## Quick start

```bash
pip install -e .            # or just use: python -m propedge ...
propedge demo               # synthetic data, runs the backtest and the scan
python -m unittest discover -s tests
```

## Live scanning (The Odds API)

1. Get a key at https://the-odds-api.com (props need a paid tier for real volume).
2. Run:

```bash
export ODDS_API_KEY=your_key
propedge scan --sport basketball_nba \
  --markets player_points,player_rebounds,player_assists,player_points_rebounds_assists \
  --books draftkings,fanduel,betmgm \
  --bankroll 2000 --save today.json
propedge scan --file today.json --min-ev 0.04  # re-run offline without spending credits
```

Each event costs about `markets × regions` credits. Use `--max-events` and `--bookmakers` to limit it.
Sport keys: `basketball_nba`, `americanfootball_nfl`, `baseball_mlb`, `icehockey_nhl`, `basketball_wnba`, …

Output columns: **Price** (the book's price), **Fair** (no-vig fair odds), **WinP**, **EV**, **Stake**
(fractional Kelly), **MktProj** (the market-implied mean), **ModelProj / ModelEV** (with `--logs`), and
**Sharp** (Y = the fair price came from a sharp book).

## Adding the model

```bash
pip install nba_api pandas
python scripts/fetch_nba_logs.py --season 2026-27 --out nba_logs.csv
propedge scan --file today.json --logs nba_logs.csv --require-model   # market and model both agree
propedge project --logs nba_logs.csv --player "Jayson Tatum" --market pra --line 40.5 \
  --over -115 --under -105 --opp "Miami Heat" --minutes 36
```

You can use game logs for any sport. The CSV needs one row per player-game with `player,date`
and, optionally, `team,opponent,minutes`, plus stat columns. Each market's stat columns are listed
in `propedge/markets.py`, for example NFL `rec_yds`, MLB `so`, NHL `sog`. For a custom combination,
use `--market col:pts+ast`.

## Prove it before betting real money

```bash
propedge backtest --logs nba_logs.csv --history my_prop_history.csv --min-ev 0.04
```

The history CSV needs `date,player,market,line,over_odds,under_odds,actual` (`opponent` is optional).
The backtest **compares the model's Brier score to the market's**. If the model can't beat the
market's own no-vig probabilities, any +EV it shows against real lines is probably noise.

## Track CLV

```bash
propedge bet add --player "X" --market pra --line 29.5 --side over --odds -117 --stake 30 --book fanduel --fair 0.58
propedge bet close 1 --over -135 --under +110     # closing prices from the sharp book
propedge bet settle 1 win
propedge bet report
```

Positive average CLV over a few hundred bets is the strongest early evidence that the edge is real.
Profit and loss alone need thousands of bets before they mean anything.

## Honest notes

- Real edges are usually **2–6%**. Anything above about 15% is usually a stale line, a news event
  (injury, scratch), or a mismatched market. Check before betting. `--max-ev` hides absurd results.
- The demo's soft books are deliberately exploitable, so its numbers say nothing about real markets.
- Soft books limit winning prop bettors fast. Spread action across books and expect limits.
- If no sharp book prices a prop, the fair price falls back to a consensus of all books, which is weaker.
- Only bet what you can afford to lose. US help line: 1-800-GAMBLER.
