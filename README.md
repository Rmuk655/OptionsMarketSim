# Options Market Simulator

A limit order book for options contracts with Black-Scholes fair-value reference pricing.

## What it does

Combines two ideas into one system:

| Component | What it is |
|---|---|
| **Limit Order Book** | Price-time priority matching engine — same mechanics as a real exchange |
| **BS Fair Value Engine** | Black-Scholes prices every contract in real time as the reference "fair" price |
| **IV Solver** | Back-solves implied volatility for every executed trade |
| **Greeks** | Delta, Gamma, Theta, Vega via finite differences — live per contract |
| **Analytics** | Trade edge distribution, IV dispersion, cumulative edge over session |

## Key insight

Black-Scholes gives you a theoretical "fair value" for an option. The order book shows you what people are actually willing to pay. The edge (trade price − fair value) tells you whether the market is pricing efficiently around the model — and the implied vol of each trade tells you what volatility the market is implying, versus what the model assumes.

The demo (`--demo`, seeded with `random.seed(42)`) submits randomised orders within +/-$0.30 of fair value and writes `market_analytics.png`; because the orders are random around the model price, it exercises the analytics rather than testing market efficiency.

## Usage

```bash
pip install -r requirements.txt
```

**Interactive CLI:**
```bash
python options_engine.py
```

**Automated demo (generates plots, no input needed):**
```bash
python options_engine.py --demo
```

**Custom parameters:**
```bash
python options_engine.py --underlying TSLA --spot 200 --sigma 0.35 --rate 0.05
```

## CLI Commands

```
add  <strike> <expiry_months> <call|put>   Add a new contract
buy  <symbol>  <price>  <qty>              Submit a buy  limit order
sell <symbol>  <price>  <qty>              Submit a sell limit order
cancel <order_id>                          Cancel an open order
book <symbol>                              Show order book depth
chain                                      Show full option chain
spot <price>                               Update underlying spot price
plot <symbol>                              Plot session analytics
trades [symbol]                            List all trades
help                                       Show commands
quit                                       Exit
```

## Example session

Illustrative; exact figures depend on spot, sigma and rate.

```
>>> add 150 6 call
  Added AAPL150C6M  FV=$10.333

>>> buy  AAPL150C6M 10.25 5
>>> sell AAPL150C6M 10.20 3
  TRADE  3x AAPL150C6M @ $10.250  FV=$10.333  edge=-0.083  IV=...%

>>> book AAPL150C6M
>>> spot 155
>>> plot AAPL150C6M
  Saved: market_analytics.png
```

The trade executes at the resting bid price (price-time priority).

## Order book mechanics

Price-time priority:
- **Buys** matched highest price first; ties broken by earliest timestamp
- **Sells** matched lowest price first; ties broken by earliest timestamp
- Partial fills supported — a 10-contract order can fill against multiple resting orders
- Orders identified by auto-incremented ID; cancellable by ID

Implemented with two heaps:
- `_bids`: max-heap (negate price for Python min-heap)
- `_asks`: min-heap

## What the analytics show

After a session, `plot <symbol>` generates a 4-panel dashboard:

1. **Trade price vs fair value** — do executed prices track the BS model?
2. **Edge distribution** — histogram of (trade price − fair value); should cluster near zero in an efficient market
3. **Cumulative edge** — running total of edge over the session; persistent drift signals mispricing
4. **Implied vol per trade** — IV back-solved for each trade; dispersion around model σ shows how much the market "disagrees" with the BS volatility input

## Black-Scholes assumptions (and why they matter here)

The BS fair value assumes:
1. Log-normal stock returns
2. Constant volatility
3. No dividends
4. Continuous trading, no transaction costs
5. Constant risk-free rate

The order book doesn't enforce these — participants can trade at any price. The analytics show whether aggregate trading behaviour is consistent with these assumptions (edge near zero, IV near model σ) or not.

## Files

```
options_engine.py        Main simulator: order book + BS engine + CLI + analytics
README.md                This file
market_analytics.png     Generated after running --demo or 'plot' command
```

## Limitations

- Single process, in-memory only; no persistence and no tests in the repository.
- A single constant sigma and rate are used for all contracts (no volatility surface).
- Timestamps come from `time.time()`; orders submitted within one clock tick can tie, so time priority among them is not guaranteed.
- CLI input is not validated: non-numeric arguments raise an exception.
- `market_analytics.png` is a generated sample output of the demo.
