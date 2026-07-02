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

In the demo, trades cluster within ±$0.20 of fair value with mean edge ≈ −$0.03, and implied vols stay close to the 20% model input — consistent with an efficient market around the BS model.

## Usage

```bash
pip install numpy matplotlib scipy
```

**Interactive CLI:**
```bash
python options_market.py
```

**Automated demo (generates plots, no input needed):**
```bash
python options_market.py --demo
```

**Custom parameters:**
```bash
python options_market.py --underlying TSLA --spot 200 --sigma 0.35 --rate 0.05
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

```
>>> add 150 6 call
  Added AAPL150C6M  FV=$10.333

>>> add 142 6 put
  Added AAPL142P6M  FV=$3.637

>>> chain
  OPTION CHAIN — AAPL  Spot=$150.00  σ=20%  r=5%
  AAPL150C6M  FV=$10.333  Delta=+0.598  Theta=-12.186/day
  AAPL142P6M  FV=$ 3.637  Delta=-0.263  Theta= -4.775/day

>>> buy  AAPL150C6M 10.25 5
>>> sell AAPL150C6M 10.30 3
  ✓ TRADE  3x AAPL150C6M @ $10.25  FV=$10.333  edge=-0.083  IV=19.8%

>>> book AAPL150C6M
  [shows live order book with vs-FV column]

>>> spot 155
  Spot updated → $155.00   (all fair values reprice instantly)

>>> plot AAPL150C6M
  Saved: market_analytics.png
```

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
options_market.py        Main simulator: order book + BS engine + CLI + analytics
README.md                This file
market_analytics.png     Generated after running --demo or 'plot' command
```