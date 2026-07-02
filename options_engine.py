"""
Options Market Simulator
========================
A limit order book for options contracts with Black-Scholes fair-value
reference pricing. Combines:
  - Real-time BS theoretical value as market fair value reference
  - Price-time priority matching engine (limit order book)
  - Market maker spread analysis: are trades happening near fair value?
  - Greeks computed live per contract

Author: Krishnan R

Usage
-----
  python options_market.py              # interactive CLI
  python options_market.py --demo       # run automated demo + generate plots
"""

import sys
import time
import math
import heapq
import random
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BLACK-SCHOLES PRICER  (fair-value engine)
# ─────────────────────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, opt='call'):
    """Black-Scholes closed-form price."""
    if T <= 0:
        return max(S - K, 0) if opt == 'call' else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, opt='call', h=0.01):
    """Greeks via central finite differences."""
    f  = lambda **kw: bs_price(**{'S':S,'K':K,'T':T,'r':r,'sigma':sigma,'opt':opt,**kw})
    delta = (f(S=S+h*S) - f(S=S-h*S)) / (2*h*S)
    gamma = (f(S=S+h*S) - 2*f() + f(S=S-h*S)) / (h*S)**2
    theta = (f(T=max(T-1/365,1e-6)) - f()) * 365
    vega  = (f(sigma=sigma+0.01) - f(sigma=sigma-0.01)) / 0.02 / 100
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}


def implied_vol(mkt_price, S, K, T, r, opt='call', tol=1e-6, max_iter=200):
    """Bisection IV solver."""
    lo, hi = 1e-6, 5.0
    if bs_price(S, K, T, r, lo, opt) > mkt_price: return float('nan')
    if bs_price(S, K, T, r, hi, opt) < mkt_price: return float('nan')
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        val = bs_price(S, K, T, r, mid, opt)
        if abs(val - mkt_price) < tol: return mid
        lo, hi = (mid, hi) if val < mkt_price else (lo, mid)
    return (lo + hi) / 2


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ORDER & TRADE DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

_id_counter = itertools.count(1)

@dataclass
class Contract:
    """An option contract specification."""
    underlying: str   # e.g. "AAPL"
    strike:     float
    expiry:     float  # years to expiry
    opt_type:   str    # 'call' or 'put'

    @property
    def symbol(self):
        t = 'C' if self.opt_type == 'call' else 'P'
        return f"{self.underlying}{int(self.strike)}{t}{int(self.expiry*12)}M"

    def fair_value(self, S, r=0.05, sigma=0.20):
        return bs_price(S, self.strike, self.expiry, r, sigma, self.opt_type)

    def greeks(self, S, r=0.05, sigma=0.20):
        return bs_greeks(S, self.strike, self.expiry, r, sigma, self.opt_type)


@dataclass(order=True)
class Order:
    """A single limit order."""
    # Heap sort fields (set by OrderBook)
    _priority: tuple = field(compare=True, default=(0,))

    order_id:  int   = field(compare=False, default_factory=lambda: next(_id_counter))
    side:      str   = field(compare=False, default='buy')   # 'buy' | 'sell'
    price:     float = field(compare=False, default=0.0)
    qty:       int   = field(compare=False, default=1)
    filled:    int   = field(compare=False, default=0)
    timestamp: float = field(compare=False, default_factory=time.time)
    contract:  Optional[object] = field(compare=False, default=None)

    @property
    def remaining(self): return self.qty - self.filled
    @property
    def is_done(self):   return self.filled >= self.qty


@dataclass
class Trade:
    """A completed match between a buy and sell order."""
    buy_id:    int
    sell_id:   int
    price:     float
    qty:       int
    timestamp: float = field(default_factory=time.time)
    fair_value: float = 0.0

    @property
    def edge(self):
        """Distance of executed price from fair value."""
        return self.price - self.fair_value


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LIMIT ORDER BOOK  (price-time priority)
# ─────────────────────────────────────────────────────────────────────────────

class OrderBook:
    """
    Limit order book for a single options contract.

    Price-time priority:
      Buys  — highest price first; ties broken by earliest timestamp.
      Sells — lowest  price first; ties broken by earliest timestamp.

    Implemented with two heaps:
      _bids: max-heap (negate price for Python's min-heap)
      _asks: min-heap
    """

    def __init__(self, contract: Contract):
        self.contract = contract
        self._bids: list = []   # max-heap: (-price, timestamp, order)
        self._asks: list = []   # min-heap: ( price, timestamp, order)
        self.trades: list[Trade] = []
        self._order_map: dict[int, Order] = {}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _push_bid(self, order: Order):
        heapq.heappush(self._bids, (-order.price, order.timestamp, order))

    def _push_ask(self, order: Order):
        heapq.heappush(self._asks, ( order.price, order.timestamp, order))

    def _top_bid(self) -> Optional[Order]:
        while self._bids:
            _, _, o = self._bids[0]
            if o.is_done: heapq.heappop(self._bids); continue
            return o
        return None

    def _top_ask(self) -> Optional[Order]:
        while self._asks:
            _, _, o = self._asks[0]
            if o.is_done: heapq.heappop(self._asks); continue
            return o
        return None

    # ── Public API ───────────────────────────────────────────────────────────

    def submit(self, side: str, price: float, qty: int,
               spot: float, r=0.05, sigma=0.20) -> list[Trade]:
        """
        Submit a limit order. Returns list of Trade objects executed.

        Parameters
        ----------
        side  : 'buy' or 'sell'
        price : limit price
        qty   : number of contracts
        spot  : current underlying price (for fair-value tagging)
        """
        order = Order(side=side, price=price, qty=qty,
                      contract=self.contract)
        self._order_map[order.order_id] = order
        fv = self.contract.fair_value(spot, r, sigma)

        new_trades = []
        if side == 'buy':
            self._push_bid(order)
        else:
            self._push_ask(order)

        # Try to match
        while True:
            bid = self._top_bid()
            ask = self._top_ask()
            if bid is None or ask is None: break
            if bid.price < ask.price:      break   # no cross

            # Match: use resting order's price (price-time priority)
            exec_price = ask.price if bid.timestamp > ask.timestamp else bid.price
            exec_qty   = min(bid.remaining, ask.remaining)

            bid.filled += exec_qty
            ask.filled += exec_qty

            trade = Trade(buy_id=bid.order_id, sell_id=ask.order_id,
                          price=exec_price, qty=exec_qty, fair_value=fv)
            self.trades.append(trade)
            new_trades.append(trade)

            if bid.is_done: heapq.heappop(self._bids)
            if ask.is_done: heapq.heappop(self._asks)

        return new_trades

    def cancel(self, order_id: int) -> bool:
        """Cancel an open order by ID (mark as fully filled)."""
        o = self._order_map.get(order_id)
        if o and not o.is_done:
            o.filled = o.qty
            return True
        return False

    # ── Market data ──────────────────────────────────────────────────────────

    @property
    def best_bid(self) -> Optional[float]:
        o = self._top_bid(); return o.price if o else None

    @property
    def best_ask(self) -> Optional[float]:
        o = self._top_ask(); return o.price if o else None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return round(ba - bb, 4) if bb and ba else None

    @property
    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return round((bb + ba) / 2, 4) if bb and ba else None

    def depth(self, levels=5):
        """Return top-N bid/ask levels as (price, qty) lists."""
        bids, asks = {}, {}
        for _, _, o in self._bids:
            if not o.is_done:
                bids[o.price] = bids.get(o.price, 0) + o.remaining
        for _, _, o in self._asks:
            if not o.is_done:
                asks[o.price] = asks.get(o.price, 0) + o.remaining
        bid_levels = sorted(bids.items(), reverse=True)[:levels]
        ask_levels = sorted(asks.items())[:levels]
        return bid_levels, ask_levels

    def display(self, spot: float, r=0.05, sigma=0.20):
        """Pretty-print the order book to terminal."""
        fv = self.contract.fair_value(spot, r, sigma)
        g  = self.contract.greeks(spot, r, sigma)
        bid_levels, ask_levels = self.depth()

        print(f"\n{'─'*52}")
        print(f"  {self.contract.symbol}  |  Spot=${spot:.2f}  "
              f"Fair Value=${fv:.3f}  σ=20%")
        print(f"  Δ={g['delta']:+.3f}  Γ={g['gamma']:.4f}  "
              f"θ={g['theta']:+.4f}/day  ν={g['vega']:.4f}/1%")
        print(f"{'─'*52}")
        print(f"  {'ASK QTY':>8}  {'PRICE':>8}  {'vs FV':>7}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*7}")
        for p, q in reversed(ask_levels):
            diff = p - fv
            print(f"  {q:>8}  {p:>8.3f}  {diff:>+7.3f}")
        if self.best_ask and self.best_bid:
            print(f"  {'':>8}  {'SPREAD':>8}  {self.spread:>7.3f}")
        for p, q in bid_levels:
            diff = p - fv
            print(f"  {q:>8}  {p:>8.3f}  {diff:>+7.3f}")
        print(f"  {'BID QTY':>8}  {'PRICE':>8}  {'vs FV':>7}")
        if self.trades:
            last = self.trades[-1]
            print(f"\n  Last trade: {last.qty}x @ ${last.price:.3f}  "
                  f"(edge vs FV: {last.edge:+.3f})")
        print(f"  Total trades: {len(self.trades)}  "
              f"Total volume: {sum(t.qty for t in self.trades)}")
        print(f"{'─'*52}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MARKET SIMULATOR  (multi-contract exchange)
# ─────────────────────────────────────────────────────────────────────────────

class OptionsMarket:
    """
    A simple options exchange holding multiple contracts.
    Tracks spot price, order books, and cross-contract analytics.
    """

    def __init__(self, underlying: str, spot: float,
                 r: float = 0.05, sigma: float = 0.20):
        self.underlying = underlying
        self.spot   = spot
        self.r      = r
        self.sigma  = sigma
        self.books: dict[str, OrderBook] = {}

    def add_contract(self, strike: float, expiry: float,
                     opt_type: str = 'call') -> Contract:
        c = Contract(self.underlying, strike, expiry, opt_type)
        self.books[c.symbol] = OrderBook(c)
        return c

    def submit_order(self, symbol: str, side: str,
                     price: float, qty: int) -> list[Trade]:
        book = self.books.get(symbol)
        if not book:
            print(f"  [!] Unknown contract: {symbol}")
            return []
        trades = book.submit(side, price, qty, self.spot, self.r, self.sigma)
        for t in trades:
            iv = implied_vol(t.price, self.spot, book.contract.strike,
                             book.contract.expiry, self.r, book.contract.opt_type)
            print(f"  ✓ TRADE  {t.qty}x {symbol} @ ${t.price:.3f}  "
                  f"FV=${t.fair_value:.3f}  edge={t.edge:+.3f}  "
                  f"IV={iv*100:.1f}%" if not math.isnan(iv) else
                  f"  ✓ TRADE  {t.qty}x {symbol} @ ${t.price:.3f}  "
                  f"FV=${t.fair_value:.3f}  edge={t.edge:+.3f}")
        return trades

    def update_spot(self, new_spot: float):
        self.spot = new_spot
        print(f"  Spot updated → ${new_spot:.2f}")

    def show_book(self, symbol: str):
        book = self.books.get(symbol)
        if book:
            book.display(self.spot, self.r, self.sigma)
        else:
            print(f"  [!] Unknown contract: {symbol}")

    def show_chain(self):
        """Display option chain: all contracts with fair values and Greeks."""
        print(f"\n{'═'*68}")
        print(f"  OPTION CHAIN — {self.underlying}  Spot=${self.spot:.2f}  "
              f"σ={self.sigma*100:.0f}%  r={self.r*100:.0f}%")
        print(f"{'═'*68}")
        print(f"  {'SYMBOL':<16} {'FV':>7} {'BID':>7} {'ASK':>7} "
              f"{'SPREAD':>7} {'DELTA':>7} {'THETA':>8}")
        print(f"  {'─'*16} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
        for sym, book in self.books.items():
            c  = book.contract
            fv = c.fair_value(self.spot, self.r, self.sigma)
            g  = c.greeks(self.spot, self.r, self.sigma)
            bb = f"${book.best_bid:.3f}" if book.best_bid else "  ---"
            ba = f"${book.best_ask:.3f}" if book.best_ask else "  ---"
            sp = f"{book.spread:.3f}"    if book.spread   else "  ---"
            print(f"  {sym:<16} ${fv:>6.3f} {bb:>7} {ba:>7} "
                  f"{sp:>7} {g['delta']:>+7.3f} {g['theta']:>+8.5f}")
        print(f"{'═'*68}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ANALYTICS & PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_market_analytics(market: OptionsMarket, symbol: str):
    """
    4-panel dashboard after a trading session:
      [0,0] Trade prices vs fair value over time
      [0,1] Bid-ask spread over time
      [1,0] Trade edge distribution (histogram)
      [1,1] Implied vol of each trade vs strike
    """
    book = market.books.get(symbol)
    if not book or not book.trades:
        print("  No trades to plot.")
        return

    trades = book.trades
    prices = [t.price     for t in trades]
    fvs    = [t.fair_value for t in trades]
    edges  = [t.edge       for t in trades]
    times  = list(range(len(trades)))
    ivs    = [implied_vol(t.price, market.spot, book.contract.strike,
                          book.contract.expiry, market.r,
                          book.contract.opt_type)
              for t in trades]

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#0f1117')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ACCENT = '#00d4aa'; ACCENT2 = '#ff6b6b'; ACCENT3 = '#ffd166'
    GRID = '#1e2130'; TEXT = '#e0e0e0'; BG = '#161922'

    def style(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for s in ax.spines.values(): s.set_edgecolor(GRID)
        ax.grid(True, color=GRID, lw=0.6, ls='--')

    # Panel 0: Trade price vs fair value
    ax0 = fig.add_subplot(gs[0, 0]); style(ax0)
    ax0.plot(times, prices, color=ACCENT,  lw=2, marker='o', ms=5,
             label='Trade price')
    ax0.plot(times, fvs,   color=ACCENT2, lw=1.5, ls='--', label='Fair value (BS)')
    ax0.fill_between(times, fvs, prices, alpha=0.15, color=ACCENT)
    ax0.set_title('Trade Price vs Black-Scholes Fair Value', fontsize=10)
    ax0.set_xlabel('Trade #'); ax0.set_ylabel('Price ($)')
    ax0.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    # Panel 1: Edge distribution
    ax1 = fig.add_subplot(gs[0, 1]); style(ax1)
    pos = [e for e in edges if e >= 0]
    neg = [e for e in edges if e <  0]
    ax1.hist(pos, bins=12, color=ACCENT,  alpha=0.8, label='Above FV (seller edge)')
    ax1.hist(neg, bins=12, color=ACCENT2, alpha=0.8, label='Below FV (buyer edge)')
    ax1.axvline(0, color='white', lw=1, ls='--')
    ax1.axvline(np.mean(edges), color=ACCENT3, lw=1.5, ls='-',
                label=f'Mean edge: {np.mean(edges):+.4f}')
    ax1.set_title('Trade Edge Distribution (Price − Fair Value)', fontsize=10)
    ax1.set_xlabel('Edge ($)'); ax1.set_ylabel('Count')
    ax1.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    # Panel 2: Cumulative edge
    ax2 = fig.add_subplot(gs[1, 0]); style(ax2)
    cum_edge = np.cumsum(edges)
    ax2.plot(times, cum_edge, color=ACCENT3, lw=2)
    ax2.fill_between(times, 0, cum_edge,
                     where=[e >= 0 for e in cum_edge],
                     color=ACCENT, alpha=0.3, label='Cumulative gain')
    ax2.fill_between(times, 0, cum_edge,
                     where=[e < 0 for e in cum_edge],
                     color=ACCENT2, alpha=0.3, label='Cumulative loss')
    ax2.axhline(0, color='white', lw=0.8, ls='--')
    ax2.set_title('Cumulative Edge Over Session', fontsize=10)
    ax2.set_xlabel('Trade #'); ax2.set_ylabel('Cumulative Edge ($)')
    ax2.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    # Panel 3: Implied vol of each trade
    ax3 = fig.add_subplot(gs[1, 1]); style(ax3)
    valid_ivs = [(i, iv*100) for i, iv in enumerate(ivs)
                 if not math.isnan(iv)]
    if valid_ivs:
        xi, yi = zip(*valid_ivs)
        ax3.scatter(xi, yi, color=ACCENT, s=40, zorder=3, label='Trade IV')
        ax3.axhline(market.sigma*100, color=ACCENT2, lw=1.5, ls='--',
                    label=f'Model σ ({market.sigma*100:.0f}%)')
        iv_mean = np.mean(yi)
        ax3.axhline(iv_mean, color=ACCENT3, lw=1, ls=':',
                    label=f'Mean trade IV ({iv_mean:.1f}%)')
    ax3.set_title('Implied Vol of Each Trade', fontsize=10)
    ax3.set_xlabel('Trade #'); ax3.set_ylabel('Implied Vol (%)')
    ax3.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    n_trades = len(trades)
    tot_vol  = sum(t.qty for t in trades)
    mean_edge = np.mean(edges)
    fig.suptitle(
        f'Session Analytics — {symbol}  |  '
        f'{n_trades} trades  {tot_vol} contracts  '
        f'Mean edge: {mean_edge:+.4f}',
        color=TEXT, fontsize=11, y=1.01
    )

    out = '/mnt/user-data/outputs/market_analytics.png'
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: market_analytics.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────────────────────

HELP = """
Commands
────────
  add  <strike> <expiry_months> <call|put>   Add a new contract
  buy  <symbol>  <price>  <qty>              Submit a buy  limit order
  sell <symbol>  <price>  <qty>              Submit a sell limit order
  cancel <order_id>                          Cancel an open order
  book <symbol>                              Show order book depth
  chain                                      Show full option chain
  spot <price>                               Update underlying spot price
  plot <symbol>                              Plot session analytics
  trades [symbol]                            List all trades
  help                                       Show this message
  quit                                       Exit

Example session
───────────────
  add 100 12 call          → lists contract AAPL100C12M
  buy  AAPL100C12M 10.50 5
  sell AAPL100C12M 10.45 3
  book AAPL100C12M
  plot AAPL100C12M
"""


def run_cli(market: OptionsMarket):
    print(f"\n  Options Market Simulator — {market.underlying}")
    print(f"  Spot=${market.spot:.2f}  σ={market.sigma*100:.0f}%  "
          f"r={market.r*100:.0f}%")
    print("  Type 'help' for commands.\n")

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye."); break
        if not raw: continue
        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd in ('quit', 'exit', 'q'):
            print("  Goodbye."); break

        elif cmd == 'help':
            print(HELP)

        elif cmd == 'add' and len(parts) == 4:
            strike = float(parts[1])
            expiry = float(parts[2]) / 12
            opt    = parts[3].lower()
            c = market.add_contract(strike, expiry, opt)
            fv = c.fair_value(market.spot, market.r, market.sigma)
            print(f"  Added {c.symbol}  FV=${fv:.3f}")

        elif cmd in ('buy', 'sell') and len(parts) == 4:
            sym   = parts[1]
            price = float(parts[2])
            qty   = int(parts[3])
            market.submit_order(sym, cmd, price, qty)

        elif cmd == 'cancel' and len(parts) == 2:
            oid = int(parts[1])
            for book in market.books.values():
                if book.cancel(oid):
                    print(f"  Cancelled order #{oid}"); break
            else:
                print(f"  Order #{oid} not found or already filled.")

        elif cmd == 'book' and len(parts) == 2:
            market.show_book(parts[1])

        elif cmd == 'chain':
            market.show_chain()

        elif cmd == 'spot' and len(parts) == 2:
            market.update_spot(float(parts[1]))

        elif cmd == 'plot' and len(parts) == 2:
            plot_market_analytics(market, parts[1])

        elif cmd == 'trades':
            sym = parts[1] if len(parts) > 1 else None
            books = ([market.books[sym]] if sym and sym in market.books
                     else market.books.values())
            for book in books:
                if book.trades:
                    print(f"\n  Trades for {book.contract.symbol}:")
                    for t in book.trades:
                        print(f"    #{t.buy_id}×#{t.sell_id}  "
                              f"{t.qty}x @ ${t.price:.3f}  "
                              f"FV=${t.fair_value:.3f}  "
                              f"edge={t.edge:+.4f}")

        else:
            print(f"  Unknown command: '{raw}'  (type 'help')")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  AUTOMATED DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    """
    Automated demo: sets up a market, submits 40+ randomised orders,
    shows analytics, and generates plots — no user input needed.
    """
    print("\n" + "═"*58)
    print("  OPTIONS MARKET SIMULATOR — Automated Demo")
    print("═"*58)

    random.seed(42)
    market = OptionsMarket("AAPL", spot=150.0, r=0.05, sigma=0.20)

    # List two contracts: ATM call and 5%-OTM put
    c1 = market.add_contract(150, 0.5,  'call')   # 6-month ATM call
    c2 = market.add_contract(142, 0.5,  'put')    # 6-month OTM put
    fv1 = c1.fair_value(market.spot, market.r, market.sigma)
    fv2 = c2.fair_value(market.spot, market.r, market.sigma)
    print(f"\n  Listed contracts:")
    print(f"    {c1.symbol}  FV=${fv1:.3f}")
    print(f"    {c2.symbol}  FV=${fv2:.3f}")

    market.show_chain()

    # Seed the ATM call book with resting orders near fair value
    sym = c1.symbol
    print(f"\n  Seeding order book for {sym} (FV=${fv1:.3f})...")
    for i in range(8):
        bid_p = round(fv1 - random.uniform(0.05, 0.40), 2)
        ask_p = round(fv1 + random.uniform(0.05, 0.40), 2)
        qty   = random.randint(1, 5)
        market.books[sym].submit('buy',  bid_p, qty, market.spot,
                                 market.r, market.sigma)
        market.books[sym].submit('sell', ask_p, qty, market.spot,
                                 market.r, market.sigma)

    market.show_book(sym)

    # Submit crossing orders — some will match
    print(f"\n  Submitting 30 orders (some will cross and trade)...")
    for i in range(30):
        side  = random.choice(['buy', 'sell'])
        noise = random.uniform(-0.30, 0.30)
        price = round(fv1 + noise, 2)
        qty   = random.randint(1, 4)
        trades = market.submit_order(sym, side, price, qty)
        if not trades:
            pass  # resting order, no print

    market.show_book(sym)

    # Spot update — re-prices all contracts
    print(f"\n  Spot moves: $150 → $155 (+3.3%)")
    market.update_spot(155.0)
    market.show_chain()

    # Summary
    book = market.books[sym]
    if book.trades:
        edges  = [t.edge for t in book.trades]
        print(f"\n  Session summary for {sym}:")
        print(f"    Trades executed : {len(book.trades)}")
        print(f"    Total volume    : {sum(t.qty for t in book.trades)} contracts")
        print(f"    Mean edge       : {np.mean(edges):+.4f}")
        print(f"    Max edge        : {max(edges):+.4f}")
        print(f"    Min edge        : {min(edges):+.4f}")
        pct_above_fv = sum(1 for e in edges if e > 0) / len(edges) * 100
        print(f"    Trades above FV : {pct_above_fv:.0f}%")

    # Generate plots
    print(f"\n  Generating analytics plots...")
    plot_market_analytics(market, sym)
    print("\n  Demo complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options Market Simulator')
    parser.add_argument('--demo',       action='store_true',
                        help='Run automated demo with random orders')
    parser.add_argument('--underlying', default='AAPL',  type=str)
    parser.add_argument('--spot',       default=150.0,   type=float)
    parser.add_argument('--sigma',      default=0.20,    type=float)
    parser.add_argument('--rate',       default=0.05,    type=float)
    args = parser.parse_args()

    market = OptionsMarket(args.underlying, args.spot,
                           r=args.rate, sigma=args.sigma)

    if args.demo:
        run_demo()
    else:
        run_cli(market)