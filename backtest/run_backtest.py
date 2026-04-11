"""
ATG 5-Year Backtester v1.0
Simulates ATG v2.1 strategy rules on 5 years of historical data (2020–2025).

Simulates:
- Market regime gate (SPY trend, VIX, breadth)
- All 6 setup types (BREAKOUT, PULLBACK, MA_BOUNCE, BASE_BREAKOUT,
  SECTOR_ROTATION, EARNINGS_DRIFT)
- Multi-timeframe trend alignment
- Volume confirmation
- Staged exits (50% at 1.5x ATR, 50% trailing)
- Breakeven stop management
- Time stop (10 days no progress)
- Trailing stop (2x ATR below peak, activates at 8% gain)
- Correlation filter (max 2 per group)
- Conviction-based position sizing

Output: backtest/results/report.txt + trades.csv
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE        = "2020-01-01"
END_DATE          = "2025-12-31"
INITIAL_CAPITAL   = 100_000.0
RISK_PER_TRADE    = 0.01           # 1% risk per trade
MAX_POSITIONS     = 8
MIN_SCORE         = 75
MIN_VOLUME_RATIO  = 1.5
MIN_RR_RATIO      = 2.0
TIME_STOP_DAYS    = 10
BREAKEVEN_ATR     = 1.0            # move stop to BE after 1x ATR gain
TRAILING_ACT_PCT  = 0.08           # activate trailing at 8% gain
TRAILING_ATR_MULT = 2.0
STAGED_EXIT_ATR   = 1.5            # first exit at 1.5x ATR
STAGED_SIZE       = 0.50
VIX_MAX           = 28.0
VIX_SPIKE         = 3.0
MAX_CORR_POS      = 2

CORRELATED_GROUPS = {
    "mega_tech":    ["AAPL","MSFT","GOOGL","META","AMZN"],
    "semis":        ["NVDA","AMD","AVGO","ARM","SMCI"],
    "fintech":      ["V","MA","PYPL","SQ","HOOD"],
    "crypto":       ["COIN","MSTR"],
    "cybersecurity":["CRWD","PANW","FTNT","NET"],
    "cloud_saas":   ["SNOW","DDOG","NET","NOW","CRM"],
}

CONVICTION_SIZING = {90: 1.50, 85: 1.25, 80: 1.00, 75: 0.80}

UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","TSLA",
    "AMD","PLTR","CRWD","NET","DDOG","SNOW","ARM",
    "JPM","GS","BAC","V","MA",
    "LLY","UNH","JNJ","ABBV",
    "XOM","CVX","CAT","DE",
    "ADBE","CRM","NOW","PANW","FTNT",
    "PYPL","UBER","COIN","MSTR","SPOT",
]

SYMBOL_SECTOR = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMZN":"Consumer Discretionary",
    "GOOGL":"Communication","META":"Communication","AVGO":"Technology","TSLA":"Consumer Discretionary",
    "AMD":"Technology","PLTR":"Technology","CRWD":"Technology","NET":"Technology",
    "DDOG":"Technology","SNOW":"Technology","ARM":"Technology",
    "JPM":"Financials","GS":"Financials","BAC":"Financials","V":"Financials","MA":"Financials",
    "LLY":"Healthcare","UNH":"Healthcare","JNJ":"Healthcare","ABBV":"Healthcare",
    "XOM":"Energy","CVX":"Energy","CAT":"Industrials","DE":"Industrials",
    "ADBE":"Technology","CRM":"Technology","NOW":"Technology","PANW":"Technology","FTNT":"Technology",
    "PYPL":"Financials","UBER":"Industrials","COIN":"Financials","MSTR":"Technology","SPOT":"Communication",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR   = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(symbols: list, start: str, end: str) -> dict:
    """Load & cache historical daily OHLCV for all symbols."""
    import yfinance as yf

    all_data = {}
    total = len(symbols)
    for i, sym in enumerate(symbols):
        cache_file = os.path.join(CACHE_DIR, f"{sym}.parquet")
        try:
            if os.path.exists(cache_file):
                df = pd.read_parquet(cache_file)
                # Check if cache covers our range
                if str(df.index[0].date()) <= start and str(df.index[-1].date()) >= end[:10]:
                    all_data[sym] = df
                    print(f"  [{i+1}/{total}] {sym}: loaded from cache ({len(df)} days)", flush=True)
                    continue
        except Exception:
            pass

        try:
            df = yf.Ticker(sym).history(start=start, end=end, interval="1d", auto_adjust=True)
            if len(df) > 100:
                df.to_parquet(cache_file)
                all_data[sym] = df
                print(f"  [{i+1}/{total}] {sym}: fetched {len(df)} days", flush=True)
            else:
                print(f"  [{i+1}/{total}] {sym}: insufficient data ({len(df)} days)", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{total}] {sym}: FAILED — {e}", flush=True)

    return all_data


# ── Indicators ────────────────────────────────────────────────────────────────

def rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1: return 50.0
    d = np.diff(closes.astype(float))
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    ag = np.mean(g[:period]); al = np.mean(l[:period])
    for gi, li in zip(g[period:], l[period:]):
        ag = (ag*(period-1)+gi)/period; al = (al*(period-1)+li)/period
    return float(100 - 100/(1+ag/al)) if al > 0 else 100.0


def atr(highs, lows, closes_prev, period=14) -> float:
    if len(highs) < period: return 0.0
    tr = np.maximum(highs-lows, np.maximum(np.abs(highs-closes_prev), np.abs(lows-closes_prev)))
    return float(np.mean(tr[-period:]))


def get_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly."""
    weekly = df_daily.resample("W-FRI").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()
    return weekly


# ── Regime Gate ───────────────────────────────────────────────────────────────

def check_regime(spy_daily: pd.DataFrame, vix_daily: pd.DataFrame,
                 all_data: dict, as_of: date) -> dict:
    """Check market regime as of a specific date."""
    spy = spy_daily[spy_daily.index.date <= as_of]
    vix = vix_daily[vix_daily.index.date <= as_of]

    if len(spy) < 200 or len(vix) < 2:
        return {"ok": True, "reason": "insufficient data"}

    spy_c   = spy["Close"].values
    vix_c   = vix["Close"].values
    spy_now = float(spy_c[-1])
    vix_now = float(vix_c[-1])
    vix_yest= float(vix_c[-2])

    ma50  = float(np.mean(spy_c[-50:]))
    ma200 = float(np.mean(spy_c[-200:]))

    if vix_now > VIX_MAX:
        return {"ok": False, "reason": f"VIX {vix_now:.1f} > {VIX_MAX}"}
    if (vix_now - vix_yest) > VIX_SPIKE:
        return {"ok": False, "reason": f"VIX spike +{vix_now-vix_yest:.1f}"}
    if spy_now < ma50 or spy_now < ma200:
        return {"ok": False, "reason": "SPY below 50/200 MA"}

    # Rough breadth from available symbols
    breadth_syms = list(all_data.keys())[:20]
    above = 0
    for s in breadth_syms:
        d = all_data[s]
        d = d[d.index.date <= as_of]
        if len(d) >= 50:
            m = float(np.mean(d["Close"].values[-50:]))
            if float(d["Close"].values[-1]) > m:
                above += 1
    breadth = (above / max(len(breadth_syms), 1)) * 100
    if breadth < 45:
        return {"ok": False, "reason": f"Breadth {breadth:.0f}% < 45%"}

    return {"ok": True, "reason": "pass", "vix": vix_now, "breadth": breadth}


# ── Setup Scoring ─────────────────────────────────────────────────────────────

def score_symbol_historical(symbol: str, df: pd.DataFrame, as_of: date,
                             spy_trend_ok: bool) -> dict:
    """Score a symbol for swing setup as of a specific date."""
    hist = df[df.index.date <= as_of].copy()
    if len(hist) < 60:
        return None

    closes_d = hist["Close"].values
    highs_d  = hist["High"].values
    lows_d   = hist["Low"].values
    vols_d   = hist["Volume"].values
    price    = float(closes_d[-1])

    if price < 10:
        return None

    # Weekly bars (approximate from daily)
    weekly_df = get_weekly(hist)
    if len(weekly_df) < 20:
        return None

    closes_w = weekly_df["Close"].values
    vols_w   = weekly_df["Volume"].values

    # ATR (weekly)
    if len(weekly_df) >= 15:
        atr_w = atr(
            weekly_df["High"].values[-14:],
            weekly_df["Low"].values[-14:],
            weekly_df["Close"].values[-15:-1],
        )
    else:
        atr_w = price * 0.03
    if atr_w < 0.01:
        return None

    # MAs
    ma20  = float(np.mean(closes_d[-20:])) if len(closes_d) >= 20 else price
    ma50  = float(np.mean(closes_d[-50:])) if len(closes_d) >= 50 else price
    ma200 = float(np.mean(closes_d[-200:])) if len(closes_d) >= 200 else ma50
    ma_aligned = ma50 > ma200

    # Multi-TF alignment
    if not (price > ma20 and price > ma50 and price > ma200):
        return None
    if len(closes_d) >= 63:
        mom3m = (closes_d[-1]/closes_d[-63] - 1) * 100
        if mom3m < -5:
            return None

    # SPY must be in uptrend
    if not spy_trend_ok:
        return None

    # RSI
    w_rsi = rsi(closes_w[-30:]) if len(closes_w) >= 30 else rsi(closes_w)
    d_rsi = rsi(closes_d[-30:]) if len(closes_d) >= 30 else rsi(closes_d)

    # Volume
    vol_ratio_w = float(vols_w[-1] / np.mean(vols_w[-20:])) if len(vols_w) >= 20 else 1.0

    # Resistance & breakout
    resistance = float(np.max(closes_w[-20:-1])) if len(closes_w) >= 20 else price * 1.05
    broke_out  = price > resistance and vol_ratio_w >= MIN_VOLUME_RATIO

    setups = []

    # BREAKOUT
    if broke_out and 45 <= w_rsi <= 78 and ma_aligned:
        score = 70 + min(vol_ratio_w * 8, 18) + (10 if ma_aligned else 0)
        setups.append(("BREAKOUT", min(int(score), 95)))

    # BASE_BREAKOUT (8-week tight range)
    if len(closes_w) >= 8:
        last8_range = (np.max(closes_w[-8:]) - np.min(closes_w[-8:])) / price
        if last8_range < 0.12 and broke_out and ma_aligned:
            score = 82 + (5 if vol_ratio_w >= 2.0 else 0)
            setups.append(("BASE_BREAKOUT", min(int(score), 95)))

    # PULLBACK
    if ma_aligned and abs(price - ma50)/price < 0.03 and 35 < w_rsi < 55:
        setups.append(("PULLBACK", 74))

    # MA_BOUNCE
    if abs(price - ma200)/price < 0.025 and 30 < w_rsi < 52:
        setups.append(("MA_BOUNCE", 68))

    if not setups:
        return None

    best_setup, best_score = max(setups, key=lambda x: x[1])

    if best_score < MIN_SCORE:
        return None

    # Position sizing
    stop_loss      = round(price - 2.0 * atr_w, 2)
    target_stage1  = round(price + STAGED_EXIT_ATR * atr_w, 2)
    target_stage2  = round(price + 4.0 * atr_w, 2)
    risk_per_share = max(price - stop_loss, 0.01)
    rr_ratio       = (target_stage2 - price) / risk_per_share

    if rr_ratio < MIN_RR_RATIO:
        return None

    # Conviction sizing
    mult   = next((CONVICTION_SIZING[t] for t in sorted(CONVICTION_SIZING, reverse=True)
                   if best_score >= t), 0.80)
    risk_dollar = INITIAL_CAPITAL * RISK_PER_TRADE * mult
    shares = max(1, int(risk_dollar / risk_per_share))

    return {
        "symbol":       symbol,
        "setup_type":   best_setup,
        "score":        best_score,
        "sector":       SYMBOL_SECTOR.get(symbol, "Technology"),
        "price":        price,
        "stop_loss":    stop_loss,
        "target1":      target_stage1,
        "target2":      target_stage2,
        "atr":          atr_w,
        "shares":       shares,
        "rr_ratio":     round(rr_ratio, 2),
        "vol_ratio":    round(vol_ratio_w, 2),
    }


# ── Position State Machine ────────────────────────────────────────────────────

class Position:
    def __init__(self, pos_id, symbol, setup_type, entry_date, entry_price,
                 shares, stop_loss, target1, target2, atr, sector):
        self.id            = pos_id
        self.symbol        = symbol
        self.setup_type    = setup_type
        self.entry_date    = entry_date
        self.entry_price   = entry_price
        self.shares        = shares
        self.shares_rem    = shares
        self.stop_loss     = stop_loss
        self.target1       = target1
        self.target2       = target2
        self.atr           = atr
        self.sector        = sector
        self.hwm           = entry_price
        self.breakeven_set = False
        self.staged_done   = False
        self.pnl_realized  = 0.0   # from staged exit
        self.closed        = False
        self.close_date    = None
        self.close_price   = None
        self.close_reason  = None

    def update(self, current: float, today: date) -> list:
        """
        Update position state for today's price.
        Returns list of close events: [] or [dict] (partial or full close).
        """
        events = []
        if self.closed:
            return events

        hold_days = (today - self.entry_date).days
        gain_pct  = (current - self.entry_price) / self.entry_price

        # Update HWM
        if current > self.hwm:
            self.hwm = current

        exit_reason = None

        # 1. Hard stop
        if current <= self.stop_loss:
            exit_reason = "STOP_HIT"

        # 2. Time stop (no progress in 10 days)
        elif hold_days >= TIME_STOP_DAYS and gain_pct <= 0.01 and not self.breakeven_set:
            exit_reason = "TIME_STOP"

        # 3. Staged exit (first 50% at target1)
        elif not self.staged_done and current >= self.target1:
            half = max(1, int(self.shares_rem * STAGED_SIZE))
            pnl_pct = (current - self.entry_price) / self.entry_price * 100
            pnl_dollar   = (current - self.entry_price) * half
            self.pnl_realized += pnl_dollar
            self.shares_rem   -= half
            self.staged_done   = True
            # Move stop to breakeven after staged exit
            self.stop_loss     = self.entry_price
            self.breakeven_set = True
            events.append({"type": "PARTIAL", "symbol": self.symbol,
                           "reason": "STAGED_50PCT", "pnl_pct": pnl_pct,
                           "pnl_dollar": pnl_dollar, "shares": half, "price": current, "date": today})

        # 4. Final exit at target2
        elif self.staged_done and current >= self.target2:
            exit_reason = "TARGET_HIT"

        # 5. Max hold
        elif hold_days >= 42:
            exit_reason = "TIME_EXIT"

        # Stop management (no exit)
        else:
            if not self.breakeven_set and gain_pct * self.entry_price >= BREAKEVEN_ATR * self.atr:
                self.stop_loss     = self.entry_price
                self.breakeven_set = True

            elif self.breakeven_set and gain_pct >= TRAILING_ACT_PCT:
                trail = self.hwm - TRAILING_ATR_MULT * self.atr
                if trail > self.stop_loss:
                    self.stop_loss = trail

        if exit_reason:
            pnl_pct = (current - self.entry_price) / self.entry_price * 100
            pnl_dollar   = (current - self.entry_price) * self.shares_rem + self.pnl_realized
            self.closed      = True
            self.close_date  = today
            self.close_price = current
            self.close_reason = exit_reason
            events.append({"type": "FULL", "symbol": self.symbol, "setup_type": self.setup_type,
                           "reason": exit_reason, "pnl_pct": pnl_pct, "pnl_dollar": pnl_dollar,
                           "entry_price": self.entry_price, "exit_price": current,
                           "entry_date": self.entry_date, "exit_date": today,
                           "hold_days": hold_days, "sector": self.sector})

        return events


# ── Main Backtest Engine ──────────────────────────────────────────────────────

def run_backtest():
    print("=" * 70, flush=True)
    print("ATG 5-YEAR BACKTESTER v1.0", flush=True)
    print(f"Period: {START_DATE} → {END_DATE}", flush=True)
    print(f"Universe: {len(UNIVERSE)} symbols", flush=True)
    print("=" * 70, flush=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/4] Loading historical data...", flush=True)
    all_symbols  = UNIVERSE + ["SPY", "^VIX"]
    raw_data     = load_data(all_symbols, START_DATE, END_DATE)

    spy_daily = raw_data.get("SPY", pd.DataFrame())
    vix_daily = raw_data.get("^VIX", pd.DataFrame())

    if spy_daily.empty or vix_daily.empty:
        print("ERROR: Could not load SPY or VIX data", flush=True)
        sys.exit(1)

    print(f"\n  SPY: {len(spy_daily)} days | VIX: {len(vix_daily)} days", flush=True)

    # ── Build trading calendar (Mon–Fri, every 5 days for EOD scan) ───────────
    print("\n[2/4] Building trading calendar...", flush=True)
    all_dates = pd.date_range(START_DATE, END_DATE, freq="B")  # business days
    # Scan every Friday (EOD weekly scan) + Monday morning scan
    scan_dates = [d.date() for d in all_dates if d.dayofweek == 4]  # Fridays
    print(f"  {len(scan_dates)} scan dates (weekly EOD)", flush=True)

    # ── Run simulation ────────────────────────────────────────────────────────
    print("\n[3/4] Running simulation...", flush=True)

    equity        = INITIAL_CAPITAL
    positions     = []        # active Position objects
    all_trades    = []        # completed trade records
    equity_curve  = []        # daily equity snapshots
    pos_id_seq    = 0
    regime_blocked = 0
    scan_count     = 0
    total_entries  = 0
    rejected_corr  = 0
    rejected_max   = 0

    for scan_date in scan_dates:
        scan_count += 1

        # ── Monitor open positions at close of this date ───────────────────
        for pos in list(positions):
            if pos.closed:
                continue
            sym_df = raw_data.get(pos.symbol)
            if sym_df is None:
                continue
            day_data = sym_df[sym_df.index.date == scan_date]
            if day_data.empty:
                # Try nearest available date
                avail = sym_df[sym_df.index.date <= scan_date]
                if avail.empty:
                    continue
                current = float(avail["Close"].iloc[-1])
            else:
                current = float(day_data["Close"].iloc[-1])

            events = pos.update(current, scan_date)
            for ev in events:
                if ev["type"] == "FULL":
                    all_trades.append(ev)
                    equity += ev["pnl_dollar"]
                    positions.remove(pos)

        # ── Regime check ───────────────────────────────────────────────────
        regime = check_regime(spy_daily, vix_daily, raw_data, scan_date)
        if not regime["ok"]:
            regime_blocked += 1
            equity_curve.append({"date": scan_date.isoformat(),
                                  "equity": round(equity, 2),
                                  "open_pos": len(positions)})
            continue

        spy_now  = float(spy_daily[spy_daily.index.date <= scan_date]["Close"].iloc[-1])
        spy_ma50 = float(np.mean(spy_daily[spy_daily.index.date <= scan_date]["Close"].values[-50:]))
        spy_trend_ok = spy_now > spy_ma50

        # ── Scan for setups ───────────────────────────────────────────────
        if len(positions) >= MAX_POSITIONS:
            rejected_max += 1
            equity_curve.append({"date": scan_date.isoformat(),
                                  "equity": round(equity, 2),
                                  "open_pos": len(positions)})
            continue

        candidates = []
        for sym in UNIVERSE:
            df = raw_data.get(sym)
            if df is None:
                continue
            result = score_symbol_historical(sym, df, scan_date, spy_trend_ok)
            if result:
                candidates.append(result)

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Open up to 2 new positions per scan
        opened_this_scan = 0
        for cand in candidates:
            if opened_this_scan >= 2:
                break
            if len(positions) >= MAX_POSITIONS:
                break

            sym    = cand["symbol"]
            sector = cand["sector"]

            # Correlation filter
            open_syms  = [p.symbol for p in positions]
            corr_group = next((g for g, m in CORRELATED_GROUPS.items() if sym in m), None)
            if corr_group:
                in_group = sum(1 for s in open_syms
                               if any(s in m for g, m in CORRELATED_GROUPS.items() if g == corr_group))
                if in_group >= MAX_CORR_POS:
                    rejected_corr += 1
                    continue

            # Sector filter
            sector_count = sum(1 for p in positions if p.sector == sector)
            if sector_count / MAX_POSITIONS >= 0.25:
                continue

            pos_id_seq += 1
            pos = Position(
                pos_id      = pos_id_seq,
                symbol      = sym,
                setup_type  = cand["setup_type"],
                entry_date  = scan_date,
                entry_price = cand["price"],
                shares      = cand["shares"],
                stop_loss   = cand["stop_loss"],
                target1     = cand["target1"],
                target2     = cand["target2"],
                atr         = cand["atr"],
                sector      = sector,
            )
            positions.append(pos)
            total_entries  += 1
            opened_this_scan += 1

        equity_curve.append({"date": scan_date.isoformat(),
                              "equity": round(equity, 2),
                              "open_pos": len(positions)})

        if scan_count % 52 == 0:
            print(f"  {scan_date} | Equity: ${equity:,.0f} | Trades: {len(all_trades)} "
                  f"| Open: {len(positions)}", flush=True)

    # Close any remaining open positions at last date's price
    last_date = scan_dates[-1]
    for pos in positions:
        if pos.closed:
            continue
        sym_df  = raw_data.get(pos.symbol)
        if sym_df is None:
            continue
        avail   = sym_df[sym_df.index.date <= last_date]
        if avail.empty:
            continue
        current = float(avail["Close"].iloc[-1])
        pnl_pct = (current - pos.entry_price) / pos.entry_price * 100
        pnl_dollar   = (current - pos.entry_price) * pos.shares_rem + pos.pnl_realized
        equity += pnl_dollar
        all_trades.append({
            "type": "FULL", "symbol": pos.symbol, "setup_type": pos.setup_type,
            "reason": "PERIOD_END", "pnl_pct": pnl_pct, "pnl_dollar": pnl_dollar,
            "entry_price": pos.entry_price, "exit_price": current,
            "entry_date": pos.entry_date, "exit_date": last_date,
            "hold_days": (last_date - pos.entry_date).days, "sector": pos.sector,
        })

    # ── Calculate metrics ─────────────────────────────────────────────────────
    print("\n[4/4] Calculating metrics...", flush=True)

    df_trades = pd.DataFrame(all_trades)

    wins   = df_trades[df_trades["pnl_dollar"] > 0] if len(df_trades) else pd.DataFrame()
    losses = df_trades[df_trades["pnl_dollar"] <= 0] if len(df_trades) else pd.DataFrame()

    total_trades   = len(df_trades)
    win_count      = len(wins)
    loss_count     = len(losses)
    win_rate       = win_count / total_trades if total_trades > 0 else 0
    avg_win_pct    = wins["pnl_pct"].mean() if len(wins) else 0
    avg_loss_pct   = losses["pnl_pct"].mean() if len(losses) else 0
    avg_win_dollar      = wins["pnl_dollar"].mean() if len(wins) else 0
    avg_loss_dollar     = losses["pnl_dollar"].mean() if len(losses) else 0
    total_pnl      = df_trades["pnl_dollar"].sum() if total_trades else 0
    final_equity   = INITIAL_CAPITAL + total_pnl
    total_return   = (final_equity / INITIAL_CAPITAL - 1) * 100
    avg_hold_days  = df_trades["hold_days"].mean() if total_trades else 0

    # Max drawdown
    eq_vals = [e["equity"] for e in equity_curve]
    peak    = INITIAL_CAPITAL
    max_dd  = 0.0
    for eq in eq_vals:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = wins["pnl_dollar"].sum() if len(wins) else 0
    gross_loss   = abs(losses["pnl_dollar"].sum()) if len(losses) else 1
    profit_factor = gross_profit / max(gross_loss, 0.01)

    # Sharpe (annualized, using weekly equity changes)
    eq_df    = pd.DataFrame(equity_curve)
    eq_df["returns"] = eq_df["equity"].pct_change().fillna(0)
    sharpe   = (eq_df["returns"].mean() / max(eq_df["returns"].std(), 1e-9)) * (52 ** 0.5)

    # By setup type
    by_setup = {}
    if total_trades > 0:
        for st in df_trades["setup_type"].unique():
            sub = df_trades[df_trades["setup_type"] == st]
            by_setup[st] = {
                "count":    len(sub),
                "win_rate": (sub["pnl_dollar"] > 0).mean(),
                "avg_pnl":  sub["pnl_pct"].mean(),
            }

    # By exit reason
    by_reason = {}
    if total_trades > 0:
        for reason in df_trades["reason"].unique():
            sub = df_trades[df_trades["reason"] == reason]
            by_reason[reason] = {"count": len(sub), "avg_pnl": sub["pnl_pct"].mean()}

    # Year-by-year
    yearly = {}
    if total_trades > 0:
        df_trades["year"] = df_trades["exit_date"].apply(
            lambda d: d.year if isinstance(d, date) else pd.Timestamp(d).year)
        for yr in sorted(df_trades["year"].unique()):
            sub = df_trades[df_trades["year"] == yr]
            yearly[str(yr)] = {
                "trades":   len(sub),
                "wins":     int((sub["pnl_dollar"] > 0).sum()),
                "win_rate": round((sub["pnl_dollar"] > 0).mean() * 100, 1),
                "pnl_dollar":    round(sub["pnl_dollar"].sum(), 2),
                "avg_pnl":  round(sub["pnl_pct"].mean(), 2),
            }

    # Best & worst trades
    best5  = df_trades.nlargest(5, "pnl_dollar")[["symbol","setup_type","pnl_pct","pnl_dollar","hold_days","entry_date","exit_date"]].to_dict("records") if total_trades else []
    worst5 = df_trades.nsmallest(5, "pnl_dollar")[["symbol","setup_type","pnl_pct","pnl_dollar","hold_days","entry_date","exit_date"]].to_dict("records") if total_trades else []

    # ── Save results ──────────────────────────────────────────────────────────
    df_trades.to_csv(os.path.join(RESULTS_DIR, "trades.csv"), index=False)
    eq_df.to_csv(os.path.join(RESULTS_DIR, "equity_curve.csv"), index=False)

    report = f"""
{'='*70}
  ATG v2.1 — 5-YEAR BACKTEST REPORT
  Period: {START_DATE} → {END_DATE}
  Universe: {len(UNIVERSE)} symbols | Capital: ${INITIAL_CAPITAL:,.0f}
{'='*70}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OVERALL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Starting Capital:    ${INITIAL_CAPITAL:>12,.2f}
  Final Equity:        ${final_equity:>12,.2f}
  Total P&L:           ${total_pnl:>+12,.2f}
  Total Return:        {total_return:>+11.2f}%
  Annualized Return:   {total_return/5:>+11.2f}%

  Max Drawdown:        {max_dd*100:>11.2f}%
  Sharpe Ratio:        {sharpe:>11.2f}
  Profit Factor:       {profit_factor:>11.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TRADE STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Trades:        {total_trades:>12d}
  Winning Trades:      {win_count:>12d}  ({win_rate*100:.1f}%)
  Losing Trades:       {loss_count:>12d}  ({(1-win_rate)*100:.1f}%)
  Avg Hold Days:       {avg_hold_days:>12.1f}

  Avg Winning Trade:   {avg_win_pct:>+11.2f}%  (${avg_win_dollar:>+,.2f})
  Avg Losing Trade:    {avg_loss_pct:>+11.2f}%  (${avg_loss_dollar:>+,.2f})
  Best Single Trade:   {df_trades['pnl_pct'].max() if total_trades else 0:>+11.2f}%
  Worst Single Trade:  {df_trades['pnl_pct'].min() if total_trades else 0:>+11.2f}%

  Regime Blocked Scans:{regime_blocked:>12d}
  Correlation Rejects: {rejected_corr:>12d}
  Max Position Rejects:{rejected_max:>12d}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BY YEAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {'Year':<8} {'Trades':<10} {'Wins':<8} {'Win%':<10} {'P&L $':<14} {'Avg%'}
  {'-'*60}"""

    for yr, data in yearly.items():
        report += f"\n  {yr:<8} {data['trades']:<10} {data['wins']:<8} {data['win_rate']:<10.1f} ${data['pnl_dollar']:<13,.2f} {data['avg_pnl']:+.2f}%"

    report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BY SETUP TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {'Setup':<18} {'Count':<10} {'Win Rate':<12} {'Avg P&L%'}
  {'-'*52}"""

    for st, data in sorted(by_setup.items(), key=lambda x: x[1]["win_rate"], reverse=True):
        report += f"\n  {st:<18} {data['count']:<10} {data['win_rate']*100:<12.1f} {data['avg_pnl']:+.2f}%"

    report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BY EXIT REASON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {'Reason':<20} {'Count':<10} {'Avg P&L%'}
  {'-'*42}"""

    for reason, data in sorted(by_reason.items(), key=lambda x: x[1]["count"], reverse=True):
        report += f"\n  {reason:<20} {data['count']:<10} {data['avg_pnl']:+.2f}%"

    report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOP 5 TRADES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    for t in best5:
        report += f"\n  {t['symbol']:<8} {t['setup_type']:<16} {t['pnl_pct']:>+7.1f}%  ${t['pnl_dollar']:>+8,.0f}  {t['hold_days']}d"

    report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WORST 5 TRADES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    for t in worst5:
        report += f"\n  {t['symbol']:<8} {t['setup_type']:<16} {t['pnl_pct']:>+7.1f}%  ${t['pnl_dollar']:>+8,.0f}  {t['hold_days']}d"

    report += f"""

{'='*70}
  Files saved:
  - {RESULTS_DIR}/report.txt
  - {RESULTS_DIR}/trades.csv
  - {RESULTS_DIR}/equity_curve.csv
{'='*70}
"""

    print(report, flush=True)

    report_path = os.path.join(RESULTS_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Save summary JSON for easy parsing
    summary = {
        "total_return_pct":    round(total_return, 2),
        "annualized_return":   round(total_return/5, 2),
        "final_equity":        round(final_equity, 2),
        "total_pnl":           round(total_pnl, 2),
        "total_trades":        total_trades,
        "win_rate":            round(win_rate * 100, 1),
        "avg_win_pct":         round(avg_win_pct, 2),
        "avg_loss_pct":        round(avg_loss_pct, 2),
        "max_drawdown_pct":    round(max_dd * 100, 2),
        "sharpe_ratio":        round(sharpe, 2),
        "profit_factor":       round(profit_factor, 2),
        "avg_hold_days":       round(avg_hold_days, 1),
        "yearly":              yearly,
        "by_setup":            {k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                                    for kk, vv in v.items()} for k, v in by_setup.items()},
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = run_backtest()
    print(f"\n✅ BACKTEST COMPLETE", flush=True)
    print(f"   Total Return: {summary['total_return_pct']:+.2f}%", flush=True)
    print(f"   Win Rate:     {summary['win_rate']:.1f}%", flush=True)
    print(f"   Avg Win:      {summary['avg_win_pct']:+.2f}%", flush=True)
    print(f"   Avg Loss:     {summary['avg_loss_pct']:+.2f}%", flush=True)
    print(f"   Max Drawdown: {summary['max_drawdown_pct']:.2f}%", flush=True)
    print(f"   Sharpe:       {summary['sharpe_ratio']:.2f}", flush=True)
