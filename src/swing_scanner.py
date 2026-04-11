"""
ATG Swing Scanner v3.0 — High-probability setup detection with full regime gate.

Filters applied (in order):
  1. Market regime gate (VIX, SPY trend, breadth)
  2. Multi-timeframe trend alignment (daily + weekly + monthly)
  3. Sector must be in top N by relative strength
  4. Earnings avoidance (skip if ≤ EARNINGS_BUFFER_DAYS to earnings)
  5. Minimum setup score threshold
  6. Minimum R:R ratio enforcement
"""
import logging
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict
import pytz

# Suppress yfinance's internal transient-error noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

from config.settings import (
    MIN_SETUP_SCORE,
    MIN_VOLUME_RATIO_BO,
    MIN_RR_RATIO,
    EARNINGS_BUFFER_DAYS,
    REQUIRE_SECTOR_RS,
    STAGED_EXIT_ATR,
    INITIAL_CAPITAL,
    RISK_PER_TRADE_PCT,
)
from src.market_regime import (
    is_market_open_for_trading,
    get_regime,
    get_sector_etf,
)

log = logging.getLogger(__name__)
ET  = pytz.timezone("America/New_York")

# ── Scan universe ─────────────────────────────────────────────────────────────
SWING_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
    "AMD",  "PLTR", "CRWD", "NET",  "DDOG",  "SNOW", "ARM",  "MSTR",
    "JPM",  "GS",   "BAC",  "V",    "MA",
    "LLY",  "UNH",  "JNJ",  "ABBV",
    "XOM",  "CVX",  "CAT",  "DE",
    "HOOD", "SMCI", "MELI", "SPOT", "COIN",
    "PYPL", "SQ",   "ROKU", "UBER", "LYFT",
    "ADBE", "CRM",  "NOW",  "PANW", "FTNT",
]

SYMBOL_SECTOR: Dict[str, str] = {
    "AAPL": "Technology",      "MSFT": "Technology",      "NVDA": "Technology",
    "AMZN": "Consumer Discretionary", "GOOGL": "Communication", "META": "Communication",
    "AVGO": "Technology",      "TSLA": "Consumer Discretionary", "AMD": "Technology",
    "PLTR": "Technology",      "CRWD": "Technology",      "NET": "Technology",
    "DDOG": "Technology",      "SNOW": "Technology",      "ARM": "Technology",
    "MSTR": "Technology",      "JPM": "Financials",        "GS": "Financials",
    "BAC":  "Financials",      "V":   "Financials",        "MA": "Financials",
    "LLY":  "Healthcare",      "UNH": "Healthcare",        "JNJ": "Healthcare",
    "ABBV": "Healthcare",      "XOM": "Energy",            "CVX": "Energy",
    "CAT":  "Industrials",     "DE":  "Industrials",
    "HOOD": "Financials",      "SMCI": "Technology",       "MELI": "Consumer Discretionary",
    "SPOT": "Communication",   "COIN": "Financials",
    "PYPL": "Financials",      "SQ":  "Financials",        "ROKU": "Communication",
    "UBER": "Industrials",     "LYFT": "Industrials",
    "ADBE": "Technology",      "CRM": "Technology",        "NOW": "Technology",
    "PANW": "Technology",      "FTNT": "Technology",
}


# ── Technical helpers ─────────────────────────────────────────────────────────

def _rsi(prices: np.ndarray, period: int = 14) -> float:
    """Wilder RSI from a price array. Returns 50.0 on insufficient data."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = float(np.mean(gains[:period]))
    al = float(np.mean(losses[:period]))
    for g, l in zip(gains[period:], losses[period:]):
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + l) / period
    return 100.0 - 100.0 / (1.0 + ag / al) if al > 0 else 100.0


def _atr(highs: np.ndarray, lows: np.ndarray, closes_prev: np.ndarray, period: int = 14) -> float:
    """Average True Range over the last `period` bars."""
    if len(highs) < period:
        return 0.0
    tr  = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - closes_prev), np.abs(lows - closes_prev)),
    )
    return float(np.mean(tr[-period:]))


def _days_to_earnings(ticker_obj) -> Optional[int]:
    """
    Return days until next earnings event, or None if unavailable.

    Args:
        ticker_obj: yfinance Ticker instance.

    Returns:
        Non-negative int, or None.
    """
    try:
        cal = ticker_obj.calendar
        if not isinstance(cal, dict):
            return None
        dates = cal.get("Earnings Date", [])
        if not dates:
            return None
        import pandas as pd
        next_date = pd.Timestamp(dates[0]).to_pydatetime()
        return max(0, (next_date.date() - datetime.now(ET).date()).days)
    except Exception as e:
        log.debug("Days-to-earnings lookup failed: %s", e)
        return None


def _multitf_trend_ok(daily_closes: np.ndarray) -> bool:
    """
    Multi-timeframe trend alignment check using daily closes.

    Requires:
      - Price > 20-day MA
      - Price > 50-day MA
      - Price > 200-day MA (if available)
      - 13-week (63-day) return > -5%

    Args:
        daily_closes: recent daily closing prices.

    Returns:
        True if all conditions are met.
    """
    if len(daily_closes) < 50:
        return False
    price = float(daily_closes[-1])
    ma20  = float(np.mean(daily_closes[-20:]))
    ma50  = float(np.mean(daily_closes[-50:]))
    ma200 = float(np.mean(daily_closes[-200:])) if len(daily_closes) >= 200 else ma50
    mom3m = float((daily_closes[-1] / daily_closes[-63] - 1) * 100) if len(daily_closes) >= 63 else 0.0
    return price > ma20 and price > ma50 and price > ma200 and mom3m > -5.0


def score_symbol(symbol: str, top_sectors: list, regime: dict) -> Optional[dict]:
    """
    Score a single symbol against all v3.0 entry filters.

    Returns the best setup dict if it qualifies, or None if filtered out.

    Args:
        symbol      : ticker to evaluate.
        top_sectors : list of sectors currently in the top N by RS.
        regime      : pre-fetched regime snapshot.

    Returns:
        Setup dict or None.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        weekly = ticker.history(period="52wk", interval="1wk")
        daily  = ticker.history(period="300d", interval="1d")

        if weekly.empty or len(daily) < 50:
            return None

        price      = float(daily["Close"].iloc[-1])
        closes_w   = weekly["Close"].values
        closes_d   = daily["Close"].values
        volume_w   = weekly["Volume"].values
        volume_d   = daily["Volume"].values

        if price < 10.0 or len(closes_d) < 50:
            return None

        # Weekly ATR
        if len(weekly) >= 15:
            atr_w = _atr(
                weekly["High"].values[-15:],
                weekly["Low"].values[-15:],
                weekly["Close"].values[-16:-1],
            )
        else:
            atr_w = price * 0.03
        if atr_w < 0.01:
            return None

        # Filter 1: multi-timeframe trend alignment
        if not _multitf_trend_ok(closes_d):
            log.debug("Skip %s: multi-TF trend misaligned", symbol)
            return None

        # Filter 2: sector must be in top N
        sector = SYMBOL_SECTOR.get(symbol, "Technology")
        if REQUIRE_SECTOR_RS and top_sectors and sector not in top_sectors:
            log.debug("Skip %s: sector %s not in top sectors", symbol, sector)
            return None

        # Filter 3: earnings avoidance
        dte = _days_to_earnings(ticker)
        if dte is not None and dte <= EARNINGS_BUFFER_DAYS:
            log.debug("Skip %s: earnings in %d days", symbol, dte)
            return None

        # MA levels
        ma50  = float(np.mean(closes_d[-50:]))
        ma200 = float(np.mean(closes_d[-200:])) if len(closes_d) >= 200 else ma50
        ma_aligned = ma50 > ma200

        # RSI
        w_rsi = _rsi(closes_w[-30:])
        d_rsi = _rsi(closes_d[-30:])

        # Volume
        vol_ratio_w = float(volume_w[-1] / np.mean(volume_w[-20:])) if len(volume_w) >= 20 else 1.0
        vol_ratio_d = float(volume_d[-1] / np.mean(volume_d[-20:])) if len(volume_d) >= 20 else 1.0

        # Resistance & breakout confirmation
        resistance = float(np.max(closes_w[-20:-1])) if len(closes_w) >= 20 else price * 1.05
        broke_out  = price > resistance and vol_ratio_w >= MIN_VOLUME_RATIO_BO

        # ── Setup detection ───────────────────────────────────────────────────
        setups = []

        # BREAKOUT: above resistance with volume confirmation
        if broke_out and 45 <= w_rsi <= 78 and ma_aligned:
            score = 70
            score += min(int(vol_ratio_w * 8), 18)
            score += 10 if ma_aligned else 0
            score += 5  if d_rsi > 55 else 0
            setups.append(("BREAKOUT", min(score, 95)))

        # BASE_BREAKOUT: 8+ weeks of tight compression then breakout
        if len(closes_w) >= 8:
            last8_range = (float(np.max(closes_w[-8:])) - float(np.min(closes_w[-8:]))) / price
            if last8_range < 0.12 and broke_out and ma_aligned:
                score = 82 + (5 if vol_ratio_w >= 2.0 else 0)
                setups.append(("BASE_BREAKOUT", min(score, 95)))

        # PULLBACK: retest of 50MA in established uptrend
        if ma_aligned and abs(price - ma50) / price < 0.03 and 35 < w_rsi < 55:
            score = 72 + (8 if d_rsi < 50 else 0)
            setups.append(("PULLBACK", min(score, 88)))

        # MA_BOUNCE: bouncing off 200MA with oversold signal
        if abs(price - ma200) / price < 0.025 and 30 < w_rsi < 52:
            setups.append(("MA_BOUNCE", 68))

        # SECTOR_ROTATION: sector just entered top tier with strong RS
        sector_score_val = regime.get("sector_scores", {}).get(sector, 0.0)
        ranking          = regime.get("sector_ranking", [])
        sector_rank      = next(
            (i + 1 for i, (s, _) in enumerate(ranking) if s == sector), 11
        )
        if sector_rank <= 2 and sector_score_val > 3.0 and w_rsi < 65 and ma_aligned:
            score = 75 + min(int(sector_score_val * 2), 15)
            setups.append(("SECTOR_ROTATION", min(score, 90)))

        # EARNINGS_DRIFT: strong trend stock well ahead of earnings
        if dte is not None and dte > 20 and ma_aligned and w_rsi > 55 and vol_ratio_w > 1.2:
            score = 72 + (8 if dte > 40 else 0)
            setups.append(("EARNINGS_DRIFT", min(score, 88)))

        if not setups:
            return None

        best_setup, best_score = max(setups, key=lambda x: x[1])

        # Filter 4: minimum score
        if best_score < MIN_SETUP_SCORE:
            log.debug("Skip %s: score %d < %d", symbol, best_score, MIN_SETUP_SCORE)
            return None

        # ── Position sizing: 1% risk rule ──────────────────────────────────────
        stop_loss     = round(price - 2.0 * atr_w, 2)
        target_stage1 = round(price + STAGED_EXIT_ATR * atr_w, 2)
        target_stage2 = round(price + 4.0 * atr_w, 2)
        risk_per_share = max(price - stop_loss, 0.01)
        risk_dollars   = INITIAL_CAPITAL * RISK_PER_TRADE_PCT
        shares         = max(1, int(risk_dollars / risk_per_share))

        # Filter 5: minimum R:R ratio
        rr_ratio = (target_stage2 - price) / risk_per_share
        if rr_ratio < MIN_RR_RATIO:
            log.debug("Skip %s: R:R %.2f < %.2f", symbol, rr_ratio, MIN_RR_RATIO)
            return None

        return {
            "symbol":            symbol,
            "setup_type":        best_setup,
            "score":             int(best_score),
            "sector":            sector,
            "price":             price,
            "ma50":              round(ma50, 2),
            "ma200":             round(ma200, 2),
            "weekly_rsi":        round(w_rsi, 1),
            "daily_rsi":         round(d_rsi, 1),
            "atr_weekly":        round(atr_w, 2),
            "atr_pct":           round(atr_w / price * 100, 2),
            "volume_ratio":      round(vol_ratio_w, 2),
            "stop_loss":         stop_loss,
            "target_price":      target_stage1,
            "target_stage2":     target_stage2,
            "shares":            shares,
            "ma_aligned":        ma_aligned,
            "broke_out":         broke_out,
            "days_to_earnings":  dte,
            "rr_ratio":          round(rr_ratio, 2),
        }

    except Exception as e:
        log.debug("Score failed for %s: %s", symbol, e)
        return None


def run_swing_scan(
    universe: Optional[List[str]] = None,
    min_score: Optional[int] = None,
    top_n: int = 5,
) -> dict:
    """
    Full v3.0 scan with regime gate, sector filter, and scoring.

    Args:
        universe  : list of symbols to scan (defaults to SWING_UNIVERSE).
        min_score : minimum setup score (defaults to MIN_SETUP_SCORE setting).
        top_n     : maximum number of setups to return.

    Returns:
        {
          "ok"     : bool — False if regime gate blocked the scan,
          "reason" : str,
          "setups" : list of setup dicts sorted by score descending,
        }
    """
    if min_score is None:
        min_score = MIN_SETUP_SCORE

    # Step 1: regime gate
    gate = is_market_open_for_trading()
    if not gate["ok"]:
        log.warning("Market regime gate CLOSED: %s", gate["reason"])
        return {"ok": False, "reason": gate["reason"], "setups": []}

    # Step 2: top sectors
    regime      = get_regime()
    top_sectors = regime.get("top_sectors", [])

    if universe is None:
        universe = SWING_UNIVERSE

    log.info(
        "ATG v3.0 swing scan | %d symbols | min_score=%d | top_sectors=%s",
        len(universe), min_score, top_sectors,
    )

    results: List[dict] = []
    for sym in universe:
        setup = score_symbol(sym, top_sectors, regime)
        if setup and setup["score"] >= min_score:
            results.append(setup)
            log.info(
                "  ✅ %s | %s | score=%d | R:R=%.1f | DTE=%s",
                sym, setup["setup_type"], setup["score"],
                setup["rr_ratio"], setup.get("days_to_earnings"),
            )
        else:
            log.debug("  — %s: filtered", sym)

    results.sort(key=lambda x: x["score"], reverse=True)
    log.info(
        "Scan complete: %d qualifying setups (top %d returned)",
        len(results), min(top_n, len(results)),
    )
    return {"ok": True, "reason": "scan complete", "setups": results[:top_n]}
