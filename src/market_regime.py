"""
ATG Market Regime v3.0 — Pre-scan gate: VIX, SPY trend, sector momentum, breadth.

All market data cached per calendar day to avoid redundant API calls.
Gate must pass all checks before any scan or position opening proceeds.
"""
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import pytz

# Suppress yfinance's internal ERROR/WARNING noise — it logs at ERROR level for
# transient API failures (rate limits, empty responses) before returning an empty
# DataFrame. Our code already handles empty DataFrames gracefully. We only want
# to see yfinance output at CRITICAL level (true library crashes).
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

from config.settings import (
    VIX_MAX,
    VIX_SPIKE_THRESHOLD,
    REQUIRE_SPY_UPTREND,
    MIN_MARKET_BREADTH,
    TOP_SECTORS_N,
)

log = logging.getLogger(__name__)
ET  = pytz.timezone("America/New_York")


def _yf_history(symbol: str, period: str, interval: str, retries: int = 2, delay: float = 1.5):
    """
    Fetch yfinance history with retry + backoff.

    yfinance returns an empty DataFrame (no exception) on transient API
    failures. We retry up to `retries` times before giving up.
    """
    import yfinance as yf
    for attempt in range(retries + 1):
        try:
            data = yf.Ticker(symbol).history(period=period, interval=interval)
            if not data.empty:
                return data
            if attempt < retries:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            log.debug("yfinance fetch error %s attempt %d: %s", symbol, attempt + 1, e)
            if attempt < retries:
                time.sleep(delay * (attempt + 1))
    return None   # exhausted retries

# ── Sector → ETF mapping ──────────────────────────────────────────────────────
SECTOR_ETFS: Dict[str, str] = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financials":             "XLF",
    "Energy":                 "XLE",
    "Consumer Discretionary": "XLY",
    "Industrials":            "XLI",
    "Materials":              "XLB",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Communication":          "XLC",
    "Consumer Staples":       "XLP",
}

# ── Session cache ─────────────────────────────────────────────────────────────
_cache: dict = {}
_cache_date: str = ""


def _refresh_cache() -> None:
    """Fetch market-wide regime data once per calendar day and cache the result."""
    global _cache, _cache_date
    today = datetime.now(ET).date().isoformat()
    if _cache_date == today and _cache:
        return

    log.info("Refreshing market regime cache for %s …", today)

    try:
        # ── SPY trend ─────────────────────────────────────────────────────────
        spy = _yf_history("SPY", "1y", "1d")
        if spy is not None and not spy.empty:
            closes       = spy["Close"].values
            ma50         = float(np.mean(closes[-50:])) if len(closes) >= 50  else float(closes[-1])
            ma200        = float(np.mean(closes[-200:])) if len(closes) >= 200 else float(closes[-1])
            spy4wk       = float((closes[-1] / closes[-22] - 1) * 100) if len(closes) >= 22 else 0.0
            _cache["spy_price"]    = float(closes[-1])
            _cache["spy_ma50"]     = ma50
            _cache["spy_ma200"]    = ma200
            _cache["spy_4wk_ret"]  = spy4wk
            _cache["spy_uptrend"]  = bool(closes[-1] > ma50 and closes[-1] > ma200)
            _cache["spy_momentum"] = spy4wk > 0
        else:
            _cache["spy_uptrend"]  = True   # fail-open

        # ── VIX ───────────────────────────────────────────────────────────────
        vix_hist = _yf_history("^VIX", "5d", "1d")
        if vix_hist is not None and len(vix_hist) >= 2:
            _cache["vix"]       = float(vix_hist["Close"].iloc[-1])
            _cache["vix_spike"] = float(vix_hist["Close"].iloc[-1] - vix_hist["Close"].iloc[-2])
        else:
            _cache["vix"]       = 20.0
            _cache["vix_spike"] = 0.0

        # ── Sector relative strength ──────────────────────────────────────────
        sector_scores: Dict[str, float] = {}
        spy_1m = _cache.get("spy_4wk_ret", 0.0)

        for sector, etf in SECTOR_ETFS.items():
            try:
                data = _yf_history(etf, "3mo", "1wk")
                if data is not None and len(data) >= 5:
                    ret_1m = float((data["Close"].values[-1] / data["Close"].values[-5] - 1) * 100)
                    ret_3m = float((data["Close"].values[-1] / data["Close"].values[-13] - 1) * 100) \
                             if len(data) >= 13 else ret_1m
                    sector_scores[sector] = (ret_1m - spy_1m) + (ret_3m / 3.0)
                else:
                    sector_scores[sector] = 0.0
            except Exception as sector_err:
                log.debug("Sector RS fetch failed for %s: %s", etf, sector_err)
                sector_scores[sector] = 0.0

        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        _cache["sector_scores"]  = sector_scores
        _cache["top_sectors"]    = [s[0] for s in sorted_sectors[:TOP_SECTORS_N]]
        _cache["sector_ranking"] = sorted_sectors

        # ── Market breadth (% of 50 major stocks above their 50-day MA) ───────
        breadth_universe = [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "V", "MA", "UNH",
            "LLY",  "XOM",  "CAT",  "AMD",  "TSLA",  "AVGO", "COST","PG", "JNJ","BAC",
            "GS",   "HD",   "ABBV", "MRK",  "CVX",   "PEP",  "KO",  "WMT","DIS","BRK-B",
            "CSCO", "INTC", "TXN",  "QCOM", "MU",    "AMAT", "LRCX","CRM","ADBE","NOW",
            "SNOW", "DDOG", "PANW", "CRWD", "NET",   "FTNT", "ZS",  "UBER","LYFT","COIN",
        ]
        above_50ma = 0
        checked    = 0
        failed     = 0
        for sym in breadth_universe:
            d = _yf_history(sym, "60d", "1d", retries=1, delay=0.5)
            if d is not None and len(d) >= 50:
                ma50 = float(np.mean(d["Close"].values[-50:]))
                if float(d["Close"].values[-1]) > ma50:
                    above_50ma += 1
                checked += 1
            else:
                failed += 1
                log.debug("Breadth: no data for %s (skipped)", sym)

        # Use total universe size as denominator so persistent fetch failures
        # don't inflate the breadth reading.
        # EXCEPTION: if ALL tickers failed (data outage), use neutral 60% default
        # rather than falsely reporting 0% and closing the regime gate.
        total = len(breadth_universe)
        if checked == 0 and failed == total:
            log.warning(
                "Breadth: ALL %d tickers failed to fetch — data outage, using neutral 60%% default",
                total,
            )
            _cache["market_breadth"] = 60.0
        else:
            _cache["market_breadth"] = (above_50ma / total) * 100.0
        if failed > 0:
            log.info("Breadth: %d/%d tickers fetched (%d failed — counted as below MA)",
                     checked, total, failed)

        _cache_date = today
        log.info(
            "Regime cache ready | VIX=%.1f spy_uptrend=%s top_sectors=%s breadth=%.0f%%",
            _cache.get("vix", 0), _cache.get("spy_uptrend"), _cache.get("top_sectors"), _cache.get("market_breadth", 0),
        )

    except Exception as e:
        log.warning("Regime cache refresh failed: %s — using safe defaults", e)
        _cache.setdefault("spy_uptrend",    True)
        _cache.setdefault("vix",            20.0)
        _cache.setdefault("vix_spike",      0.0)
        _cache.setdefault("top_sectors",    list(SECTOR_ETFS.keys())[:TOP_SECTORS_N])
        _cache.setdefault("market_breadth", 60.0)
        _cache.setdefault("sector_scores",  {})
        _cache.setdefault("sector_ranking", [])
        _cache_date = today


def get_regime() -> dict:
    """Return full regime snapshot (refreshed once per calendar day)."""
    _refresh_cache()
    return dict(_cache)


def is_market_open_for_trading() -> dict:
    """
    Master scan gate — all conditions must pass before trading.

    Returns:
        {"ok": bool, "reason": str}
    """
    _refresh_cache()
    vix       = _cache.get("vix",            20.0)
    vix_spike = _cache.get("vix_spike",       0.0)
    uptrend   = _cache.get("spy_uptrend",    True)
    breadth   = _cache.get("market_breadth", 60.0)

    if vix > VIX_MAX:
        return {"ok": False, "reason": f"VIX too high: {vix:.1f} > {VIX_MAX}"}

    if vix_spike > VIX_SPIKE_THRESHOLD:
        return {
            "ok": False,
            "reason": f"VIX spike: +{vix_spike:.1f} pts today (threshold: {VIX_SPIKE_THRESHOLD})",
        }

    if REQUIRE_SPY_UPTREND and not uptrend:
        return {"ok": False, "reason": "SPY below 50MA or 200MA — market in downtrend"}

    if breadth < MIN_MARKET_BREADTH:
        return {
            "ok": False,
            "reason": f"Market breadth too low: {breadth:.0f}% < {MIN_MARKET_BREADTH}%",
        }

    return {"ok": True, "reason": "All regime checks passed"}


def get_top_sectors() -> list:
    """Return the top-performing sectors from the regime cache."""
    _refresh_cache()
    return list(_cache.get("top_sectors", list(SECTOR_ETFS.keys())[:TOP_SECTORS_N]))


def get_sector_etf(sector: str) -> str:
    """Return the ETF ticker for a given sector name."""
    return SECTOR_ETFS.get(sector, "XLK")


def is_spy_deteriorating() -> bool:
    """
    True if SPY has broken below its 50-day MA (market deterioration signal).
    Used to trigger early exits on all long positions.
    """
    _refresh_cache()
    return not _cache.get("spy_uptrend", True)
