"""
ATG Context Builder v3.0 — Fully populated 32-dim swing context vector.

All 32 features populated from yfinance + FRED + regime cache.
Returns a float32 numpy array clipped to [0, 1] for LinUCB compatibility.
"""
import numpy as np
import logging
import requests
from datetime import datetime
from typing import Optional
import pytz

# Suppress yfinance's internal transient-error noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

def _fetch_bars_alpaca(symbol: str, timeframe: str, limit: int) -> "pd.DataFrame":
    """
    Fetch historical bars from Alpaca Data API.
    Returns DataFrame with Open/High/Low/Close/Volume columns (matching yfinance format).
    Falls back to empty DataFrame on failure.
    """
    import os, pandas as pd
    api_key    = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not api_secret:
        return pd.DataFrame()
    try:
        resp = requests.get(
            f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret},
            params={"timeframe": timeframe, "limit": limit, "adjustment": "all"},
            timeout=10,
        )
        if resp.status_code != 200:
            return pd.DataFrame()
        bars = resp.json().get("bars", [])
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "Date"})
        df["Close"] = df["Close"].astype(float)
        df["Volume"] = df["Volume"].astype(float)
        return df
    except Exception as _e:
        log.debug("Alpaca bars fetch failed for %s: %s", symbol, _e)
        return pd.DataFrame()


logging.getLogger("peewee").setLevel(logging.CRITICAL)

from config.settings import CONTEXT_DIM, FRED_API_KEY

log = logging.getLogger(__name__)
ET  = pytz.timezone("America/New_York")

SECTOR_ETFS = {
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

_yield_cache: dict = {}
_yield_cache_date: str = ""


def _get_yield_spread() -> float:
    """
    Fetch 10Y – 2Y Treasury yield spread from FRED.
    Cached once per calendar day.

    Returns:
        Spread in percentage points (e.g. 0.5 for 50 bps).
    """
    global _yield_cache, _yield_cache_date
    today = datetime.now(ET).date().isoformat()
    if _yield_cache_date == today and "spread" in _yield_cache:
        return _yield_cache["spread"]

    if not FRED_API_KEY:
        return 0.5

    try:
        def _fred_latest(series_id: str) -> Optional[float]:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id":  series_id,
                    "api_key":    FRED_API_KEY,
                    "file_type":  "json",
                    "limit":      5,
                    "sort_order": "desc",
                },
                timeout=15,
            )
            r.raise_for_status()
            for obs in r.json().get("observations", []):
                try:
                    return float(obs["value"])
                except (ValueError, KeyError):
                    continue
            return None

        y10 = _fred_latest("DGS10")
        y2  = _fred_latest("DGS2")
        if y10 is not None and y2 is not None:
            spread = y10 - y2
            _yield_cache      = {"spread": spread}
            _yield_cache_date = today
            return spread
    except requests.RequestException as e:
        log.debug("FRED yield fetch failed: %s", e)

    return 0.5  # neutral fallback


def _safe(val: float, lo: float = 0.0, hi: float = 1.0, default: float = 0.5) -> float:
    """
    Min-max scale val ∈ [lo, hi] → [0, 1]; clamp and handle NaN/Inf.

    Args:
        val    : raw value.
        lo     : expected lower bound.
        hi     : expected upper bound.
        default: returned when val is not finite.

    Returns:
        Scaled float in [0.0, 1.0].
    """
    try:
        v = float(val)
        if not np.isfinite(v):
            return default
        return float(np.clip((v - lo) / (hi - lo + 1e-9), 0.0, 1.0))
    except (TypeError, ValueError):
        return default


def _rsi(prices: np.ndarray, period: int = 14) -> float:
    """
    Wilder RSI from a 1-D price array.

    Args:
        prices : closing prices (at least period + 1 values).
        period : RSI lookback.

    Returns:
        RSI in [0, 100].
    """
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


def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Exponential moving average of array arr with period n.

    Args:
        arr: input 1-D array.
        n  : EMA period.

    Returns:
        1-D array of same length as arr.
    """
    arr    = np.asarray(arr, dtype=float)
    result = np.zeros_like(arr)
    result[0] = arr[0]
    k = 2.0 / (n + 1)
    for i in range(1, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1 - k)
    return result


def _fear_greed_proxy(
    vix: float,
    vix_ma20: float,
    spy_mom: float,
    breadth: float = 60.0,
    spy_vs_ma200: float = 1.0,
) -> float:
    """
    Multi-factor Fear & Greed proxy scaled to [0, 1] (0 = extreme fear, 1 = greed).

    Factors and weights:
      1. VIX vs 20-day MA  (25%)
      2. VIX absolute level (20%)
      3. SPY 4-week momentum (20%)
      4. Market breadth % above 50MA (20%)
      5. SPY price vs 200MA (15%)

    Args:
        vix        : current VIX level.
        vix_ma20   : 20-day VIX moving average.
        spy_mom    : SPY 4-week % return.
        breadth    : % of stocks above 50-day MA.
        spy_vs_ma200: SPY price / 200-day MA ratio.

    Returns:
        Composite score in [0.0, 1.0].
    """
    vix_vs_ma   = 1.0 if vix < vix_ma20 else 0.0
    vix_level_s = 1.0 - _safe(vix, 10, 45)
    spy_mom_s   = _safe(spy_mom, -15, 15)
    breadth_s   = _safe(breadth, 20, 80)
    spy_ma200_s = _safe(spy_vs_ma200, 0.85, 1.15)
    composite   = (
        0.25 * vix_vs_ma
        + 0.20 * vix_level_s
        + 0.20 * spy_mom_s
        + 0.20 * breadth_s
        + 0.15 * spy_ma200_s
    )
    return float(np.clip(composite, 0.0, 1.0))


def build_context(
    symbol: str,
    sector: str = "Technology",
    regime: Optional[dict] = None,
) -> np.ndarray:
    """
    Construct a fully populated 32-dimensional context vector for the bandit.

    Feature order (matches CONTEXT_FEATURES in settings.py):
      [0]  weekly_rsi            [1]  monthly_rsi
      [2]  weekly_macd_hist      [3]  atr_ratio
      [4]  volume_ratio          [5]  price_to_50ma
      [6]  price_to_200ma        [7]  ma_alignment
      [8]  breakout_strength     [9]  price_mom_4wk
      [10] price_mom_12wk        [11] sector_rs_1m
      [12] sector_rs_3m          [13] sector_momentum
      [14] days_to_earnings      [15] earnings_surprise
      [16] revenue_growth        [17] earnings_growth
      [18] spy_weekly_trend      [19] vix_level
      [20] fear_greed            [21] market_breadth
      [22] yield_curve           [23] institutional_own
      [24] short_interest        [25] insider_activity
      [26] volume_accum          [27] debt_to_equity
      [28] profit_margin         [29] day_of_week
      [30] week_of_month         [31] market_cap_tier

    Args:
        symbol : ticker symbol.
        sector : sector name for ETF relative-strength lookup.
        regime : pre-fetched regime dict (avoids redundant refresh).

    Returns:
        Float32 numpy array of shape (CONTEXT_DIM,) with values in [0, 1].
    """
    ctx = np.full(CONTEXT_DIM, 0.5, dtype=np.float32)

    try:
        import yfinance as yf

        if regime is None:
            try:
                from src.market_regime import get_regime
                regime = get_regime()
            except Exception as regime_err:
                log.debug("Could not fetch regime in context builder: %s", regime_err)
                regime = {}

        # Alpaca-first price data; yfinance fallback for earnings/fundamentals
        weekly  = _fetch_bars_alpaca(symbol, "1Week", 56)
        daily   = _fetch_bars_alpaca(symbol, "1Day",  300)
        monthly = _fetch_bars_alpaca(symbol, "1Month", 24)

        # Fallback to yfinance if Alpaca unavailable
        if weekly.empty or daily.empty:
            log.debug("Alpaca bars empty for %s — falling back to yfinance", symbol)
            ticker  = yf.Ticker(symbol)
            weekly  = ticker.history(period="52wk", interval="1wk")
            daily   = ticker.history(period="300d", interval="1d")
            monthly = ticker.history(period="24mo", interval="1mo")
        else:
            ticker = None  # Alpaca succeeded — yfinance only needed for earnings/fundamentals

        if weekly.empty or daily.empty:
            log.warning("No price data for %s (Alpaca + yfinance both failed)", symbol)
            return ctx

        closes_w = weekly["Close"].values
        closes_d = daily["Close"].values
        volume_w = weekly["Volume"].values
        volume_d = daily["Volume"].values
        price    = float(closes_d[-1])

        # [0] Weekly RSI
        ctx[0] = _safe(_rsi(closes_w[-30:]), 0, 100)

        # [1] Monthly RSI
        if not monthly.empty:
            ctx[1] = _safe(_rsi(monthly["Close"].values[-20:]), 0, 100)

        # [2] Weekly MACD histogram
        if len(closes_w) >= 26:
            macd_line = _ema(closes_w, 12) - _ema(closes_w, 26)
            signal    = _ema(macd_line, 9)
            hist_val  = float(macd_line[-1] - signal[-1])
            ctx[2]    = _safe(hist_val, -price * 0.05, price * 0.05)

        # [3] ATR ratio (weekly ATR / price)
        atr = price * 0.03  # fallback
        if len(weekly) >= 15:
            h   = weekly["High"].values[-14:]
            l   = weekly["Low"].values[-14:]
            c   = weekly["Close"].values[-15:-1]
            tr  = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
            atr = float(np.mean(tr))
            ctx[3] = _safe(atr / price, 0, 0.15)

        # [4] Volume ratio (last week vs 20-week avg)
        if len(volume_w) >= 20:
            ctx[4] = _safe(float(volume_w[-1]) / float(np.mean(volume_w[-20:])), 0, 3.0)

        # [5-7] MA structure
        if len(closes_d) >= 50:
            ma50  = float(np.mean(closes_d[-50:]))
            ma200 = float(np.mean(closes_d[-200:])) if len(closes_d) >= 200 else ma50
            ctx[5] = _safe(price / ma50,  0.7, 1.3)
            ctx[6] = _safe(price / ma200, 0.7, 1.5)
            ctx[7] = 1.0 if ma50 > ma200 else 0.0

        # [8] Breakout strength (price vs 20-week resistance)
        if len(closes_w) >= 20:
            resistance = float(np.max(closes_w[-20:-1]))
            ctx[8]     = _safe((price - resistance) / (atr + 1e-9), -2, 2)

        # [9-10] Price momentum
        if len(closes_w) >= 13:
            ctx[9]  = _safe(float((closes_w[-1] / closes_w[-5]  - 1) * 100), -20, 20)
            ctx[10] = _safe(float((closes_w[-1] / closes_w[-13] - 1) * 100), -40, 40)

        # [11-13] Sector relative strength
        sector_etf = SECTOR_ETFS.get(sector, "XLK")
        sec_scores = regime.get("sector_scores", {})
        if sector in sec_scores:
            rs_1m  = float(sec_scores[sector])
            ctx[11] = _safe(rs_1m, -10, 10)
            ctx[12] = _safe(rs_1m, -20, 20)
            ctx[13] = 1.0 if rs_1m > 0 else 0.0
        else:
            try:
                sec_d  = yf.Ticker(sector_etf).history(period="13wk", interval="1wk")
                spy_d  = yf.Ticker("SPY").history(period="13wk", interval="1wk")
                if len(sec_d) >= 5 and len(spy_d) >= 5:
                    s1m    = float((sec_d["Close"].values[-1] / sec_d["Close"].values[-5] - 1) * 100)
                    p1m    = float((spy_d["Close"].values[-1] / spy_d["Close"].values[-5] - 1) * 100)
                    ctx[11] = _safe(s1m - p1m, -10, 10)
                    ctx[12] = _safe(s1m - p1m, -20, 20)
                    ctx[13] = 1.0 if s1m > p1m else 0.0
            except Exception as sec_err:
                log.debug("Sector RS fallback failed for %s: %s", sector_etf, sec_err)

        # [14] Days to next earnings
        try:
            cal = ticker.calendar
            if isinstance(cal, dict) and "Earnings Date" in cal:
                import pandas as pd
                dates = cal["Earnings Date"]
                if dates:
                    next_e = pd.Timestamp(dates[0]).to_pydatetime()
                    dte    = max(0, (next_e.date() - datetime.now(ET).date()).days)
                    ctx[14] = _safe(dte, 0, 60)
        except Exception as cal_err:
            log.debug("Earnings calendar failed for %s: %s", symbol, cal_err)
            ctx[14] = 0.8  # no near-term earnings → somewhat bullish

        # [15] Earnings surprise (last quarter)
        try:
            earnings_hist = ticker.earnings_history
            if earnings_hist is not None and not earnings_hist.empty:
                last     = earnings_hist.iloc[-1]
                actual   = float(last.get("epsActual",   0) or 0)
                estimate = float(last.get("epsEstimate", 0) or 1e-9)
                surprise = (actual - estimate) / abs(estimate + 1e-9)
                ctx[15]  = _safe(surprise, -0.5, 0.5)
        except Exception as earn_err:
            log.debug("Earnings history failed for %s: %s", symbol, earn_err)

        # [16-17] Fundamental growth
        try:
            info    = ticker.info
            ctx[16] = _safe((info.get("revenueGrowth",  0) or 0) * 100, -20, 50)
            ctx[17] = _safe((info.get("earningsGrowth", 0) or 0) * 100, -50, 100)
        except Exception as info_err:
            log.debug("Fundamental info failed for %s: %s", symbol, info_err)

        # [18] SPY weekly trend (4-week return)
        spy_mom = regime.get("spy_4wk_ret", 0.0)
        ctx[18] = _safe(spy_mom, -15, 15)

        # [19] VIX level
        vix = regime.get("vix", 20.0)
        ctx[19] = _safe(vix, 10, 40)

        # [20] Fear/Greed proxy (multi-factor)
        try:
            vix_hist_df = _fetch_bars_alpaca("VIX", "1Day", 30)
            if vix_hist_df.empty:
                vix_hist_df = yf.Ticker("^VIX").history(period="30d")  # fallback
            if len(vix_hist_df) >= 20:
                vix_ma20    = float(np.mean(vix_hist_df["Close"].values[-20:]))
                breadth_val = regime.get("market_breadth", 60.0)
                spy_price   = regime.get("spy_price", 500.0)
                spy_ma200   = regime.get("spy_ma200", 450.0)
                spy_vs_200  = spy_price / max(spy_ma200, 1.0)
                ctx[20]     = _fear_greed_proxy(vix, vix_ma20, spy_mom, breadth_val, spy_vs_200)
        except Exception as fg_err:
            log.debug("Fear/Greed proxy failed: %s", fg_err)

        # [21] Market breadth
        breadth = regime.get("market_breadth", 60.0)
        ctx[21] = _safe(breadth, 0, 100)

        # [22] Yield curve (10Y – 2Y spread)
        spread  = _get_yield_spread()
        ctx[22] = _safe(spread, -2, 3)

        # [23] Institutional ownership
        try:
            info     = ticker.info
            inst_pct = float((info.get("heldPercentInstitutions", 0.5) or 0.5)) * 100.0
            ctx[23]  = _safe(inst_pct, 0, 100)
        except Exception as inst_err:
            log.debug("Institutional ownership failed for %s: %s", symbol, inst_err)

        # [24] Short interest (days to cover — high = bearish pressure)
        try:
            info    = ticker.info
            si      = float(info.get("shortRatio", 3.0) or 3.0)
            ctx[24] = 1.0 - _safe(si, 0, 15)  # invert: low short interest = bullish
        except Exception as si_err:
            log.debug("Short interest failed for %s: %s", symbol, si_err)

        # [25] Insider activity (net buys fraction of last transactions)
        try:
            insider = ticker.insider_transactions
            if insider is not None and not insider.empty and "Shares" in insider.columns:
                shares_col = insider["Shares"].fillna(0)
                buys  = int((shares_col > 0).sum())
                sells = int((shares_col < 0).sum())
                total = buys + sells
                if total > 0:
                    ctx[25] = _safe((buys - sells) / total, -1, 1)
        except Exception as ins_err:
            log.debug("Insider transactions failed for %s: %s", symbol, ins_err)

        # [26] Volume accumulation trend (up-volume / total volume over 20 days)
        if len(volume_d) >= 21 and len(closes_d) >= 21:
            up_vol   = 0.0
            down_vol = 0.0
            for i in range(1, 21):
                if closes_d[-i] > closes_d[-i - 1]:
                    up_vol   += float(volume_d[-i])
                else:
                    down_vol += float(volume_d[-i])
            total_vol = up_vol + down_vol + 1e-9
            ctx[26]   = _safe(up_vol / total_vol, 0, 1)

        # [27-28] Fundamental quality
        try:
            info = ticker.info
            de   = float((info.get("debtToEquity", 50) or 50)) / 100.0
            pm   = float((info.get("profitMargins", 0.1) or 0.1)) * 100.0
            ctx[27] = 1.0 - _safe(de, 0, 3)      # less debt = higher score
            ctx[28] = _safe(pm, -10, 50)
        except Exception as fund_err:
            log.debug("Fundamental quality failed for %s: %s", symbol, fund_err)

        # [29-31] Time features
        now     = datetime.now(ET)
        ctx[29] = float(now.weekday()) / 4.0
        ctx[30] = float((now.day - 1) // 7) / 3.0

        try:
            mktcap  = float(ticker.info.get("marketCap", 5e9) or 5e9)
            ctx[31] = 1.0 if mktcap > 10e9 else 0.5 if mktcap > 2e9 else 0.0
        except Exception as cap_err:
            log.debug("Market cap failed for %s: %s", symbol, cap_err)
            ctx[31] = 0.5

    except Exception as e:
        log.warning("Context build partially failed for %s: %s", symbol, e)

    return np.clip(ctx, 0.0, 1.0).astype(np.float32)
