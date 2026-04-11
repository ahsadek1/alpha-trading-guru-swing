"""
ATG DeepSeek Analyst v3.0 — Standalone qualitative thesis layer.

Called before Quad-Intelligence for a quick pre-filter on individual setups.
Non-blocking: always returns a usable dict even when the API is unavailable.
"""
import json
import logging
import re
import requests

from config.settings import DEEPSEEK_API_KEY

log = logging.getLogger(__name__)

_DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


def analyze_swing_setup(scan_result: dict, bandit_selection: dict) -> dict:
    """
    Ask DeepSeek-V3 to evaluate a swing setup candidate.

    Args:
        scan_result      : scanner output dict (symbol, price, setup_type, …).
        bandit_selection : bandit selection dict (setup_type, stop_multiplier).

    Returns:
        Dict with keys: conviction (HIGH/MEDIUM/LOW), thesis (str),
        proceed (bool), key_risk (str).
        Falls back gracefully if the API is unavailable.
    """
    if not DEEPSEEK_API_KEY:
        log.debug("DEEPSEEK_API_KEY not set — skipping DeepSeek analysis")
        return {
            "conviction": "MEDIUM",
            "thesis":     "DeepSeek not configured",
            "proceed":    True,
            "key_risk":   "Unknown",
        }

    symbol     = scan_result["symbol"]
    setup_type = scan_result["setup_type"]
    score      = scan_result["score"]
    price      = scan_result["price"]
    stop       = scan_result["stop_loss"]
    target     = scan_result["target_price"]
    rsi        = scan_result.get("weekly_rsi", "N/A")
    ma_aligned = scan_result.get("ma_aligned", False)
    atr_pct    = scan_result.get("atr_pct", "N/A")
    vol_ratio  = scan_result.get("volume_ratio", 1.0)
    stop_mult  = bandit_selection.get("stop_multiplier", 2.0)
    risk_pct   = ((price - stop) / price * 100) if price > 0 else 0

    prompt = f"""You are a swing trade analyst evaluating a candidate for paper trading.

SYMBOL:          {symbol}
SETUP TYPE:      {setup_type}
SCANNER SCORE:   {score}/100
PRICE:           ${price:.2f}
STOP LOSS:       ${stop:.2f}  (risk: {risk_pct:.1f}%)
TARGET:          ${target:.2f} (reward: {((target - price) / price * 100):.1f}%)
WEEKLY RSI:      {rsi}
MA ALIGNED:      {ma_aligned}
WEEKLY ATR:      {atr_pct}% of price
VOLUME RATIO:    {vol_ratio:.2f}x average
STOP MULTIPLIER: {stop_mult}x weekly ATR

Evaluate this setup for a 3–8 week swing trade.

Reply ONLY in this exact JSON format (no other text):
{{"conviction": "HIGH" | "MEDIUM" | "LOW", "thesis": "max 20 words", "proceed": true | false, "key_risk": "max 15 words"}}"""

    try:
        resp = requests.post(
            _DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       "deepseek-chat",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  150,
                "temperature": 0.1,
            },
            timeout=15,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        match   = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            result = json.loads(match.group())
            log.info(
                "DeepSeek analyst | %s %s | conviction=%s proceed=%s",
                symbol, setup_type, result.get("conviction"), result.get("proceed"),
            )
            return result

    except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
        log.warning("DeepSeek analysis failed for %s: %s — proceeding", symbol, e)

    return {
        "conviction": "MEDIUM",
        "thesis":     "DeepSeek analysis unavailable",
        "proceed":    True,
        "key_risk":   "Unknown",
    }
