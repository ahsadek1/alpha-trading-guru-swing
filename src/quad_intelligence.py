"""
ATG Quad-Intelligence Validator v3.1

FIX [F14]: Implemented INV-5 graduated failure policy.
           Brain failures now distinguished from brain votes.
           0 failures → normal voting | 1 failure → 2-brain (max MODERATE)
           2 failures → DEGRADED (0.25x) + Ahmed alert
           3 failures → SUSPENDED (no trade) + Ahmed alert

Decision matrix (0 failures):
  3/3 proceed  → STRONG    → 1.5x size
  2/3 proceed  → MODERATE  → 1.0x size
  1/3 proceed  → BLOCKED
  0/3 proceed  → BLOCKED + alert Ahmed

Decision matrix (1 failure):
  2/2 proceed  → MODERATE  → 1.0x (NOT STRONG — per INV-5)
  1/2 or 0/2   → BLOCKED

Decision matrix (2 failures):
  1/1 proceed  → DEGRADED  → 0.25x + alert Ahmed
  0/1          → BLOCKED + alert Ahmed

Decision matrix (3 failures):
  SUSPENDED → no trade + alert Ahmed
"""
import os
import json
import logging
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from config.settings import DEEPSEEK_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY

log = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
OPENAI_URL   = "https://api.openai.com/v1/chat/completions"

TIMEOUT_S   = 20
MAX_TOKENS  = 200
MIN_POSITION_USD = 200.0   # INV-6: skip trade if sizing falls below this

# INV-5: Exit reason weights for bandit update (used by trade_executor)
EXIT_REASON_WEIGHTS = {
    "PROFIT_TARGET":    1.0,
    "STAGED_EXIT":      1.0,
    "STOP_HIT":         1.0,
    "TIERED_STOP":      1.0,
    "BREAKEVEN_STOP":   0.8,
    "TIME_EXIT":        0.8,
    "TIME_STOP":        0.5,
    "DTE_FORCED":       0.5,
    "TRAILING_STOP":    0.9,
    "CIRCUIT_BREAKER":  0.0,
    "MANUAL_CLOSE":     0.0,
    "RECONCILER_VOID":  0.0,
    "UNKNOWN":          0.5,
}


def get_exit_reason_weight(exit_reason: str) -> float:
    """
    Return the bandit update weight for a given exit reason.
    INV-2: Only high-quality exit signals update bandit posterior.
    """
    weight = EXIT_REASON_WEIGHTS.get(exit_reason.upper() if exit_reason else "UNKNOWN", 0.5)
    if exit_reason and exit_reason.upper() not in EXIT_REASON_WEIGHTS:
        log.warning("Unknown exit reason '%s' — using weight=0.5", exit_reason)
    return weight


def _alert_ahmed(message: str) -> None:
    """Send alert to Ahmed. Non-blocking, never raises."""
    try:
        token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_AHMED_ID", "8573754783")
        if not token:
            return
        import urllib.request
        payload = json.dumps({"chat_id": chat_id, "text": message}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def _build_prompt(setup: dict, selection: dict, brain_role: str) -> str:
    symbol     = setup.get("symbol", "?")
    setup_type = setup.get("setup_type", "?")
    score      = setup.get("score", 0)
    price      = setup.get("price", 0)
    stop       = setup.get("stop_loss", 0)
    t1         = setup.get("target_price", 0)
    t2         = setup.get("target_stage2", 0)
    rsi_w      = setup.get("weekly_rsi", 0)
    rsi_d      = setup.get("daily_rsi", 0)
    vol        = setup.get("volume_ratio", 1)
    atr_pct    = setup.get("atr_pct", 0)
    ma_aligned = setup.get("ma_aligned", False)
    sector     = setup.get("sector", "?")
    dte        = setup.get("days_to_earnings", "N/A")
    rr         = setup.get("rr_ratio", 0)
    stop_mult  = selection.get("stop_multiplier", 2.0)
    risk_pct   = abs(price - stop) / price * 100 if price > 0 else 0

    return f"""You are a {brain_role} evaluating a swing trade for a self-learning algorithmic trading system.

TRADE SETUP:
  Symbol:          {symbol} ({sector})
  Setup Type:      {setup_type}
  Scanner Score:   {score}/100
  Price:           ${price:.2f}
  Stop Loss:       ${stop:.2f}  (risk: {risk_pct:.1f}%)
  Target 1 (50%):  ${t1:.2f}
  Target 2 (ride): ${t2:.2f}
  Risk:Reward:     {rr:.1f}:1
  Weekly RSI:      {rsi_w:.0f}
  Daily RSI:       {rsi_d:.0f}
  Volume Ratio:    {vol:.1f}x avg
  ATR:             {atr_pct:.1f}% of price
  MA Aligned:      {ma_aligned} (50MA > 200MA)
  Days to Earnings:{dte}
  Stop Multiplier: {stop_mult:.1f}x ATR

Evaluate whether this setup has high probability of success in the next 3-8 weeks.

Reply ONLY in this exact JSON format (no other text):
{{"conviction": "HIGH" | "MEDIUM" | "LOW", "proceed": true | false, "thesis": "one sentence max 20 words", "key_risk": "biggest risk in max 15 words"}}"""


def _parse_json_response(content: str) -> dict:
    try:
        match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if match:
            d = json.loads(match.group())
            d["proceed"]    = bool(d.get("proceed", True))
            d["conviction"] = str(d.get("conviction", "MEDIUM")).upper()
            d["failed"]     = False
            return d
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"conviction": "LOW", "proceed": False, "thesis": "parse error",
            "key_risk": "unknown", "failed": False}


def _call_deepseek(prompt: str) -> dict:
    if not DEEPSEEK_API_KEY:
        return {"conviction": "LOW", "proceed": False, "thesis": "DeepSeek not configured",
                "key_risk": "Brain unavailable", "brain": "deepseek", "failed": True}
    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": MAX_TOKENS, "temperature": 0.1},
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        result = _parse_json_response(resp.json()["choices"][0]["message"]["content"].strip())
        result["brain"] = "deepseek"
        return result
    except Exception as e:
        log.warning("DeepSeek brain FAILED: %s", e)
        # FIX [F14]: failed=True marks this as API failure, not a vote against
        return {"conviction": "LOW", "proceed": False, "thesis": "timeout",
                "key_risk": "API failure", "brain": "deepseek", "failed": True}


def _call_gemini(prompt: str) -> dict:
    if not GEMINI_API_KEY:
        return {"conviction": "LOW", "proceed": False, "thesis": "Gemini not configured",
                "key_risk": "Brain unavailable", "brain": "gemini", "failed": True}
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": MAX_TOKENS, "temperature": 0.1}},
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        result = _parse_json_response(resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip())
        result["brain"] = "gemini"
        return result
    except Exception as e:
        log.warning("Gemini brain FAILED: %s", e)
        return {"conviction": "LOW", "proceed": False, "thesis": "timeout",
                "key_risk": "API failure", "brain": "gemini", "failed": True}


def _call_gpt4o(prompt: str) -> dict:
    if not OPENAI_API_KEY:
        return {"conviction": "LOW", "proceed": False, "thesis": "GPT-4o not configured",
                "key_risk": "Brain unavailable", "brain": "gpt4o", "failed": True}
    try:
        resp = requests.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": MAX_TOKENS, "temperature": 0.1},
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        result = _parse_json_response(resp.json()["choices"][0]["message"]["content"].strip())
        result["brain"] = "gpt4o"
        return result
    except Exception as e:
        log.warning("GPT-4o brain FAILED: %s", e)
        return {"conviction": "LOW", "proceed": False, "thesis": "timeout",
                "key_risk": "API failure", "brain": "gpt4o", "failed": True}


def _aggregate(votes: List[dict], symbol: str = "?") -> dict:
    """
    FIX [F14]: Full INV-5 graduated failure policy.
    Distinguishes brain failures (failed=True) from brain votes (failed=False).

    Failure modes:
      0 failures → normal 3-brain voting (STRONG/MODERATE/BLOCKED)
      1 failure  → 2-brain system (max MODERATE, no STRONG allowed)
      2 failures → DEGRADED (0.25x size), alert Ahmed
      3 failures → SUSPENDED (no trade), alert Ahmed
    """
    failures = [v for v in votes if v.get("failed", False)]
    actives  = [v for v in votes if not v.get("failed", False)]
    n_fail   = len(failures)
    n_active = len(actives)

    # SUSPENDED: all brains failed
    if n_fail == 3:
        log.error("⚠️ QI SUSPENDED — all 3 brains failed for %s", symbol)
        _alert_ahmed(
            f"⚠️ ATG QI SUSPENDED: All 3 AI brains unavailable for {symbol}. "
            f"No trade executed. Check API keys."
        )
        return {
            "proceed": False, "consensus": "SUSPENDED", "size_multiplier": 0.0,
            "proceed_count": 0, "total_brains": 3, "failures": 3,
            "thesis": "All AI brains unavailable", "key_risks": ["QI fully down"],
            "votes": votes,
        }

    # DEGRADED: 2 brains failed
    if n_fail == 2:
        remaining_vote = actives[0] if actives else None
        if remaining_vote and remaining_vote.get("proceed", False):
            consensus     = "DEGRADED"
            size_mult     = 0.25
            final_proceed = True
            thesis        = remaining_vote.get("thesis", "—")
        else:
            consensus     = "BLOCKED"
            size_mult     = 0.0
            final_proceed = False
            thesis        = "Degraded QI — remaining brain blocked"
        failed_names = [v.get("brain", "?") for v in failures]
        log.warning("⚠️ QI DEGRADED for %s — brains failed: %s", symbol, failed_names)
        _alert_ahmed(
            f"⚠️ ATG QI DEGRADED for {symbol}: {', '.join(failed_names)} unavailable. "
            f"Decision: {consensus} ({size_mult:.0%} size). Check API health."
        )
        return {
            "proceed": final_proceed, "consensus": consensus, "size_multiplier": size_mult,
            "proceed_count": 1 if final_proceed else 0, "total_brains": 3, "failures": n_fail,
            "thesis": thesis, "key_risks": ["QI degraded"],
            "votes": votes,
        }

    # 1 failure: 2-brain system (max MODERATE, not STRONG per INV-5)
    if n_fail == 1:
        proceed_count = sum(1 for v in actives if v.get("proceed", False))
        if proceed_count == 2:
            # Both active brains agree → MODERATE (NOT STRONG per INV-5)
            consensus = "MODERATE"
            size_mult = 1.0
            final_proceed = True
        else:
            consensus = "BLOCKED"
            size_mult = 0.0
            final_proceed = False
        log.info("QI 2-brain system for %s (1 failure) → %s", symbol, consensus)
        thesis = max(actives, key=lambda v: {"HIGH":3,"MEDIUM":2,"LOW":1}.get(v.get("conviction","MEDIUM"),2)).get("thesis","—")
        return {
            "proceed": final_proceed, "consensus": consensus, "size_multiplier": size_mult,
            "proceed_count": proceed_count, "total_brains": 3, "failures": n_fail,
            "thesis": thesis,
            "key_risks": [v.get("key_risk","") for v in actives if not v.get("proceed",False)],
            "votes": votes,
        }

    # 0 failures: normal 3-brain voting
    proceed_count = sum(1 for v in actives if v.get("proceed", False))

    if proceed_count == 3:
        consensus = "STRONG"; size_mult = 1.5; final_proceed = True
    elif proceed_count == 2:
        consensus = "MODERATE"; size_mult = 1.0; final_proceed = True
    elif proceed_count == 1:
        consensus = "BLOCKED"; size_mult = 0.0; final_proceed = False
    else:
        consensus = "BLOCKED"; size_mult = 0.0; final_proceed = False
        _alert_ahmed(f"⚠️ ATG QI: all 3 brains voted AGAINST {symbol}.")

    best = max(actives, key=lambda v: {"HIGH":3,"MEDIUM":2,"LOW":1}.get(v.get("conviction","MEDIUM"),2))
    return {
        "proceed": final_proceed, "consensus": consensus, "size_multiplier": size_mult,
        "proceed_count": proceed_count, "total_brains": 3, "failures": 0,
        "thesis": best.get("thesis","—"),
        "key_risks": [v.get("key_risk","") for v in actives if not v.get("proceed",False)],
        "votes": votes,
    }


def quad_validate(setup: dict, selection: dict) -> dict:
    """
    Run all 3 AI brains in parallel. Return INV-5 graduated consensus.

    FIX [F14]: Uses graduated failure policy with explicit failure tracking.
    """
    symbol = setup.get("symbol", "?")
    log.info("Quad-Intelligence validating %s %s …", symbol, setup.get("setup_type","?"))

    brain_calls = {
        "deepseek": (_call_deepseek, _build_prompt(setup, selection,
            "quantitative swing trade analyst specialising in price action and technical patterns")),
        "gemini":   (_call_gemini,   _build_prompt(setup, selection,
            "macro-aware market analyst specialising in sector dynamics and market regime")),
        "gpt4o":    (_call_gpt4o,    _build_prompt(setup, selection,
            "fundamental analyst specialising in earnings quality and business momentum")),
    }

    votes: List[dict] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn, prompt): name for name, (fn, prompt) in brain_calls.items()}
        for future in as_completed(futures, timeout=TIMEOUT_S + 5):
            brain_name = futures[future]
            try:
                result = future.result(timeout=2)
                votes.append(result)
            except Exception as e:
                log.warning("Brain %s result retrieval exception: %s", brain_name, e)
                votes.append({"conviction": "LOW", "proceed": False, "thesis": "exception",
                               "key_risk": "N/A", "brain": brain_name, "failed": True})

    consensus = _aggregate(votes, symbol=symbol)

    vote_str = " | ".join(
        f"{v.get('brain','?')}: {'💀' if v.get('failed') else '✅' if v.get('proceed') else '❌'} "
        f"{v.get('conviction','?')}"
        for v in consensus["votes"]
    )
    log.info(
        "QI %s | %d/%d proceed | %d failed | %s | %s",
        symbol, consensus["proceed_count"], consensus["total_brains"],
        consensus.get("failures", 0), consensus["consensus"], vote_str,
    )
    return consensus
