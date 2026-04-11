"""
ATG Quad-Intelligence Validator v3.0

Runs 3 independent AI brains in parallel to validate each swing trade setup.
Consensus drives position sizing; divergence surfaces key risks.

Decision matrix:
  3/3 proceed  → STRONG    → 1.5x size boost
  2/3 proceed  → MODERATE  → 1.0x standard size
  1/3 proceed  → WEAK      → skip trade (0.5x)
  0/3 proceed  → BLOCKED   → skip trade (0.0x)

Brains:
  1. DeepSeek V3  — quantitative patterns, price action, technical structure
  2. Gemini 2.5   — macro context, sector dynamics, broader market regime
  3. GPT-4o-mini  — fundamental analysis, earnings quality, valuation

Claude is embedded in every design decision as the architecture brain.
"""
import os
import json
import logging
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from config.settings import DEEPSEEK_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY

log = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
OPENAI_URL   = "https://api.openai.com/v1/chat/completions"

TIMEOUT_S   = 20
MAX_TOKENS  = 200


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(setup: dict, selection: dict, brain_role: str) -> str:
    """
    Construct a brain-specific trade evaluation prompt.

    Args:
        setup      : scanner result dict.
        selection  : bandit selection dict (setup_type, stop_multiplier).
        brain_role : role description for the AI persona.

    Returns:
        Prompt string requesting JSON output.
    """
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
Consider: sector strength, market conditions, setup quality, risk/reward, momentum.

Reply ONLY in this exact JSON format (no other text):
{{"conviction": "HIGH" | "MEDIUM" | "LOW", "proceed": true | false, "thesis": "one sentence max 20 words", "key_risk": "biggest risk in max 15 words"}}"""


# ── Individual brain callers ──────────────────────────────────────────────────

def _parse_json_response(content: str) -> dict:
    """
    Safely extract and parse the first JSON object from an LLM response.

    Args:
        content: raw text response from the LLM.

    Returns:
        Parsed dict with normalised keys, or safe fallback.
    """
    try:
        match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if match:
            d = json.loads(match.group())
            d["proceed"]    = bool(d.get("proceed", True))
            d["conviction"] = str(d.get("conviction", "MEDIUM")).upper()
            return d
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"conviction": "LOW", "proceed": False, "thesis": "parse error", "key_risk": "unknown"}


def _call_deepseek(prompt: str) -> dict:
    """
    Query DeepSeek V3 for a trade evaluation.

    Args:
        prompt: formatted evaluation prompt.

    Returns:
        Parsed response dict with "brain" key set to "deepseek".
    """
    if not DEEPSEEK_API_KEY:
        log.warning("DeepSeek key not configured — BLOCKING (fail-closed)")
        return {
            "conviction": "LOW", "proceed": False,
            "thesis": "DeepSeek not configured", "key_risk": "Brain unavailable", "brain": "deepseek",
        }
    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       "deepseek-chat",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  MAX_TOKENS,
                "temperature": 0.1,
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        result  = _parse_json_response(content)
        result["brain"] = "deepseek"
        log.info("DeepSeek: proceed=%s conviction=%s", result.get("proceed"), result.get("conviction"))
        return result
    except (requests.RequestException, KeyError, IndexError) as e:
        log.warning("DeepSeek call failed: %s — defaulting to proceed", e)
        return {
            "conviction": "LOW", "proceed": False,
            "thesis": "timeout", "key_risk": "API error", "brain": "deepseek",
        }


def _call_gemini(prompt: str) -> dict:
    """
    Query Gemini 2.0 Flash for a trade evaluation.

    Args:
        prompt: formatted evaluation prompt.

    Returns:
        Parsed response dict with "brain" key set to "gemini".
    """
    if not GEMINI_API_KEY:
        log.warning("Gemini key not configured — BLOCKING (fail-closed)")
        return {
            "conviction": "LOW", "proceed": False,
            "thesis": "Gemini not configured", "key_risk": "Brain unavailable", "brain": "gemini",
        }
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": MAX_TOKENS, "temperature": 0.1},
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        content = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        result  = _parse_json_response(content)
        result["brain"] = "gemini"
        log.info("Gemini: proceed=%s conviction=%s", result.get("proceed"), result.get("conviction"))
        return result
    except (requests.RequestException, KeyError, IndexError) as e:
        log.warning("Gemini call failed: %s — defaulting to proceed", e)
        return {
            "conviction": "LOW", "proceed": False,
            "thesis": "timeout", "key_risk": "API error", "brain": "gemini",
        }


def _call_gpt4o(prompt: str) -> dict:
    """
    Query GPT-4o-mini for a trade evaluation.

    Args:
        prompt: formatted evaluation prompt.

    Returns:
        Parsed response dict with "brain" key set to "gpt4o".
    """
    if not OPENAI_API_KEY:
        log.debug("OpenAI key not configured — defaulting to proceed")
        return {
            "conviction": "MEDIUM", "proceed": True,
            "thesis": "GPT-4o not configured", "key_risk": "N/A", "brain": "gpt4o",
        }
    try:
        resp = requests.post(
            OPENAI_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       "gpt-4o-mini",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  MAX_TOKENS,
                "temperature": 0.1,
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        result  = _parse_json_response(content)
        result["brain"] = "gpt4o"
        log.info("GPT-4o: proceed=%s conviction=%s", result.get("proceed"), result.get("conviction"))
        return result
    except (requests.RequestException, KeyError, IndexError) as e:
        log.warning("GPT-4o call failed: %s — defaulting to proceed", e)
        return {
            "conviction": "LOW", "proceed": False,
            "thesis": "timeout", "key_risk": "API error", "brain": "gpt4o",
        }


# ── Aggregator ────────────────────────────────────────────────────────────────

def _aggregate(votes: List[dict]) -> dict:
    """
    Aggregate votes from all brains into a consensus decision.

    Decision matrix:
      3/3 → STRONG   (1.5x size)
      2/3 → MODERATE (1.0x size)
      1/3 → WEAK     (0.5x, blocked)
      0/3 → BLOCKED  (0.0x)

    Args:
        votes: list of per-brain response dicts.

    Returns:
        Aggregated consensus dict.
    """
    proceed_count = sum(1 for v in votes if v.get("proceed", True))
    total         = len(votes)

    if proceed_count == total:
        consensus     = "STRONG"
        size_mult     = 1.50
        final_proceed = True
    elif proceed_count >= max(1, int(total * 0.67)):   # ≥ 2/3
        consensus     = "MODERATE"
        size_mult     = 1.00
        final_proceed = True
    elif proceed_count >= 1:
        consensus     = "WEAK"
        size_mult     = 0.50
        final_proceed = False
    else:
        consensus     = "BLOCKED"
        size_mult     = 0.00
        final_proceed = False

    dissents  = [v.get("key_risk", "") for v in votes if not v.get("proceed", True)]
    key_risks = list({r for r in dissents if r and r not in ("N/A", "unknown", "API error")})

    best_vote = max(
        votes,
        key=lambda v: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(v.get("conviction", "MEDIUM"), 2),
    )
    thesis = best_vote.get("thesis", "—")

    return {
        "proceed":         final_proceed,
        "consensus":       consensus,
        "size_multiplier": size_mult,
        "proceed_count":   proceed_count,
        "total_brains":    total,
        "thesis":          thesis,
        "key_risks":       key_risks,
        "votes":           votes,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def quad_validate(setup: dict, selection: dict) -> dict:
    """
    Run all 3 external AI brains in parallel and return aggregated consensus.

    Args:
        setup     : scanner result dict (symbol, price, setup_type, …).
        selection : bandit selection dict (setup_type, stop_multiplier).

    Returns:
        Consensus dict with keys:
          proceed, consensus, size_multiplier, proceed_count, total_brains,
          thesis, key_risks, votes.
    """
    symbol = setup.get("symbol", "?")
    log.info("Quad-Intelligence validating %s %s …", symbol, setup.get("setup_type", "?"))

    brain_calls = {
        "deepseek": (
            _call_deepseek,
            _build_prompt(
                setup, selection,
                "quantitative swing trade analyst specialising in price action, volume, and technical patterns",
            ),
        ),
        "gemini": (
            _call_gemini,
            _build_prompt(
                setup, selection,
                "macro-aware market analyst specialising in sector dynamics, market regime, and broader context",
            ),
        ),
        "gpt4o": (
            _call_gpt4o,
            _build_prompt(
                setup, selection,
                "fundamental analyst specialising in business quality, earnings catalysts, and valuation",
            ),
        ),
    }

    votes: List[dict] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(fn, prompt): name
            for name, (fn, prompt) in brain_calls.items()
        }
        for future in as_completed(futures, timeout=TIMEOUT_S + 5):
            brain_name = futures[future]
            try:
                result = future.result(timeout=2)
                votes.append(result)
            except Exception as e:
                log.warning("Brain %s result retrieval failed: %s — defaulting to proceed", brain_name, e)
                votes.append({
                    "conviction": "LOW", "proceed": False,
                    "thesis": "timeout", "key_risk": "N/A", "brain": brain_name,
                })

    consensus = _aggregate(votes)

    vote_str = " | ".join(
        f"{v['brain']}: {'✅' if v['proceed'] else '❌'} {v['conviction']}"
        for v in consensus["votes"]
    )
    log.info(
        "Quad-Intelligence %s | %d/%d proceed | %s | %s",
        symbol, consensus["proceed_count"], consensus["total_brains"],
        consensus["consensus"], vote_str,
    )

    return consensus
