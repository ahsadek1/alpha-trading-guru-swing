"""
src/startup_guard.py — Validates required environment variables at startup.
Crashes immediately with a clear error if anything is missing.
Validates AI brain API keys with live test calls — alerts Ahmed if any brain is dark.
Prevents silent failures where systems run but can't trade (B05 permanent fix).
"""
import os
import sys
import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

REQUIRED_VARS = [
    ("ALPACA_API_KEY",    "Alpaca API credentials"),
    ("ALPACA_SECRET_KEY", "Alpaca secret credentials"),
    ("ALPACA_BASE_URL",   "Alpaca base URL (paper or live)"),
]

OPTIONAL_VARS = [
    ("CAPITAL_ROUTER_URL", "Capital Router — trade allocation"),
    ("DEEPSEEK_API_KEY",   "DeepSeek AI — trade intelligence"),
    ("GEMINI_API_KEY",     "Gemini AI — trade intelligence"),
    ("OPENAI_API_KEY",     "OpenAI — trade intelligence"),
    ("TELEGRAM_BOT_TOKEN", "Telegram alerts"),
    ("TELEGRAM_AHMED_ID",  "Ahmed DM target"),
]


def _test_deepseek(key: str) -> tuple:
    """Ping DeepSeek API with minimal request. Returns (ok, error_msg)."""
    try:
        payload = json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }).encode()
        req = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            return True, None
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)[:60]


def _test_gemini(key: str) -> tuple:
    """Ping Gemini API with minimal request. Returns (ok, error_msg)."""
    try:
        payload = json.dumps({"contents": [{"parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 1}}).encode()
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            return True, None
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)[:60]


def _test_openai(key: str) -> tuple:
    """Ping OpenAI API with minimal request. Returns (ok, error_msg)."""
    try:
        payload = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            return True, None
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)[:60]


def _alert_ahmed(message: str):
    """Send Telegram alert to Ahmed about brain failures."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    ahmed_id  = os.getenv("TELEGRAM_AHMED_ID", "8573754783")
    if not bot_token:
        return
    try:
        payload = json.dumps({"chat_id": ahmed_id, "text": message}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # Telegram failure never blocks startup


def validate_api_keys():
    """
    B05 Permanent Fix: Test each AI brain API key with a live call.
    Logs WARNING for dark brains. Alerts Ahmed via Telegram.
    Does NOT crash startup — degraded QI is better than no trading.
    """
    dark_brains = []

    brains = [
        ("DeepSeek",  "DEEPSEEK_API_KEY", _test_deepseek),
        ("Gemini",    "GEMINI_API_KEY",   _test_gemini),
        ("OpenAI",    "OPENAI_API_KEY",   _test_openai),
    ]

    for name, env_var, test_fn in brains:
        key = os.getenv(env_var, "").strip()
        if not key:
            logger.warning("BRAIN DARK: %s — %s not set", name, env_var)
            dark_brains.append(f"{name} (key missing)")
            continue
        ok, err = test_fn(key)
        if ok:
            logger.info("BRAIN OK: %s ✅", name)
        else:
            logger.warning("BRAIN DARK: %s — API key invalid or unreachable: %s", name, err)
            dark_brains.append(f"{name} ({err})")

    if dark_brains:
        alert = (
            "\u26a0\ufe0f *QI BRAIN ALERT — STARTUP*\n"
            + "\n".join(f"\u274c {b}" for b in dark_brains)
            + "\n\nSystem running in degraded mode. Fix API keys in Railway."
        )
        _alert_ahmed(alert)
        logger.warning("QI degraded: %d brain(s) dark: %s", len(dark_brains), dark_brains)
    else:
        logger.info("All 3 AI brains validated OK \u2705")


def validate():
    """
    Call this at application startup.
    Raises SystemExit if any required var is missing.
    Logs warnings for missing optional vars.
    Validates AI brain API keys with live test calls.
    """
    missing_required = []
    for var, description in REQUIRED_VARS:
        val = os.getenv(var, "").strip()
        if not val:
            missing_required.append(f"  \u274c {var} — {description}")

    for var, description in OPTIONAL_VARS:
        val = os.getenv(var, "").strip()
        if not val:
            logger.warning("Optional env var missing: %s (%s) — degraded mode", var, description)

    if missing_required:
        error_msg = "\n".join([
            "=" * 60,
            "STARTUP ABORTED — Required environment variables missing:",
            *missing_required,
            "Set these in Railway → Service → Variables and redeploy.",
            "=" * 60,
        ])
        logger.critical(error_msg)
        print(error_msg, flush=True)
        sys.exit(1)

    logger.info("Startup guard: all required env vars present \u2705")

    # B05: Validate AI brain keys with live API calls (non-blocking)
    validate_api_keys()
