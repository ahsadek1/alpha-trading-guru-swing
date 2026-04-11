"""
Alpha Trading Guru — Conversational Language Module
─────────────────────────────────────────────────────
Adds natural language understanding. Ahmed can message ATG in plain
English and receive context-aware, personality-driven responses.

Brain: DeepSeek V3
Personality: Patient swing trader. Macro-aware. Chart-reader. Never rushes.
─────────────────────────────────────────────────────
"""
import os
import time
import logging
import threading
import requests

log = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL     = "https://api.deepseek.com/v1/chat/completions"
TELEGRAM_TOKEN   = os.getenv("ATG_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("ATG_TELEGRAM_CHAT_ID", "8573754783")
TELEGRAM_API     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

_histories: dict = {}
MAX_TURNS = 8

ATG_SYSTEM_PROMPT = """You are Alpha Trading Guru (ATG), a patient swing trading intelligence.

YOUR DOMAIN: Swing trades (3–10 day holds), equity setups, weekly charts. You think in weekly RSI, breakouts, pullbacks, base patterns, relative strength. You are NOT an options specialist (that's ATM). You are the equity swing specialist.

YOUR PERSONALITY:
- Unhurried, measured, zen-like. You don't chase. You wait for your pitch.
- You think in weeks, not minutes. You respect the chart.
- Calm under volatility. You've seen every market condition.
- Direct. No false certainty. When a setup is unclear, you say: "Not clear yet. Let it develop."
- You think risk-first: "Where's my stop before I think about my target."

YOUR CAPABILITIES:
- Explain current swing setups and why they scored the way they did
- Evaluate a stock setup Ahmed describes (sector, chart pattern, risk/reward)
- Review open swing positions and comment on setup integrity
- Explain the scan criteria (weekly RSI, MA alignment, volume, ATR)
- Advise on patience — when to wait vs when to act
- Walk through historical closed positions and lessons

CURRENT SYSTEM CONTEXT:
{context}

RULES:
- Always ground responses in the context above
- Never fabricate position data
- Keep responses mobile-friendly (under 300 words)
- Think out loud about risk before reward"""


def _get_live_context() -> str:
    try:
        from src.database import get_trade_stats, get_open_positions
        from src.phase_manager import get_current_phase
        from datetime import datetime
        import pytz
        ET  = pytz.timezone("America/New_York")
        now = datetime.now(ET).strftime("%a %b %d %Y %I:%M %p ET")
        stats = get_trade_stats()
        phase = get_current_phase()
        try:
            positions = get_open_positions()
        except Exception:
            positions = []
        pos_lines = []
        for p in (positions or []):
            pos_lines.append(
                f"  {p.get('symbol','?')} {p.get('setup_type','?')} | "
                f"Entry: ${p.get('entry',0):.2f} | Stop: ${p.get('stop',0):.2f} | "
                f"P&L: {p.get('pnl_pct',0):+.1f}%"
            )
        return (
            f"Time: {now}\n"
            f"Phase: {phase}\n"
            f"Closed trades: {stats.get('total_closed',0)} | "
            f"Win rate: {stats.get('win_rate',0):.1%} | "
            f"Total P&L: ${stats.get('total_pnl',0):+,.0f}\n"
            f"Open swings ({len(positions or [])}):\n" +
            ("\n".join(pos_lines) if pos_lines else "  None — waiting for the right setup")
        )
    except Exception as e:
        return f"Context unavailable: {e}"


def _call_deepseek(messages: list, system: str) -> str:
    if not DEEPSEEK_API_KEY:
        return "DeepSeek API key not configured."
    try:
        r = requests.post(DEEPSEEK_URL, json={
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": 600,
            "temperature": 0.5,
        }, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        timeout=25)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error("DeepSeek call failed: %s", e)
        return f"AI unavailable right now. ({e})"


def respond(user_text: str, chat_id: str = TELEGRAM_CHAT_ID) -> str:
    history = _histories.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})
    while len(history) > MAX_TURNS * 2:
        history.pop(0)
    context = _get_live_context()
    system  = ATG_SYSTEM_PROMPT.format(context=context)
    reply   = _call_deepseek(history.copy(), system)
    history.append({"role": "assistant", "content": reply})
    return reply


def send_reply(chat_id: str, text: str):
    if not TELEGRAM_TOKEN:
        return
    try:
        requests.post(f"{TELEGRAM_API}/sendMessage", json={
            "chat_id": chat_id, "text": text
        }, timeout=10)
    except Exception as e:
        log.error("Telegram reply failed: %s", e)


def _is_command(text: str) -> bool:
    return text.strip().startswith("/")


def _polling_loop():
    offset = 0
    log.info("[ATG-LANG] Polling loop started")
    while True:
        try:
            r = requests.get(f"{TELEGRAM_API}/getUpdates",
                             params={"offset": offset, "timeout": 20},
                             timeout=30)
            if r.status_code != 200:
                time.sleep(5)
                continue
            updates = r.json().get("result", [])
            for upd in updates:
                offset = upd["update_id"] + 1
                msg    = upd.get("message", {})
                text   = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))
                if not text or not chat_id:
                    continue
                if chat_id != TELEGRAM_CHAT_ID:
                    continue
                if _is_command(text):
                    continue
                reply = respond(text, chat_id)
                send_reply(chat_id, reply)
        except Exception as e:
            log.warning("[ATG-LANG] Polling error: %s", e)
            time.sleep(5)


def start():
    if not TELEGRAM_TOKEN:
        log.warning("[ATG-LANG] TELEGRAM_TOKEN not set — language module disabled")
        return
    t = threading.Thread(target=_polling_loop, name="atg-lang-poll", daemon=True)
    t.start()
    log.info("[ATG-LANG] Language module started (DeepSeek V3)")
