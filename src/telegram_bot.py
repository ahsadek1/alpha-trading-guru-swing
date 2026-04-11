"""
ATG Telegram Bot v3.0 — Rich alerts for Alpha Trading Guru swing system.

All messages sent via Telegram Bot API with Markdown parse mode.
Token sourced from ATG_TELEGRAM_TOKEN env var (no hardcoded fallback).
"""
import logging
import requests
from datetime import datetime
from typing import Optional
import pytz

from config.settings import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYSTEM_NAME

log = logging.getLogger(__name__)
ET  = pytz.timezone("America/New_York")

_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def send(msg: str, parse_mode: str = "Markdown") -> bool:
    """
    Send a message to the configured Telegram chat.

    Args:
        msg        : message text.
        parse_mode : "Markdown" or "HTML".

    Returns:
        True if delivered, False on any failure (non-raising).
    """
    if not TELEGRAM_TOKEN:
        log.warning("ATG_TELEGRAM_TOKEN not set — skipping Telegram send")
        return False
    try:
        r = requests.post(
            f"{_API_BASE}/sendMessage",
            json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       msg,
                "parse_mode": parse_mode,
            },
            timeout=15,
        )
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error("Telegram send failed: %s", e)
        return False


# ── Startup ───────────────────────────────────────────────────────────────────

def send_startup_card(phase: int, total_trades: int, win_rate: float) -> None:
    """
    Send system startup notification card.

    Args:
        phase        : current learning phase (1–4).
        total_trades : total closed trades to date.
        win_rate     : historical win rate (0–1 float).
    """
    phase_names = {
        1: "LINEAR BANDIT",
        2: "NEURAL BANDIT",
        3: "DISTRIBUTIONAL RL",
        4: "CAUSAL DISCOVERY",
    }
    now = datetime.now(ET).strftime("%b %d %Y  %I:%M %p ET")
    send(f"""🧘 *{SYSTEM_NAME} — ONLINE*
━━━━━━━━━━━━━━━━━━━━━━━
🕐 `{now}`
⚡ Phase {phase} | {phase_names.get(phase, 'ADVANCED')}
📊 Trades: {total_trades} | Win Rate: {win_rate:.1%}
📋 Paper Mode: ✅ ACTIVE
━━━━━━━━━━━━━━━━━━━━━━━
_Swing trading engine v3.0 initialized_""")


# ── Trade events ──────────────────────────────────────────────────────────────

def send_trade_opened(result: dict, qi: Optional[dict] = None) -> None:
    """
    Send a rich trade-opened card with Quad-Intelligence summary.

    Args:
        result : open_swing_position return dict.
        qi     : quad_validate return dict (optional).
    """
    qi        = qi or {}
    consensus = qi.get("conviction", qi.get("consensus", "—"))
    votes_str = f"{qi.get('proceed_count','?')}/{qi.get('total_brains','3')} brains"
    thesis    = qi.get("thesis", "—")
    risks     = " | ".join(qi.get("key_risks", [])) or "—"
    size_mult = qi.get("size_multiplier", 1.0)
    now       = datetime.now(ET).strftime("%I:%M %p ET")
    send(f"""✅ *SWING POSITION OPENED* | {now}
━━━━━━━━━━━━━━━━━━━━━━━
📌 Symbol:    *{result['symbol']}*
📋 Setup:     {result['setup_type']}
📊 Shares:    {result['shares']} (QI ×{size_mult:.2f})
💵 Entry:     ${result['entry']:.2f}
🛑 Stop:      ${result['stop']:.2f}
🎯 Target 1:  ${result['target']:.2f}
🎯 Target 2:  ${result.get('target2', '—')}
━━━━━━━━━━━━━━━━━━━━━━━
🧠 *Quad-Intel: {consensus}* ({votes_str})
📝 _{thesis}_
⚠️ Risk: _{risks}_
━━━━━━━━━━━━━━━━━━━━━━━
_Paper trade logged_""")


def send_trade_closed(result: dict) -> None:
    """
    Send trade-closed notification.

    Args:
        result : _close_position return dict.
    """
    now   = datetime.now(ET).strftime("%I:%M %p ET")
    emoji = "💚" if result.get("pnl_pct", 0) > 0 else "🔴"
    send(f"""{emoji} *SWING POSITION CLOSED* | {now}
━━━━━━━━━━━━━━━━━━━━━━━
📌 Symbol:   *{result['symbol']}*
📋 Reason:   {result.get('exit_reason', '—')}
⏱ Hold:     {result.get('hold_days', 0)} days
💵 P&L:     {result.get('pnl_pct', 0):+.2f}% (${result.get('pnl_dollars', 0):+.2f})
━━━━━━━━━━━━━━━━━━━━━━━""")


def send_phase_transition(from_phase: int, to_phase: int, stats: dict) -> None:
    """
    Send phase transition alert.

    Args:
        from_phase : previous phase number.
        to_phase   : new phase number.
        stats      : trade stats dict from get_trade_stats().
    """
    names = {
        1: "LINEAR BANDIT",
        2: "NEURAL BANDIT",
        3: "DISTRIBUTIONAL RL",
        4: "CAUSAL DISCOVERY",
    }
    send(f"""🚀 *ATG PHASE TRANSITION*
━━━━━━━━━━━━━━━━━━━━━━━
{names.get(from_phase, 'Phase ' + str(from_phase))} → *{names.get(to_phase, 'Phase ' + str(to_phase))}*

📊 Trades:   {stats.get('total_closed', 0)}
📈 Win Rate: {stats.get('win_rate', 0):.1%}
💰 Avg P&L:  {stats.get('avg_pnl_pct', 0):+.2f}%
━━━━━━━━━━━━━━━━━━━━━━━
_ATG is evolving — criteria met automatically_""")


def send_daily_summary(
    phase: int, open_count: int, stats: dict, best_setup: dict
) -> None:
    """
    Send daily performance summary.

    Args:
        phase      : current phase.
        open_count : number of open positions.
        stats      : trade stats from get_trade_stats().
        best_setup : best_setup() dict from bandit.
    """
    now = datetime.now(ET).strftime("%b %d %Y")
    send(f"""🧘 *ATG DAILY SUMMARY* | {now}
━━━━━━━━━━━━━━━━━━━━━━━
⚡ Phase {phase} | Open: {open_count}
📊 Total Trades:  {stats.get('total_closed', 0)}
📈 Win Rate:      {stats.get('win_rate', 0):.1%}
💰 Avg P&L:       {stats.get('avg_pnl_pct', 0):+.2f}%
💵 Total P&L:     ${stats.get('total_pnl', 0):+.2f}
🏆 Best Setup:    {best_setup.get('setup_type', '—')} @ {best_setup.get('stop_multiplier', '—')}x ATR
━━━━━━━━━━━━━━━━━━━━━━━""")


def send_weekly_report(stats: dict) -> None:
    """
    Send Friday weekly performance report.

    Args:
        stats : dict with total_trades, wins, losses, win_rate, avg_win_pct,
                avg_loss_pct, total_pnl, phase.
    """
    now   = datetime.now(ET).strftime("%b %d %Y")
    wr    = stats.get("win_rate", 0)
    emoji = "📈" if wr >= 0.55 else "⚠️" if wr >= 0.40 else "🔴"
    send(f"""{emoji} *ATG WEEKLY REPORT* | {now}
━━━━━━━━━━━━━━━━━━━━━━━
📊 Trades This Week: {stats.get('total_trades', 0)}
✅ Wins:  {stats.get('wins', 0)} | ❌ Losses: {stats.get('losses', 0)}
📈 Win Rate:   {wr:.1%}
💚 Avg Win:    {stats.get('avg_win_pct', 0):+.2f}%
🔴 Avg Loss:   {stats.get('avg_loss_pct', 0):+.2f}%
💰 Week P&L:   ${stats.get('total_pnl', 0):+.2f}
⚡ Phase:      {stats.get('phase', 1)}
━━━━━━━━━━━━━━━━━━━━━━━""")


def send_position_aging_alert(
    symbol: str, hold_days: int, gain_pct: float, reason: str
) -> None:
    """
    Alert when a position is aging without hitting its target.

    Args:
        symbol    : ticker.
        hold_days : days held.
        gain_pct  : current gain/loss percentage.
        reason    : "approaching_max_hold" or "stagnating_loss".
    """
    if reason == "approaching_max_hold":
        send(
            f"⏰ *Position Aging Alert*\n"
            f"*{symbol}* — {hold_days}d held (max 42)\n"
            f"Current P&L: {gain_pct:+.2f}%\n"
            f"_Consider closing or tightening trailing stop_"
        )
    else:
        send(
            f"⚠️ *Stagnating Position*\n"
            f"*{symbol}* — {hold_days}d, {gain_pct:+.2f}%\n"
            f"_Time stop approaching — reassess_"
        )


def send_scan_result(setups: list, scan_type: str = "EOD", gate_reason: str = "") -> None:
    """
    Send scan result summary card.

    Args:
        setups     : list of setup dicts from the scanner.
        scan_type  : "EOD" or "WEEKLY".
        gate_reason: reason string if gate blocked the scan.
    """
    now = datetime.now(ET).strftime("%I:%M %p ET")

    if gate_reason and not setups:
        send(f"🚫 *ATG {scan_type} SCAN GATED* | {now}\n_{gate_reason}_")
        return

    if not setups:
        send(f"🧘 *ATG {scan_type} SCAN* | {now}\n\n_No qualifying swing setups today_")
        return

    lines = [f"🧘 *ATG {scan_type} SCAN* | {now}", f"Found *{len(setups)}* setup(s):\n"]
    for s in setups:
        emoji   = "🔼" if s["setup_type"] in ("BREAKOUT", "BASE_BREAKOUT") else \
                  "↩️"  if "PULLBACK" in s["setup_type"] else "🔄"
        dte_str = f"| DTE {s.get('days_to_earnings', '?')}d" if s.get("days_to_earnings") else ""
        lines.append(
            f"{emoji} *{s['symbol']}* — {s['setup_type']} {dte_str}\n"
            f"   Score: {s['score']} | R:R {s.get('rr_ratio','?')} | Vol: {s.get('volume_ratio',1):.1f}x\n"
            f"   Entry ~${s['price']:.2f} | Stop ${s['stop_loss']:.2f} | T1 ${s['target_price']:.2f}"
        )
    send("\n".join(lines))


def send_alert(msg: str) -> None:
    """
    Generic alert wrapper (circuit breakers, add-ons, regime changes).

    Args:
        msg: alert message text (may include Markdown).
    """
    send(f"🧘 *ATG ALERT*\n{msg}")
