"""
ATG (Alpha Trading Guru) — Configuration v3.0
Swing trading system: clean rebuild, all env-based, no hardcoded secrets.
"""
import os
import numpy as np

# ── Identity ──────────────────────────────────────────────────────────────────
SYSTEM_NAME    = "ALPHA TRADING GURU"
SYSTEM_VERSION = "3.0.0"
SYSTEM_EMOJI   = "🧘"

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("ATG_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("ATG_TELEGRAM_CHAT_ID", "-5130564161")

# ── Alpaca (Paper Trading) ────────────────────────────────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
PAPER_MODE        = os.getenv("PAPER_MODE", "true").lower() == "true"

# ── Data APIs ─────────────────────────────────────────────────────────────────
POLYGON_API_KEY   = os.getenv("POLYGON_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

# ── Capital Router ────────────────────────────────────────────────────────────
CAPITAL_ROUTER_URL  = os.getenv("CAPITAL_ROUTER_URL", "http://localhost:8000")
CAPITAL_ROUTER_NAME = "ATG_SWING"   # registered system name in Capital Router

# ── Capital & Risk ────────────────────────────────────────────────────────────
INITIAL_CAPITAL    = float(os.getenv("INITIAL_CAPITAL",    "100000"))
# FIX [F26]: MAX_POSITIONS must match nexus-alpha execute_endpoint.py. Set via env var.
MAX_POSITIONS      = int(os.getenv("MAX_POSITIONS",        "3"))  # unified default: 3
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.01"))   # 1% per trade
MAX_SECTOR_PCT     = float(os.getenv("MAX_SECTOR_PCT",     "0.25"))   # 25% per sector
MAX_HOLD_DAYS      = int(os.getenv("MAX_HOLD_DAYS",        "42"))     # 6 weeks hard stop

# ── Entry Filters ─────────────────────────────────────────────────────────────
MIN_SETUP_SCORE      = int(os.getenv("MIN_SETUP_SCORE",      "75"))
MIN_VOLUME_RATIO_BO  = float(os.getenv("MIN_VOLUME_RATIO_BO", "1.5"))
MIN_RR_RATIO         = float(os.getenv("MIN_RR_RATIO",        "2.0"))
EARNINGS_BUFFER_DAYS = int(os.getenv("EARNINGS_BUFFER_DAYS",  "3"))
TOP_SECTORS_N        = int(os.getenv("TOP_SECTORS_N",         "4"))

# ── Market Regime Gate ────────────────────────────────────────────────────────
REQUIRE_SPY_UPTREND   = os.getenv("REQUIRE_SPY_UPTREND",   "true").lower() == "true"
REQUIRE_SECTOR_RS     = os.getenv("REQUIRE_SECTOR_RS",     "true").lower() == "true"
VIX_MAX               = float(os.getenv("VIX_MAX",           "28"))
VIX_SPIKE_THRESHOLD   = float(os.getenv("VIX_SPIKE_THRESHOLD", "3.0"))
MIN_MARKET_BREADTH    = float(os.getenv("MIN_MARKET_BREADTH",  "45"))

# ── Dynamic Exit Management ───────────────────────────────────────────────────
# FIX [F29-PENDING]: Ahmed confirm: extend to 21? Backtest shows 10d kills 54% of trades.
TIME_STOP_DAYS          = int(os.getenv("TIME_STOP_DAYS",           "10"))  # TODO: Ahmed to set to 21

# FIX [F38]: DTE 1-day gap — submissions validated at DTE≥22 to guarantee ≥21 at next-morning execution
MIN_DTE_AT_SUBMISSION   = int(os.getenv("MIN_DTE_AT_SUBMISSION",    "22"))  # +1 buffer for overnight gap
MIN_DTE_AT_EXECUTION    = int(os.getenv("MIN_DTE_AT_EXECUTION",     "21"))  # hard floor at execution time
BREAKEVEN_TRIGGER_ATR   = float(os.getenv("BREAKEVEN_TRIGGER_ATR",  "1.0"))
TRAILING_ACTIVATION_PCT = float(os.getenv("TRAILING_ACTIVATION_PCT","0.08"))
TRAILING_ATR_MULTIPLE   = float(os.getenv("TRAILING_ATR_MULTIPLE",  "2.0"))
STAGED_EXIT_ATR         = float(os.getenv("STAGED_EXIT_ATR",        "1.5"))
STAGED_EXIT_SIZE        = float(os.getenv("STAGED_EXIT_SIZE",       "0.50"))
ADDON_MIN_GAIN_PCT      = float(os.getenv("ADDON_MIN_GAIN_PCT",     "0.08"))
ADDON_SIZE_PCT          = float(os.getenv("ADDON_SIZE_PCT",         "0.50"))

# ── Bandit / RL ───────────────────────────────────────────────────────────────
SETUP_TYPES = [
    "BREAKOUT",        # Multi-week consolidation breakout + volume confirm
    "PULLBACK",        # Pullback to key MA in established uptrend
    "EARNINGS_DRIFT",  # Strong trend stock heading into favorable earnings
    "SECTOR_ROTATION", # Early entry in sector showing relative strength surge
    "MA_BOUNCE",       # Bounce off 50/200 MA with reversal signal
    "BASE_BREAKOUT",   # Breaking out of long base (8+ weeks compression)
]
NUM_SETUP_TYPES = len(SETUP_TYPES)

STOP_MULTIPLIERS     = np.linspace(1.0, 3.0, 11)
NUM_STOP_MULTIPLIERS = len(STOP_MULTIPLIERS)
NUM_BANDIT_ARMS      = NUM_SETUP_TYPES * NUM_STOP_MULTIPLIERS   # 66

CONTEXT_DIM = 32

CONTEXT_FEATURES = [
    "weekly_rsi", "monthly_rsi", "weekly_macd_hist", "atr_ratio", "volume_ratio",
    "price_to_50ma", "price_to_200ma", "ma_alignment", "breakout_strength",
    "price_mom_4wk", "price_mom_12wk",
    "sector_rs_1m", "sector_rs_3m", "sector_momentum",
    "days_to_earnings", "earnings_surprise", "revenue_growth", "earnings_growth",
    "spy_weekly_trend", "vix_level", "fear_greed", "market_breadth",
    "yield_curve", "institutional_own", "short_interest", "insider_activity",
    "volume_accum", "debt_to_equity", "profit_margin",
    "day_of_week", "week_of_month", "market_cap_tier",
]

# ── Conviction sizing (score → risk multiplier) ───────────────────────────────
CONVICTION_SIZING = {
    90: 1.50,
    85: 1.25,
    80: 1.00,
    75: 0.80,
}

# ── Correlation filter ────────────────────────────────────────────────────────
MAX_CORRELATED_POSITIONS = int(os.getenv("MAX_CORRELATED_POSITIONS", "2"))
CORRELATED_GROUPS = {
    "mega_tech":     ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "semis":         ["NVDA", "AMD",  "AVGO",  "ARM",  "SMCI"],
    "fintech":       ["V",    "MA",   "PYPL",  "SQ",   "HOOD"],
    "crypto":        ["COIN", "MSTR"],
    "cybersecurity": ["CRWD", "PANW", "FTNT",  "NET"],
    "cloud_saas":    ["SNOW", "DDOG", "NET",   "NOW",  "CRM"],
}

# ── Phase Thresholds ──────────────────────────────────────────────────────────
PHASE_1_MIN_TRADES   = int(os.getenv("PHASE_1_MIN_TRADES",   "100"))
PHASE_1_MIN_WIN_RATE = float(os.getenv("PHASE_1_MIN_WIN_RATE","0.55"))
PHASE_1_MIN_DAYS     = int(os.getenv("PHASE_1_MIN_DAYS",     "30"))
PHASE_2_MIN_TRADES   = int(os.getenv("PHASE_2_MIN_TRADES",   "250"))
PHASE_2_MIN_WIN_RATE = float(os.getenv("PHASE_2_MIN_WIN_RATE","0.58"))
PHASE_2_MIN_DAYS     = int(os.getenv("PHASE_2_MIN_DAYS",     "60"))
PHASE_3_MIN_TRADES   = int(os.getenv("PHASE_3_MIN_TRADES",   "500"))
PHASE_3_MIN_WIN_RATE = float(os.getenv("PHASE_3_MIN_WIN_RATE","0.60"))
PHASE_3_MIN_DAYS     = int(os.getenv("PHASE_3_MIN_DAYS",     "90"))
PERFORMANCE_WINDOW   = int(os.getenv("PERFORMANCE_WINDOW",   "50"))

# ── Circuit Breakers ──────────────────────────────────────────────────────────
CIRCUIT_DAILY_LOSS_PCT  = float(os.getenv("CIRCUIT_DAILY_LOSS_PCT",  "0.03"))
CIRCUIT_WEEKLY_LOSS_PCT = float(os.getenv("CIRCUIT_WEEKLY_LOSS_PCT", "0.08"))
CIRCUIT_DRAWDOWN_PCT    = float(os.getenv("CIRCUIT_DRAWDOWN_PCT",    "0.05"))

# ── Scan Universe ─────────────────────────────────────────────────────────────
SCAN_UNIVERSE_SIZE = int(os.getenv("SCAN_UNIVERSE_SIZE", "50"))

# ── Database ──────────────────────────────────────────────────────────────────
DATA_PATH = os.getenv("DATA_PATH", "/data")
DB_PATH   = os.environ.get("DB_PATH", "/data/atg_swing.db")
