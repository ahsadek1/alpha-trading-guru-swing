"""
ATG v3.0 — Startup validator.

Verifies all modules import cleanly, DB initialises, bandit loads, and
warm-start runs without error.  Run locally before deploying.
"""
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("validate")

# ── Set up a temp data path ────────────────────────────────────────────────────
os.environ.setdefault("DATA_PATH", "/tmp/atg_validate_data")
os.makedirs(os.environ["DATA_PATH"], exist_ok=True)

errors = []

# ── Import checks ──────────────────────────────────────────────────────────────
modules_to_check = [
    "config.settings",
    "src.database",
    "src.bandit",
    "src.market_regime",
    "src.context_builder",
    "src.capital_router",
    "src.swing_scanner",
    "src.trade_executor",
    "src.quad_intelligence",
    "src.deepseek_analyst",
    "src.telegram_bot",
    "src.phase_manager",
    "src.backtest_warmstart",
    "src.self_evolving_orchestrator",
]

for mod in modules_to_check:
    try:
        __import__(mod)
        log.info("  ✅ %s", mod)
    except Exception as e:
        log.error("  ❌ %s: %s", mod, e)
        errors.append((mod, str(e)))

# ── DB init ────────────────────────────────────────────────────────────────────
try:
    from src.database import initialize_database, get_trade_stats
    initialize_database()
    stats = get_trade_stats()
    log.info("  ✅ Database init OK | stats=%s", stats)
except Exception as e:
    log.error("  ❌ Database init: %s", e)
    errors.append(("database_init", str(e)))

# ── Bandit init ────────────────────────────────────────────────────────────────
try:
    from src.bandit import AutonomousSwingBandit
    import numpy as np
    b = AutonomousSwingBandit(alpha=1.0)
    ctx = np.full(32, 0.5, dtype=np.float32)
    arm = b.select_arm(ctx)
    b.update(arm, ctx, 0.05)
    setup, stop = b.decode_arm(arm)
    log.info("  ✅ Bandit OK | arm=%d setup=%s stop=%.2f", arm, setup, stop)
except Exception as e:
    log.error("  ❌ Bandit: %s", e)
    errors.append(("bandit", str(e)))

# ── Warm-start ─────────────────────────────────────────────────────────────────
try:
    from src.bandit import AutonomousSwingBandit
    from src.backtest_warmstart import warmstart_bandit
    b2 = AutonomousSwingBandit()
    n  = warmstart_bandit(b2)
    log.info("  ✅ Warm-start OK | %d rows processed", n)
except Exception as e:
    log.error("  ❌ Warm-start: %s", e)
    errors.append(("warmstart", str(e)))

# ── Settings sanity ────────────────────────────────────────────────────────────
try:
    from config.settings import (
        NUM_BANDIT_ARMS, SETUP_TYPES, CONTEXT_DIM, CAPITAL_ROUTER_NAME
    )
    assert NUM_BANDIT_ARMS == 66, f"Expected 66 arms, got {NUM_BANDIT_ARMS}"
    assert CAPITAL_ROUTER_NAME == "ATG_SWING", f"Router name wrong: {CAPITAL_ROUTER_NAME}"
    assert CONTEXT_DIM == 32
    log.info("  ✅ Settings OK | arms=%d router_name=%s", NUM_BANDIT_ARMS, CAPITAL_ROUTER_NAME)
except (AssertionError, Exception) as e:
    log.error("  ❌ Settings: %s", e)
    errors.append(("settings", str(e)))

# ── Result ─────────────────────────────────────────────────────────────────────
print()
if errors:
    log.error("VALIDATION FAILED — %d error(s):", len(errors))
    for mod, err in errors:
        log.error("  %s: %s", mod, err)
    sys.exit(1)
else:
    log.info("✅ VALIDATION PASSED — all checks green")
    sys.exit(0)
