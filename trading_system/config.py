"""Settings for my KIS-style classroom trading demo.

I keep the defaults boring on purpose:
- MOCK_MODE=True means the demo uses sample prices, not a real brokerage API.
- DRY_RUN=True means the program explains a decision but never sends an order.
"""

import os


def read_bool_env(name, default):
    """Read a true/false setting without making mock mode depend on a .env file."""
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in ["1", "true", "yes", "on"]


# The project should run immediately after download, so mock mode stays on.
MOCK_MODE = read_bool_env("MOCK_MODE", True)

# Dry run is like practice mode: explain the idea, do not trade.
DRY_RUN = read_bool_env("DRY_RUN", True)

# This extra switch makes accidental real trading harder.
REAL_TRADING_ENABLED = read_bool_env("REAL_TRADING_ENABLED", False)

# These are only read as empty optional values. The demo never needs real secrets.
KIS_APP_KEY = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET", "")
KIS_ACCOUNT_NUMBER = os.getenv("KIS_ACCOUNT_NUMBER", "")
KIS_BASE_URL = os.getenv("KIS_BASE_URL", "")

# Public example stock symbol. 005930 is Samsung Electronics in Korea.
STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "005930")

# A five-price average is small enough to calculate by hand during an oral test.
MOVING_AVERAGE_WINDOW = 5

# 1.8% gives the strategy a small buffer before it prints BUY or SELL.
THRESHOLD_PERCENT = 0.018


def validate_safety_settings():
    """Stop the program if someone combines settings in an unsafe way."""
    if not MOCK_MODE and not DRY_RUN and not REAL_TRADING_ENABLED:
        raise RuntimeError(
            "Unsafe settings: real mode with DRY_RUN=False requires "
            "REAL_TRADING_ENABLED=True. This project is safe by default."
        )
