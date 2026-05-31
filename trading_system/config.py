"""Configuration for the KIS-style mock trading demo.

The safe defaults are important:
- MOCK_MODE is True, so the project does not need real API credentials.
- DRY_RUN is True, so the program only prints simulated decisions.
"""

import os


def read_bool_env(name, default):
    """Read a boolean environment variable in a beginner-friendly way."""
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in ["1", "true", "yes", "on"]


# Default mode: safe mock mode.
MOCK_MODE = read_bool_env("MOCK_MODE", True)

# Dry-run mode means "show what would happen" instead of placing an order.
DRY_RUN = read_bool_env("DRY_RUN", True)

# Real trading is disabled unless a user edits the code and environment on purpose.
REAL_TRADING_ENABLED = read_bool_env("REAL_TRADING_ENABLED", False)

# Optional values for a real KIS API template. They are not required in mock mode.
KIS_APP_KEY = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET", "")
KIS_ACCOUNT_NUMBER = os.getenv("KIS_ACCOUNT_NUMBER", "")
KIS_BASE_URL = os.getenv("KIS_BASE_URL", "")

# Public example stock symbol. 005930 is Samsung Electronics in Korea.
STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "005930")

# Strategy settings.
MOVING_AVERAGE_WINDOW = 5
THRESHOLD_PERCENT = 0.02


def validate_safety_settings():
    """Stop the program if settings look unsafe for an educational demo."""
    if not MOCK_MODE and not DRY_RUN and not REAL_TRADING_ENABLED:
        raise RuntimeError(
            "Unsafe settings: real mode with DRY_RUN=False requires "
            "REAL_TRADING_ENABLED=True. This project is safe by default."
        )

