"""Simple moving-average trading strategy.

This file is intentionally small so each function is easy to explain during an
oral test.
"""


def calculate_moving_average(prices, window):
    """Calculate the average of the latest `window` prices."""
    if window <= 0:
        raise ValueError("window must be greater than zero.")

    if len(prices) < window:
        raise ValueError("not enough prices to calculate the moving average.")

    recent_prices = prices[-window:]
    return sum(recent_prices) / window


def generate_signal(current_price, moving_average, threshold_percent):
    """Return BUY, SELL, or HOLD using a simple threshold rule."""
    buy_line = moving_average * (1 - threshold_percent)
    sell_line = moving_average * (1 + threshold_percent)

    if current_price < buy_line:
        explanation = (
            "Current price is below the moving average by more than the "
            "threshold, so the mock strategy prints BUY."
        )
        return "BUY", explanation

    if current_price > sell_line:
        explanation = (
            "Current price is above the moving average by more than the "
            "threshold, so the mock strategy prints SELL."
        )
        return "SELL", explanation

    explanation = (
        "Current price is close to the moving average, so the mock strategy "
        "prints HOLD."
    )
    return "HOLD", explanation

