"""Small moving-average strategy for the mock trading demo.

I wrote this like a notebook example: calculate one average, compare today's
price, and print the signal. There is no hidden trading logic.
"""


def calculate_moving_average(prices, window):
    """Average the latest prices, just like a simple calculator exercise."""
    if window <= 0:
        raise ValueError("window must be greater than zero.")

    if len(prices) < window:
        raise ValueError("not enough prices to calculate the moving average.")

    recent_prices = prices[-window:]
    return sum(recent_prices) / window


def generate_signal(current_price, moving_average, threshold_percent):
    """Return BUY, SELL, or HOLD using one easy threshold rule."""
    buy_line = moving_average * (1 - threshold_percent)
    sell_line = moving_average * (1 + threshold_percent)
    gap_percent = abs(current_price - moving_average) / moving_average

    if current_price < buy_line:
        explanation = (
            f"The price is {gap_percent:.2%} below the moving average, which is "
            f"more than the {threshold_percent:.1%} buffer. My demo prints BUY."
        )
        return "BUY", explanation

    if current_price > sell_line:
        explanation = (
            f"The price is {gap_percent:.2%} above the moving average, which is "
            f"more than the {threshold_percent:.1%} buffer. My demo prints SELL."
        )
        return "SELL", explanation

    explanation = (
        f"The price is only {gap_percent:.2%} away from the moving average, "
        f"inside the {threshold_percent:.1%} buffer. My demo prints HOLD."
    )
    return "HOLD", explanation
