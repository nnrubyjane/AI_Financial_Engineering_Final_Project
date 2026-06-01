"""Run my mock trading workflow from token idea to printed signal."""

from pathlib import Path

from trading_system import config
from trading_system.kis_auth import get_access_token
from trading_system.kis_market import get_stock_price
from trading_system.strategy import calculate_moving_average, generate_signal


def build_output_text(symbol, current_price, moving_average, signal, explanation, token):
    """Create the console report I can read aloud during the oral test."""
    risk_note = (
        "Risk note: A real system should check cash, position size, stop-loss, "
        "and daily loss limit before any order. This demo stops at a printed signal."
    )

    return "\n".join(
        [
            "=== My Mock KIS Trading Walkthrough ===",
            f"Practice mode: {'mock prices' if config.MOCK_MODE else 'real API template'}",
            f"Safety switch: dry_run={config.DRY_RUN}",
            f"Lesson token: {token}",
            f"Classroom stock code: {symbol}",
            f"Latest mock price: {current_price:,.0f} KRW",
            f"Five-price average: {moving_average:,.2f} KRW",
            f"My strategy signal: {signal}",
            f"Reason I can explain: {explanation}",
            risk_note,
            "Execution note: No real order was placed. This is a simulation only.",
        ]
    )


def save_sample_output(output_text):
    """Save the same text that appears in the terminal."""
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "results" / "trading_sample_output.txt"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(output_text + "\n", encoding="utf-8")


def run_trading_demo():
    """Run the small demo in the same order I would explain it."""
    config.validate_safety_settings()

    token = get_access_token()
    market_data = get_stock_price(token, config.STOCK_SYMBOL)
    prices = market_data["price_history"]
    current_price = market_data["current_price"]

    moving_average = calculate_moving_average(prices, config.MOVING_AVERAGE_WINDOW)
    signal, explanation = generate_signal(
        current_price,
        moving_average,
        config.THRESHOLD_PERCENT,
    )

    output_text = build_output_text(
        market_data["symbol"],
        current_price,
        moving_average,
        signal,
        explanation,
        token,
    )
    save_sample_output(output_text)
    return output_text


if __name__ == "__main__":
    print(run_trading_demo())
