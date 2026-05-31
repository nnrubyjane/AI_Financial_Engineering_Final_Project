"""Run the full mock trading workflow."""

from pathlib import Path

from trading_system import config
from trading_system.kis_auth import get_access_token
from trading_system.kis_market import get_stock_price
from trading_system.strategy import calculate_moving_average, generate_signal


def build_output_text(symbol, current_price, moving_average, signal, explanation, token):
    """Create clear console text for the trading result."""
    return "\n".join(
        [
            "=== Mock KIS API Trading System Demo ===",
            f"Mode: {'MOCK' if config.MOCK_MODE else 'REAL TEMPLATE'}",
            f"Dry run: {config.DRY_RUN}",
            f"Access token concept: {token}",
            f"Stock symbol: {symbol}",
            f"Current price: {current_price:,.0f} KRW",
            f"Moving average: {moving_average:,.2f} KRW",
            f"Generated signal: {signal}",
            f"Explanation: {explanation}",
            "Order status: No real order was placed. This is a simulation only.",
        ]
    )


def save_sample_output(output_text):
    """Save the demo output to the results folder."""
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "results" / "trading_sample_output.txt"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(output_text + "\n", encoding="utf-8")


def run_trading_demo():
    """Run authentication, market data, strategy, and reporting steps."""
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

