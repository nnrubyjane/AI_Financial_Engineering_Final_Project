"""Market price request example for a KIS-style API workflow."""

from trading_system import config


SAMPLE_PRICE_DATA = {
    "005930": [73500, 73800, 73600, 73900, 73700, 71500],
    "000660": [184000, 185500, 186000, 184500, 185000, 188900],
}


def get_stock_price(access_token, symbol):
    """Get stock prices from mock data or a safe real-mode template."""
    if config.MOCK_MODE:
        return get_mock_stock_price(symbol)

    return request_real_stock_price_template(access_token, symbol)


def get_mock_stock_price(symbol):
    """Return fake stock price history for the demo."""
    prices = SAMPLE_PRICE_DATA.get(symbol, SAMPLE_PRICE_DATA["005930"])

    return {
        "symbol": symbol,
        "current_price": prices[-1],
        "price_history": prices,
    }


def request_real_stock_price_template(access_token, symbol):
    """Show how a real market request would be structured, but keep it disabled.

    A real implementation would usually:
    1. Build request headers with the access token.
    2. Send the stock symbol to the official KIS market price endpoint.
    3. Parse the JSON response and return the current price.

    This project does not include a real endpoint or real request code because
    it must be safe and runnable without credentials.
    """
    if not access_token:
        raise ValueError("Real mode needs an access token before requesting prices.")

    if not symbol:
        raise ValueError("A stock symbol is required.")

    raise NotImplementedError(
        "Real KIS market requests are a template only. Mock data is used by default."
    )

