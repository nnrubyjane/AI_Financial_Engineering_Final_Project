"""Mock market data for the KIS-style API workflow."""

from trading_system import config


SAMPLE_PRICE_DATA = {
    # My classroom story: Samsung moved mostly sideways, then dipped today.
    # That makes the BUY signal easy to explain with a moving average.
    "005930": [73200, 73500, 73400, 73600, 73300, 71700],
    "000660": [181500, 182300, 181800, 183200, 182700, 185900],
}


def get_stock_price(access_token, symbol):
    """Get price data from mock samples unless real mode is manually enabled."""
    if config.MOCK_MODE:
        return get_mock_stock_price(symbol)

    return request_real_stock_price_template(access_token, symbol)


def get_mock_stock_price(symbol):
    """Return one small fake price history that is easy to inspect by eye."""
    prices = SAMPLE_PRICE_DATA.get(symbol, SAMPLE_PRICE_DATA["005930"])

    return {
        "symbol": symbol,
        "current_price": prices[-1],
        "price_history": prices,
    }


def request_real_stock_price_template(access_token, symbol):
    """Show how a real market request would be structured, but keep it disabled.

    If this were a real project, the steps would usually be:
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
