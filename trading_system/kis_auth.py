"""Access token example for a Korea Investment Securities API concept.

In a real API, an access token proves that the program is allowed to make
requests. This file keeps that idea visible, but the default behavior is mock
mode and does not contact any real financial service.
"""

from trading_system import config


def get_access_token():
    """Return an access token for the demo workflow."""
    if config.MOCK_MODE:
        return "mock_access_token"

    return request_real_access_token_template()


def request_real_access_token_template():
    """Show the safe structure for real authentication without implementing it.

    A real implementation would usually:
    1. Read KIS_APP_KEY and KIS_APP_SECRET from environment variables.
    2. Send them to the official KIS authentication endpoint.
    3. Receive an access token in the API response.

    This educational project intentionally leaves the network request disabled.
    """
    if not config.KIS_APP_KEY or not config.KIS_APP_SECRET:
        raise ValueError(
            "Real mode needs KIS_APP_KEY and KIS_APP_SECRET from the environment. "
            "Do not write secrets into source code."
        )

    raise NotImplementedError(
        "Real KIS authentication is a template only. Keep MOCK_MODE=True for "
        "the university demo."
    )

