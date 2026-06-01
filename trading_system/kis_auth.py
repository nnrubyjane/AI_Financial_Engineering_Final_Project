"""Access token idea for the KIS-style classroom demo.

The real KIS API uses authentication before data requests. In this project I
only imitate that shape, so I can explain the workflow without handling secrets.
"""

from trading_system import config


def get_access_token():
    """Return a fake token in mock mode, like a practice permission pass."""
    if config.MOCK_MODE:
        return "mock_access_token"

    return request_real_access_token_template()


def request_real_access_token_template():
    """Show the safe structure for real authentication without implementing it.

    If this were connected to the real service, the steps would usually be:
    1. Read KIS_APP_KEY and KIS_APP_SECRET from environment variables.
    2. Send them to the official KIS authentication endpoint.
    3. Receive an access token in the API response.

    I intentionally leave the network request disabled for this final project.
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
