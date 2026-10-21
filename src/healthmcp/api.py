"""
Where the data comes from. Two interchangeable backends with one method, `get(url, params) -> JSON`:

  LiveAPI : the real openFDA and ClinicalTrials.gov APIs, over HTTP. Needs no key.
  DemoAPI : answers the same requests with made-up data in the same response shape (see demo_data.py), so the server runs anywhere and is
            testable without a network.

The tools are written once, against that JSON shape, so the same parsing code runs in both modes.
"""

import httpx

OPENFDA = "https://api.fda.gov"
CTGOV = "https://clinicaltrials.gov/api/v2"


class SourceError(RuntimeError):
    """The data source could not answer. The message is written for the model and the person to read."""


class NoMatch(SourceError):
    pass


class LiveAPI:
    name = "live (openFDA + ClinicalTrials.gov)"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 15.0):
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport, headers={"User-Agent": "healthmcp-demo"})

    async def get(self, url: str, params: dict | None = None) -> dict:
        try:
            r = await self._client.get(url, params=params)
        except httpx.HTTPError as e:
            raise SourceError(f"could not reach the data source ({type(e).__name__}). Try again in a moment.") from e
        if r.status_code == 404:                       # openFDA answers 404 when a search has no results
            raise NoMatch("no matches")
        if r.status_code == 429:
            raise SourceError("the data source is rate limiting requests. Wait a minute and try again.")
        if r.status_code >= 400:
            raise SourceError(f"the data source returned an error ({r.status_code}).")
        try:
            return r.json()
        except ValueError as e:
            raise SourceError("the data source returned something that was not JSON.") from e
