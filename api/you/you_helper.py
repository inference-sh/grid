"""
Shared helper for You.com API apps.
Provides HTTP client setup and common request handling.
"""

import os
import logging
import requests
from typing import Optional


def get_api_key() -> str:
    key = os.environ.get("YOU_KEY")
    if not key:
        raise RuntimeError("YOU_KEY environment variable is required.")
    return key


def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)


def search_request(
    api_key: str,
    query: str,
    count: int = 10,
    country: Optional[str] = None,
    search_lang: Optional[str] = None,
    safe_search: Optional[str] = None,
    livecrawl: bool = False,
    include_domains: Optional[list] = None,
    exclude_domains: Optional[list] = None,
    boost_domains: Optional[list] = None,
    recent_past_day: bool = False,
    recent_past_week: bool = False,
    recent_past_month: bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Call the You.com Search API."""
    log = logger or logging.getLogger(__name__)

    params = {"query": query, "count": count}

    if country:
        params["country"] = country
    if search_lang:
        params["search_lang"] = search_lang
    if safe_search:
        params["safe_search"] = safe_search
    if livecrawl:
        params["livecrawl"] = "always"
    if include_domains:
        params["include_domains"] = ",".join(include_domains)
    if exclude_domains:
        params["exclude_domains"] = ",".join(exclude_domains)
    if boost_domains:
        params["boost_domains"] = ",".join(boost_domains)
    if recent_past_day:
        params["recent_past_day"] = "true"
    if recent_past_week:
        params["recent_past_week"] = "true"
    if recent_past_month:
        params["recent_past_month"] = "true"

    headers = {"X-API-Key": api_key}

    log.info(f"Search query: {query}, count: {count}")
    response = requests.get(
        "https://api.you.com/v1/agents/search",
        params=params,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def research_request(
    api_key: str,
    query: str,
    research_effort: str = "standard",
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Call the You.com Research API."""
    log = logger or logging.getLogger(__name__)

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "input": query,
        "research_effort": research_effort,
    }

    log.info(f"Research query: {query[:100]}, effort: {research_effort}")
    response = requests.post(
        "https://api.you.com/v1/research",
        json=payload,
        headers=headers,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def finance_research_request(
    api_key: str,
    query: str,
    research_effort: str = "deep",
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Call the You.com Finance Research API."""
    log = logger or logging.getLogger(__name__)

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "input": query,
        "research_effort": research_effort,
    }

    log.info(f"Finance research: {query[:100]}, effort: {research_effort}")
    response = requests.post(
        "https://api.you.com/v1/finance_research",
        json=payload,
        headers=headers,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def contents_request(
    api_key: str,
    urls: list,
    format: str = "markdown",
    crawl_timeout: int = 10,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Call the You.com Contents API."""
    log = logger or logging.getLogger(__name__)

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "urls": urls,
        "formats": [format],
        "crawl_timeout": crawl_timeout,
    }

    log.info(f"Fetching contents for {len(urls)} URL(s)")
    response = requests.post(
        "https://ydc-index.io/v1/contents",
        json=payload,
        headers=headers,
        timeout=max(crawl_timeout + 10, 30),
    )
    response.raise_for_status()
    return response.json()
