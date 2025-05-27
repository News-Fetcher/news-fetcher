# fetcher/topic_fetcher.py
import logging
import os
from typing import List
from firecrawl import FirecrawlApp, ScrapeOptions

DEFAULT_WAIT_MS = 2000
DEFAULT_TIMEOUT_MS = 40000

logger = logging.getLogger(__name__)


def fetch_articles_from_json(json_file: str) -> List[dict]:
    """Read a JSON file containing article URLs and scrape them."""
    import json

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"{json_file} not found")

    with open(json_file, "r", encoding="utf-8") as f:
        try:
            url_list = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")
            url_list = []

    if not isinstance(url_list, list):
        raise ValueError(f"{json_file} should contain a JSON list of URLs")

    api_key_firecrawl = os.getenv("FIRECRAWL_API_KEY")
    logger.info(f"api_key_firecrawl: {api_key_firecrawl}")
    if not api_key_firecrawl:
        raise ValueError("Firecrawl API key not set in environment variables.")

    wait_ms = int(os.getenv("SCRAPE_WAIT_MS", DEFAULT_WAIT_MS))
    timeout_ms = int(os.getenv("SCRAPE_TIMEOUT_MS", DEFAULT_TIMEOUT_MS))

    app = FirecrawlApp(api_key=api_key_firecrawl)
    articles = []

    for url in url_list:
        try:
            logger.info(f"[Topic] Fetching URL: {url}")
            # options = ScrapeOptions(formats=["markdown", "html"], waitFor=wait_ms, timeout=timeout_ms)
            # result = app.scrape_url(url, options=options)

            result = app.scrape_url(
                url,		
                formats= [ 'markdown' ],
                only_main_content= True,
                wait_for= wait_ms
            )

            if result and not isinstance(result, dict):
                result = result.model_dump()
            if "markdown" in result:
                articles.append(result)
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")

    logger.info(f"[Topic] Total articles fetched: {len(articles)}")
    return articles
