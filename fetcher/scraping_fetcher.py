# fetcher/scraping_fetcher.py
import logging
import os
from firecrawl import FirecrawlApp, ScrapeOptions

DEFAULT_WAIT_MS = 2000
DEFAULT_TIMEOUT_MS = 40000

logger = logging.getLogger(__name__)


def fetch_articles_by_scraping(news_websites_scraping: dict):
    """
    根据 scraping 配置 (多个 URL) 来抓取文章。
    返回一个所有文章对象组成的列表，每个元素都应包含必要的 'markdown', 'metadata' 字段。
    """
    api_key_firecrawl = os.getenv("FIRECRAWL_API_KEY")
    logger.info(f"api_key_firecrawl: {api_key_firecrawl}")
    if not api_key_firecrawl:
        raise ValueError("Firecrawl API key not set in environment variables.")

    wait_ms = int(os.getenv("SCRAPE_WAIT_MS", DEFAULT_WAIT_MS))
    timeout_ms = int(os.getenv("SCRAPE_TIMEOUT_MS", DEFAULT_TIMEOUT_MS))

    app = FirecrawlApp(api_key=api_key_firecrawl)
    all_articles = []

    for url in news_websites_scraping:
        try:
            logger.info(f"[Scraping] Fetching URL: {url}")
            options = ScrapeOptions(
                formats=["markdown", "html"], waitFor=wait_ms, timeout=timeout_ms
            )
            scrape_result = app.scrape_url(url, options=options)

            if scrape_result and not isinstance(scrape_result, dict):
                scrape_result = scrape_result.model_dump()

            if "markdown" in scrape_result:
                all_articles.append(scrape_result)
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")

    logger.info(f"[Scraping] Total articles fetched: {len(all_articles)}")
    return all_articles
