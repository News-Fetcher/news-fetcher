# fetcher/scraping_fetcher.py
import logging
import os
from firecrawl import FirecrawlApp, ScrapeOptions

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

    app = FirecrawlApp(api_key=api_key_firecrawl)
    all_articles = []

    for url in news_websites_scraping:
        try:
            logger.info(f"[Scraping] Fetching URL: {url}")
            scrape_options = ScrapeOptions(formats=["markdown", "html"])
            # firecrawl-py may not handle custom objects when serializing the
            # request payload. Convert ScrapeOptions to a plain dict first to
            # avoid "Object of type ScrapeOptions is not JSON serializable"
            # errors.
            if hasattr(scrape_options, "model_dump"):
                scrape_options = scrape_options.model_dump()
            else:
                try:
                    from dataclasses import asdict, is_dataclass
                    if is_dataclass(scrape_options):
                        scrape_options = asdict(scrape_options)
                    else:
                        scrape_options = scrape_options.__dict__
                except Exception:  # pragma: no cover - fallback for unexpected cases
                    scrape_options = scrape_options.__dict__

            scrape_result = app.scrape_url(
                url,
                scrape_options=scrape_options,
            )

            # firecrawl-py >= 2 returns a model; convert to dict so existing
            # code using dict access continues to work
            if scrape_result:
                if not isinstance(scrape_result, dict):
                    scrape_result = scrape_result.model_dump()

                # 如果有 markdown，就代表成功抓取到内容
                if "markdown" in scrape_result:
                    # 保持和原有逻辑类似，封装成一个 list
                    news_articles = [scrape_result]
                    all_articles.extend(news_articles)
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")

    logger.info(f"[Scraping] Total articles fetched: {len(all_articles)}")
    return all_articles
