# fetcher/crawling_fetcher.py
import logging
import os
from firecrawl import FirecrawlApp

logger = logging.getLogger(__name__)


def fetch_articles_by_crawling(news_websites_crawl: dict,
                               reuters_date_array: list,
                               coindesk_date_array: list):
    """
    根据 crawling 配置来爬取文章。可根据日期等参数对 includePaths 做动态扩展。
    返回一个列表，内部是所有网站的所有文章。
    """
    api_key_firecrawl = os.getenv("FIRECRAWL_API_KEY")
    logger.info(f"api_key_firecrawl: {api_key_firecrawl}")
    if not api_key_firecrawl:
        raise ValueError("Firecrawl API key not set in environment variables.")

    app = FirecrawlApp(api_key=api_key_firecrawl)
    all_articles = []

    for website, rules in news_websites_crawl.items():
        include_paths = rules.get('includePaths', [])

        # 根据特定关键字对 includePaths 做动态操作
        if "reuters" in website:
            include_paths.extend(reuters_date_array)
        if "coindesk" in website:
            include_paths.extend(coindesk_date_array)
        # 这里也可添加别的网站的日期处理逻辑

        try:
            logger.info(f"[Crawling] Website: {website}, with rules: {rules}")
            crawl_status = app.crawl_url(
                website,
                params={
                    'limit': rules.get('limit', 2),
                    'scrapeOptions': {'formats': ['markdown', 'html']},
                    'includePaths': include_paths,
                    'excludePaths': rules.get('excludePaths', []),
                },
                poll_interval=1
            )

            news_articles = crawl_status.get('data', [])
            for article in news_articles:
                logger.info(f"Article URL: {article.get('metadata', {}).get('sourceURL', '')}")
            if news_articles:
                all_articles.extend(news_articles)
            else:
                logger.warning(f"No articles found for {website}.")

        except Exception as e:
            logger.error(f"Error while crawling {website}: {e}")

    logger.info(f"[Crawling] Total articles fetched: {len(all_articles)}")
    return all_articles