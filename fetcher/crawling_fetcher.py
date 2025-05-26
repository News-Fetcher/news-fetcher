# fetcher/crawling_fetcher.py
import logging
import os
from firecrawl import FirecrawlApp

logger = logging.getLogger(__name__)


def fetch_articles_by_crawling(news_websites_crawl: dict,
                               dynamic_paths: dict):
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
        logger.info(f"website: {website}, rules: {rules}")
        include_paths = list(rules.get('includePaths', []))

        # 根据配置中的关键字对 includePaths 做动态操作
        for keyword, paths in dynamic_paths.items():
            logger.info(f"keyword: {keyword}, paths: {paths}")
            if keyword in website:
                logger.info(f"extend paths: {paths}")
                include_paths.extend(paths)

        rules['includePaths'] = include_paths
        try:
            logger.info(f"[Crawling] Website: {website}, with rules: {rules}")
            crawl_status = app.crawl_url(
                website,
                params={
                    "limit": rules.get('limit', 2),
                    "includePaths": include_paths,
                    "excludePaths": rules.get('excludePaths', []),
                    "scrapeOptions": {"formats": ["markdown", "html"]},
                },
                poll_interval=1,
            )

            # firecrawl-py >= 2 returns Pydantic models; convert to dict for
            # compatibility with existing dict-based code
            status_dict = crawl_status.model_dump()
            news_articles = status_dict.get("data", [])
            normalized_articles = []
            for article in news_articles:
                if not isinstance(article, dict):
                    article = article.model_dump()
                logger.info(
                    f"Article URL: {article.get('metadata', {}).get('sourceURL', '')}"
                )
                normalized_articles.append(article)
            if normalized_articles:
                all_articles.extend(normalized_articles)
            else:
                logger.warning(f"No articles found for {website}.")

        except Exception as e:
            logger.error(f"Error while crawling {website}: {e}")

    logger.info(f"[Crawling] Total articles fetched: {len(all_articles)}")
    return all_articles
