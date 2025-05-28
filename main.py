# main.py
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

from fetcher.scraping_fetcher import fetch_articles_by_scraping
from fetcher.crawling_fetcher import fetch_articles_by_crawling
from fetcher.topic_fetcher import fetch_articles_from_json
from utils.common_utils import load_json_config
from utils.firebase_utils import initialize_firebase
from podcast_generator import generate_full_podcast, generate_topic_podcast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def main():
    # 1. 初始化 Firebase
    service_account_path = "./serviceAccountKey.json"
    if not os.path.exists(service_account_path):
        raise ValueError("serviceAccountKey.json not found in the current directory.")

    initialize_firebase(service_account_path)

    # 2. 读取环境变量
    fetcher_method = os.getenv("FETCHER_METHOD", "crawling")  # 默认用 crawling
    logger.info(f"FETCHER_METHOD: {fetcher_method}")

    # 3. 根据配置动态生成 includePaths
    config_file = os.getenv("CONFIG_FILE", "config.json")
    try:
        global_cfg = load_json_config(config_file)
        logger.info(f"Successfully loaded {config_file} configuration.")
    except Exception as e:
        logger.error(f"Failed to load {config_file}: {e}")
        global_cfg = {}

    dynamic_paths = {}
    dynamic_config = global_cfg.get("news_dynamic_paths", {})
    for keyword, info in dynamic_config.items():
        fmt = info.get("date_format", "%Y-%m-%d")
        offsets = info.get("days_offset", [0])
        dynamic_paths[keyword] = [
            (datetime.now() + timedelta(days=off)).strftime(fmt)
            for off in offsets
        ]
    
    logger.info(f"dynamic_paths: {dynamic_paths}")

    # 4. 加载或解析爬取/抓取配置
    crawl_config_file = os.getenv("CRAWL_CONFIG_FILE", "news_websites_crawl_coindesk.json")
    try:
        news_websites_crawl = load_json_config(crawl_config_file)
        logger.info(f"Successfully loaded {crawl_config_file} configuration.")
    except Exception as e:
        logger.error(f"Failed to load {crawl_config_file}: {e}")
        news_websites_crawl = {}

    # 5. 根据环境变量或本地文件加载 scraping 配置
    try:
        import json
        scraping_config_str = os.getenv("SCRAPING_CONFIG")
        if scraping_config_str and scraping_config_str != "{}":
            news_websites_scraping = json.loads(scraping_config_str)
            logger.info("Successfully loaded news_websites_scraping from environment.")
        else:
            news_websites_scraping = load_json_config("news_websites_scraping.json")
            logger.info("Successfully loaded local news_websites_scraping.json config.")
    except Exception as e:
        logger.error(f"Failed to load news_websites_scraping configuration: {e}")
        news_websites_scraping = {}

    # 6. 获取新闻文章（抓取/爬取/专题列表）
    if fetcher_method == "scraping":
        all_articles = fetch_articles_by_scraping(news_websites_scraping)
    elif fetcher_method == "crawling":
        all_articles = fetch_articles_by_crawling(news_websites_crawl,
                                                 dynamic_paths)
    elif fetcher_method == "topic":
        topic_file = os.getenv("TOPIC_URLS_FILE", "news_websites_scraping.json")
        all_articles = fetch_articles_from_json(topic_file)
    else:
        logger.warning("Unknown FETCHER_METHOD. No articles fetched.")
        all_articles = []

    # 7. 调用播客生成主流程（包含摘要、TTS、音频合并、封面生成、上传等）
    output_folder = "podcast_audio"
    Path(output_folder).mkdir(exist_ok=True)

    if fetcher_method == "topic":
        generate_topic_podcast(all_articles, output_folder)
    else:
        generate_full_podcast(all_articles, output_folder)

    logger.info("All done!")


if __name__ == "__main__":
    main()