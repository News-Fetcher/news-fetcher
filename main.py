# main.py
import os
import logging
import re
import hashlib
from pathlib import Path
from pydub import AudioSegment
from openai import OpenAI
import requests  # 导入 Requests 库
from firecrawl import FirecrawlApp
from datetime import datetime, timedelta
from github import Github
import json
import time
import firebase_admin
from firebase_admin import credentials, db, storage

# 如果 use_scraping 为 True，则直接使用 scrape_url，不使用 crawl_url
use_scraping = False
# 如果 be_concise 为 True，则输出简洁的内容
be_concise = True

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 生成 SHA256 哈希
def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# 记录元数据到 Firebase Realtime Database
def record_metadata_to_firebase(title, description, sha256, img_url=None):
    try:
        ref = db.reference("podcasts")
        existing_data = ref.get() or []

        # 确保数据为列表格式
        if isinstance(existing_data, dict):
            existing_data = list(existing_data.values())

        metadata = {
            "title": title,
            "description": description,
            "sha256": sha256,
        }

        if img_url:
            metadata["img_url"] = img_url

        existing_data.append(metadata)
        ref.set(existing_data)
        logger.info(f"Metadata recorded to Firebase as array: {metadata}")
    except Exception as e:
        logger.error(f"Error recording metadata to Firebase: {e}")

# 上传最终播客到 Firebase Storage，并以 SHA256 作为文件名
def upload_to_firebase_storage(local_file_path, title, description, img_url):
    try:
        bucket = storage.bucket()
        sha256_hash = calculate_sha256(local_file_path)
        storage_path = f"podcasts/{sha256_hash}.mp3"

        blob = bucket.blob(storage_path)
        blob.upload_from_filename(local_file_path)
        logger.info(f"File uploaded to Firebase Storage: {storage_path}")

        record_metadata_to_firebase(title, description, sha256_hash, img_url)

        return sha256_hash, blob.public_url
    except Exception as e:
        logger.error(f"Error uploading file to Firebase Storage: {e}")
        return None, None

def is_url_fetched(url):
    try:
        ref = db.reference("fetched_urls")
        fetched_urls = ref.get() or []
        return url in fetched_urls
    except Exception as e:
        logger.error(f"Error checking URL in Firebase: {e}")
        return False

def add_url_to_fetched(url):
    try:
        ref = db.reference("fetched_urls")
        fetched_urls = ref.get() or []
        if url not in fetched_urls:
            fetched_urls.append(url)
            ref.set(fetched_urls)
    except Exception as e:
        logger.error(f"Error adding URL to Firebase: {e}")

def extract_domain(url):
    match = re.search(r'https?://([^/]+)', url)
    if match:
        return match.group(1)
    return "unknown_domain"

def retry_github_action(action, max_retries=3):
    for attempt in range(max_retries):
        try:
            return action()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e

def process_articles(news_articles, website, client, output_folder, need_add_to_fetched_url=True, need_skip_same_url=True):
    logger.info(f"Processing articles from {website}...")
    local_mp3_files = []
    local_all_summaries = []

    for idx, article in enumerate(news_articles):
        url = article.get('metadata', {}).get('sourceURL', '')
        if need_skip_same_url and (url == website):
            logger.info(f"Skipping article {idx + 1} from {website} because it's the same as the website URL.")
            continue

        if is_url_fetched(url):
            logger.info(f"Article {idx + 1} already fetched, skipping: {url}")
            continue

        markdown_content = article.get('markdown', '')
        if markdown_content:
            meta_data = article.get('metadata', {})
            title = meta_data.get('title', '')
            url = meta_data.get('sourceURL', '')

            if be_concise:
                summary_require_prompt = "Summarize this single article into a conversational, podcast-friendly style in Chinese. Please be very concise" 
            else:
                summary_require_prompt = "Summarize this single article into a conversational, podcast-friendly style in Chinese. Explain the content in detail without an introduction or conclusion:"

            single_article_prompt = f"""
            {summary_require_prompt}

            Article Title: {title}
            Article URL: {url}

            Article Content:
            {markdown_content}
            """

            logger.info(f"Summarizing article {idx + 1} from {website} with title: {title} and URL: {url}...")
            try:
                article_response = client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一位新闻工作者或播客主持人，负责播报新闻或提供信息。 \
                                        在回答问题时，使用简洁、有条理、流畅且富有感染力的语言风格。 \
                                        确保语气专业、清晰且富有吸引力，适合公众广播或播客节目。\
                                        总结新闻的时候，先说类似“接下来的这则（来自xxx网站）新闻讲的是/描述了...” 之类的，\
                                        引出接下来的内容, "
                        },
                        {
                            "role": "user",
                            "content": single_article_prompt,
                        }
                    ]
                )
                article_summary = article_response.choices[0].message.content
                logger.info(f"Summary for article {idx + 1}:\n{article_summary}")

                local_all_summaries.append(article_summary)
                if need_add_to_fetched_url:
                    add_url_to_fetched(url)

                # 根据url生成一个哈希，避免和index挂钩，同时也避免中文等问题
                url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()[:8]
                domain = extract_domain(website)
                # 将domain和hash组合成为文件名，并去除特殊字符
                safe_domain = re.sub(r'[^\w.-]', '_', domain)
                filename = f"summary_{safe_domain}_{url_hash}.mp3"
                speech_file_path = Path(output_folder) / filename

                tts_response = client.audio.speech.create(
                    model="tts-1",
                    voice="echo",
                    speed=1.3,
                    input=article_summary
                )
                tts_response.stream_to_file(speech_file_path)
                local_mp3_files.append(speech_file_path)
                logger.info(f"Audio saved for article: {speech_file_path}")

            except Exception as e:
                logger.error(f"Error during summarization or TTS for article {idx + 1} from {website}: {e}")
                continue

    return local_mp3_files, local_all_summaries

# 加载 JSON 配置文件
def load_json_config(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件 {file_path} 未找到。")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 主函数入口
def main():
    # 初始化 Firebase
    if not os.path.exists("./serviceAccountKey.json"):
        raise ValueError("serviceAccountKey.json not found in the current directory.")  

    cred = credentials.Certificate("./serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://news-fetcher-platform-default-rtdb.asia-southeast1.firebasedatabase.app',
        'storageBucket': 'news-fetcher-platform.firebasestorage.app'
    })

    client = OpenAI()
    api_key_firecrawl = os.getenv("FIRECRAWL_API_KEY")
    if not api_key_firecrawl:
        raise ValueError("Firecrawl API key not set in environment variables.")
    app = FirecrawlApp(api_key=api_key_firecrawl)

    # 动态生成日期
    reuters_date = datetime.now().strftime("%Y-%m-%d")
    reuters_date_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    reuters_date_tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    coindesk_date = datetime.now().strftime("/%Y/%m/%d")
    coindesk_date_yesterday = (datetime.now() - timedelta(days=1)).strftime("/%Y/%m/%d")
    coindesk_date_tomorrow = (datetime.now() + timedelta(days=1)).strftime("/%Y/%m/%d")

    try:
        news_websites_crawl = load_json_config("news_websites_crawl.json")
        logger.info("Successfully loaded news_websites_crawl.json configuration.")
    except Exception as e:
        logger.error(f"Failed to load news_websites_crawl.json: {e}")
        news_websites_crawl = {}

    # Load configuration
    try:
        # Attempt to read scraping configuration from environment variable
        scraping_config_str = os.getenv("SCRAPING_CONFIG")
        if scraping_config_str and scraping_config_str != '{}':
            news_websites_scraping = json.loads(scraping_config_str)
            logger.info("Successfully loaded news_websites_scraping configuration from environment variable.")
        else:
            # If environment variable is not provided, load local configuration
            news_websites_scraping = load_json_config("news_websites_scraping.json")
            logger.info("Successfully loaded local news_websites_scraping.json configuration.")
    except Exception as e:
        logger.error(f"Failed to load news_websites_scraping configuration: {e}")
        news_websites_scraping = {}

    output_folder = "podcast_audio"
    os.makedirs(output_folder, exist_ok=True)

    logger.info("Starting to gather, summarize, and generate audio for news articles...")

    mp3_files = []
    all_summaries = []

    if use_scraping:
        # 使用 scraping 的方式获取数据
        for single_url in news_websites_scraping:
            try:
                logger.info(f"Scraping single URL: {single_url}")
                scrape_result = app.scrape_url(single_url, params={'formats': ['markdown', 'html']})
                if scrape_result:
                    # 构造与爬取数据类似的结构
                    news_articles = [scrape_result] if 'markdown' in scrape_result else []
                    logger.info(f"Found {len(news_articles)} articles from {single_url}.")
                    if news_articles:
                        local_mp3_files, local_all_summaries = process_articles(news_articles, single_url, client, output_folder, False, False)
                        mp3_files.extend(local_mp3_files)
                        all_summaries.extend(local_all_summaries)
            except Exception as e:
                logger.error(f"Error scraping {single_url}: {e}")
    else:
        # 使用 crawling 的方式获取数据
        for website, rules in news_websites_crawl.items():
            # 根据网站动态生成包含的路径日期
            includePaths = rules.get('includePaths', [])
            if "reuters" in website:
                includePaths.extend([
                    reuters_date,
                    reuters_date_yesterday,
                    reuters_date_tomorrow,
                ])
            elif "coindesk" in website:
                includePaths.extend([
                    coindesk_date,
                    coindesk_date_yesterday,
                    coindesk_date_tomorrow,
                ])

            try:
                logger.info(f"Crawling website: {website} with rules {rules}")
                crawl_status = app.crawl_url(
                    website,
                    params={
                        'limit': rules.get('limit', 2),
                        'scrapeOptions': {'formats': ['markdown', 'html']},
                        'includePaths': includePaths,
                        'excludePaths': rules.get('excludePaths', []),
                    },
                    poll_interval=1
                )

                news_articles = crawl_status.get('data', [])

                if not news_articles:
                    logger.warning(f"No news articles found in the crawl data for {website}.")
                    continue

                logger.info(f"Found {len(news_articles)} articles from {website}.")

                local_mp3_files, local_all_summaries = process_articles(news_articles, website, client, output_folder)
                mp3_files.extend(local_mp3_files)
                all_summaries.extend(local_all_summaries)

            except Exception as e:
                logger.error(f"Error while crawling {website}: {e}")
                continue

    logger.info("Generating introduction for the podcast...")

    title = ""
    intro_data = {}
    img_url = ""

    try:
        example_json = {
            "opening": "",
            "title": "",
            "description": "",
            "ending": ""
        }
        intro_prompt = f"""
        Please follow the following instructions to generate a podcast introduction:

        the opening:
        Combine the following article summaries into an introduction for today's news podcast. Start with a greeting and summarize the main topics
        the important thing is need to be detailed, not too short

        the title:
        Provide a podcast title

        description:
        Provide a one-sentence description for the podcast, Be a little more detailed, but no more than 200 words. 

        the ending:
        Provide a one-sentence ending for the podcast

        Here is a JSON structure for your podcast introduction that includes the opening, title, and one-sentence description:

        {json.dumps(example_json)}

        This JSON provides a structured format for integrating the introduction, title, and description into your script or further processes.

        So the summaries is:
        {chr(10).join(all_summaries)}
        """
        intro_response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system",
                    "content": "按要求输出，仅仅给出json格式，不要输出其他内容，中文输出"
                },
                {
                    "role": "user",
                    "content": intro_prompt,
                }
            ]
        )

        intro_json = intro_response.choices[0].message.content
        logger.info(f"Introduction generated:\n{intro_json}")

        if intro_json.strip().startswith("{") and intro_json.strip().endswith("}"):
            cleaned_json = intro_json.strip()
        elif "```json" in intro_json:
            match = re.search(r"```json(.*?)```", intro_json, re.DOTALL)
            cleaned_json = match.group(1).strip() if match else intro_json.strip()
        elif "```" in intro_json:
            match = re.search(r"```(.*?)```", intro_json, re.DOTALL)
            cleaned_json = match.group(1).strip() if match else intro_json.strip()
        else:
            cleaned_json = intro_json.strip()

        intro_data = json.loads(cleaned_json)

        opening = intro_data.get('opening', '欢迎收听今天的新闻播客，我们将为您带来最新的新闻动态。')
        title = intro_data.get('title', '今日新闻播客News~')
        description = intro_data.get('description', '这是一个关于今日新闻的播客，涵盖了重要的新闻事件。')
        ending = intro_data.get('ending', '感谢您的收听，我们下期节目再见。')

        intro_audio_path = Path(output_folder) / f"{title}_intro.mp3"
        intro_tts_response = client.audio.speech.create(
            model="tts-1",
            voice="echo",
            speed=1.3,
            input=opening
        )
        intro_tts_response.stream_to_file(intro_audio_path)
        mp3_files.insert(0, intro_audio_path)
        logger.info(f"Introduction audio saved: {intro_audio_path}")

        ending_audio_path = Path(output_folder) / f"{title}_ending.mp3"
        ending_tts_response = client.audio.speech.create(
            model="tts-1",
            voice="echo",
            speed=1.3,
            input=ending
        )
        ending_tts_response.stream_to_file(ending_audio_path)
        mp3_files.append(ending_audio_path)
        logger.info(f"Ending audio saved: {ending_audio_path}")

    except Exception as e:
        logger.error(f"Error generating podcast introduction: {e}")

    logger.info("Merging all audio files into a final podcast...")
    today_date = datetime.now().strftime("%Y-%m-%d")
    podcast_name = f"{today_date}_{title}.mp3"
    final_podcast = Path(output_folder) / podcast_name

    try:
        if mp3_files:
            combined_audio = AudioSegment.from_file(mp3_files[0])
            for mp3_file in mp3_files[1:]:
                combined_audio += AudioSegment.from_file(mp3_file)

            combined_audio.export(final_podcast, format="mp3")
            logger.info(f"Final podcast saved as: {final_podcast}")
        else:
            logger.warning("No audio files were generated to merge.")
    except Exception as e:
        logger.error(f"Error combining MP3 files: {e}")

    logger.info("Committing the final podcast to GitHub...")
    GH_ACCESS_TOKEN = os.getenv("GH_ACCESS_TOKEN")
    if not GH_ACCESS_TOKEN:
        raise ValueError("GitHub access token not set in environment variables.")

    REPO_NAME = "nagisa77/posts"
    COMMIT_FILE_PATH = f"podcasts/{podcast_name}"

    try:
        g = Github(GH_ACCESS_TOKEN)
        repo = g.get_repo(REPO_NAME)

        branch_name = "main"
        try:
            repo.get_branch(branch_name)
        except Exception:
            repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=repo.get_branch("master").commit.sha)

        if os.path.getsize(final_podcast) > 100 * 1024 * 1024:
            raise ValueError("File size too large for GitHub commit.")

        with open(final_podcast, "rb") as podcast_file:
            content = podcast_file.read()

        def commit_action():
            try:
                existing_file = repo.get_contents(COMMIT_FILE_PATH, ref=branch_name)
                return repo.update_file(
                    COMMIT_FILE_PATH,
                    f"Update podcast for {today_date}",
                    content,
                    existing_file.sha,
                    branch=branch_name
                )
            except:
                return repo.create_file(
                    COMMIT_FILE_PATH,
                    f"Add podcast for {today_date}",
                    content,
                    branch=branch_name
                )

        retry_github_action(commit_action)
        logger.info(f"Podcast committed successfully to GitHub: {COMMIT_FILE_PATH}")

    except Exception as e:
        logger.error(f"Error committing podcast to GitHub: {e}")

    # 开始生成博客封面图像并上传到 Firebase Storage
    try:
        # 从 intro_data 中提取标题和描述
        title = intro_data.get('title', '未命名播客')
        description = intro_data.get('description', '暂无描述。')

        # 定义用于生成图像的提示语
        image_prompt = f"为播客《{title}》创建一个专业且具有视觉吸引力的博客封面，反映主题：{description}。设计应现代、引人注目，适合新闻播客。"

        logger.info(f"使用提示语生成博客封面图像：{image_prompt}")

        # 使用 OpenAI 的 DALL·E 模型生成图像
        intro_response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            n=1,
            size="512x512",
            response_format="url" 
        )   

        # 从响应中提取图像 URL
        image_url = intro_response.data[0].url
        logger.info(f"图像生成成功：{image_url}")

        # 下载图像数据
        image_data = requests.get(image_url).content
        image_filename = f"{title.replace(' ', '_')}_cover.png"
        image_path = Path(output_folder) / image_filename

        # 将图像保存到本地
        with open(image_path, 'wb') as f:
            f.write(image_data)
        logger.info(f"图像已下载并保存到本地：{image_path}")

        # 上传图像到 Firebase Storage 的 'podcasts_image/' 文件夹
        bucket = storage.bucket()
        firebase_image_path = f"podcasts_image/{image_filename}"
        blob = bucket.blob(firebase_image_path)
        blob.upload_from_filename(image_path)
        logger.info(f"图像已上传到 Firebase Storage：{firebase_image_path}")

        # 使图像公开可访问，并获取公共 URL
        blob.make_public()
        img_url = blob.public_url
        logger.info(f"图像的公共 URL：{img_url}")

        # 可选：上传后删除本地图像文件以节省空间
        os.remove(image_path)
        logger.info(f"已删除本地图像文件：{image_path}")

    except Exception as e:
        logger.error(f"生成或上传博客封面图像时出错：{e}")
        img_url = None  # 确保 img_url 变量在出错时有定义

    try:
        if mp3_files:
            combined_audio = AudioSegment.from_file(mp3_files[0])
            for mp3_file in mp3_files[1:]:
                combined_audio += AudioSegment.from_file(mp3_file)

            combined_audio.export(final_podcast, format="mp3")
            logger.info(f"Final podcast saved as: {final_podcast}")

            # 上传到 Firebase Storage
            firebase_hash, firebase_url = upload_to_firebase_storage(final_podcast, podcast_name, description, img_url)
            if firebase_url:
                logger.info(f"Podcast uploaded to Firebase Storage successfully: {firebase_url}")

        else:
            logger.warning("No audio files were generated to merge.")
    except Exception as e:
        logger.error(f"Error combining MP3 files: {e}")

if __name__ == "__main__":
    main()
