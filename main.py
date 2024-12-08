import os
import logging
import re
from pathlib import Path
from pydub import AudioSegment
from openai import OpenAI
from firecrawl import FirecrawlApp
from datetime import datetime, timedelta
from github import Github
import json
import time
import re
import firebase_admin
from firebase_admin import credentials, db

# Check and update URLs in Firebase
def is_url_fetched(url):
    try:
        ref = db.reference("fetched_urls")
        fetched_urls = ref.get() or []  # Get existing URLs or initialize with an empty list
        return url in fetched_urls
    except Exception as e:
        logger.error(f"Error checking URL in Firebase: {e}")
        return False

def add_url_to_fetched(url):
    try:
        ref = db.reference("fetched_urls")
        fetched_urls = ref.get() or []  # Get existing URLs or initialize with an empty list
        if url not in fetched_urls:
            fetched_urls.append(url)
            ref.set(fetched_urls)  # Update Firebase with the new list
    except Exception as e:
        logger.error(f"Error adding URL to Firebase: {e}")

# Extract domain function
def extract_domain(url):
    match = re.search(r'https?://([^/]+)', url)
    if match:
        return match.group(1)
    return "unknown_domain"

# Retry decorator for GitHub actions
def retry_github_action(action, max_retries=3):
    for attempt in range(max_retries):
        try:
            return action()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# check serviceAccountKey exists
if not os.path.exists("./serviceAccountKey.json"):
    raise ValueError("serviceAccountKey.json not found in the current directory.")  

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://news-fetcher-platform-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# Initialize OpenAI and Firecrawl clients
client = OpenAI()
api_key_firecrawl = os.getenv("FIRECRAWL_API_KEY")
if not api_key_firecrawl:
    raise ValueError("Firecrawl API key not set in environment variables.")
app = FirecrawlApp(api_key=api_key_firecrawl)

reuters_date = datetime.now().strftime("%Y-%m-%d")
reuters_date_yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
reuters_date_two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
coindesk_date = datetime.now().strftime("/%Y/%m/%d")
coindesk_date_yesterday = (datetime.now() - timedelta(days=1)).strftime("/%Y/%m/%d")
coindesk_date_two_days_ago = (datetime.now() - timedelta(days=2)).strftime("/%Y/%m/%d")

# Custom rules for each website
news_websites = {
    'https://www.reuters.com': {
        'limit': 14,
        'includePaths': [
            f'{reuters_date}',
            f'{reuters_date_yesterday}',
            # f'{reuters_date_two_days_ago}',
        ],
        'excludePaths': [
            'wrapup',
            'podcasts',
        ]
    },
    # 'https://cointelegraph.com': {
    #     'limit': 7,
    #     'includePaths': [
    #         'news/*',
    #         # 'technology/*',
    #     ],
    #     'excludePaths': []
    # },
    # 'https://apnews.com': {
    #     'limit': 7,
    #     'includePaths': [
    #         # 'article',
    #     ],
    #     'excludePaths': []
    # },
    'https://www.coindesk.com': {
        'limit': 14,
        'includePaths': [
            f'{coindesk_date}',
            f'{coindesk_date_yesterday}',
            # f'{coindesk_date_two_days_ago}',
        ],
        'excludePaths': []
    }
}

# Folder for saving audio files
output_folder = "podcast_audio"
os.makedirs(output_folder, exist_ok=True)

logger.info("Starting to crawl, summarize, and generate audio for news websites...")

# Crawl and process each website
mp3_files = []
all_summaries = []

for website, rules in news_websites.items():
    try:
        logger.info(f"Crawling website: {website} with rules {rules}")
        crawl_status = app.crawl_url(
            website,
            params={
                'limit': rules['limit'],
                'scrapeOptions': {'formats': ['markdown', 'html']},
                'includePaths': rules['includePaths'],
                'excludePaths': rules['excludePaths'],
            },
            poll_interval=1
        )

        news_articles = crawl_status.get('data', [])

        if not news_articles:
            logger.warning(f"No news articles found in the crawl data for {website}.")
            continue

        logger.info(f"Found {len(news_articles)} articles from {website}.")

        for article in news_articles:
            logger.info(f"Article URL: {article.get('metadata', {}).get('sourceURL', '')}")

        # Summarize and generate audio for each article
        for idx, article in enumerate(news_articles):
            url = article.get('metadata', {}).get('sourceURL', '')
            if (url == website):
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
                single_article_prompt = f"""
                Summarize this single article into a conversational, podcast-friendly style in Chinese. Explain the content in detail without an introduction or conclusion:

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

                    all_summaries.append(article_summary)

                    # Mark the URL as fetched
                    add_url_to_fetched(url)

                    # Generate TTS audio
                    domain = extract_domain(website)
                    speech_file_path = Path(output_folder) / f"summary_{domain}_{idx + 1}.mp3"

                    tts_response = client.audio.speech.create(
                        model="tts-1",
                        voice="echo",
                        speed=1.3,
                        input=article_summary
                    )
                    tts_response.stream_to_file(speech_file_path)
                    mp3_files.append(speech_file_path)
                    logger.info(f"Audio saved for article {idx + 1}: {speech_file_path}")

                except Exception as e:
                    logger.error(f"Error during summarization or TTS for article {idx + 1} from {website}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error while crawling {website}: {e}")
        continue

# Generate podcast introduction
logger.info("Generating introduction for the podcast...")
title = ""

try:
    example_json = {
        "opening": "",
        "title": "",
        "description": ""
    }
    intro_prompt = f"""
    Please follow the following instructions to generate a podcast introduction:
    
    the opening:
    Combine the following article summaries into an introduction for today's news podcast. Start with a greeting and summarize the main topics
    the important thing is need to be detailed, not too short

    the title:
    Provide a podcast title

    description:
    Provide a one-sentence description for the podcast  

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

    # 匹配不同的可能情况
    if intro_json.strip().startswith("{") and intro_json.strip().endswith("}"):
        # Case 1: 直接是 JSON 数据
        cleaned_json = intro_json.strip()
    elif "```json" in intro_json:
        # Case 2: ```json 开头的 JSON 数据
        match = re.search(r"```json(.*?)```", intro_json, re.DOTALL)
        cleaned_json = match.group(1).strip() if match else intro_json.strip()
    elif "```" in intro_json:
        # Case 3: ``` 开头的 JSON 数据
        match = re.search(r"```(.*?)```", intro_json, re.DOTALL)
        cleaned_json = match.group(1).strip() if match else intro_json.strip()
    else:
        # 如果没有匹配到特定情况，保留原始内容
        cleaned_json = intro_json.strip()
    
    intro_data = json.loads(cleaned_json)
    
    opening = intro_data.get('opening', '欢迎收听今天的新闻播客，我们将为您带来最新的新闻动态。')
    title = intro_data.get('title', '今日新闻播客News~')
    description = intro_data.get('description', '这是一个关于今日新闻的播客，涵盖了重要的新闻事件。')

    intro_audio_path = Path(output_folder) / f"{title}.mp3"
    intro_tts_response = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        speed=1.3,
        input=opening
    )
    intro_tts_response.stream_to_file(intro_audio_path)
    mp3_files.insert(0, intro_audio_path)
    logger.info(f"Introduction audio saved: {intro_audio_path}")

except Exception as e:
    logger.error(f"Error generating podcast introduction: {e}")

# Merge audio files into a single podcast
logger.info("Merging all audio files into a final podcast...")
today_date = datetime.now().strftime("%Y-%m-%d")
final_podcast = Path(output_folder) / f"{today_date}.mp3"

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

# Commit the podcast to GitHub
logger.info("Committing the final podcast to GitHub...")

GH_ACCESS_TOKEN = os.getenv("GH_ACCESS_TOKEN")
if not GH_ACCESS_TOKEN:
    raise ValueError("GitHub access token not set in environment variables.")

REPO_NAME = "nagisa77/posts"
COMMIT_FILE_PATH = f"podcasts/{today_date}-{title}.mp3"

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