import os
import logging
import re
from pathlib import Path
from pydub import AudioSegment
from openai import OpenAI
from firecrawl import FirecrawlApp
from datetime import datetime
from github import Github

def extract_domain(url):
    # 使用正则提取主域名
    match = re.search(r'https?://([^/]+)', url)
    if match:
        return match.group(1)  # 返回匹配的域名部分
    return "unknown_domain"

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
client = OpenAI()

# Initialize Firecrawl App with your Firecrawl API key
api_key_firecrawl = os.getenv("FIRECRAWL_API_KEY")
app = FirecrawlApp(api_key=api_key_firecrawl)

# List of news websites to crawl
news_websites = [
  'https://www.reuters.com',
  'https://cointelegraph.com',
  'https://apnews.com',
]

# Folder for saving audio files
output_folder = "podcast_audio"
os.makedirs(output_folder, exist_ok=True)

logger.info("Starting to crawl, summarize, and generate audio for news websites...")

# Crawl and process each website
mp3_files = []  # Store paths to individual MP3 files for merging later
all_summaries = []  # Store all summaries for the introduction

for website in news_websites:
    try:
        logger.info(f"Crawling website: {website}")
        crawl_status = app.crawl_url(
            website,
            params={
                'limit': 10,
                'scrapeOptions': {'formats': ['markdown', 'html']}
            },
            poll_interval=1
        )
        
        # Extract the list of news articles from the crawl status
        news_articles = crawl_status.get('data', [])
        
        if not news_articles:
            logger.warning(f"No news articles found in the crawl data for {website}.")
            continue
        
        logger.info(f"Found {len(news_articles)} articles from {website}.")
        
        # Summarize each article individually and generate audio
        for idx, article in enumerate(news_articles):
            markdown_content = article.get('markdown', '')
            if markdown_content:
                single_article_prompt = f"""
                Summarize this single article into a conversational, podcast-friendly style in Chinese, Explain the content in as much detail as possible, and This article is just the middle of the report, so there is no need for the opening and closing greetings, just report the content directly.:

                Article Content:
                {markdown_content}
                """
                logger.info(f"Summarizing article {idx+1} from {website}...")
                try:
                    # Generate summary using GPT
                    article_response = client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=[
                            {
                              "role": "system",
                              "content": "你是一位新闻工作者或播客主持人，负责播报新闻或提供信息。在回答问题时，使用简洁、有条理、流畅且富有感染力的语言风格。确保语气专业、清晰且富有吸引力，适合公众广播或播客节目。"
                            }, 
                            {
                                "role": "user",
                                "content": single_article_prompt,
                            }
                        ]
                    )
                    article_summary = article_response.choices[0].message.content
                    logger.info(f"Summary for article {idx+1}:\n{article_summary}")

                    all_summaries.append(article_summary)

                    # Generate TTS audio for the summary
                    domain = extract_domain(website)
                    speech_file_path = Path(output_folder) / f"summary_{domain}_{idx + 1}.mp3"

                    tts_response = client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",
                        input=article_summary
                    )
                    tts_response.stream_to_file(speech_file_path)
                    mp3_files.append(speech_file_path)
                    logger.info(f"Audio saved for article {idx+1}: {speech_file_path}")

                except Exception as e:
                    logger.error(f"Error during summarization or TTS for article {idx+1} from {website}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error while crawling {website}: {e}")
        continue

# Generate introduction for the podcast
logger.info("Generating introduction for the podcast...")
try:
    intro_prompt = f"""
    Combine the following article summaries into an introduction for today's news podcast.
    The introduction should be warm and conversational. It should start with a greeting and give a brief overview of the main topics covered in today's news.

    Summaries:
    {chr(10).join(all_summaries)}
    """
    intro_response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
              "role": "system",
              "content": "你是一位新闻播音员，为今日新闻播客生成暖心开场白，并引导进入第一个新闻。"
            }, 
            {
                "role": "user",
                "content": intro_prompt,
            }
        ]
    )
    intro_summary = intro_response.choices[0].message.content
    logger.info(f"Introduction generated:\n{intro_summary}")

    # Generate TTS audio for the introduction
    intro_audio_path = Path(output_folder) / "intro.mp3"
    intro_tts_response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=intro_summary
    )
    intro_tts_response.stream_to_file(intro_audio_path)
    mp3_files.insert(0, intro_audio_path)
    logger.info(f"Introduction audio saved: {intro_audio_path}")

except Exception as e:
    logger.error(f"Error generating podcast introduction: {e}")

# Merge all MP3 files into one podcast
logger.info("Merging all audio files into a final podcast...")
today_date = datetime.now().strftime("%Y-%m-%d")
final_podcast = Path(output_folder) / f"{today_date}.mp3"

try:
    if mp3_files:
        combined_audio = AudioSegment.from_file(mp3_files[0])
        for mp3_file in mp3_files[1:]:
            combined_audio += AudioSegment.from_file(mp3_file)

        # Export the final combined podcast file
        combined_audio.export(final_podcast, format="mp3")
        logger.info(f"Final podcast saved as: {final_podcast}")
    else:
        logger.warning("No audio files were generated to merge.")
except Exception as e:
    logger.error(f"Error combining MP3 files: {e}")

# Commit the final podcast to GitHub
logger.info("Committing the final podcast to GitHub...")

GIT_ACCESS_TOKEN = os.getenv("GIT_ACCESS_TOKEN")
if not GIT_ACCESS_TOKEN:
    raise ValueError("GitHub access token not set in environment variables.")

REPO_NAME = "nagisa77/posts"
COMMIT_FILE_PATH = f"podcasts/{today_date}.mp3"

try:
    g = Github(GIT_ACCESS_TOKEN)
    repo = g.get_repo(REPO_NAME)

    # Read the podcast file
    with open(final_podcast, "rb") as podcast_file:
        content = podcast_file.read()

    # Commit to GitHub
    repo.create_file(
        COMMIT_FILE_PATH,
        f"Add podcast for {today_date}",
        content,
        branch="main"
    )
    logger.info(f"Podcast committed successfully to GitHub: {COMMIT_FILE_PATH}")
except Exception as e:
    logger.error(f"Error committing podcast to GitHub: {e}")
