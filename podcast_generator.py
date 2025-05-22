# podcast_generator.py
import os
import re
import uuid
import json
import requests
import logging
import hashlib
from pathlib import Path
import tiktoken
from datetime import datetime
from openai import OpenAI
from utils.firebase_utils import is_url_fetched, add_url_to_fetched, upload_to_firebase_storage
from utils.cos_utils import upload_file_to_cos
from utils.image_utils import compress_image
from utils.audio_utils import merge_audio_files, extract_domain, calculate_sha256
from pydub import AudioSegment
# 导入阿里云百炼语音合成SDK
from dashscope.audio.tts_v2 import *
import time

logger = logging.getLogger(__name__)

# 根据环境变量初始化 LLM 客户端
def initialize_llm_client():
    """Return (client, model_name) based on SUMMARY_PROVIDER env."""
    provider = os.getenv("SUMMARY_PROVIDER", "tongyi").lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        client = OpenAI(api_key=api_key)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    else:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set in environment variables.")
        client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_name = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
    return client, model_name

# 获取模型编码，用于计算 tokens
def num_tokens_from_string(string: str, model_name: str = "gpt-4") -> int:
    """
    计算给定文本在指定模型下的 token 数量
    注意：gpt-4o 自定义模型可能需要换成类似 gpt-3.5-turbo 或别的可识别模型名称
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # 如果自定义模型不被识别，就用一个通用编码器
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

def safe_chat_completion_create(
    client, model, messages, max_retries=3, initial_delay=1.0, max_tokens=2048
):
    """
    包装对 chat.completions.create 的调用，
    - 做指数退避，避免频繁 429
    - 可以指定 max_tokens 以限制返回大小
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens  # 注意根据需要设置
            )
            return response
        except Exception as e:
            err_msg = str(e)
            # 如果是 429 或者其他速率限制相关信息，做退避重试
            if "429" in err_msg or "rate_limit" in err_msg:
                logger.warning(f"Got 429/rate limit error: {err_msg}, retry in {delay} s")
                time.sleep(delay)
                delay *= 2  # 指数退避
            else:
                # 其他错误直接抛出
                raise e

    # 超过最大重试次数，抛出异常
    raise RuntimeError("Max retries exceeded for chat.completions.create()")


def truncate_text_to_fit_model(prompt_text: str, max_prompt_tokens: int, model_name: str = "gpt-4"):
    """
    简单截断方式：确保 prompt 的 token 总数不超过 max_prompt_tokens
    若超过则直接把末尾砍掉。
    更复杂的场景可以考虑分块总结。
    """
    tokens = num_tokens_from_string(prompt_text, model_name)
    if tokens <= max_prompt_tokens:
        return prompt_text

    # 如果过长，就做简单截断
    truncated = prompt_text
    while tokens > max_prompt_tokens:
        truncated = truncated[:-200]  # 每次砍掉一定字符，避免循环次数过多
        tokens = num_tokens_from_string(truncated, model_name)

    # 可选：给截断内容加个提示
    truncated += "\n\n(以上内容已被截断)..."
    return truncated


def summarize_and_tts_articles(news_articles, client, model_name, output_folder, be_concise=False):
    local_mp3_files = []
    all_summaries = []

    for idx, article in enumerate(news_articles):
        # results from firecrawl-py may be Pydantic models; convert to dict
        if not isinstance(article, dict):
            article = article.model_dump()

        url = article.get('metadata', {}).get('sourceURL', '')

        logger.info(f"Summarizing article {idx + 1}: {url}")    
        # 跳过已经抓取过的文章
        # if is_url_fetched(url):
        #     logger.info(f"Article {idx + 1} already fetched, skipping: {url}")
        #     continue

        markdown_content = article.get('markdown', '')
        if not markdown_content:
            logger.info(f"Article {idx + 1} has no markdown content. Skipping.")
            continue

        title = article.get('metadata', {}).get('title', '')

        # 生成摘要提示语
        if be_concise:
            summary_require_prompt = "Summarize this single article into a conversational, podcast-friendly style in Chinese. Please be very very concise"
        else:
            summary_require_prompt = "Summarize this single article into a conversational, podcast-friendly style in Chinese. Explain the content in detail without an introduction or conclusion:"

        single_article_prompt = f"""
        {summary_require_prompt}

        Article Title: {title}
        Article URL: {url}

        Article Content:
        {markdown_content}
        """

        # ========== 新增：截断处理，假设我们想让 prompt + 回答 <= 8000 tokens, 其中回答max 2048 tokens => prompt部分留 5000 tokens 左右 ==========
        truncated_prompt = truncate_text_to_fit_model(
            single_article_prompt,
            max_prompt_tokens=20000,   # 这里可根据需要调整
            model_name=model_name
        )
        # =================================================================================================

        try:
            # ========== 新增：使用 safe_chat_completion_create，带重试 + 限制max_tokens ==========
            article_response = safe_chat_completion_create(
                client=client,
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位新闻工作者或播客主持人，负责播报新闻或提供信息。 \
                                        在回答问题时，使用简洁、有条理、流畅且富有感染力的语言风格。 \
                                        确保语气专业、清晰且富有吸引力，适合公众广播或播客节目。\
                                        总结新闻的时候，先说类似“接下来是一则（来自xxx网站）的新闻，...”，\
                                        引出接下来的内容, "
                    },
                    {
                        "role": "user",
                        "content": truncated_prompt,
                    }
                ],
                max_retries=3,
                initial_delay=1.0,
                max_tokens=2048  # 对生成结果做一个上限, 防止太长
            )
            # ============================================================================================

            article_summary = article_response.choices[0].message.content
            logger.info(f"Summary for article {idx + 1}:\n{article_summary}")

            all_summaries.append(article_summary)
            add_url_to_fetched(url)  # 将 URL 加入已抓取列表

            # 调用阿里云百炼语音合成
            url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()[:8]
            domain = extract_domain(url)
            safe_domain = re.sub(r'[^\w.-]', '_', domain)
            filename = f"summary_{safe_domain}_{url_hash}.mp3"
            speech_file_path = Path(output_folder) / filename
            
            # 使用阿里云百炼语音合成API - 使用同步方式
            try:
                # 限制文本长度，避免过长导致错误
                max_text_length = 5000
                if len(article_summary) > max_text_length:
                    logger.warning(f"Article {idx + 1} summary too long ({len(article_summary)} chars), truncating to {max_text_length} chars")
                    article_summary = article_summary[:max_text_length]
                
                # 创建语音合成器并调用 - 使用同步方式
                tts_synthesizer = SpeechSynthesizer(
                    model="cosyvoice-v1", 
                    voice="longxiaochun",  # 默认使用龙小淳音色
                    format=AudioFormat.MP3_22050HZ_MONO_256KBPS,  # 使用MP3格式，22050Hz采样率
                    speech_rate=1.0  # 语速1.0倍，对应OpenAI的speed=1.0
                )
                
                # 调用语音合成 - 同步方式
                logger.info(f"Starting TTS for article {idx + 1} with length {len(article_summary)} chars")
                audio_data = tts_synthesizer.call(article_summary)
                
                # 检查音频数据是否为空
                if audio_data is None:
                    logger.error(f"No audio data received for article {idx + 1}")
                    continue
                
                # 将合成的音频保存到文件
                with open(speech_file_path, 'wb') as f:
                    f.write(audio_data)
                
                # 验证文件是否正确保存
                if os.path.exists(speech_file_path) and os.path.getsize(speech_file_path) > 0:
                    local_mp3_files.append(speech_file_path)
                    logger.info(f"Audio saved for article {idx + 1}: {speech_file_path}, size: {os.path.getsize(speech_file_path)} bytes")
                else:
                    logger.error(f"Failed to save audio for article {idx + 1} or file is empty")
                
            except Exception as e:
                logger.error(f"Error in CosyVoice TTS for article {idx + 1}: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error summarizing or TTS for article {idx + 1}: {e}")

    return local_mp3_files, all_summaries


def generate_intro_ending(all_summaries, client, model_name, output_folder):
    """
    调用 GPT 生成播客的开场白、标题、描述、结束语等，并生成对应 MP3 文件, 注意开场白应该尽量简洁，控制在100字左右。
    返回： (mp3_files: list, title: str, description: str, tags: list, img_url: str)
    """
    mp3_files = []
    title = ""
    description = ""
    tags = []
    img_url = None

    # 1. 生成开场白 & 结束语
    try:
        response = requests.get("https://gettags-a6lubplbza-uc.a.run.app")
        response.raise_for_status()
        history_tags = response.json()

        example_json = {
            "opening": "",
            "title": "",
            "description": "",
            "ending": "",
            "tags": ["tag1", "tag2", "tag3"]
        }

        intro_prompt = f"""
        Please follow the following instructions to generate a podcast introduction:

        the opening:
        Combine the following article summaries into an introduction for today's news podcast. Start with a greeting and then provide a comprehensive summary of the main topics.
        The important thing is to create a true synthesis that captures trends and significance, rather than simply listing each news item briefly. the opening should start with "欢迎收听今天的UC播客"

        the title:
        Provide a podcast title

        description:
        Provide a one-sentence description for the podcast, Be a little more detailed, but no more than 200 words. 

        the ending:
        Provide a one-sentence ending for the podcast

        the tags: 
        Extract the most relevant keywords to represent the content direction of this podcast in a JSON array format, 
        returning no more than 4 keywords, and do not include sentences, only words. Historical tags can be referenced 
        but should not be limited to them. Historical tags: {', '.join(history_tags)}

        Here is a JSON structure for your podcast introduction that includes the opening, title, and one-sentence description:

        {json.dumps(example_json)}

        This JSON provides a structured format for integrating the introduction, title, and description into your script or further processes.

        So the summaries is:
        {chr(10).join(all_summaries)}
        """

        intro_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "按要求输出，仅仅给出json格式，不要输出其他内容，中文输出"},
                {"role": "user", "content": intro_prompt}
            ]
        )

        intro_json = intro_response.choices[0].message.content
        logger.info(f"Introduction generated:\n{intro_json}")

        # 尝试解析成 JSON
        cleaned_json = extract_json_content(intro_json)
        intro_data = json.loads(cleaned_json)

        opening = intro_data.get('opening', '欢迎收听...')
        title = intro_data.get('title', '今日新闻播客News')
        description = intro_data.get('description', '这是一个关于今日新闻的播客...')
        ending = intro_data.get('ending', '感谢您的收听...')
        tags = intro_data.get('tags', [])

        intro_audio_path = Path(output_folder) / f"{title}_intro.mp3"
        ending_audio_path = Path(output_folder) / f"{title}_ending.mp3"

        # 使用阿里云百炼语音合成API生成开场白音频 - 同步方式
        try:
            # 创建语音合成器 - 同步方式
            tts_synthesizer = SpeechSynthesizer(
                model="cosyvoice-v1", 
                voice="longxiaochun",  # 默认使用龙小淳音色
                format=AudioFormat.MP3_22050HZ_MONO_256KBPS,  # 使用MP3格式，22050Hz采样率
                speech_rate=1.0  # 语速1.0倍，对应OpenAI的speed=1.0
            )
            
            # 合成开场白 - 同步方式
            logger.info(f"Generating intro audio with length {len(opening)} chars")
            intro_audio = tts_synthesizer.call(opening)
            
            if intro_audio is not None:
                with open(intro_audio_path, 'wb') as f:
                    f.write(intro_audio)
                mp3_files.append(intro_audio_path)
                logger.info(f"Intro audio saved: {intro_audio_path}, size: {os.path.getsize(intro_audio_path)} bytes")
            else:
                logger.error("No audio data received for intro")

            tts_synthesizer = SpeechSynthesizer(
                model="cosyvoice-v1", 
                voice="longxiaochun",  # 默认使用龙小淳音色
                format=AudioFormat.MP3_22050HZ_MONO_256KBPS,  # 使用MP3格式，22050Hz采样率
                speech_rate=1.0  # 语速1.0倍，对应OpenAI的speed=1.0
            )
            
            # 合成结束语 - 同步方式
            logger.info(f"Generating ending audio with length {len(ending)} chars")
            ending_audio = tts_synthesizer.call(ending)
            
            if ending_audio is not None:
                with open(ending_audio_path, 'wb') as f:
                    f.write(ending_audio)
                mp3_files.append(ending_audio_path)
                logger.info(f"Ending audio saved: {ending_audio_path}, size: {os.path.getsize(ending_audio_path)} bytes")
            else:
                logger.error("No audio data received for ending")
                
        except Exception as e:
            logger.error(f"Error in CosyVoice TTS for intro/ending: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error generating intro/ending: {e}")

    # 2. 生成封面图并上传
    try:
        img_url = generate_and_upload_cover_image(title, description, client, output_folder)
    except Exception as e:
        logger.error(f"Error generating podcast cover image: {e}")
        return mp3_files, title, description, tags, ""

    return mp3_files, title, description, tags, img_url


def extract_json_content(response_text):
    """
    从返回文本中提取 JSON 字符串：常见情况包括 ```json ... ``` 或仅 { ... }
    """
    text = response_text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    import re
    # 尝试提取 ```json ``` 块
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 尝试提取 ``` ```
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def generate_and_upload_cover_image(title, description, client, output_folder):
    """使用 OpenAI 生成播客封面图并上传到腾讯云COS"""

    # 使用独立的 OpenAI 客户端生成图片，避免受到传入 client 的 base_url 影响
    image_client = OpenAI()  # 依赖 OPENAI_API_KEY 环境变量

    image_prompt = f"为播客《{title}》创建一个专业且具有视觉吸引力的封面..."
    logger.info(f"[Cover] Using prompt: {image_prompt}")

    # 调用图像生成
    intro_response = image_client.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        n=1,
        size="1024x1024",
        response_format="url"
    )
    image_url = intro_response.data[0].url
    logger.info(f"Image generated from DALL-E: {image_url}")

    # 下载并压缩图像
    image_data = requests.get(image_url).content
    image_filename = f"{uuid.uuid4().hex}.jpg"
    image_path = Path(output_folder) / image_filename
    compress_image(image_data, image_path)
    logger.info(f"Cover image saved: {image_path}")

    # 上传到腾讯云COS
    cos_key = f"podcasts_image/{image_filename}"
    final_img_url = upload_file_to_cos(str(image_path), cos_key)
    logger.info(f"Cover image uploaded to COS {cos_key}, public URL: {final_img_url}")
    os.remove(image_path)  # 可选：删除本地文件
    return final_img_url


def generate_full_podcast(all_articles, output_folder):
    """
    处理所有文章 => 生成音频 => 生成开场&结束 => 合并音频 => 上传到 Firebase
    """
    if not all_articles:
        logger.warning("No articles to process. Aborting podcast generation.")
        return

    # 根据环境变量初始化 LLM 客户端
    client, model_name = initialize_llm_client()
    # 是否"简洁"
    be_concise = (os.getenv("BE_CONCISE") == "true")

    # 1. 对文章逐篇处理并生成音频
    articles_mp3, articles_summaries = summarize_and_tts_articles(all_articles, client, model_name, output_folder, be_concise)

    # 2. 生成节目开场 & 结束音频
    intro_ending_mp3, title, description, tags, img_url = generate_intro_ending(articles_summaries, client, model_name, output_folder)

    # 3. 合并全部音频
    #   将开场放最前，结束放最后
    mp3_to_merge = [intro_ending_mp3[0]] + articles_mp3 + [intro_ending_mp3[1]]
    today_date = datetime.now().strftime("%Y-%m-%d")
    final_podcast_filename = f"{today_date}_{title}.mp3"
    final_podcast_path = Path(output_folder) / final_podcast_filename
    merge_audio_files(mp3_to_merge, final_podcast_path)

    # 4. 上传到 Firebase Storage
    sha256_hash = calculate_sha256(str(final_podcast_path))
    from utils.firebase_utils import upload_to_firebase_storage
    audio_public_url, file_sha256 = upload_to_firebase_storage(
        local_file_path=str(final_podcast_path),
        title=title,
        description=description,
        sha256_hash=sha256_hash,
        img_url=img_url,
        tags=tags
    )

    if audio_public_url:
        logger.info(f"Podcast uploaded successfully! {audio_public_url}")
    else:
        logger.warning("Failed to upload the final podcast.")
