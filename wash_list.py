import json
import re
import os
from openai import OpenAI
import logging
from pathlib import Path
import requests
import time
import firebase_admin
from firebase_admin import credentials, db
import uuid
from utils.cos_utils import upload_file_to_cos
from utils.image_utils import compress_image
from mutagen.mp3 import MP3
from io import BytesIO  # 用于处理内存中的MP3数据

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
client = OpenAI()

# ===== 新增/调整标志位 =====
WASH_IMAGES = True             # 本次是否wash图片
WASH_TAGS = True               # 本次是否wash tags
WASH_TOTAL_DURATION = True     # 本次是否wash total_duration
DELETE_SHORT_AUDIO = True      # 本次是否删除时长小于2分钟（120秒）的播客
MIGRATE_IMAGES = True          # 是否迁移已有图片到COS
COMPRESS_EXISTING_IMAGES = False # 当图片已经在COS中时是否仍重新压缩并上传

# 已上传至腾讯云COS的图片基准 URL 前缀
COS_IMAGE_BASE_URL = "https://news-fetcher-1307107697.cos.ap-guangzhou.myqcloud.com/"

# MP3文件的基础URL
MP3_BASE_URL = "https://downloadfile-a6lubplbza-uc.a.run.app?filename="

def exponential_backoff_retry(func, *args, max_retries=5, **kwargs):
    """
    通用指数退避重试函数:
    初始等待1秒，每次重试等待时间倍增，最多5次。
    """
    delay = 1
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if i == max_retries - 1:
                logger.error(f"重试次数已达上限，错误：{e}")
                raise
            else:
                logger.warning(f"请求出现错误：{e}，将在 {delay} 秒后重试...")
                time.sleep(delay)
                delay *= 2

def generate_img_url(title, description):
    logging.info("生成图片URL...")
    output_folder = "wash_list_image"
    os.makedirs(output_folder, exist_ok=True)

    image_prompt = f"为播客《{title}》创建一个专业且具有视觉吸引力的博客封面，反映主题：{description} 设计应现代、引人注目，适合新闻播客。注意内容不要太过杂乱，简洁"
    logger.info(f"使用提示语生成博客封面图像：{image_prompt}")

    # 使用指数退避重试生成图像
    intro_response = exponential_backoff_retry(
        client.images.generate,
        model="gpt-image-1",
        prompt=image_prompt,
        n=1,
        size="1024x1024",
        quality="standard",
        response_format="url"
    )

    image_url = intro_response.data[0].url
    logger.info(f"图像生成成功：{image_url}")

    # 下载并压缩图像数据
    image_data = requests.get(image_url).content
    image_filename = f"{uuid.uuid4().hex}.jpg"
    image_path = Path(output_folder) / image_filename
    compress_image(image_data, image_path)
    logger.info(f"图像已下载并压缩保存到本地：{image_path}")

    # 上传图像到腾讯云COS
    cos_key = f"podcasts_image/{image_filename}"
    img_url = upload_file_to_cos(str(image_path), cos_key)
    logger.info(f"图像已上传到 COS：{img_url}")
    os.remove(image_path)

    return img_url

def generate_tags_by_description(description, historical_tags):
    logging.info(f"生成标签，历史标签: {', '.join(historical_tags)}，描述: {description}")
    prompt = f"""
    请从以下描述中提取最相关的关键词，用于表示本期播客的内容方向，以Json数组形式返回，不要超过4个,不要包含句子只要词语。 历史标签可供参考选择，但不要局限于历史标签。
    历史标签: {', '.join(historical_tags)}
    描述: {description}
    """
    completion = exponential_backoff_retry(
        client.chat.completions.create,
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    content = completion.choices[0].message.content.strip()

    if content.strip().startswith("[") and content.strip().endswith("]"):
        cleaned_json = content.strip()
    elif "```json" in content:
        match = re.search(r"```json(.*?)```", content, re.DOTALL)
        cleaned_json = match.group(1).strip() if match else content.strip()
    elif "```" in content:
        match = re.search(r"```(.*?)```", content, re.DOTALL)
        cleaned_json = match.group(1).strip() if match else content.strip()
    else:
        cleaned_json = content.strip()

    logging.info(f"生成的标签: {cleaned_json}")
    try:
        tags = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {e}")
        tags = []
    
    tags = tags[:4]

    for tag in tags:
        if tag not in historical_tags:
            historical_tags.append(tag)

    logging.info(f"生成的最终标签列表: {tags}")
    return tags

def calculate_mp3_duration(mp3_url):
    """
    通过URL下载MP3文件并计算其时长（秒）。
    """
    try:
        logger.info(f"正在下载MP3文件：{mp3_url}")
        response = requests.get(mp3_url, timeout=10)
        response.raise_for_status()
        audio = MP3(BytesIO(response.content))
        duration = int(audio.info.length)
        logger.info(f"MP3文件 {mp3_url} 的时长为 {duration} 秒")
        return duration
    except Exception as e:
        logger.error(f"无法计算MP3文件 {mp3_url} 的时长，错误：{e}")
        return None


def migrate_image_to_cos(image_url, recompress=False):
    """下载并上传现有图片到COS，返回新的URL

    参数 recompress 为 True 时，即使图片已在 COS 也会重新压缩并上传。
    """
    if image_url.startswith(COS_IMAGE_BASE_URL) and not recompress:
        logger.info("图片已在腾讯云COS，无需迁移")
        return image_url
    try:
        logger.info(f"开始迁移图片：{image_url}")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image_data = response.content
        filename = f"{uuid.uuid4().hex}.jpg"
        tmp_path = Path("wash_list_image") / filename
        compress_image(image_data, tmp_path)
        cos_key = f"podcasts_image/{filename}"
        new_url = upload_file_to_cos(str(tmp_path), cos_key)
        tmp_path.unlink()
        logger.info(f"图片迁移完成：{new_url}")
        return new_url
    except Exception as e:
        logger.error(f"迁移图片失败 {image_url}：{e}")
        return image_url

def main():
    logging.info("开始处理播客列表...")

    cred = credentials.Certificate("./serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://news-fetcher-platform-default-rtdb.asia-southeast1.firebasedatabase.app',
        'storageBucket': 'news-fetcher-platform.firebasestorage.app'
    })

    with open("list_to_wash.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    historical_tags = []

    for podcast in data.get("podcasts", []):
        title = podcast.get("title", "")
        logging.info(f"处理播客标题: {title}")

        # 从title中提取日期
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})_", title)
        if date_match:
            date_str = date_match.group(1)
            podcast["date"] = date_str
            logging.info(f"提取的日期: {date_str}")

        # 清理标题
        cleaned_title = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", title)
        cleaned_title = re.sub(r"\.mp3$", "", cleaned_title)
        podcast["title"] = cleaned_title
        logging.info(f"清理后的标题: {cleaned_title}")

        # 若已有img_url且需要迁移到COS
        if MIGRATE_IMAGES and podcast.get("img_url"):
            podcast["img_url"] = migrate_image_to_cos(
                podcast["img_url"], recompress=COMPRESS_EXISTING_IMAGES
            )

        # 若缺img_url且WASH_IMAGES为True则生成
        if WASH_IMAGES and "img_url" not in podcast:
            podcast["img_url"] = generate_img_url(cleaned_title, podcast.get("description", ""))

        # 生成标签
        description = podcast.get("description", "")
        if WASH_TAGS and ("tags" not in podcast or not podcast["tags"]):
            podcast["tags"] = generate_tags_by_description(description, historical_tags)

        # 计算 total_duration
        if WASH_TOTAL_DURATION and "total_duration" not in podcast:
            # 构造MP3文件的下载URL
            filename = podcast.get("sha256", "")
            if filename:
                mp3_url = f"{MP3_BASE_URL}{filename}.mp3"
                duration = calculate_mp3_duration(mp3_url)
                if duration is not None:
                    podcast["total_duration"] = duration
                    logger.info(f"播客 {cleaned_title} 的 total_duration 计算成功: {duration} 秒")
            else:
                logger.warning(f"播客 {cleaned_title} 缺少 sha256 字段，无法构造MP3 URL。")

    # 在所有数据处理完成后，再执行删除小于2分钟时长的逻辑
    if DELETE_SHORT_AUDIO:
        original_count = len(data["podcasts"])
        data["podcasts"] = [
            p for p in data["podcasts"]
            if p.get("total_duration", 0) >= 120
        ]
        new_count = len(data["podcasts"])
        logging.info(f"已删除小于2分钟时长的播客 {original_count - new_count} 个。")

    with open("list_to_wash_processed.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info("处理完成，已保存到 list_to_wash_processed.json")

if __name__ == "__main__":
    main()
