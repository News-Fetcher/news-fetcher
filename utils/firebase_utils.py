# utils/firebase_utils.py
import logging
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime

logger = logging.getLogger(__name__)


def initialize_firebase(service_account_path: str):
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://news-fetcher-platform-default-rtdb.asia-southeast1.firebasedatabase.app',
        'storageBucket': 'news-fetcher-platform.firebasestorage.app'
    })
    logger.info("Firebase initialized successfully.")


def is_url_fetched(url: str) -> bool:
    try:
        ref = db.reference("fetched_urls")
        fetched_urls = ref.get() or []
        return url in fetched_urls
    except Exception as e:
        logger.error(f"Error checking URL in Firebase: {e}")
        return False


def add_url_to_fetched(url: str):
    try:
        ref = db.reference("fetched_urls")
        fetched_urls = ref.get() or []
        if url not in fetched_urls:
            fetched_urls.append(url)
            ref.set(fetched_urls)
    except Exception as e:
        logger.error(f"Error adding URL to Firebase: {e}")


def record_metadata_to_firebase(title, description, sha256, img_url=None, tags=None, total_duration=None):
    if tags is None:
        tags = []
    try:
        ref = db.reference("podcasts")
        existing_data = ref.get() or []

        # 如果数据库里存的是 dict，而不是 list，则转换成 list
        if isinstance(existing_data, dict):
            existing_data = list(existing_data.values())

        metadata = {
            "title": title,
            "description": description,
            "sha256": sha256,
            "tags": tags,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }

        if img_url:
            metadata["img_url"] = img_url

        if total_duration is not None:
            metadata["total_duration"] = total_duration

        existing_data.append(metadata)
        ref.set(existing_data)
        logger.info(f"Metadata recorded to Firebase: {metadata}")
    except Exception as e:
        logger.error(f"Error recording metadata to Firebase: {e}")


def upload_to_firebase_storage(local_file_path, title, description, sha256_hash, img_url, tags):
    """
    根据 local_file_path 上传到 Storage 并记录 Podcast 元信息到 Realtime Database
    返回 (audio_public_url, sha256_hash) 供后续使用
    """
    try:
        bucket = storage.bucket()
        storage_path = f"podcasts/{sha256_hash}.mp3"

        blob = bucket.blob(storage_path)
        blob.upload_from_filename(local_file_path)

        # 生成公共访问链接
        blob.make_public()
        audio_public_url = blob.public_url
        logger.info(f"File uploaded to Firebase Storage: {storage_path}")

        from pydub import AudioSegment
        audio = AudioSegment.from_file(local_file_path)
        duration_seconds = int(len(audio) / 1000.0)
        record_metadata_to_firebase(title, description, sha256_hash, img_url, tags, duration_seconds)

        return audio_public_url, sha256_hash
    except Exception as e:
        logger.error(f"Error uploading file to Firebase Storage: {e}")
        return None, None