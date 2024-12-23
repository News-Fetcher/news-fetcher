# utils/audio_utils.py
import hashlib
import logging
import re
from pathlib import Path
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def calculate_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def merge_audio_files(mp3_file_paths, output_file_path):
    """
    将给定的 MP3 文件依次合并并输出到 output_file_path
    """
    if not mp3_file_paths:
        logger.warning("No MP3 files provided to merge.")
        return

    try:
        combined_audio = AudioSegment.from_file(mp3_file_paths[0])
        for mp3_file in mp3_file_paths[1:]:
            combined_audio += AudioSegment.silent(duration=1000)  # 1秒静音
            combined_audio += AudioSegment.from_file(mp3_file)

        combined_audio.export(output_file_path, format="mp3")
        logger.info(f"Merged audio saved as: {output_file_path}")
    except Exception as e:
        logger.error(f"Error merging MP3 files: {e}")


def extract_domain(url: str) -> str:
    import re
    match = re.search(r'https?://([^/]+)', url)
    if match:
        return match.group(1)
    return "unknown_domain"