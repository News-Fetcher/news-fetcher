import os
import hashlib
import re
from pathlib import Path
from openai import OpenAI
from utils.audio_utils import extract_domain

# 初始化OpenAI客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 测试用的数据
url = "https://example.com/news/1"
article_summary = """这是一个测试摘要，用于生成测试音频。这个摘要包含了文章的主要内容和关键点。
文章讨论了人工智能在现代社会中的应用和影响，特别是在自然语言处理和语音合成技术方面的进展。
此外，文章还探讨了这些技术如何改变我们的日常生活和工作方式，以及未来可能的发展方向。
这个测试摘要将被用来测试文本转语音功能的效果和质量。"""
output_folder = "./output"
os.makedirs(output_folder, exist_ok=True)

# 开始调试指定代码段
url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()[:8]
domain = extract_domain(url)
safe_domain = re.sub(r'[^\w.-]', '_', domain)
filename = f"summary_{safe_domain}_{url_hash}.mp3"
speech_file_path = Path(output_folder) / filename
instructions = """语气非常非常轻，非常温柔, 轻柔, 有一种温柔姐姐, 睡前讲故事的感觉, 轻到几乎只有气流声"""

tts_response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="sage",
    speed=1.3,
    input=article_summary,
    instructions=instructions,
)
tts_response.stream_to_file(speech_file_path)

print("生成的音频文件路径：", speech_file_path)
