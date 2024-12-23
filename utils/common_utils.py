# utils/common_utils.py
import os
import json


def load_json_config(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件 {file_path} 未找到。")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)