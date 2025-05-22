# News Podcast Generator

本项目用于自动抓取新闻网站的最新文章，并生成每日播客。流程包括新闻抓取、摘要生成、文本转语音、音频合并以及上传到 Firebase 等步骤，便于快速制作播客节目。

## 功能特点
- **新闻抓取**：使用 Firecrawl 根据配置文件爬取或抓取指定网站的文章。
- **摘要生成**：调用 OpenAI 或阿里云百炼模型生成适合播客的中文摘要。
- **文本转语音**：利用 OpenAI TTS 将摘要转换成 MP3 文件。
- **节目开场与结束**：自动生成开场与结束词并添加到最终音频。
- **音频合并与上传**：将所有音频合并成单个文件，并上传到 Firebase Storage（工作流可选地推送到 GitHub）。

## 环境依赖
- Python 3.8 及以上
- FFmpeg
- 其它依赖见 `requirements.txt`

## 安装
```bash
git clone https://github.com/nagisa77/news-fetcher.git
cd news-fetcher
pip install -r requirements.txt
```
如系统未安装 FFmpeg，可在 macOS 使用 `brew install ffmpeg`，Ubuntu 使用 `sudo apt install ffmpeg`。

## 配置
1. 将 Firebase 的 `serviceAccountKey.json` 放在项目根目录。
2. 设置环境变量：
   - `FIRECRAWL_API_KEY`：Firecrawl API Key。
   - `OPENAI_API_KEY` 或 `DASHSCOPE_API_KEY`：用于摘要和语音合成的 Key。
   - `GH_ACCESS_TOKEN`：如果需通过 GitHub Actions 推送结果，则需要该 Token。
3. 修改 `news_websites_crawl_*.json` 或 `news_websites_scraping.json` 以配置要抓取的网站和路径。

这些变量可在终端导出，也可以写入 `.env` 文件后在运行前加载。

## 使用方法
运行以下命令开始生成播客：
```bash
python main.py
```
脚本执行后会：
1. 根据配置抓取新闻文章；
2. 生成摘要并转换成语音；
3. 将所有音频合并为单个 MP3；
4. 上传到 Firebase Storage，并在本地 `podcast_audio` 目录保存最终文件。

## 输出示例
```text
podcast_audio/2024-11-28_Daily_News.mp3
```
上传后的文件也可在 Firebase 或（若启用工作流）GitHub 仓库中查看。

## 其他说明
- 请确保 API Key 具备足够的调用额度。
- 配置文件中的 `limit` 和 `includePaths` 可控制抓取范围和数量。

## License
本项目采用 MIT License，详见 LICENSE 文件。

