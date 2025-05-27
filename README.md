# News Podcast Generator

本项目可以将新闻网站的最新文章自动生成播客，实现从抓取到发布的完整流程：新闻抓取、摘要生成、文本转语音、音频合并以及上传到 Firebase，帮助你快速制作每日节目。

## 项目亮点
- **端到端自动化**：一条命令即可完成爬取新闻、AI 摘要、语音合成、音频合并和上传。
- **动态日期抓取**：配置支持按日期扩展 URL，适配 Reuters、CoinDesk、news.smol.ai 等需要日期参数的网站。
- **灵活的 LLM**：可选择 OpenAI 或阿里云百炼模型，并自动处理 token 限制和重试逻辑。
- **自然的中文语音**：使用阿里云 cosyvoice 生成高质量中文音频，也支持自定义音色。
- **自动封面与标签**：借助 GPT 生成开场、结束语、播客封面及关键标签，提升节目专业度。
- **多平台发布**：合成后的 MP3 可上传到 Firebase Storage 或腾讯云，并可通过 GitHub Actions 自动化推送。

## 功能特点
- **新闻抓取**：使用 Firecrawl 根据配置文件爬取或抓取指定网站的文章。
- **摘要生成**：调用 OpenAI 或阿里云百炼模型生成适合播客的中文摘要。
- **新闻分析**：基于所有摘要生成整体 AI 分析。
- **文本转语音**：利用 TTS 将摘要转换成 MP3 文件。
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
   - `OPENAI_API_KEY`、`DASHSCOPE_API_KEY` 或 `DEEPSEEK_API_KEY`：用于摘要和语音合成的 Key。
   - `GH_ACCESS_TOKEN`：如果需通过 GitHub Actions 推送结果，则需要该 Token。
3. 修改 `news_websites_crawl_*.json` 或 `news_websites_scraping.json` 以配置要抓取的网站和路径。
4. 如需按日期扩展抓取路径，可编辑 `news_dynamic_paths.json`（或通过环境变量 `DYNAMIC_DATE_CONFIG` 指定其他文件），默认已包含 Reuters、CoinDesk、news.smol.ai 的示例。

这些变量可在终端导出，也可以写入 `.env` 文件后在运行前加载。

### 配置文件说明

- **news_websites_crawl_*.json**：定义爬取站点及规则。键为网站域名，值中可设置：
  - `limit`：每个网站最多抓取的文章数量。
  - `includePaths`：指定需要包含的路径，可结合动态日期扩展。
  - `excludePaths`：排除不需要的路径。
- **news_websites_scraping.json**：直接抓取的文章 URL 列表，用于无需站点爬取时的场景。
- **news_dynamic_paths.json**：根据站点关键字提供日期格式和 `days_offset`，在运行时自动生成 `includePaths`。
- **serviceAccountKey.json**：Firebase 凭证文件，需放在项目根目录。
- **image_generation_config.json**：配置播客封面图片生成的模型、尺寸和质量。

### 环境变量说明

- `FIRECRAWL_API_KEY`：Firecrawl API Key，用于爬取或抓取文章。
- `OPENAI_API_KEY`：当 `SUMMARY_PROVIDER=openai` 时使用的 API Key。
- `OPENAI_MODEL`：OpenAI 模型名称，默认 `gpt-4o`。
- 图片生成配置可在 `image_generation_config.json` 中调整，包括模型、尺寸和质量。
- `DASHSCOPE_API_KEY`：阿里云百炼 API Key，`SUMMARY_PROVIDER=tongyi` 时使用。
- `DASHSCOPE_MODEL`：阿里云模型名称，默认 `qwen-plus`。
- `DEEPSEEK_API_KEY`：DeepSeek API Key，`SUMMARY_PROVIDER=deepseek` 时使用。
- `DEEPSEEK_MODEL`：DeepSeek 模型名称，默认 `deepseek-chat`。
- `SCRAPE_WAIT_MS`：Firecrawl 抓取时额外等待的毫秒数，默认 `2000`。
- `SCRAPE_TIMEOUT_MS`：Firecrawl 抓取的超时时间毫秒数，默认 `40000`。
- `SUMMARY_PROVIDER`：选择 `openai`、`tongyi` 或 `deepseek` 作为摘要和文生图的提供方，默认 `tongyi`。
- `BE_CONCISE`：设为 `true` 时生成更简洁的摘要。
- `FETCHER_METHOD`：`crawling`（默认）或 `scraping`，决定新闻获取方式。
- `CRAWL_CONFIG_FILE`：爬取配置文件路径，默认为 `news_websites_crawl_coindesk.json`。
- `SCRAPING_CONFIG`：直接传入的抓取配置 JSON 字符串，优先级高于本地文件。
- `DYNAMIC_DATE_CONFIG`：动态日期配置文件路径，默认为 `news_dynamic_paths.json`。
- `GH_ACCESS_TOKEN`：如需通过 GitHub Actions 推送结果时使用。
- `IMAGE_GEN_CONFIG_FILE`：图片生成配置文件路径，默认为 `image_generation_config.json`。
- `COS_SECRET_ID`、`COS_SECRET_KEY`、`COS_REGION`、`COS_BUCKET`、`COS_PATH`：配置腾讯云 COS 上传所需的凭据及路径。

## 使用方法
运行以下命令开始生成播客：
```bash
python main.py
```
脚本执行后会：
1. 根据配置抓取新闻文章；
2. 生成摘要并转换成语音；
3. 生成 AI 新闻分析；
4. 将所有音频合并为单个 MP3；
5. 上传到 Firebase Storage，并在本地 `podcast_audio` 目录保存最终文件。

## 输出示例
```text
podcast_audio/2024-11-28_Daily_News.mp3
```
上传后的文件也可在 Firebase 或（若启用工作流）GitHub 仓库中查看。

## 其他说明
- 请确保 API Key 具备足够的调用额度。
- 配置文件中的 `limit` 和 `includePaths` 可控制抓取范围和数量。

## License
本项目采用 MIT License，同时声明支持 GPL 协议（GNU General Public License），详见 LICENSE 文件。

