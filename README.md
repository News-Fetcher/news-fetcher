# News Podcast Generator

This script automates the process of crawling news websites, summarizing articles, generating audio using text-to-speech (TTS), and compiling a daily podcast. Finally, it commits the generated podcast to a GitHub repository.

## Features
- Crawls news websites for relevant articles.
- Summarizes articles into a podcast-friendly format.
- Converts summaries into MP3 files using TTS.
- Generates an introduction for the podcast.
- Merges all MP3 files into a single podcast episode.
- Commits the podcast to a GitHub repository.

## Requirements
- Python 3.8+
- `pydub` for audio manipulation.
- `openai` for text and audio generation.
- `firecrawl` for web crawling.
- `PyGithub` for GitHub API integration.
- FFmpeg installed for audio processing.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nagisa77/news-fetcher.git
   cd news-fetcher
	 ```

2.	Install dependencies:

	  ```
	  pip install -r requirements.txt
	  ```


3.	Install FFmpeg (if not already installed):
	•	macOS: brew install ffmpeg
	•	Ubuntu: sudo apt install ffmpeg
	•	Windows: Download and install FFmpeg from FFmpeg.org.

## Setup

1.	Set up environment variables:

	•	FIRECRAWL_API_KEY: Your Firecrawl API key.

	•	GH_ACCESS_TOKEN: Your GitHub access token.
	•	DASHSCOPE_API_KEY: Your DashScope API key.

2.	Update the script:

	•	Replace news_websites with the desired list of news website URLs.

	•	Ensure the REPO_NAME variable is set to nagisa77/news-fetcher.

## Usage

Run the script:

```
python news_podcast_generator.py
```

The script will:

1.	Crawl the specified news websites for articles.

2.	Generate summaries and convert them into audio files.

3.	Combine all audio files into a single MP3 podcast.

4.	Commit the podcast to your GitHub repository.


## Output

•	Podcasts are saved locally in the podcast_audio folder.

•	The final merged podcast is committed to the podcasts folder in the specified GitHub repository.

## Example

The podcast for today’s news will be saved locally as:

```
podcast_audio/2024-11-28.mp3
```

and committed to GitHub at:

```
podcasts/2024-11-28.mp3
```

## Notes

•	Ensure that your GitHub token has the appropriate permissions to create and push files to the repository.

•	Configure limit and includePaths in the FirecrawlApp parameters to control the number and type of articles crawled.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

