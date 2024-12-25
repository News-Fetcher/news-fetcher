import requests
import json
import os
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("scraping.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_websites(file_path):
    """
    Read the websites.txt file and return a list of URLs.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # Read all non-empty lines and strip whitespace
        websites = [line.strip() for line in file if line.strip()]
    
    if not websites:
        logger.error("websites.txt is empty or contains no valid URLs.")
        raise ValueError("websites.txt is empty or contains no valid URLs.")
    
    return websites

def send_post_request(url, payload, headers):
    """
    Send a POST request to the specified URL.
    """
    logger.info(f"Sending POST request to {url} with payload: {payload}")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        logger.info(f"Received response with status code: {response.status_code}")
        logger.debug(f"Response headers: {response.headers}")
        logger.debug(f"Response content: {response.text}")

        response.raise_for_status()  # Raise HTTPError for bad responses

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.Timeout:
        logger.error("Request timed out.")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error occurred: {req_err}")

def main():
    # Configuration parameters
    api_endpoint = "https://triggerpodcastgeneration-a6lubplbza-uc.a.run.app"
    headers = {
        "Content-Type": "application/json"
    }
    email = "1216414009@qq.com"  # You can modify this email address as needed
    websites_file = "./websites.txt"

    method = "scraping"  
    be_concise = False   

    try:
        # Read the list of websites
        websites = read_websites(websites_file)
        logger.info(f"Read {len(websites)} websites.")

        # Construct the payload
        payload = {
            "news_websites_scraping": websites,
            "email": email,
            "method": method,
            "be_concise": be_concise
        }

        # Send the POST request
        logger.info("Sending POST request...")
        send_post_request(api_endpoint, payload, headers)

    except Exception as e:
        logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()