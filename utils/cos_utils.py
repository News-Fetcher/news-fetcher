import os
import logging
from qcloud_cos import CosConfig, CosS3Client

logger = logging.getLogger(__name__)


def get_cos_client():
    secret_id = os.getenv("COS_SECRET_ID")
    secret_key = os.getenv("COS_SECRET_KEY")
    region = os.getenv("COS_REGION")
    if not all([secret_id, secret_key, region]):
        raise ValueError("COS credentials or region not set in environment variables")
    config = CosConfig(
        Region=region,
        SecretId=secret_id,
        SecretKey=secret_key,
        Token=None,
        Scheme="https",
    )
    return CosS3Client(config)


def upload_file_to_cos(local_file_path: str, cos_key: str) -> str:
    """Upload a local file to COS and return its public URL."""
    client = get_cos_client()
    bucket = os.getenv("COS_BUCKET")
    if not bucket:
        raise ValueError("COS_BUCKET not set in environment variables")

    with open(local_file_path, "rb") as fp:
        client.put_object(Bucket=bucket, Body=fp, Key=cos_key)

    base_path = os.getenv("COS_PATH", "").rstrip("/")
    return f"{base_path}/{cos_key}"
