import logging
from io import BytesIO
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


def compress_image(image_bytes: bytes, output_path: Path, quality: int = 80) -> Path:
    """Compress image bytes and save as JPEG."""
    try:
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="JPEG", optimize=True, quality=quality)
        logger.info(f"Compressed image saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to compress image, saving original. Error: {e}")
        with open(output_path, "wb") as f:
            f.write(image_bytes)
    return output_path

