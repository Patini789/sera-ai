"""Screenshot utility for Gaming Mode.

Captures the screen, resizes, compresses, and encodes to Base64.
"""
import base64
import io
from PIL import ImageGrab
from ..core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_DIMENSION = 768
DEFAULT_JPEG_QUALITY = 70


def capture_screenshot_b64(
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> str | None:
    """Capture a screenshot and return it as a Base64 JPEG string.

    The image is resized to fit within *max_dimension* x *max_dimension*
    while preserving aspect ratio, then compressed as JPEG.

    Args:
        max_dimension: Max width/height in pixels before resizing.
        jpeg_quality: JPEG quality (1-100). Lower = smaller payload.

    Returns:
        Base64-encoded JPEG string, or None on failure.
    """
    try:
        screenshot = ImageGrab.grab()
    except Exception as e:
        logger.error(f"Failed to capture screenshot: {e}")
        return None

    try:
        width, height = screenshot.size
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension / width, max_dimension / height)
            new_size = (int(width * ratio), int(height * ratio))
            screenshot = screenshot.resize(new_size)
    except Exception as e:
        logger.warning(f"Failed to resize screenshot: {e}")

    try:
        buffer = io.BytesIO()
        screenshot.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        logger.info("Screenshot captured, resized and encoded successfully.")
        return b64
    except Exception as e:
        logger.error(f"Failed to encode screenshot: {e}")
        return None
