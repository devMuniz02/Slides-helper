"""
Utility functions for Slides-helper.
"""
import base64
from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image


def image_to_base64(image: Union[Image.Image, Path, str]) -> str:
    """
    Convert an image to base64 string.
    
    Args:
        image: PIL Image object or path to image file
        
    Returns:
        Base64 encoded string of the image
    """
    if isinstance(image, (Path, str)):
        image = Image.open(image)
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def resize_image(image: Image.Image, max_size: int = 1920) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension (width or height)
        
    Returns:
        Resized PIL Image object
    """
    # Calculate new size maintaining aspect ratio
    width, height = image.size
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            return image
    else:
        if height > max_size:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            return image
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
