"""
Configuration module for Slides-helper.

Manages application settings, environment variables, and model configurations.
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class LMStudioConfig(BaseModel):
    """Configuration for LM Studio connection."""
    
    base_url: str = Field(
        default_factory=lambda: os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    )
    vision_model_name: str = Field(
        default_factory=lambda: os.getenv("VISION_MODEL_NAME", "Qwen2.5-VL")
    )
    api_key: str = Field(default="not-needed")  # LM Studio doesn't require API key
    timeout: int = Field(default=120)  # 2 minutes timeout


class TTSConfig(BaseModel):
    """Configuration for Text-to-Speech engine."""
    
    engine: str = Field(
        default_factory=lambda: os.getenv("TTS_ENGINE", "edge-tts")
    )
    voice: str = Field(
        default_factory=lambda: os.getenv("TTS_VOICE", "en-US-AriaNeural")
    )
    rate: str = Field(default="+0%")  # Speech rate
    volume: str = Field(default="+0%")  # Speech volume


class ProcessingConfig(BaseModel):
    """Configuration for slide processing."""
    
    max_slides_per_batch: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SLIDES_PER_BATCH", "5"))
    )
    image_quality: str = Field(
        default_factory=lambda: os.getenv("IMAGE_QUALITY", "high")
    )
    output_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output"))
    )
    temp_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("TEMP_DIR", "./temp"))
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


class Config(BaseModel):
    """Main configuration class."""
    
    lm_studio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)


# Global config instance
config = Config()
