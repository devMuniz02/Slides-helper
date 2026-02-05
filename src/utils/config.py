"""
Configuration module for Slides-helper.

Manages application settings from config files.
"""
import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {config_path}: {e}, using defaults")
        return {}


# Load configuration from files
config_dir = Path(__file__).parent.parent.parent / "config"
rag_config = load_config_from_file(config_dir / "rag_config.json")


class LMStudioConfig(BaseModel):
    """Configuration for LM Studio connection."""

    base_url: str = Field(default=rag_config.get("lm_studio", {}).get("base_url", "http://localhost:1234/v1"))
    vision_model_name: str = Field(default=rag_config.get("lm_studio", {}).get("vision_model_name", "Qwen2.5-VL"))
    model_name: str = Field(default=rag_config.get("lm_studio", {}).get("model_name", "Qwen2.5-0.5B-Instruct"))
    embedding_model_name: str = Field(default=rag_config.get("lm_studio", {}).get("embedding_model_name", "text-embedding-ada-002"))
    api_key: str = Field(default="not-needed")  # LM Studio doesn't require API key
    timeout: int = Field(default=120)  # 2 minutes timeout


class TTSConfig(BaseModel):
    """Configuration for Text-to-Speech engine."""

    engine: str = Field(default=rag_config.get("tts", {}).get("engine", "edge-tts"))
    voice: str = Field(default=rag_config.get("tts", {}).get("voice", "en-US-AriaNeural"))
    rate: str = Field(default=rag_config.get("tts", {}).get("rate", "+0%"))  # Speech rate
    volume: str = Field(default=rag_config.get("tts", {}).get("volume", "+0%"))  # Speech volume


class WebInterfaceConfig(BaseModel):
    """Configuration for web interface."""

    host: str = Field(default=rag_config.get("web_interface", {}).get("host", "localhost"))
    port: int = Field(default=rag_config.get("web_interface", {}).get("port", 8000))
    reload: bool = Field(default=rag_config.get("web_interface", {}).get("reload", True))


class ProcessingConfig(BaseModel):
    """Configuration for slide processing."""

    max_slides_per_batch: int = Field(default=rag_config.get("processing", {}).get("max_slides_per_batch", 5))
    image_quality: str = Field(default=rag_config.get("processing", {}).get("image_quality", "high"))
    output_dir: Path = Field(default=Path(rag_config.get("processing", {}).get("output_dir", "./output")))
    temp_dir: Path = Field(default=Path(rag_config.get("processing", {}).get("temp_dir", "./temp")))

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
    web_interface: WebInterfaceConfig = Field(default_factory=WebInterfaceConfig)


# Global config instance
config = Config()
