"""Slides-helper main package."""
from .slide_processor import SlideProcessor, SlideContent
from .vision_analyzer import VisionAnalyzer
from .tts_engine import TTSEngine, TTSEngineFactory
from .orchestrator import SlidesOrchestrator, ProcessingResult
from .powerpoint_connector import PowerPointConnector
from .gui import SlidesHelperGUI
from .utils import config

__version__ = "0.1.0"

__all__ = [
    "SlideProcessor",
    "SlideContent",
    "VisionAnalyzer",
    "TTSEngine",
    "TTSEngineFactory",
    "SlidesOrchestrator",
    "ProcessingResult",
    "PowerPointConnector",
    "SlidesHelperGUI",
    "config",
]
