"""
TTS (Text-to-Speech) Engine Module

This module handles text-to-speech conversion using edge-tts or other TTS libraries.
"""
import asyncio
from pathlib import Path
from typing import Optional, List

import edge_tts

from ..utils.config import config


class TTSEngine:
    """
    Text-to-Speech engine using edge-tts.
    
    This class provides functionality to:
    - Convert text to speech audio
    - Save audio files
    - Support multiple voices and languages
    """
    
    def __init__(self, voice: Optional[str] = None):
        """
        Initialize the TTS engine.
        
        Args:
            voice: Voice to use (defaults to config setting)
        """
        self.voice = voice or config.tts.voice
        self.rate = config.tts.rate
        self.volume = config.tts.volume
    
    async def _generate_speech_async(
        self,
        text: str,
        output_path: Path,
    ) -> None:
        """
        Asynchronously generate speech from text.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
        """
        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
        )
        await communicate.save(str(output_path))
    
    def generate_speech(
        self,
        text: str,
        output_path: Path,
    ) -> Path:
        """
        Generate speech from text and save to file.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
            
        Returns:
            Path to the saved audio file
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run async function
        asyncio.run(self._generate_speech_async(text, output_path))
        
        return output_path
    
    def generate_speech_batch(
        self,
        texts: List[str],
        output_dir: Path,
        filename_prefix: str = "narration",
    ) -> List[Path]:
        """
        Generate speech for multiple texts.
        
        Args:
            texts: List of texts to convert
            output_dir: Directory to save audio files
            filename_prefix: Prefix for output filenames
            
        Returns:
            List of paths to saved audio files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        for idx, text in enumerate(texts, start=1):
            output_path = output_dir / f"{filename_prefix}_{idx:03d}.mp3"
            self.generate_speech(text, output_path)
            output_paths.append(output_path)
        
        return output_paths
    
    async def _list_voices_async(self) -> List[dict]:
        """
        Asynchronously list available voices.
        
        Returns:
            List of voice dictionaries
        """
        voices = await edge_tts.list_voices()
        return voices
    
    def list_available_voices(self) -> List[dict]:
        """
        List all available voices.
        
        Returns:
            List of voice dictionaries with name, gender, locale info
        """
        return asyncio.run(self._list_voices_async())
    
    def set_voice(self, voice: str) -> None:
        """
        Change the voice for speech generation.
        
        Args:
            voice: Voice name (e.g., 'en-US-AriaNeural')
        """
        self.voice = voice
    
    def set_rate(self, rate: str) -> None:
        """
        Set speech rate.
        
        Args:
            rate: Rate modifier (e.g., '+0%', '+50%', '-25%')
        """
        self.rate = rate
    
    def set_volume(self, volume: str) -> None:
        """
        Set speech volume.
        
        Args:
            volume: Volume modifier (e.g., '+0%', '+50%', '-25%')
        """
        self.volume = volume


class TTSEngineFactory:
    """Factory for creating TTS engines."""
    
    @staticmethod
    def create_engine(engine_type: Optional[str] = None) -> TTSEngine:
        """
        Create a TTS engine instance.
        
        Args:
            engine_type: Type of engine to create (defaults to config setting)
            
        Returns:
            TTSEngine instance
        """
        engine_type = engine_type or config.tts.engine
        
        if engine_type == "edge-tts":
            return TTSEngine()
        # Future: Add support for other engines like Kokoro-82M
        # elif engine_type == "kokoro":
        #     return KokoroTTSEngine()
        else:
            raise ValueError(f"Unsupported TTS engine: {engine_type}")
