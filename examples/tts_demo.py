#!/usr/bin/env python3
"""
Example: TTS engine usage.

This example demonstrates how to use the TTS engine to convert text to speech.
"""
from pathlib import Path
from src.tts_engine import TTSEngineFactory


def main():
    """Demonstrate TTS engine usage."""
    # Create TTS engine
    print("Initializing TTS engine (edge-tts)...")
    tts = TTSEngineFactory.create_engine()
    
    # Example texts
    texts = [
        "Hello! This is slide 1. Welcome to our presentation about AI and machine learning.",
        "Slide 2 covers the key concepts of neural networks and deep learning architectures.",
        "In slide 3, we explore practical applications of AI in various industries.",
    ]
    
    # Output directory
    output_dir = Path("./temp/audio_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating speech for {len(texts)} examples...")
    print(f"Voice: {tts.voice}")
    print(f"Output directory: {output_dir}")
    
    # Generate speech files
    audio_files = tts.generate_speech_batch(
        texts,
        output_dir,
        filename_prefix="example",
    )
    
    print(f"\n✓ Generated {len(audio_files)} audio files:")
    for audio_file in audio_files:
        print(f"  - {audio_file}")
    
    # List available voices
    print("\nListing some available voices...")
    voices = tts.list_available_voices()
    
    # Filter for English voices
    en_voices = [v for v in voices if v.get('Locale', '').startswith('en')][:10]
    
    print(f"\nSample English voices (showing {len(en_voices)} of {len([v for v in voices if v.get('Locale', '').startswith('en')])}):")
    for voice in en_voices:
        print(f"  - {voice.get('ShortName', 'N/A')}: {voice.get('Gender', 'N/A')}")


if __name__ == "__main__":
    main()
