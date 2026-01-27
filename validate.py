#!/usr/bin/env python3
"""
Simple validation script to check the installation and basic functionality.

This script validates that:
1. All dependencies are installed
2. Modules can be imported
3. Basic functionality works (without requiring LM Studio or a real .pptx file)
"""
import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils import config, image_to_base64, resize_image
        from src.slide_processor import SlideProcessor, SlideContent
        from src.vision_analyzer import VisionAnalyzer
        from src.tts_engine import TTSEngine, TTSEngineFactory
        from src.orchestrator import SlidesOrchestrator, ProcessingResult
        print("✓ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.utils import config
        
        print(f"  LM Studio URL: {config.lm_studio.base_url}")
        print(f"  Vision Model: {config.lm_studio.vision_model_name}")
        print(f"  TTS Engine: {config.tts.engine}")
        print(f"  TTS Voice: {config.tts.voice}")
        print(f"  Output Dir: {config.processing.output_dir}")
        print(f"  Temp Dir: {config.processing.temp_dir}")
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_image_utils():
    """Test image utilities."""
    print("\nTesting image utilities...")
    
    try:
        from PIL import Image
        from src.utils.image_utils import resize_image, image_to_base64
        
        # Create a test image
        img = Image.new('RGB', (1920, 1080), color='red')
        
        # Test resize
        resized = resize_image(img, max_size=640)
        assert resized.size[0] <= 640 or resized.size[1] <= 640
        print(f"  ✓ Image resized from {img.size} to {resized.size}")
        
        # Test base64 conversion
        b64 = image_to_base64(img)
        assert len(b64) > 0
        print(f"  ✓ Image converted to base64 ({len(b64)} chars)")
        
        print("✓ Image utilities working correctly")
        return True
    except Exception as e:
        print(f"✗ Image utilities error: {e}")
        return False


def test_tts_engine():
    """Test TTS engine initialization (without actually generating speech)."""
    print("\nTesting TTS engine...")
    
    try:
        from src.tts_engine import TTSEngineFactory
        
        tts = TTSEngineFactory.create_engine()
        print(f"  ✓ TTS engine created: {tts.__class__.__name__}")
        print(f"  ✓ Voice: {tts.voice}")
        print(f"  ✓ Rate: {tts.rate}")
        print(f"  ✓ Volume: {tts.volume}")
        
        # Test voice listing (requires internet, but should work)
        try:
            voices = tts.list_available_voices()
            print(f"  ✓ Found {len(voices)} available voices")
        except Exception as e:
            print(f"  ⚠ Could not list voices (internet required): {e}")
        
        print("✓ TTS engine working correctly")
        return True
    except Exception as e:
        print(f"✗ TTS engine error: {e}")
        return False


def test_directory_structure():
    """Test that all required directories and files exist."""
    print("\nTesting project structure...")
    
    required_files = [
        "src/__init__.py",
        "src/utils/__init__.py",
        "src/utils/config.py",
        "src/utils/image_utils.py",
        "src/slide_processor/__init__.py",
        "src/slide_processor/processor.py",
        "src/vision_analyzer/__init__.py",
        "src/vision_analyzer/analyzer.py",
        "src/tts_engine/__init__.py",
        "src/tts_engine/engine.py",
        "src/orchestrator/__init__.py",
        "src/orchestrator/orchestrator.py",
        "main.py",
        "requirements.txt",
        "README.md",
        ".env.example",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {', '.join(missing_files)}")
        return False
    
    print(f"✓ All {len(required_files)} required files present")
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Slides-helper Validation Script")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Image Utilities", test_image_utils),
        ("TTS Engine", test_tts_engine),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All validation tests passed!")
        print("\nNext steps:")
        print("1. Set up LM Studio (see docs/SETUP.md)")
        print("2. Run: python main.py your_presentation.pptx")
        return 0
    else:
        print("\n✗ Some validation tests failed")
        print("\nPlease fix the issues above before proceeding.")
        print("See docs/SETUP.md for installation instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
