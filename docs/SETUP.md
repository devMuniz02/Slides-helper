# Setup Guide

## Setting up LM Studio

### 1. Download and Install LM Studio

1. Visit https://lmstudio.ai/
2. Download the version for your operating system
3. Install and launch LM Studio

### 2. Download a Vision Model

1. In LM Studio, click on the "Search" tab
2. Search for "Qwen2.5-VL" or another Vision model
3. Download a model compatible with your VRAM (12GB):
   - Recommended: `Qwen2.5-VL-7B-Instruct-GGUF` (Q4 or Q5 quantization)
   - Alternative: `llava-v1.6-mistral-7b-gguf`

### 3. Start the Server

1. Go to the "Local Server" tab in LM Studio
2. Select your downloaded Vision model
3. Click "Start Server"
4. Verify it's running at `http://localhost:1234/v1`
5. Test the endpoint:
   ```bash
   curl http://localhost:1234/v1/models
   ```

### 4. Configure for Vision

- Enable "Vision" support in the model settings
- Adjust context length based on your needs (default: 2048 is fine)
- Set temperature to 0.7 for balanced creativity

## Python Environment Setup

### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using conda

```bash
# Create conda environment
conda create -n slides-helper python=3.10

# Activate it
conda activate slides-helper

# Install dependencies
pip install -r requirements.txt
```

## First Run

### 1. Test Individual Modules

```bash
# Test TTS (doesn't require LM Studio)
python examples/tts_demo.py

# Test slide processor (provide your own .pptx file)
python examples/basic_processor.py
```

### 2. Verify LM Studio Connection

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

# Test connection
models = client.models.list()
print("Available models:", [m.id for m in models.data])
```

### 3. Run Full Pipeline

```bash
# Replace with your actual .pptx file
python main.py your_presentation.pptx
```

## Optimizing for Your Hardware

### Memory Optimization (32GB RAM)

In `.env`:
```env
MAX_SLIDES_PER_BATCH=3  # Process fewer slides at once
IMAGE_QUALITY=medium     # Reduce image size
```

### VRAM Optimization (12GB)

1. **Use quantized models**: Q4 or Q5 quantization is ideal for 12GB
2. **Reduce context length**: Set to 2048 or 4096 in LM Studio
3. **Close other GPU applications**: Ensure no other apps use VRAM

### Speed Optimization

1. **Use SSD**: Store temp files on SSD for faster I/O
2. **Batch processing**: Increase `MAX_SLIDES_PER_BATCH` if RAM allows
3. **Lower image quality**: Use "low" for faster processing

## Troubleshooting

### "Model not found" error

**Solution**: Check that LM Studio server is running and the model name in `.env` matches exactly.

```bash
# Check running models
curl http://localhost:1234/v1/models
```

### "Connection refused" error

**Solution**: Ensure LM Studio server is started on port 1234.

1. Open LM Studio
2. Go to "Local Server" tab
3. Click "Start Server"

### Out of memory errors

**Solution**: Reduce batch size and image quality in `.env`:

```env
MAX_SLIDES_PER_BATCH=1
IMAGE_QUALITY=low
```

### TTS not working

**Solution**: edge-tts needs internet for first use to download voices.

```bash
# List available voices
edge-tts --list-voices

# Test TTS manually
edge-tts --text "Hello world" --write-media test.mp3
```

### PowerPoint file not opening

**Solution**: Ensure the file is a valid .pptx (not .ppt):

```bash
# Convert .ppt to .pptx using PowerPoint or LibreOffice
```

## Advanced Configuration

### Custom TTS Voice

List all available voices:
```python
from src.tts_engine import TTSEngineFactory

tts = TTSEngineFactory.create_engine()
voices = tts.list_available_voices()

# Filter for specific language
en_voices = [v for v in voices if v['Locale'].startswith('en-US')]
for voice in en_voices:
    print(f"{voice['ShortName']}: {voice['Gender']}")
```

Update `.env`:
```env
TTS_VOICE=en-US-JennyNeural  # Change to your preferred voice
```

### Custom Prompts

Modify prompts in `src/vision_analyzer/analyzer.py`:

```python
def _build_analysis_prompt(self, slide_text: Optional[str] = None) -> str:
    """Customize this method for different analysis styles."""
    prompt = """Your custom prompt here..."""
    return prompt
```

### Adding New TTS Engines

1. Create a new class inheriting from `TTSEngine`
2. Implement required methods
3. Add to `TTSEngineFactory`

Example for Kokoro-82M:
```python
class KokoroTTSEngine(TTSEngine):
    def generate_speech(self, text: str, output_path: Path) -> Path:
        # Implement Kokoro integration
        pass
```

## Performance Benchmarks

Typical performance on recommended hardware:

- **Slide Extraction**: ~0.5s per slide
- **Vision Analysis**: ~2-5s per slide (depends on model and image size)
- **TTS Generation**: ~1-2s per slide
- **Total**: ~3-8s per slide

For a 20-slide presentation: **1-3 minutes total**
