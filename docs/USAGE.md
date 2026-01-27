# Usage Guide

## Quick Start Guide

### Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Set Up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a Vision model (e.g., Qwen2.5-VL-7B-Instruct)
3. Start the local server at `http://localhost:1234/v1`

See [docs/SETUP.md](docs/SETUP.md) for detailed instructions.

### Step 3: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env if needed (default values work for most setups)
```

### Step 4: Run the Application

```bash
# Process a PowerPoint presentation
python main.py your_presentation.pptx

# With custom output directory
python main.py your_presentation.pptx --output-dir ./my_output

# With streaming progress
python main.py your_presentation.pptx --stream
```

## Usage Scenarios

### Scenario 1: Extract Slide Content Only

If you just need to extract text and images from slides without AI analysis:

```python
from src.slide_processor import SlideProcessor

# Load presentation
processor = SlideProcessor("presentation.pptx")

# Get all slides
slides = processor.process_all_slides()

# Access content
for slide in slides:
    print(f"Slide {slide.slide_number}: {slide.title}")
    print(f"Text: {slide.text_content}")
    print(f"Images: {len(slide.images)}")
    if slide.notes:
        print(f"Notes: {slide.notes}")
```

**Use cases:**
- Content extraction for documentation
- Analyzing presentation structure
- Migrating content to other formats

### Scenario 2: Analyze Slides with Vision AI

Use the vision model to understand slide visuals:

```python
from src.slide_processor import SlideProcessor
from src.vision_analyzer import VisionAnalyzer

processor = SlideProcessor("presentation.pptx")
analyzer = VisionAnalyzer()

slides = processor.process_all_slides()

for slide in slides:
    if slide.images:  # Only analyze slides with images
        analysis = analyzer.analyze_slide_content(slide)
        print(f"Slide {slide.slide_number}:")
        print(f"Visual Analysis: {analysis['visual_analysis']}")
        print(f"Key Points: {analysis['key_points']}")
```

**Use cases:**
- Understanding complex diagrams
- Extracting insights from charts
- Generating slide summaries

### Scenario 3: Generate Narrations

Create audio narrations for slides:

```python
from src.slide_processor import SlideProcessor
from src.vision_analyzer import VisionAnalyzer
from src.tts_engine import TTSEngineFactory
from pathlib import Path

processor = SlideProcessor("presentation.pptx")
analyzer = VisionAnalyzer()
tts = TTSEngineFactory.create_engine()

slides = processor.process_all_slides()

# Generate summaries
summaries = [analyzer.generate_slide_summary(slide) for slide in slides]

# Create audio files
audio_files = tts.generate_speech_batch(
    summaries,
    Path("./output/audio"),
    filename_prefix="slide"
)

print(f"Generated {len(audio_files)} audio files")
```

**Use cases:**
- Creating video presentations
- Accessibility (audio descriptions)
- Study materials

### Scenario 4: Full Automated Workflow

Use the orchestrator for complete end-to-end processing:

```python
from src.orchestrator import SlidesOrchestrator

orchestrator = SlidesOrchestrator()
result = orchestrator.process_presentation("presentation.pptx")

if result.success:
    print(f"✓ Processed {result.total_slides} slides")
    print(f"✓ Generated {len(result.audio_files)} audio files")
    print(f"✓ Output: {result.output_dir}")
    
    # Access results
    for i, (slide, summary, audio) in enumerate(
        zip(result.slides_content, result.summaries, result.audio_files)
    ):
        print(f"\nSlide {i+1}: {slide.title}")
        print(f"Summary: {summary[:100]}...")
        print(f"Audio: {audio.name}")
else:
    print(f"✗ Error: {result.error}")
```

**Use cases:**
- Batch processing presentations
- Creating complete narrated presentations
- Automated content generation

### Scenario 5: Streaming Progress

Monitor processing in real-time:

```python
from src.orchestrator import SlidesOrchestrator

orchestrator = SlidesOrchestrator()

for state in orchestrator.process_presentation_streaming("presentation.pptx"):
    for node_name, node_state in state.items():
        if "current_slide_index" in node_state:
            idx = node_state["current_slide_index"]
            total = node_state.get("total_slides", 0)
            print(f"Processing slide {idx}/{total}...")
        
        if "error" in node_state and node_state["error"]:
            print(f"ERROR: {node_state['error']}")
```

**Use cases:**
- Long-running batch jobs
- User interfaces with progress bars
- Monitoring and logging

## Advanced Usage

### Custom Voice Selection

```python
from src.tts_engine import TTSEngineFactory

tts = TTSEngineFactory.create_engine()

# List available voices
voices = tts.list_available_voices()
en_voices = [v for v in voices if v['Locale'].startswith('en')]

# Print voice options
for voice in en_voices[:5]:
    print(f"{voice['ShortName']}: {voice['Gender']}")

# Change voice
tts.set_voice("en-US-JennyNeural")

# Adjust speech rate and volume
tts.set_rate("+25%")  # 25% faster
tts.set_volume("+10%")  # 10% louder

# Generate speech with new settings
tts.generate_speech("Test message", Path("output/test.mp3"))
```

### Custom Analysis Prompts

```python
from src.vision_analyzer import VisionAnalyzer

analyzer = VisionAnalyzer()

# Use custom prompt for specific analysis
custom_prompt = """
Analyze this slide as if explaining to a technical audience.
Focus on:
1. Technical details in diagrams
2. Code snippets or formulas
3. Architecture or system design elements
Provide a concise technical summary.
"""

result = analyzer.analyze_slide_image(
    image=slide.images[0],
    custom_prompt=custom_prompt
)
print(result)
```

### Batch Processing Multiple Files

```python
from pathlib import Path
from src.orchestrator import SlidesOrchestrator

orchestrator = SlidesOrchestrator()

# Get all .pptx files in a directory
pptx_dir = Path("./presentations")
pptx_files = list(pptx_dir.glob("*.pptx"))

# Process each file
results = []
for pptx_file in pptx_files:
    print(f"Processing {pptx_file.name}...")
    result = orchestrator.process_presentation(str(pptx_file))
    results.append(result)
    
    if result.success:
        print(f"✓ {pptx_file.name}: {result.total_slides} slides")
    else:
        print(f"✗ {pptx_file.name}: {result.error}")

# Summary
successful = sum(1 for r in results if r.success)
print(f"\nProcessed {successful}/{len(results)} files successfully")
```

### Processing Specific Slides

```python
from src.slide_processor import SlideProcessor
from src.vision_analyzer import VisionAnalyzer

processor = SlideProcessor("presentation.pptx")
analyzer = VisionAnalyzer()

# Process only slides 1, 3, and 5
slide_indices = [0, 2, 4]  # 0-based

for idx in slide_indices:
    slide = processor.process_slide(idx)
    summary = analyzer.generate_slide_summary(slide)
    print(f"Slide {slide.slide_number}: {summary}")
```

### Saving Analysis Results

```python
import json
from pathlib import Path
from src.slide_processor import SlideProcessor
from src.vision_analyzer import VisionAnalyzer

processor = SlideProcessor("presentation.pptx")
analyzer = VisionAnalyzer()

slides = processor.process_all_slides()
analyses = [analyzer.analyze_slide_content(slide) for slide in slides]

# Save to JSON
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "analysis.json", "w") as f:
    json.dump(analyses, f, indent=2, default=str)

print(f"Analysis saved to {output_dir / 'analysis.json'}")
```

## Tips and Best Practices

### Performance Tips

1. **Use appropriate image quality**: Set `IMAGE_QUALITY=medium` for faster processing
2. **Batch slides wisely**: Adjust `MAX_SLIDES_PER_BATCH` based on RAM
3. **Use streaming for large presentations**: Monitor progress without blocking

### Quality Tips

1. **Check LM Studio model**: Use Q4 or Q5 quantization for balance
2. **Review generated summaries**: Vision models may miss context
3. **Customize prompts**: Tailor analysis to your specific needs
4. **Test voices**: Different voices work better for different content

### Error Handling

```python
from src.orchestrator import SlidesOrchestrator

orchestrator = SlidesOrchestrator()

try:
    result = orchestrator.process_presentation("presentation.pptx")
    
    if result.success:
        print("Success!")
    else:
        print(f"Processing failed: {result.error}")
        # Handle error appropriately
        
except FileNotFoundError:
    print("File not found!")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from src.orchestrator import SlidesOrchestrator
from pathlib import Path

app = Flask(__name__)
orchestrator = SlidesOrchestrator()

@app.route('/process', methods=['POST'])
def process_presentation():
    file = request.files['file']
    temp_path = Path(f"/tmp/{file.filename}")
    file.save(temp_path)
    
    result = orchestrator.process_presentation(str(temp_path))
    
    return jsonify({
        'success': result.success,
        'total_slides': result.total_slides,
        'audio_files': [str(f) for f in result.audio_files],
        'error': result.error
    })

if __name__ == '__main__':
    app.run(port=5000)
```

### CLI Tool with Click

```python
import click
from pathlib import Path
from src.orchestrator import SlidesOrchestrator

@click.command()
@click.argument('pptx_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def process(pptx_file, output, verbose):
    """Process PowerPoint presentation."""
    orchestrator = SlidesOrchestrator(output_dir=Path(output) if output else None)
    
    if verbose:
        click.echo(f"Processing {pptx_file}...")
    
    result = orchestrator.process_presentation(pptx_file)
    
    if result.success:
        click.echo(f"✓ Success! Generated {len(result.audio_files)} files")
    else:
        click.echo(f"✗ Failed: {result.error}", err=True)

if __name__ == '__main__':
    process()
```

## Troubleshooting Common Issues

See [docs/SETUP.md](docs/SETUP.md) for detailed troubleshooting.

**Quick fixes:**
- LM Studio not connecting → Check server is running on port 1234
- Out of memory → Reduce `IMAGE_QUALITY` and `MAX_SLIDES_PER_BATCH`
- TTS fails → Ensure internet connection for first-time voice download
- Slow processing → Use lower quality settings or smaller model

## Next Steps

1. Explore the [API documentation](docs/API.md) for detailed reference
2. Check out the [examples](examples/) directory for more code samples
3. Customize prompts and settings for your specific use case
4. Consider adding new TTS engines or vision models
