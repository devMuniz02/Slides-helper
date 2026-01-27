# Slides-helper 🎤📊

A local-first, multimodal AI agent that transforms PowerPoint slides into narrated experiences. Built with LangGraph & LM Studio to analyze visuals, generate executive summaries, and provide real-time presentation assistance.

## Features

- 🔍 **Slide Processing**: Extract text, images, and metadata from PowerPoint (.pptx) files
- 👁️ **Vision Analysis**: Analyze slide visuals using local Vision models (Qwen2.5-VL via LM Studio)
- 🗣️ **Text-to-Speech**: Generate natural narrations using edge-tts or other TTS engines
- 🔄 **LangGraph Orchestration**: Modular workflow management for complex processing pipelines
- 💻 **Local-First**: All processing happens on your machine - no cloud dependencies
- ⚡ **Efficient**: Optimized for 32GB RAM and 12GB VRAM systems

## Architecture

```
┌─────────────────┐
│  PowerPoint     │
│  (.pptx file)   │
└────────┬────────┘
         │
         v
┌─────────────────────────────────────────────────────────┐
│              Slide Processor Module                      │
│  • Extract text, images, speaker notes                  │
│  • Parse slide structure and metadata                   │
└────────────────┬────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────┐
│           Vision Analyzer Module                         │
│  • LM Studio Integration (Qwen2.5-VL)                   │
│  • Visual content analysis                              │
│  • Generate descriptions and summaries                  │
└────────────────┬────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────┐
│             TTS Engine Module                            │
│  • edge-tts for speech synthesis                        │
│  • Generate audio narrations                            │
└────────────────┬────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────┐
│          LangGraph Orchestrator                          │
│  • Coordinate workflow                                   │
│  • Manage state and dependencies                        │
│  • Stream progress updates                              │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### Hardware Requirements
- **RAM**: 32GB recommended
- **VRAM**: 12GB recommended for Vision model
- **Storage**: 10GB+ for models and dependencies

### Software Requirements
- **Python**: 3.9 or higher
- **LM Studio**: Running locally with Vision model loaded
  - Download from: https://lmstudio.ai/
  - Load a Vision model (e.g., Qwen2.5-VL)
  - Ensure server is running at `http://localhost:1234/v1`

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/devMuniz02/Slides-helper.git
   cd Slides-helper
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings if needed
   ```

## Usage

### Quick Start

Process a PowerPoint presentation:

```bash
python main.py presentation.pptx
```

With custom output directory:

```bash
python main.py presentation.pptx --output-dir ./my_output
```

Stream progress updates:

```bash
python main.py presentation.pptx --stream
```

### Modular Usage

#### 1. Slide Processor Only

```python
from src.slide_processor import SlideProcessor

# Initialize processor
processor = SlideProcessor("presentation.pptx")

# Get presentation info
info = processor.get_presentation_info()
print(f"Total slides: {info['total_slides']}")

# Process all slides
slides = processor.process_all_slides()

# Process specific slide
slide = processor.process_slide(0)
print(f"Title: {slide.title}")
print(f"Content: {slide.text_content}")
print(f"Images: {len(slide.images)}")
```

#### 2. Vision Analysis

```python
from src.slide_processor import SlideProcessor
from src.vision_analyzer import VisionAnalyzer

processor = SlideProcessor("presentation.pptx")
analyzer = VisionAnalyzer()

slide = processor.process_slide(0)
analysis = analyzer.analyze_slide_content(slide)

print(f"Visual analysis: {analysis['visual_analysis']}")
print(f"Key points: {analysis['key_points']}")
```

#### 3. Text-to-Speech

```python
from src.tts_engine import TTSEngineFactory
from pathlib import Path

tts = TTSEngineFactory.create_engine()

# Generate single audio file
text = "Welcome to slide 1"
output = Path("./output/slide1.mp3")
tts.generate_speech(text, output)

# Generate multiple audio files
texts = ["Slide 1", "Slide 2", "Slide 3"]
audio_files = tts.generate_speech_batch(
    texts,
    Path("./output"),
    filename_prefix="narration"
)
```

#### 4. Full Orchestration

```python
from src.orchestrator import SlidesOrchestrator

orchestrator = SlidesOrchestrator()
result = orchestrator.process_presentation("presentation.pptx")

if result.success:
    print(f"Generated {len(result.audio_files)} audio files")
    print(f"Output: {result.output_dir}")
else:
    print(f"Error: {result.error}")
```

## Examples

The `examples/` directory contains standalone scripts demonstrating each module:

- `basic_processor.py` - Slide extraction and processing
- `vision_analysis.py` - Vision model integration
- `tts_demo.py` - Text-to-speech generation
- `full_workflow.py` - Complete orchestrated workflow

Run any example:
```bash
python examples/basic_processor.py
```

## Configuration

Edit `.env` to customize settings:

```env
# LM Studio Configuration
LM_STUDIO_BASE_URL=http://localhost:1234/v1
VISION_MODEL_NAME=Qwen2.5-VL

# TTS Configuration
TTS_ENGINE=edge-tts
TTS_VOICE=en-US-AriaNeural

# Output Configuration
OUTPUT_DIR=./output
TEMP_DIR=./temp

# Processing Configuration
MAX_SLIDES_PER_BATCH=5
IMAGE_QUALITY=high  # Options: high, medium, low
```

## Module Documentation

### Slide Processor
Located in `src/slide_processor/`

**Key Classes:**
- `SlideProcessor`: Main class for processing PowerPoint files
- `SlideContent`: Data class representing slide content

**Features:**
- Extract text from all shapes (including tables, groups)
- Extract embedded images
- Parse speaker notes
- Retrieve slide metadata

### Vision Analyzer
Located in `src/vision_analyzer/`

**Key Classes:**
- `VisionAnalyzer`: Integrates with LM Studio's vision model

**Features:**
- Analyze slide images using Vision models
- Generate visual descriptions
- Extract key points from content
- Create narration-ready summaries

### TTS Engine
Located in `src/tts_engine/`

**Key Classes:**
- `TTSEngine`: Text-to-speech using edge-tts
- `TTSEngineFactory`: Factory for creating TTS engines

**Features:**
- Convert text to natural-sounding speech
- Multiple voice options
- Batch processing support
- Async operations

### Orchestrator
Located in `src/orchestrator/`

**Key Classes:**
- `SlidesOrchestrator`: LangGraph-based workflow orchestration
- `AgentState`: State management for the workflow
- `ProcessingResult`: Result container

**Features:**
- Coordinate multi-step workflows
- State management
- Error handling
- Progress streaming

## Project Structure

```
Slides-helper/
├── src/
│   ├── slide_processor/    # PowerPoint processing
│   ├── vision_analyzer/    # Vision model integration
│   ├── tts_engine/         # Text-to-speech
│   ├── orchestrator/       # LangGraph workflow
│   └── utils/              # Shared utilities
├── examples/               # Example scripts
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
├── .env.example           # Configuration template
└── README.md              # This file
```

## Troubleshooting

### LM Studio Connection Issues
- Ensure LM Studio is running: Check `http://localhost:1234/v1`
- Verify the Vision model is loaded in LM Studio
- Check firewall settings if connection fails

### Memory Issues
- Reduce `IMAGE_QUALITY` in `.env` (use "medium" or "low")
- Process fewer slides at once by adjusting `MAX_SLIDES_PER_BATCH`
- Ensure no other heavy applications are running

### TTS Issues
- edge-tts requires internet for first-time voice downloads
- Check available voices with: `edge-tts --list-voices`
- Try different voices if one doesn't work

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See LICENSE file for details.

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for orchestration
- [LM Studio](https://lmstudio.ai/) for local LLM serving
- [edge-tts](https://github.com/rany2/edge-tts) for text-to-speech
- [python-pptx](https://python-pptx.readthedocs.io/) for PowerPoint processing
