# Slides-helper 🎤📊

A local-first, multimodal AI agent that transforms PowerPoint slides into narrated experiences. Built with LangGraph & LM Studio to analyze visuals, generate executive summaries, and provide real-time presentation assistance.

## Features

- 🔍 **Slide Processing**: Extract text, images, and metadata from PowerPoint (.pptx) files
- 👁️ **Vision Analysis**: Analyze slide visuals using local Vision models (Qwen2.5-VL via LM Studio)
- 🗣️ **Text-to-Speech**: Generate natural narrations using edge-tts or other TTS engines
- 🔄 **LangGraph Orchestration**: Modular workflow management for complex processing pipelines
- 💻 **Local-First**: All processing happens on your machine - no cloud dependencies
- ⚡ **Efficient**: Optimized for 32GB RAM and 12GB VRAM systems
- 🖥️ **PowerPoint Integration**: Real-time connection to active PowerPoint presentations
- 🎛️ **Graphical Interface**: Modern GUI for live slide monitoring and analysis
- 📊 **Live Monitoring**: Automatically tracks current slide changes during presentations
- 🎤 **TTS with Progressive Subtitles**: Generate summaries and speak them with synchronized, sentence-by-sentence subtitle overlay

## Prerequisites

- **Python 3.8+**
- **LM Studio** running locally with a vision-capable model (e.g., Qwen2.5-VL)
- **PowerPoint** (for GUI integration features)
- **Windows** (required for PowerPoint COM automation)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd slides-helper
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up LM Studio:**
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Download a vision model (e.g., Qwen2.5-VL)
   - Start LM Studio and load the model
   - Ensure LM Studio is running on `http://localhost:1234` (default)

## Configuration

Create a `.env` file in the project root (optional, defaults will work):

```env
# LM Studio Configuration
LM_STUDIO_BASE_URL=http://localhost:1234/v1
VISION_MODEL_NAME=Qwen2.5-VL

# TTS Configuration
TTS_ENGINE=edge-tts
TTS_VOICE=en-US-AriaNeural

# Processing Configuration
MAX_SLIDES_PER_BATCH=5
IMAGE_QUALITY=high
OUTPUT_DIR=./output
TEMP_DIR=./temp
```

## Usage

### Command Line Interface

Process a PowerPoint file:
```bash
python main.py path/to/your/presentation.pptx
```

With options:
```bash
python main.py path/to/presentation.pptx --output-dir ./my_output --stream
```

Available options:
- `--output-dir`: Specify output directory (default: ./output)
- `--stream`: Enable streaming progress updates
- `--gui`: Launch graphical interface instead

### Graphical User Interface

Launch the GUI for real-time PowerPoint integration:
```bash
python gui_launcher.py
```

Or from main script:
```bash
python main.py --gui
```

The GUI provides:
- Live slide monitoring during presentations
- Real-time analysis and narration
- Visual feedback and controls

## Architecture

```
┌─────────────────┐    ┌──────────────────────┐
│  PowerPoint     │    │   PowerPoint GUI     │
│  (.pptx file)   │    │   (Live Integration) │
└────────┬────────┘    └──────────┬───────────┘
         │                       │
         └───────────────────────┼───────────────────────┐
                                 v                       v
┌─────────────────────────────────────────────────────────┐
│              Slide Processor Module                      │
│  • Extract text, images, speaker notes                  │
│  • Parse slide structure and metadata                   │
│  • Real-time slide extraction from active presentations │
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
│            Orchestrator Module                           │
│  • LangGraph workflow management                        │
│  • Coordinate all processing steps                      │
│  • Handle errors and state management                   │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
slides-helper/
├── main.py                 # Command-line interface
├── gui_launcher.py         # GUI launcher script
├── requirements.txt        # Python dependencies
└── src/
    ├── gui/                # Graphical user interface
    ├── orchestrator/       # LangGraph orchestration
    ├── powerpoint_connector/ # PowerPoint integration
    ├── slide_processor/    # Slide content extraction
    ├── tts_engine/         # Text-to-speech synthesis
    ├── utils/              # Configuration and utilities
    └── vision_analyzer/    # AI vision analysis
```

## Troubleshooting

### Common Issues

1. **LM Studio Connection Failed**
   - Ensure LM Studio is running and accessible at `http://localhost:1234`
   - Check that the vision model is loaded and active

2. **PowerPoint Integration Issues**
   - Ensure PowerPoint is installed and running
   - Run as administrator if COM automation fails

3. **TTS Engine Problems**
   - Verify internet connection for edge-tts
   - Check voice availability with `edge-tts --list-voices`

### Performance Optimization

- Use models optimized for your hardware (12GB VRAM recommended)
- Process fewer slides per batch for lower-end systems
- Enable streaming mode for progress feedback

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.</content>
<parameter name="filePath">c:\Users\emman\Desktop\PROYECTOS_VS_CODE\PRUEBAS_DE_PYTHON\Slides-helper\README.md
