[![LinkedIn](https://img.shields.io/badge/LinkedIn-devmuniz-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/devmuniz)
[![GitHub Profile](https://img.shields.io/badge/GitHub-devMuniz02-181717?logo=github&logoColor=white)](https://github.com/devMuniz02)
[![Portfolio](https://img.shields.io/badge/Portfolio-devmuniz02.github.io-0F172A?logo=googlechrome&logoColor=white)](https://devmuniz02.github.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-manu02-FFD21E?logoColor=black)](https://huggingface.co/manu02)

# Slides-helper 🎤📊

A local-first, multimodal AI agent that transforms PowerPoint slides into narrated experiences. Built with LangGraph & LM Studio to analyze visuals, generate executive summaries, and provide real-time presentation assistance.

- 🔍 **Slide Processing**: Extract text, images, and metadata from PowerPoint (.pptx) files - 👁️ **Vision Analysis**: Analyze slide visuals using local Vision models (Qwen2.5-VL via LM Studio) - 🗣️ **Text-to-Speech**: Generate natural narrations using edge-tts or other TTS engines - 🔄 **LangGraph Orchestration**: Modular workflow management for complex processing pipelines - 💻 **Local-First**: All processing happens on your machine - no cloud dependencies - ⚡ **Efficient**: Optimized for 32GB RAM and 12GB VRAM systems - 🖥️ **PowerPoint Integration**: Real-time connection to active PowerPoint presentations - 🎛️ **Graphical Interface**: Modern GUI for live slide monitoring and analysis - 📊 **Live Monitoring**: Automatically tracks current slide changes during presentations - 🎤 **TTS with Progressive Subtitles**: Generate summaries and speak them with synchronized, sentence-by-sentence subtitle overlay

## Overview

A local-first, multimodal AI agent that transforms slides into narrated experiences. Built with LangGraph & LM Studio to analyze visuals, generate executive summaries, and provide real-time presentation assistance.

## Repository Structure

| Path | Description |
| --- | --- |
| `assets/` | Images, figures, or other supporting media used by the project. |
| `config/` | Top-level project directory containing repository-specific resources. |
| `docs/` | Project documentation and supporting written material. |
| `src/` | Primary source code for the application or library. |
| `templates/` | Top-level project directory containing repository-specific resources. |
| `.gitignore` | Top-level file included in the repository. |
| `gui_launcher.py` | Top-level file included in the repository. |
| `LICENSE` | Repository license information. |
| `main.py` | Top-level file included in the repository. |
| `rag_system.py` | Top-level file included in the repository. |

## Getting Started

1. Clone the repository.

   ```bash
   git clone https://github.com/devMuniz02/Slides-helper.git
   cd Slides-helper
   ```

2. Prepare the local environment.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run or inspect the project entry point.

Run the main Python entry point:
```bash
python main.py
```

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

## Usage

### Command Line Interface

Process a PowerPoint file:
```bash
python main.py <path-to-pptx-file>
```

With options:
```bash
python main.py <path-to-pptx-file> --output-dir ./my_output --stream
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

### RAG (Retrieval-Augmented Generation) System

The RAG system combines the power of information retrieval with AI language generation to provide intelligent question-answering over your presentations. Instead of just searching for keywords, it understands the semantic meaning of your questions, retrieves the most relevant content from your slides (including both text and visual information), and generates natural, contextual answers backed by source citations.

**How it works:**
1. **Ingestion**: Your PowerPoint presentations are processed, with text extracted and images analyzed by vision models
2. **Embedding**: Content is converted into semantic vectors and stored in a local ChromaDB database
3. **Retrieval**: When you ask a question, the system finds the most relevant slides and content
4. **Generation**: A local language model synthesizes a comprehensive answer using the retrieved context
5. **Citation**: Every answer includes references to the specific slides where the information was found

Ask questions about your PowerPoint presentations using natural language:

**Process presentations for Q&A:**
```bash
python rag_system.py --pptx <path-to-pptx-file>
```

**Start the interactive chat interface:**
```bash
python rag_system.py --chat
```

**Or launch from main script:**
```bash
python main.py --rag
```

**RAG Features:**
- 🔍 **Intelligent Search**: Find relevant information across all slides
- 🖼️ **Multimodal Understanding**: Search through both text and image content
- 📍 **Source Citations**: Always shows which presentation and slide the information came from
- 💬 **Interactive Chat**: Web-based interface with real-time responses
- 🔄 **Async Processing**: Fluid responses with progressive token streaming
- 🗄️ **Vector Database**: Persistent storage using ChromaDB
- 🎯 **Contextual Answers**: AI-generated responses based on retrieved content

**RAG Configuration:**
The RAG system uses `config/rag_config.json` for model configuration. Copy and modify this file to customize model settings:

```json
{
  "models": {
    "vision": "qwen/qwen2.5-vl-7b",
    "text_generation": "qwen2.5-7b-instruct",
    "embedding": "local"
  },
  "lm_studio": {
    "base_url": "http://localhost:1234/v1",
    "vision_model_name": "qwen/qwen2.5-vl-7b",
    "model_name": "qwen2.5-7b-instruct",
    "embedding_model_name": "text-embedding-ada-002"
  }
}
```

See `docs/RAG_CONFIG_README.md` for detailed configuration options.

**Example queries:**
- "What are the main benefits of our product?"
- "Show me the quarterly sales figures"
- "Explain the technical architecture diagram"
- "What were the key discussion points from last month's meeting?"

## Testing with Local Files

**Note:** This repository does not include example PowerPoint files due to size constraints.

To test the system:

1. **Place your .pptx files** in the root directory of the project
2. **Use the GUI** to select files from your local system
3. **Or use the command line** with paths to your presentations

Example:
```bash
# Place your presentation in the root directory
cp <path-to-your-pptx-file> .

# Then process it
python main.py <pptx-file>

# Or use the GUI to browse and select files
python gui_launcher.py
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Prerequisites

- **Python 3.8+**
- **LM Studio** running locally with a vision-capable model (e.g., Qwen2.5-VL)
- **PowerPoint** (for GUI integration features)
- **Windows** (required for PowerPoint COM automation)

## Project Structure

```
slides-helper/
├── config/                 # Configuration files
│   └── rag_config.json     # RAG system model configuration
├── docs/                   # Documentation
│   └── RAG_CONFIG_README.md # RAG configuration guide
├── output/                 # Output files and results
├── rag_db/                 # RAG database and extracted data
│   ├── chroma_db/          # Vector database storage
│   └── *_extracted.json    # Extracted slide data
├── src/                    # Source code
│   ├── gui/                # Graphical user interface
│   ├── orchestrator/       # Main processing orchestration
│   ├── powerpoint_connector/ # PowerPoint integration
│   ├── rag_system/         # RAG system components
│   ├── slide_processor/    # Slide text/image extraction
│   ├── tts_engine/         # Text-to-speech functionality
│   ├── utils/              # Utility functions and configuration
│   └── vision_analyzer/    # Image analysis with vision models
├── temp/                   # Temporary files
├── templates/              # HTML templates for web interface
│   ├── chat.html
│   └── workflow.html
├── uploads/                # Uploaded files
├── gui_launcher.py         # GUI application entry point
├── list_files.py           # File listing utility
├── main.py                 # CLI application entry point
├── rag_system.py           # RAG system entry point
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── LICENSE                 # License information
```
