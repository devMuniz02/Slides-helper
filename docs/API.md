# API Documentation

## Core Modules

### SlideProcessor

Extract content from PowerPoint presentations.

#### Class: `SlideProcessor`

**Initialization:**
```python
SlideProcessor(pptx_path: str)
```

**Parameters:**
- `pptx_path` (str): Path to the PowerPoint file

**Methods:**

##### `process_slide(slide_index: int) -> SlideContent`

Process a single slide and extract all content.

**Parameters:**
- `slide_index` (int): Zero-based index of the slide

**Returns:**
- `SlideContent`: Object containing slide information

**Example:**
```python
processor = SlideProcessor("presentation.pptx")
slide = processor.process_slide(0)  # First slide
print(f"Title: {slide.title}")
```

##### `process_all_slides() -> List[SlideContent]`

Process all slides in the presentation.

**Returns:**
- `List[SlideContent]`: List of all slides

**Example:**
```python
slides = processor.process_all_slides()
for slide in slides:
    print(f"Slide {slide.slide_number}: {slide.title}")
```

##### `get_presentation_info() -> Dict[str, Any]`

Get metadata about the presentation.

**Returns:**
- Dictionary with keys: `file_path`, `file_name`, `total_slides`, `has_notes`

**Example:**
```python
info = processor.get_presentation_info()
print(f"Total slides: {info['total_slides']}")
```

#### Class: `SlideContent`

Dataclass representing slide content.

**Attributes:**
- `slide_number` (int): 1-based slide number
- `title` (Optional[str]): Slide title
- `text_content` (List[str]): Text content from slide
- `images` (List[Image.Image]): PIL Image objects
- `notes` (Optional[str]): Speaker notes
- `metadata` (Dict[str, Any]): Additional metadata

**Methods:**

##### `get_full_text() -> str`

Get all text content as a single string.

**Returns:**
- Formatted string with title, content, and notes

---

### VisionAnalyzer

Analyze slide visuals using LM Studio's vision model.

#### Class: `VisionAnalyzer`

**Initialization:**
```python
VisionAnalyzer()
```

Uses configuration from `config.lm_studio` (base_url, model_name).

**Methods:**

##### `analyze_slide_image(image: Image.Image, slide_text: Optional[str] = None, custom_prompt: Optional[str] = None) -> str`

Analyze a single slide image.

**Parameters:**
- `image` (Image.Image): PIL Image object
- `slide_text` (Optional[str]): Text content from slide
- `custom_prompt` (Optional[str]): Override default prompt

**Returns:**
- Analysis result as string

**Example:**
```python
analyzer = VisionAnalyzer()
result = analyzer.analyze_slide_image(
    image=slide.images[0],
    slide_text=slide.get_full_text()
)
print(result)
```

##### `analyze_slide_content(slide_content: SlideContent) -> Dict[str, Any]`

Analyze complete slide including text and images.

**Parameters:**
- `slide_content` (SlideContent): Slide to analyze

**Returns:**
- Dictionary with keys:
  - `slide_number` (int)
  - `title` (str)
  - `text_summary` (str)
  - `visual_analysis` (List[Dict])
  - `key_points` (List[str])

**Example:**
```python
analysis = analyzer.analyze_slide_content(slide)
for visual in analysis['visual_analysis']:
    print(visual['analysis'])
```

##### `generate_slide_summary(slide_content: SlideContent) -> str`

Generate a comprehensive summary for narration.

**Parameters:**
- `slide_content` (SlideContent): Slide to summarize

**Returns:**
- Summary text suitable for TTS

**Example:**
```python
summary = analyzer.generate_slide_summary(slide)
print(summary)
```

---

### TTSEngine

Convert text to speech using edge-tts.

#### Class: `TTSEngine`

**Initialization:**
```python
TTSEngine(voice: Optional[str] = None)
```

**Parameters:**
- `voice` (Optional[str]): Voice name (defaults to config)

**Methods:**

##### `generate_speech(text: str, output_path: Path) -> Path`

Generate speech from text.

**Parameters:**
- `text` (str): Text to convert
- `output_path` (Path): Where to save audio file

**Returns:**
- Path to saved audio file

**Example:**
```python
from pathlib import Path
tts = TTSEngine()
audio_path = tts.generate_speech(
    "Hello world",
    Path("output/hello.mp3")
)
```

##### `generate_speech_batch(texts: List[str], output_dir: Path, filename_prefix: str = "narration") -> List[Path]`

Generate speech for multiple texts.

**Parameters:**
- `texts` (List[str]): List of texts
- `output_dir` (Path): Output directory
- `filename_prefix` (str): Prefix for filenames

**Returns:**
- List of paths to generated files

**Example:**
```python
texts = ["Slide 1", "Slide 2", "Slide 3"]
audio_files = tts.generate_speech_batch(
    texts,
    Path("output/audio"),
    filename_prefix="slide"
)
```

##### `list_available_voices() -> List[dict]`

List all available voices.

**Returns:**
- List of voice dictionaries with metadata

**Example:**
```python
voices = tts.list_available_voices()
en_voices = [v for v in voices if v['Locale'].startswith('en')]
```

##### `set_voice(voice: str)`

Change the voice.

**Parameters:**
- `voice` (str): Voice name

**Example:**
```python
tts.set_voice("en-US-JennyNeural")
```

##### `set_rate(rate: str)`

Set speech rate.

**Parameters:**
- `rate` (str): Rate modifier (e.g., "+0%", "+50%", "-25%")

**Example:**
```python
tts.set_rate("+25%")  # 25% faster
```

##### `set_volume(volume: str)`

Set speech volume.

**Parameters:**
- `volume` (str): Volume modifier (e.g., "+0%", "+50%")

**Example:**
```python
tts.set_volume("+10%")  # 10% louder
```

#### Class: `TTSEngineFactory`

Factory for creating TTS engines.

##### `create_engine(engine_type: Optional[str] = None) -> TTSEngine`

Create TTS engine instance.

**Parameters:**
- `engine_type` (Optional[str]): Engine type (defaults to config)

**Returns:**
- TTSEngine instance

**Example:**
```python
tts = TTSEngineFactory.create_engine("edge-tts")
```

---

### SlidesOrchestrator

Orchestrate the complete workflow using LangGraph.

#### Class: `SlidesOrchestrator`

**Initialization:**
```python
SlidesOrchestrator(output_dir: Optional[Path] = None)
```

**Parameters:**
- `output_dir` (Optional[Path]): Output directory (defaults to config)

**Methods:**

##### `process_presentation(pptx_path: str) -> ProcessingResult`

Process a presentation end-to-end.

**Parameters:**
- `pptx_path` (str): Path to PowerPoint file

**Returns:**
- `ProcessingResult` object

**Example:**
```python
orchestrator = SlidesOrchestrator()
result = orchestrator.process_presentation("presentation.pptx")

if result.success:
    print(f"Generated {len(result.audio_files)} audio files")
    print(f"Output: {result.output_dir}")
else:
    print(f"Error: {result.error}")
```

##### `process_presentation_streaming(pptx_path: str)`

Process with streaming progress updates.

**Parameters:**
- `pptx_path` (str): Path to PowerPoint file

**Yields:**
- State updates during processing

**Example:**
```python
for state in orchestrator.process_presentation_streaming("presentation.pptx"):
    for node_name, node_state in state.items():
        if "current_slide_index" in node_state:
            print(f"Processing slide {node_state['current_slide_index']}...")
```

#### Class: `ProcessingResult`

Result of presentation processing.

**Attributes:**
- `pptx_path` (Path): Input file path
- `total_slides` (int): Number of slides
- `slides_content` (List[SlideContent]): All slides
- `analyses` (List[Dict]): Analysis results
- `summaries` (List[str]): Generated summaries
- `audio_files` (List[Path]): Generated audio files
- `output_dir` (Path): Output directory
- `success` (bool): Whether processing succeeded
- `error` (Optional[str]): Error message if failed

---

### Configuration

#### Class: `Config`

Main configuration class.

**Attributes:**
- `lm_studio` (LMStudioConfig): LM Studio settings
- `tts` (TTSConfig): TTS settings
- `processing` (ProcessingConfig): Processing settings

**Global Instance:**
```python
from src.utils import config

# Access configuration
print(config.lm_studio.base_url)
print(config.tts.voice)
```

#### Class: `LMStudioConfig`

**Attributes:**
- `base_url` (str): LM Studio server URL
- `vision_model_name` (str): Vision model name
- `api_key` (str): API key (not needed for LM Studio)
- `timeout` (int): Request timeout in seconds

#### Class: `TTSConfig`

**Attributes:**
- `engine` (str): TTS engine name
- `voice` (str): Voice name
- `rate` (str): Speech rate
- `volume` (str): Speech volume

#### Class: `ProcessingConfig`

**Attributes:**
- `max_slides_per_batch` (int): Slides per batch
- `image_quality` (str): "high", "medium", or "low"
- `output_dir` (Path): Output directory
- `temp_dir` (Path): Temporary directory

---

## Utility Functions

### Image Utilities

Located in `src.utils.image_utils`

##### `image_to_base64(image: Union[Image.Image, Path, str]) -> str`

Convert image to base64 string.

**Parameters:**
- `image`: PIL Image or path to image

**Returns:**
- Base64 encoded string

**Example:**
```python
from src.utils.image_utils import image_to_base64
from PIL import Image

img = Image.open("slide.png")
b64_str = image_to_base64(img)
```

##### `resize_image(image: Image.Image, max_size: int = 1920) -> Image.Image`

Resize image maintaining aspect ratio.

**Parameters:**
- `image`: PIL Image object
- `max_size`: Maximum dimension

**Returns:**
- Resized PIL Image

**Example:**
```python
from src.utils.image_utils import resize_image

resized = resize_image(img, max_size=1280)
```

---

## Type Definitions

### AgentState (TypedDict)

State for LangGraph workflow.

**Fields:**
- `pptx_path` (str): Input file path
- `slides` (List[SlideContent]): Processed slides
- `analyses` (List[Dict[str, Any]]): Analysis results
- `summaries` (List[str]): Generated summaries
- `audio_files` (List[Path]): Audio file paths
- `current_slide_index` (int): Current processing index
- `total_slides` (int): Total slide count
- `error` (Optional[str]): Error message if any
