# RAG Configuration Guide

This document explains the configuration options for the RAG (Retrieval-Augmented Generation) system in Slides-helper.

## Configuration File: `config/rag_config.json`

The RAG system uses a JSON configuration file located at `config/rag_config.json`. This file contains all settings for the RAG functionality.

## Configuration Sections

### Models Configuration

```json
{
  "models": {
    "vision": "qwen/qwen2.5-vl-7b",
    "text_generation": "qwen2.5-7b-instruct",
    "embedding": "local"
  }
}
```

- **vision**: Model used for analyzing slide images
- **text_generation**: Model used for generating responses to user queries
- **embedding**: Embedding model for vectorizing text content

### LM Studio Configuration

```json
{
  "lm_studio": {
    "base_url": "http://localhost:1234",
    "vision_model_name": "qwen/qwen2.5-vl-7b",
    "model_name": "qwen2.5-7b-instruct",
    "embedding_model_name": "text-embedding-ada-002"
  }
}
```

- **base_url**: URL where LM Studio is running
- **vision_model_name**: Name of the vision model loaded in LM Studio
- **model_name**: Name of the text generation model
- **embedding_model_name**: Name of the embedding model

### Processing Configuration

```json
{
  "processing": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_slides_per_batch": 5
  }
}
```

- **chunk_size**: Maximum characters per text chunk for embedding
- **chunk_overlap**: Characters to overlap between chunks
- **max_slides_per_batch**: Maximum slides to process in a single batch

### Database Configuration

```json
{
  "database": {
    "collection_name": "slides_collection",
    "persist_directory": "./rag_db"
  }
}
```

- **collection_name**: Name of the ChromaDB collection
- **persist_directory**: Directory where the vector database is stored

### Web Interface Configuration

```json
{
  "web_interface": {
    "host": "localhost",
    "port": 8000,
    "reload": true
  }
}
```

- **host**: Host address for the web interface
- **port**: Port number for the web interface
- **reload**: Whether to enable auto-reload during development

## Usage Examples

### Basic RAG Query
```bash
python rag_system.py --chat
```

### Process Presentations for Q&A
```bash
python rag_system.py --pptx presentation.pptx
```

### Start Web Interface
```bash
python rag_system.py --web
```

## Model Recommendations

### Vision Models (via LM Studio)
- **Qwen2.5-VL**: Excellent for slide image analysis
- **LLaVA**: Good alternative for vision tasks
- **GPT-4V**: If using OpenAI API instead

### Text Generation Models
- **Qwen2.5-7B-Instruct**: Balanced performance and speed
- **Llama-3-8B-Instruct**: Good alternative
- **GPT-4**: For highest quality responses

### Embedding Models
- **text-embedding-ada-002**: OpenAI's embedding model
- **sentence-transformers**: Local embedding models
- **all-MiniLM-L6-v2**: Fast local embeddings

## Troubleshooting

### Common Issues

1. **Connection Failed**: Ensure LM Studio is running and models are loaded
2. **Empty Results**: Check that presentations were properly processed
3. **Slow Responses**: Consider using smaller models or reducing chunk size
4. **Memory Issues**: Reduce `max_slides_per_batch` or use smaller models

### Performance Tuning

- **For Speed**: Use smaller models and increase chunk_size
- **For Accuracy**: Use larger models and decrease chunk_size
- **For Memory**: Reduce batch sizes and use CPU embeddings

## Advanced Configuration

### Custom Embedding Models

To use local sentence transformers:

```json
{
  "models": {
    "embedding": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

### Multiple Model Support

The system supports switching between different model configurations for different tasks.