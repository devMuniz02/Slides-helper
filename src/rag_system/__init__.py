"""
RAG System Module

This module implements the Retrieval-Augmented Generation system for PowerPoint presentations.
It allows users to ask questions about their slides and get AI-powered answers based on
the content of their presentations.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ..slide_processor.processor import SlideContent
from ..utils.config import config


class RAGSystem:
    """
    Retrieval-Augmented Generation system for PowerPoint presentations.

    This class provides functionality to:
    - Process and store presentation content in a vector database
    - Retrieve relevant information based on user queries
    - Generate AI-powered responses using retrieved context
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG system.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._load_config()
        self.client = OpenAI(
            base_url=self.config.get("lm_studio", {}).get("base_url", "http://localhost:1234/v1"),
            api_key="not-needed"  # LM Studio doesn't require API key
        )

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.get("database", {}).get("persist_directory", "./rag_db")
        )
        self.collection_name = self.config.get("database", {}).get("collection_name", "slides_collection")

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
        except ValueError:
            self.collection = self.chroma_client.create_collection(name=self.collection_name)

        # Initialize embedding model
        self.embedding_model = self._init_embedding_model()

        # Model configurations
        self.vision_model = self.config.get("lm_studio", {}).get("vision_model_name", "qwen/qwen2.5-vl-7b")
        self.text_model = self.config.get("lm_studio", {}).get("model_name", "qwen2.5-7b-instruct")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path("config/rag_config.json")
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")

        # Return default configuration
        return {
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
            },
            "processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_slides_per_batch": 5
            },
            "database": {
                "collection_name": "slides_collection",
                "persist_directory": "./rag_db"
            }
        }

    def _init_embedding_model(self):
        """Initialize the embedding model."""
        embedding_config = self.config.get("models", {}).get("embedding", "local")

        if embedding_config == "local":
            try:
                return SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load local embedding model: {e}")
                return None
        else:
            # Use OpenAI embeddings
            return None

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        if self.embedding_model:
            # Use local embeddings
            return self.embedding_model.encode(texts, convert_to_numpy=True).tolist()
        else:
            # Use OpenAI embeddings (fallback)
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.config.get("lm_studio", {}).get("embedding_model_name", "text-embedding-ada-002")
                )
                return [embedding.embedding for embedding in response.data]
            except Exception as e:
                print(f"Error getting embeddings: {e}")
                return []

    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into chunks with overlap."""
        if chunk_size is None:
            chunk_size = self.config.get("processing", {}).get("chunk_size", 1000)
        if overlap is None:
            overlap = self.config.get("processing", {}).get("chunk_overlap", 200)

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If we're not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                last_newline = text.rfind('\n', end - 100, end)

                # Use the latest sentence ending found
                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap

        return chunks

    def add_presentation(self, slides_content: List[SlideContent], presentation_name: str):
        """
        Add a presentation to the RAG system.

        Args:
            slides_content: List of SlideContent objects
            presentation_name: Name of the presentation
        """
        documents = []
        metadatas = []
        ids = []

        for i, slide in enumerate(slides_content):
            slide_id = f"{presentation_name}_slide_{i+1}"

            # Add text content
            if slide.text_content:
                chunks = self._chunk_text(slide.text_content)
                for j, chunk in enumerate(chunks):
                    chunk_id = f"{slide_id}_text_{j}"
                    documents.append(chunk)
                    metadatas.append({
                        "presentation": presentation_name,
                        "slide_number": i + 1,
                        "content_type": "text",
                        "chunk_index": j,
                        "total_chunks": len(chunks)
                    })
                    ids.append(chunk_id)

            # Add image descriptions if available
            if slide.image_descriptions:
                for j, description in enumerate(slide.image_descriptions):
                    image_id = f"{slide_id}_image_{j}"
                    documents.append(description)
                    metadatas.append({
                        "presentation": presentation_name,
                        "slide_number": i + 1,
                        "content_type": "image",
                        "image_index": j
                    })
                    ids.append(image_id)

        if documents:
            # Get embeddings for all documents
            embeddings = self._get_embeddings(documents)

            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

            print(f"Added {len(documents)} content pieces from {presentation_name}")

    def _retrieve_relevant_content(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant content for a query.

        Args:
            query: User query
            n_results: Number of results to retrieve

        Returns:
            Dictionary with retrieved content and metadata
        """
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Search the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        return results

    def _generate_response(self, query: str, context: List[str], metadata: List[Dict]) -> str:
        """
        Generate a response using the retrieved context.

        Args:
            query: User query
            context: Retrieved context documents
            metadata: Metadata for the retrieved documents

        Returns:
            Generated response
        """
        # Prepare context string
        context_str = "\n\n".join([
            f"From {meta.get('presentation', 'Unknown')} - Slide {meta.get('slide_number', 'Unknown')}:\n{doc}"
            for doc, meta in zip(context, metadata)
        ])

        # Create prompt
        prompt = f"""You are an AI assistant helping users with questions about their PowerPoint presentations.

Based on the following content from the user's presentations, please answer their question.
If the information is not available in the provided context, say so clearly.

Context from presentations:
{context_str}

User question: {query}

Please provide a helpful, accurate answer based on the presentation content:"""

        try:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about PowerPoint presentations based on their content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def query_async(self, query: str, n_results: int = 5) -> str:
        """
        Query the RAG system asynchronously.

        Args:
            query: User query
            n_results: Number of results to retrieve

        Returns:
            Generated response
        """
        try:
            # Retrieve relevant content
            results = self._retrieve_relevant_content(query, n_results)

            if not results['documents'] or not results['documents'][0]:
                return "I couldn't find any relevant information in your presentations. Please make sure you've processed some presentations first using the --pptx option."

            # Generate response
            response = self._generate_response(
                query,
                results['documents'][0],
                results['metadatas'][0]
            )

            return response

        except Exception as e:
            return f"Error processing query: {str(e)}"

    def query(self, query: str, n_results: int = 5) -> str:
        """
        Query the RAG system (synchronous wrapper).

        Args:
            query: User query
            n_results: Number of results to retrieve

        Returns:
            Generated response
        """
        return asyncio.run(self.query_async(query, n_results))

    def reset_database(self):
        """Reset the vector database by deleting and recreating the collection."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # Collection might not exist

        # Recreate collection
        self.collection = self.chroma_client.create_collection(name=self.collection_name)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {"error": str(e)}

    def list_presentations(self) -> List[str]:
        """List all presentations in the system."""
        try:
            results = self.collection.get(include=['metadatas'])
            presentations = set()

            for metadata in results['metadatas']:
                if metadata.get('presentation'):
                    presentations.add(metadata['presentation'])

            return sorted(list(presentations))

        except Exception as e:
            print(f"Error listing presentations: {e}")
            return []