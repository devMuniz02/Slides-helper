"""
Vision Analyzer Module

This module handles image analysis using LM Studio's vision model API.
"""
import base64
from io import BytesIO
from typing import List, Optional, Dict, Any

from openai import OpenAI
from PIL import Image

from ..utils.config import config
from ..utils.image_utils import image_to_base64
from ..slide_processor.processor import SlideContent


class VisionAnalyzer:
    """
    Analyzes slide images using a vision model via LM Studio.
    
    This class provides functionality to:
    - Analyze slide visuals using a vision model
    - Generate descriptions of slide content
    - Extract key information from slide images
    """
    
    def __init__(self):
        """Initialize the vision analyzer with LM Studio connection."""
        self.client = OpenAI(
            base_url=config.lm_studio.base_url,
            api_key=config.lm_studio.api_key,
        )
        self.model_name = config.lm_studio.vision_model_name
    
    def analyze_slide_image(
        self,
        image: Image.Image,
        slide_text: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """
        Analyze a slide image and generate a description.
        
        Args:
            image: PIL Image object of the slide
            slide_text: Optional text content from the slide
            custom_prompt: Custom prompt to use instead of default
            
        Returns:
            Analysis result as a string
        """
        # Convert image to base64
        img_base64 = image_to_base64(image)
        
        # Build the prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_analysis_prompt(slide_text)
        
        # Create messages for the vision model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        },
                    },
                ],
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def analyze_slide_content(self, slide_content: SlideContent) -> Dict[str, Any]:
        """
        Analyze a complete slide including text and images.
        
        Args:
            slide_content: SlideContent object containing all slide information
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            "slide_number": slide_content.slide_number,
            "title": slide_content.title,
            "text_summary": slide_content.get_full_text(),
            "visual_analysis": [],
            "key_points": [],
        }
        
        # Analyze each image in the slide
        for idx, image in enumerate(slide_content.images):
            analysis = self.analyze_slide_image(
                image,
                slide_text=slide_content.get_full_text(),
            )
            results["visual_analysis"].append({
                "image_index": idx,
                "analysis": analysis,
            })
        
        # Generate key points if no images (text-only slide)
        if not slide_content.images:
            key_points = self._extract_key_points(slide_content.get_full_text())
            results["key_points"] = key_points
        
        return results
    
    def generate_slide_summary(self, slide_content: SlideContent) -> str:
        """
        Generate a comprehensive summary of a slide for narration.
        
        Args:
            slide_content: SlideContent object
            
        Returns:
            Summary text suitable for TTS narration
        """
        analysis = self.analyze_slide_content(slide_content)
        
        # Build the summary
        summary_parts = []
        
        # Add slide number
        summary_parts.append(f"Slide {slide_content.slide_number}.")
        
        # Add title if present
        if slide_content.title:
            summary_parts.append(f"Title: {slide_content.title}.")
        
        # Add visual analysis if images present
        if analysis["visual_analysis"]:
            summary_parts.append("Visual content:")
            for visual in analysis["visual_analysis"]:
                summary_parts.append(visual["analysis"])
        
        # Add text content summary
        if slide_content.text_content:
            summary_parts.append("Key points:")
            for point in slide_content.text_content[:5]:  # Limit to first 5 points
                summary_parts.append(f"- {point}")
        
        # Add notes if present
        if slide_content.notes:
            summary_parts.append(f"Speaker notes: {slide_content.notes}")
        
        return " ".join(summary_parts)
    
    def _build_analysis_prompt(self, slide_text: Optional[str] = None) -> str:
        """
        Build a prompt for slide image analysis.
        
        Args:
            slide_text: Optional text content from the slide
            
        Returns:
            Formatted prompt string
        """
        prompt = """Analyze this presentation slide image. Describe:
1. The main visual elements (charts, diagrams, images, icons)
2. The layout and design approach
3. Key information conveyed visually
4. How the visuals support the message

Keep the description concise and focused on what's important for understanding the slide."""
        
        if slide_text:
            prompt += f"\n\nThe slide contains this text:\n{slide_text}\n\nDescribe how the visuals complement this text."
        
        return prompt
    
    def _extract_key_points(self, text: str) -> List[str]:
        """
        Extract key points from text using the language model.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of key points
        """
        if not text.strip():
            return []
        
        prompt = f"""Extract the 3-5 most important key points from this slide text.
Return only the key points as a bulleted list.

Text:
{text}

Key points:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.5,
            )
            
            content = response.choices[0].message.content
            # Parse bullet points
            points = [
                line.strip().lstrip('-•* ').strip()
                for line in content.split('\n')
                if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•') or line.strip().startswith('*'))
            ]
            return points
        
        except Exception as e:
            print(f"Error extracting key points: {e}")
            # Fallback: split text into sentences
            return [s.strip() for s in text.split('.') if s.strip()][:5]
