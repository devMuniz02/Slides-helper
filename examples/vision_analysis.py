#!/usr/bin/env python3
"""
Example: Vision analyzer usage.

This example demonstrates how to use the VisionAnalyzer module to analyze
slides with LM Studio's vision model.
"""
from pathlib import Path
from src.slide_processor import SlideProcessor
from src.vision_analyzer import VisionAnalyzer


def main():
    """Demonstrate vision analyzer usage."""
    # Example PPTX file path (replace with your file)
    pptx_file = "example_presentation.pptx"
    
    if not Path(pptx_file).exists():
        print(f"Please provide a valid PPTX file. Current path: {pptx_file}")
        print("You can modify this script to point to your file.")
        return
    
    print("Note: Make sure LM Studio is running at http://localhost:1234/v1")
    print("with the Qwen2.5-VL model loaded.\n")
    
    # Initialize components
    print(f"Loading presentation: {pptx_file}")
    processor = SlideProcessor(pptx_file)
    analyzer = VisionAnalyzer()
    
    # Process first slide as example
    slide = processor.process_slide(0)
    
    print(f"\nAnalyzing Slide 1...")
    print(f"Title: {slide.title}")
    
    # Analyze the slide
    analysis = analyzer.analyze_slide_content(slide)
    
    print(f"\n{'=' * 60}")
    print("Analysis Results")
    print(f"{'=' * 60}")
    
    print(f"\nSlide Number: {analysis['slide_number']}")
    print(f"Title: {analysis['title']}")
    
    if analysis['visual_analysis']:
        print(f"\nVisual Analysis:")
        for visual in analysis['visual_analysis']:
            print(f"  Image {visual['image_index'] + 1}:")
            print(f"    {visual['analysis']}")
    
    if analysis['key_points']:
        print(f"\nKey Points:")
        for point in analysis['key_points']:
            print(f"  - {point}")
    
    # Generate narration summary
    print(f"\n{'=' * 60}")
    print("Narration Summary")
    print(f"{'=' * 60}")
    summary = analyzer.generate_slide_summary(slide)
    print(summary)


if __name__ == "__main__":
    main()
