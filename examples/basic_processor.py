#!/usr/bin/env python3
"""
Example: Basic slide processor usage.

This example demonstrates how to use the SlideProcessor module independently
to extract content from PowerPoint files.
"""
from pathlib import Path
from src.slide_processor import SlideProcessor


def main():
    """Demonstrate basic slide processor usage."""
    # Example PPTX file path (replace with your file)
    pptx_file = "example_presentation.pptx"
    
    if not Path(pptx_file).exists():
        print(f"Please provide a valid PPTX file. Current path: {pptx_file}")
        print("You can modify this script to point to your file.")
        return
    
    # Initialize the processor
    print(f"Loading presentation: {pptx_file}")
    processor = SlideProcessor(pptx_file)
    
    # Get presentation info
    info = processor.get_presentation_info()
    print(f"\nPresentation Info:")
    print(f"  File: {info['file_name']}")
    print(f"  Total Slides: {info['total_slides']}")
    print(f"  Has Notes: {info['has_notes']}")
    
    # Process all slides
    print(f"\nProcessing {info['total_slides']} slides...")
    slides = processor.process_all_slides()
    
    # Display slide content
    for slide in slides:
        print(f"\n{'=' * 60}")
        print(f"Slide {slide.slide_number}")
        print(f"{'=' * 60}")
        
        if slide.title:
            print(f"Title: {slide.title}")
        
        if slide.text_content:
            print(f"\nContent:")
            for text in slide.text_content:
                print(f"  - {text}")
        
        if slide.images:
            print(f"\nImages: {len(slide.images)} found")
            for idx, img in enumerate(slide.images):
                print(f"  Image {idx + 1}: {img.size[0]}x{img.size[1]} pixels")
        
        if slide.notes:
            print(f"\nSpeaker Notes:")
            print(f"  {slide.notes}")
        
        print(f"\nMetadata: {slide.metadata}")


if __name__ == "__main__":
    main()
