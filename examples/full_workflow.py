#!/usr/bin/env python3
"""
Example: Full orchestrator workflow.

This example demonstrates the complete workflow using LangGraph orchestration.
"""
from pathlib import Path
from src.orchestrator import SlidesOrchestrator


def main():
    """Demonstrate full orchestrator workflow."""
    # Example PPTX file path (replace with your file)
    pptx_file = "example_presentation.pptx"
    
    if not Path(pptx_file).exists():
        print(f"Please provide a valid PPTX file. Current path: {pptx_file}")
        print("You can modify this script to point to your file.")
        print("\nNote: Make sure LM Studio is running with Qwen2.5-VL model.")
        return
    
    print("Initializing orchestrator...")
    print("Note: Make sure LM Studio is running at http://localhost:1234/v1\n")
    
    # Create orchestrator
    orchestrator = SlidesOrchestrator()
    
    # Process presentation
    print(f"Processing: {pptx_file}")
    print("=" * 60)
    
    result = orchestrator.process_presentation(pptx_file)
    
    # Display results
    if result.success:
        print("\n✓ Processing completed successfully!\n")
        print(f"Presentation: {result.pptx_path.name}")
        print(f"Total slides: {result.total_slides}")
        print(f"Output directory: {result.output_dir}")
        print(f"Audio files: {len(result.audio_files)}")
        
        print("\n" + "=" * 60)
        print("Generated Audio Files:")
        print("=" * 60)
        for audio_file in result.audio_files:
            print(f"  {audio_file}")
        
        print("\n" + "=" * 60)
        print("Slide Summaries:")
        print("=" * 60)
        for i, (slide, summary) in enumerate(zip(result.slides_content, result.summaries), start=1):
            print(f"\nSlide {i}: {slide.title or '(No title)'}")
            print(f"  Summary: {summary[:150]}...")
    else:
        print(f"\n✗ Processing failed: {result.error}")


if __name__ == "__main__":
    main()
