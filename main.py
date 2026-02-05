#!/usr/bin/env python3
"""
Main entry point for Slides-helper.

This script demonstrates how to use the Slides-helper to process PowerPoint
presentations and generate narrated summaries.
"""
import argparse
import sys
from pathlib import Path

from src import SlidesOrchestrator, SlidesHelperGUI, config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Process PowerPoint presentations and generate narrated summaries"
    )
    parser.add_argument(
        "pptx_file",
        type=str,
        nargs="?",  # Make it optional
        help="Path to the PowerPoint (.pptx) file to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated files (default: ./output)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream progress updates during processing",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical user interface for PowerPoint integration",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Launch the RAG (Retrieval-Augmented Generation) system for PowerPoint Q&A",
    )

    args = parser.parse_args()

    if args.gui:
        # Launch GUI mode
        print("Launching Slides Helper GUI...")
        gui = SlidesHelperGUI()
        gui.run()
        return

    if args.rag:
        # Launch RAG mode
        print("Launching RAG System...")
        import subprocess
        import sys
        try:
            # Run the RAG system with chat interface
            subprocess.run([sys.executable, "rag_system.py", "--chat"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error launching RAG system: {e}")
            sys.exit(1)
        return

    # File processing mode
    if not args.pptx_file:
        parser.error("pptx_file is required when not using --gui")

    # Validate input file
    pptx_path = Path(args.pptx_file)
    if not pptx_path.exists():
        print(f"Error: File not found: {pptx_path}", file=sys.stderr)
        sys.exit(1)

    if not pptx_path.suffix.lower() == ".pptx":
        print(f"Error: File must be a .pptx file, got: {pptx_path.suffix}", file=sys.stderr)
        sys.exit(1)

    # Set up output directory if provided
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Create orchestrator
    print("Initializing Slides-helper...")
    print(f"LM Studio URL: {config.lm_studio.base_url}")
    print(f"Vision Model: {config.lm_studio.vision_model_name}")
    print(f"TTS Engine: {config.tts.engine}")
    print(f"TTS Voice: {config.tts.voice}")
    print()

    orchestrator = SlidesOrchestrator(output_dir=output_dir)

    # Process the presentation
    print(f"Processing presentation: {pptx_path.name}")
    print("-" * 60)

    if args.stream:
        # Streaming mode
        for state_update in orchestrator.process_presentation_streaming(str(pptx_path)):
            # Print progress updates
            for node_name, node_state in state_update.items():
                if "error" in node_state and node_state["error"]:
                    print(f"ERROR in {node_name}: {node_state['error']}")
                elif "current_slide_index" in node_state:
                    idx = node_state["current_slide_index"]
                    total = node_state.get("total_slides", 0)
                    if total > 0:
                        print(f"Processing slide {idx}/{total}...")
    else:
        # Standard mode
        result = orchestrator.process_presentation(str(pptx_path))

        # Print results
        print()
        print("=" * 60)
        if result.success:
            print("✓ Processing completed successfully!")
            print()
            print(f"Total slides processed: {result.total_slides}")
            print(f"Output directory: {result.output_dir}")
            print(f"Audio files generated: {len(result.audio_files)}")
            print()
            print("Generated files:")
            for audio_file in result.audio_files:
                print(f"  - {audio_file.name}")
            print()
            print("Summary of slides:")
            for i, summary in enumerate(result.summaries, start=1):
                print(f"\nSlide {i}:")
                print(f"  {summary[:200]}...")  # First 200 chars
        else:
            print("✗ Processing failed!")
            print(f"Error: {result.error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
