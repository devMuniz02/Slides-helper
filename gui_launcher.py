#!/usr/bin/env python3
"""
GUI Launcher for Slides-helper

Launches the graphical user interface for PowerPoint integration.
"""
from src.gui import SlidesHelperGUI


def main():
    """Launch the GUI application."""
    print("Starting Slides Helper GUI...")
    print("Make sure PowerPoint is running with an active presentation.")
    print()

    gui = SlidesHelperGUI()
    gui.run()


if __name__ == "__main__":
    main()