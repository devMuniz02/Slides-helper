"""
PowerPoint Connector

Uses pywin32 to connect to active PowerPoint presentations and extract slide data.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import win32com.client as win32
import pythoncom

from ..slide_processor import SlideContent


class PowerPointConnector:
    """Connector for Microsoft PowerPoint using COM automation."""

    def __init__(self):
        self.powerpoint = None
        self.presentation = None
        self.temp_dir = None
        self._com_initialized = False

    def connect(self) -> bool:
        """
        Connect to the active PowerPoint application.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            pythoncom.CoInitialize()
            self._com_initialized = True
            self.powerpoint = win32.Dispatch("PowerPoint.Application")
            if self.powerpoint.Presentations.Count > 0:
                self.presentation = self.powerpoint.ActivePresentation
                return True
            return False
        except Exception as e:
            print(f"Failed to connect to PowerPoint: {e}")
            return False

    def disconnect(self):
        """Disconnect from PowerPoint."""
        if self.powerpoint:
            self.powerpoint = None
            self.presentation = None
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
        if self._com_initialized:
            try:
                pythoncom.CoUninitialize()
            except:
                pass  # Ignore if CoUninitialize fails
            self._com_initialized = False

    def get_active_presentation_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the active presentation.

        Returns:
            Dict with presentation info or None if no active presentation
        """
        if not self.presentation:
            return None

        return {
            "name": self.presentation.Name,
            "path": self.presentation.FullName if hasattr(self.presentation, 'FullName') else None,
            "slide_count": self.presentation.Slides.Count,
            "current_slide": self.presentation.SlideShowWindow.View.CurrentShowPosition if self.presentation.SlideShowWindow else None
        }

    def extract_slide_content(self, slide_index: int) -> Optional[SlideContent]:
        """
        Extract content from a specific slide.

        Args:
            slide_index: 1-based slide index

        Returns:
            SlideContent object or None if failed
        """
        if not self.presentation or slide_index < 1 or slide_index > self.presentation.Slides.Count:
            return None

        try:
            slide = self.presentation.Slides(slide_index)

            # Extract text content
            text_content = []
            for shape in slide.Shapes:
                if shape.HasTextFrame and shape.TextFrame.HasText:
                    text_content.append(shape.TextFrame.TextRange.Text)

            # Extract images
            images = []
            if not self.temp_dir:
                self.temp_dir = tempfile.mkdtemp()

            for i, shape in enumerate(slide.Shapes):
                if shape.Type == 13:  # Picture type
                    try:
                        image_path = os.path.join(self.temp_dir, f"slide_{slide_index}_image_{i}.png")
                        shape.Export(image_path, 2)  # 2 = PNG format
                        images.append(Path(image_path))
                    except Exception as e:
                        print(f"Failed to export image: {e}")

            # Extract speaker notes
            notes = ""
            if slide.HasNotesPage:
                notes_page = slide.NotesPage
                for shape in notes_page.Shapes:
                    if shape.PlaceholderFormat.Type == 2:  # Notes placeholder
                        if shape.HasTextFrame and shape.TextFrame.HasText:
                            notes = shape.TextFrame.TextRange.Text
                            break

            return SlideContent(
                slide_number=slide_index,
                title=text_content[0] if text_content else f"Slide {slide_index}",
                text_content=text_content,
                images=images,
                notes=notes,
                metadata={
                    "slide_id": slide.SlideID,
                    "slide_index": slide_index,
                    "layout": slide.Layout if hasattr(slide, 'Layout') else None
                }
            )

        except Exception as e:
            print(f"Failed to extract slide {slide_index}: {e}")
            return None

    def get_all_slides(self) -> List[SlideContent]:
        """
        Extract content from all slides in the active presentation.

        Returns:
            List of SlideContent objects
        """
        if not self.presentation:
            return []

        slides = []
        for i in range(1, self.presentation.Slides.Count + 1):
            slide_content = self.extract_slide_content(i)
            if slide_content:
                slides.append(slide_content)

        return slides

    def get_current_slide(self) -> Optional[SlideContent]:
        """
        Get content of the currently displayed slide in slideshow mode.

        Returns:
            SlideContent of current slide or None
        """
        if not self.presentation or not self.presentation.SlideShowWindow:
            return None

        current_index = self.presentation.SlideShowWindow.View.CurrentShowPosition
        return self.extract_slide_content(current_index)

    def save_presentation_as_pptx(self, output_path: str) -> bool:
        """
        Save the active presentation as a .pptx file.

        Args:
            output_path: Path where to save the .pptx file

        Returns:
            bool: True if successful
        """
        if not self.presentation:
            return False

        try:
            self.presentation.SaveAs(output_path, 24)  # 24 = pptx format
            return True
        except Exception as e:
            print(f"Failed to save presentation: {e}")
            return False