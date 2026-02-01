"""
Main GUI Window for Slides-helper

Provides an interface that can overlay on PowerPoint or monitor the current slide.
"""
import customtkinter as ctk
from tkinter import messagebox, filedialog
import threading
from pathlib import Path
from typing import Optional

from ..powerpoint_connector import PowerPointConnector
from ..orchestrator import SlidesOrchestrator
from ..slide_processor import SlideContent


class SlidesHelperGUI:
    """Main GUI application for Slides-helper."""

    def __init__(self):
        self.connector = PowerPointConnector()
        self.orchestrator = None
        self.current_slide_content = None
        self.monitoring = False
        self.monitor_timer = None
        self.last_slide_number = None

        # Initialize GUI
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Slides Helper")
        self.root.geometry("800x600")
        self.root.attributes("-topmost", True)  # Always on top
        self.root.attributes("-alpha", 0.95)  # Slightly transparent

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="🎤📊 Slides Helper",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=10)

        # Connection status
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Not connected to PowerPoint",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=5)

        # Connect button
        self.connect_btn = ctk.CTkButton(
            main_frame,
            text="Connect to PowerPoint",
            command=self.connect_to_powerpoint
        )
        self.connect_btn.pack(pady=10)

        # Current slide info
        slide_frame = ctk.CTkFrame(main_frame)
        slide_frame.pack(fill="x", padx=10, pady=10)

        slide_title = ctk.CTkLabel(
            slide_frame,
            text="Current Slide",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        slide_title.pack(pady=5)

        self.slide_info_text = ctk.CTkTextbox(
            slide_frame,
            height=100,
            wrap="word"
        )
        self.slide_info_text.pack(fill="x", padx=10, pady=10)
        self.slide_info_text.insert("0.0", "No slide data available")

        # Control buttons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=10)

        self.monitor_btn = ctk.CTkButton(
            button_frame,
            text="Start Monitoring",
            command=self.toggle_monitoring,
            state="disabled"
        )
        self.monitor_btn.pack(side="left", padx=5)

        self.analyze_btn = ctk.CTkButton(
            button_frame,
            text="Analyze Current Slide",
            command=self.analyze_current_slide,
            state="disabled"
        )
        self.analyze_btn.pack(side="left", padx=5)

        self.summarize_btn = ctk.CTkButton(
            button_frame,
            text="Summarize & Speak",
            command=self.summarize_and_speak,
            state="disabled"
        )
        self.summarize_btn.pack(side="left", padx=5)

        self.process_btn = ctk.CTkButton(
            button_frame,
            text="Process Full Presentation",
            command=self.process_presentation,
            state="disabled"
        )
        self.process_btn.pack(side="left", padx=5)

        # Analysis results
        analysis_frame = ctk.CTkFrame(main_frame)
        analysis_frame.pack(fill="both", expand=True, padx=10, pady=10)

        analysis_title = ctk.CTkLabel(
            analysis_frame,
            text="Analysis Results",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        analysis_title.pack(pady=5)

        self.analysis_text = ctk.CTkTextbox(
            analysis_frame,
            wrap="word"
        )
        self.analysis_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.analysis_text.insert("0.0", "Analysis results will appear here...")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(main_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.progress_bar.set(0)

    def connect_to_powerpoint(self):
        """Connect to PowerPoint application."""
        if self.connector.connect():
            info = self.connector.get_active_presentation_info()
            if info:
                self.status_label.configure(
                    text=f"Connected: {info['name']} ({info['slide_count']} slides)"
                )
                self.connect_btn.configure(text="Disconnect", command=self.disconnect_from_powerpoint)
                self.monitor_btn.configure(state="normal")
                self.analyze_btn.configure(state="normal")
                self.summarize_btn.configure(state="normal")
                self.process_btn.configure(state="normal")
                # Initialize current slide content
                self.current_slide_content = self.connector.get_current_slide()
                self.update_slide_info()
            else:
                messagebox.showwarning("Warning", "Connected to PowerPoint but no active presentation found.")
        else:
            messagebox.showerror("Error", "Failed to connect to PowerPoint. Make sure PowerPoint is running.")

    def disconnect_from_powerpoint(self):
        """Disconnect from PowerPoint."""
        self.stop_monitoring()
        self.connector.disconnect()
        self.status_label.configure(text="Not connected to PowerPoint")
        self.connect_btn.configure(text="Connect to PowerPoint", command=self.connect_to_powerpoint)
        self.monitor_btn.configure(state="disabled")
        self.analyze_btn.configure(state="disabled")
        self.summarize_btn.configure(state="disabled")
        self.process_btn.configure(state="disabled")
        self.slide_info_text.delete("0.0", "end")
        self.slide_info_text.insert("0.0", "No slide data available")

    def update_slide_info(self):
        """Update the current slide information display."""
        slide_content = self.current_slide_content
        if slide_content:
            info_text = f"Slide {slide_content.slide_number}: {slide_content.title}\n\n"
            info_text += "Content:\n" + "\n".join(slide_content.text_content) + "\n\n"
            if slide_content.notes:
                info_text += f"Speaker Notes:\n{slide_content.notes}"
            self.slide_info_text.delete("0.0", "end")
            self.slide_info_text.insert("0.0", info_text)
        else:
            self.slide_info_text.delete("0.0", "end")
            self.slide_info_text.insert("0.0", "Unable to get current slide information")

    def toggle_monitoring(self):
        """Start or stop monitoring the current slide."""
        if self.monitoring:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        """Start monitoring the current slide for changes."""
        self.monitoring = True
        self.monitor_btn.configure(text="Stop Monitoring")
        self.last_slide_number = None
        self._schedule_monitor_check()

    def stop_monitoring(self):
        """Stop monitoring the current slide."""
        self.monitoring = False
        self.monitor_btn.configure(text="Start Monitoring")
        if self.monitor_timer:
            self.root.after_cancel(self.monitor_timer)
            self.monitor_timer = None

    def _schedule_monitor_check(self):
        """Schedule the next monitor check."""
        if self.monitoring:
            self._check_slide_change()
            self.monitor_timer = self.root.after(1000, self._schedule_monitor_check)  # Check every second

    def _check_slide_change(self):
        """Check if the current slide has changed and update if needed."""
        try:
            current_slide = self.connector.get_current_slide()
            if current_slide and current_slide.slide_number != self.last_slide_number:
                self.current_slide_content = current_slide
                self.update_slide_info()
                self.last_slide_number = current_slide.slide_number
        except Exception as e:
            print(f"Monitoring error: {e}")
            # Don't break, just continue monitoring

    def summarize_and_speak(self):
        """Generate summary and speak it with subtitles."""
        if not self.current_slide_content:
            messagebox.showwarning("Warning", "No current slide to summarize.")
            return

        # Run summarization and TTS in a separate thread
        threading.Thread(target=self._summarize_and_speak_thread, daemon=True).start()

    def _summarize_and_speak_thread(self):
        """Thread function for summarization and TTS."""
        try:
            self.root.after(0, lambda: self.analysis_text.delete("0.0", "end"))
            self.root.after(0, lambda: self.analysis_text.insert("0.0", "Generating summary..."))
            self.root.after(0, lambda: self.progress_bar.set(0.2))

            if not self.orchestrator:
                self.orchestrator = SlidesOrchestrator()
                self.root.after(0, lambda: self.progress_bar.set(0.4))

            # Generate summary
            summary = self.orchestrator.vision_analyzer.generate_slide_summary(self.current_slide_content)
            self.root.after(0, lambda: self.progress_bar.set(0.6))

            # Display summary
            summary_text = f"Summary for Slide {self.current_slide_content.slide_number}:\n\n{summary}"
            self.root.after(0, lambda: self.analysis_text.insert("0.0", summary_text))
            self.root.after(0, lambda: self.progress_bar.set(0.8))

            # Generate TTS audio
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                audio_path = Path(temp_file.name)

            self.orchestrator.tts_engine.generate_speech(summary, audio_path)
            self.root.after(0, lambda: self.progress_bar.set(1.0))

            # Start speaking with subtitles
            self.root.after(0, lambda: self._start_speaking_with_subtitles(summary, audio_path))

        except Exception as e:
            error_msg = f"Summarization failed: {str(e)}"
            self.root.after(0, lambda: self.analysis_text.insert("0.0", error_msg))
            self.root.after(0, lambda: self.progress_bar.set(0))

    def _start_speaking_with_subtitles(self, text: str, audio_path: Path):
        """Start speaking the text with progressive subtitle overlay."""
        # Split text into sentences for progressive display
        sentences = self._split_into_sentences(text)

        # Create subtitle window
        self._create_progressive_subtitle_window(sentences)

        # Play audio in a separate thread
        threading.Thread(target=self._play_audio, args=(audio_path,), daemon=True).start()

        # Start progressive subtitle display
        self._start_progressive_subtitles(sentences)

    def _create_subtitle_window(self, text: str):
        """Create a subtitle window at the bottom of the screen."""
        # Destroy existing subtitle window if it exists
        if hasattr(self, 'subtitle_window') and self.subtitle_window:
            self.subtitle_window.destroy()

        # Create new subtitle window
        self.subtitle_window = ctk.CTkToplevel(self.root)
        self.subtitle_window.title("")
        self.subtitle_window.attributes("-topmost", True)
        self.subtitle_window.attributes("-alpha", 0.9)  # Semi-transparent
        self.subtitle_window.overrideredirect(True)  # Remove window decorations

        # Get screen dimensions
        screen_width = self.subtitle_window.winfo_screenwidth()
        screen_height = self.subtitle_window.winfo_screenheight()

        # Position at bottom of screen
        window_width = min(800, screen_width - 100)
        window_height = 100
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 50

        self.subtitle_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Create subtitle label
        subtitle_label = ctk.CTkLabel(
            self.subtitle_window,
            text=text,
            font=ctk.CTkFont(size=16, weight="bold"),
            wraplength=window_width - 40,
            justify="center"
        )
        subtitle_label.pack(expand=True, fill="both", padx=20, pady=10)

        # Make window closable by clicking
        subtitle_label.bind("<Button-1>", lambda e: self._close_subtitle_window())
        self.subtitle_window.bind("<Button-1>", lambda e: self._close_subtitle_window())

        # Auto-close after estimated speaking time (roughly 150 words per minute)
        word_count = len(text.split())
        estimated_seconds = max(5, word_count * 60 // 150)  # At least 5 seconds
    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences for progressive display."""
        import re
        # Split on sentence endings, but keep the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out empty sentences and limit to reasonable chunks
        sentences = [s.strip() for s in sentences if s.strip()]
        # If we have very long sentences, break them into smaller chunks
        result = []
        for sentence in sentences:
            if len(sentence.split()) > 20:  # If more than 20 words
                # Break into smaller chunks
                words = sentence.split()
                for i in range(0, len(words), 15):  # 15 words per chunk
                    chunk = ' '.join(words[i:i+15])
                    if i + 15 < len(words):
                        chunk += '...'
                    result.append(chunk)
            else:
                result.append(sentence)
        return result

    def _create_progressive_subtitle_window(self, sentences: list):
        """Create a subtitle window for progressive display."""
        # Destroy existing subtitle window if it exists
        if hasattr(self, 'subtitle_window') and self.subtitle_window:
            self.subtitle_window.destroy()

        # Create new subtitle window
        self.subtitle_window = ctk.CTkToplevel(self.root)
        self.subtitle_window.title("")
        self.subtitle_window.attributes("-topmost", True)
        self.subtitle_window.attributes("-alpha", 0.9)  # Semi-transparent
        self.subtitle_window.overrideredirect(True)  # Remove window decorations

        # Get screen dimensions
        screen_width = self.subtitle_window.winfo_screenwidth()
        screen_height = self.subtitle_window.winfo_screenheight()

        # Position at bottom of screen
        window_width = min(800, screen_width - 100)
        window_height = 100
        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 50

        self.subtitle_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Create subtitle label
        self.subtitle_label = ctk.CTkLabel(
            self.subtitle_window,
            text="",
            font=ctk.CTkFont(size=16, weight="bold"),
            wraplength=window_width - 40,
            justify="center"
        )
        self.subtitle_label.pack(expand=True, fill="both", padx=20, pady=10)

        # Make window closable by clicking
        self.subtitle_label.bind("<Button-1>", lambda e: self._close_subtitle_window())
        self.subtitle_window.bind("<Button-1>", lambda e: self._close_subtitle_window())

    def _start_progressive_subtitles(self, sentences: list):
        """Start displaying subtitles progressively."""
        self.current_sentence_index = 0
        self._show_next_sentence(sentences)

    def _show_next_sentence(self, sentences: list):
        """Show the next sentence in the subtitle window."""
        if self.current_sentence_index < len(sentences):
            sentence = sentences[self.current_sentence_index]
            self.subtitle_label.configure(text=sentence)
            self.current_sentence_index += 1

            # Calculate timing based on word count (roughly 150 words per minute)
            word_count = len(sentence.split())
            delay_ms = max(2000, word_count * 60 * 1000 // 150)  # At least 2 seconds per sentence

            # Schedule next sentence
            if hasattr(self, 'subtitle_window') and self.subtitle_window:
                self.subtitle_window.after(delay_ms, lambda: self._show_next_sentence(sentences))
        else:
            # All sentences shown, keep the last one visible briefly then close
            if hasattr(self, 'subtitle_window') and self.subtitle_window:
                self.subtitle_window.after(2000, self._close_subtitle_window)

    def _close_subtitle_window(self):
        """Close the subtitle window."""
        if hasattr(self, 'subtitle_window') and self.subtitle_window:
            self.subtitle_window.destroy()
            self.subtitle_window = None

    def _play_audio(self, audio_path: Path):
        """Play the audio file."""
        try:
            import subprocess
            import platform

            if platform.system() == "Windows":
                # Use Windows Media Player or default player
                subprocess.run(["start", str(audio_path)], shell=True, check=True)
            else:
                # For other platforms, try common audio players
                subprocess.run(["mpg123", str(audio_path)], check=False) or \
                subprocess.run(["afplay", str(audio_path)], check=False) or \
                subprocess.run(["aplay", str(audio_path)], check=False)

        except Exception as e:
            print(f"Failed to play audio: {e}")
        finally:
            # Clean up audio file after a delay
            import time
            time.sleep(2)  # Give time for the audio to start playing
            try:
                audio_path.unlink()
            except:
                pass

    def analyze_current_slide(self):
        """Analyze the current slide using the vision analyzer."""
        if not self.current_slide_content:
            messagebox.showwarning("Warning", "No current slide to analyze.")
            return

        # Run analysis in a separate thread to avoid blocking GUI
        threading.Thread(target=self._analyze_slide_thread, daemon=True).start()

    def _analyze_slide_thread(self):
        """Thread function for slide analysis."""
        try:
            self.root.after(0, lambda: self.analysis_text.delete("0.0", "end"))
            self.root.after(0, lambda: self.analysis_text.insert("0.0", "Analyzing slide..."))
            self.root.after(0, lambda: self.progress_bar.set(0.3))

            if not self.current_slide_content:
                self.root.after(0, lambda: self.analysis_text.insert("0.0", "No slide content available to analyze."))
                self.root.after(0, lambda: self.progress_bar.set(0))
                return

            # Initialize orchestrator if needed to get the vision analyzer
            if not self.orchestrator:
                self.orchestrator = SlidesOrchestrator()
                self.root.after(0, lambda: self.progress_bar.set(0.6))

            # Perform vision analysis on the current slide
            analysis = self.orchestrator.vision_analyzer.analyze_slide_content(self.current_slide_content)
            self.root.after(0, lambda: self.progress_bar.set(0.9))

            # Format the analysis results
            result_text = f"Analysis Results for Slide {self.current_slide_content.slide_number}:\n\n"
            result_text += f"Title: {self.current_slide_content.title}\n\n"
            result_text += f"Text Content: {self.current_slide_content.get_full_text()}\n\n"

            # Add visual analysis results
            if analysis.get("visual_analysis"):
                result_text += "Visual Analysis:\n"
                for visual in analysis["visual_analysis"]:
                    result_text += f"Image {visual['image_index'] + 1}: {visual['analysis']}\n\n"

            # Add key points for text-only slides
            if analysis.get("key_points"):
                result_text += "Key Points:\n"
                for point in analysis["key_points"]:
                    result_text += f"• {point}\n"
                result_text += "\n"

            self.root.after(0, lambda: self.analysis_text.insert("0.0", result_text))
            self.root.after(0, lambda: self.progress_bar.set(1.0))

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.root.after(0, lambda: self.analysis_text.insert("0.0", error_msg))
            self.root.after(0, lambda: self.progress_bar.set(0))

    def process_presentation(self):
        """Process the entire presentation."""
        # Save current presentation as pptx and process it
        output_dir = filedialog.askdirectory(title="Select output directory")
        if not output_dir:
            return

        temp_pptx = Path(output_dir) / "temp_presentation.pptx"
        if self.connector.save_presentation_as_pptx(str(temp_pptx)):
            # Process using the existing orchestrator
            threading.Thread(target=self._process_presentation_thread,
                           args=(temp_pptx, Path(output_dir)), daemon=True).start()
        else:
            messagebox.showerror("Error", "Failed to save presentation.")

    def _process_presentation_thread(self, pptx_path: Path, output_dir: Path):
        """Thread function for presentation processing."""
        try:
            self.root.after(0, lambda: self.analysis_text.delete("0.0", "end"))
            self.root.after(0, lambda: self.analysis_text.insert("0.0", "Processing presentation..."))
            self.root.after(0, lambda: self.progress_bar.set(0.2))

            if not self.orchestrator:
                self.orchestrator = SlidesOrchestrator(output_dir=output_dir)

            result = self.orchestrator.process_presentation(str(pptx_path))

            self.root.after(0, lambda: self.progress_bar.set(1.0))
            if result.success:
                summary = f"Successfully processed {result.total_slides} slides.\n"
                summary += f"Output saved to: {result.output_dir}\n"
                summary += f"Generated {len(result.audio_files)} audio files."
                self.root.after(0, lambda: self.analysis_text.insert("0.0", summary))
            else:
                self.root.after(0, lambda: self.analysis_text.insert("0.0", "Processing failed."))

        except Exception as e:
            self.root.after(0, lambda: self.analysis_text.insert("0.0", f"Processing failed: {str(e)}"))
            self.root.after(0, lambda: self.progress_bar.set(0))
        finally:
            # Clean up temp file
            if pptx_path.exists():
                pptx_path.unlink()

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_monitoring()
        self._close_subtitle_window()
        self.connector.disconnect()