"""
Orchestrator Module

This module uses LangGraph to orchestrate the workflow of processing slides,
analyzing content, and generating narrations.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from ..slide_processor import SlideProcessor, SlideContent
from ..vision_analyzer import VisionAnalyzer
from ..tts_engine import TTSEngineFactory
from ..utils.config import config

import os

class AgentState(TypedDict):
    """State for the LangGraph agent."""
    
    pptx_path: str
    slides: List[SlideContent]
    analyses: List[Dict[str, Any]]
    summaries: List[str]
    audio_files: List[Path]
    current_slide_index: int
    total_slides: int
    error: Optional[str]


@dataclass
class ProcessingResult:
    """Result of processing a presentation."""
    
    pptx_path: Path
    total_slides: int
    slides_content: List[SlideContent]
    analyses: List[Dict[str, Any]]
    summaries: List[str]
    audio_files: List[Path]
    output_dir: Path
    success: bool
    error: Optional[str] = None


class SlidesOrchestrator:
    """
    Orchestrates the entire slide processing workflow using LangGraph.
    
    Workflow:
    1. Load presentation
    2. Extract slides
    3. Analyze each slide (text + vision)
    4. Generate summaries
    5. Convert summaries to speech
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the orchestrator.
        
        Args:
            output_dir: Directory for output files (defaults to config)
        """
        self.output_dir = output_dir or config.processing.output_dir
        self.vision_analyzer = VisionAnalyzer()
        self.tts_engine = TTSEngineFactory.create_engine(engine_type="edge-tts")
        self.tts_engine.voice = config.tts.voice
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Configured StateGraph
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("load_presentation", self._load_presentation)
        workflow.add_node("extract_slides", self._extract_slides)
        workflow.add_node("analyze_slide", self._analyze_slide)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("generate_audio", self._generate_audio)
        
        # Define the flow
        workflow.set_entry_point("load_presentation")
        workflow.add_edge("load_presentation", "extract_slides")
        workflow.add_edge("extract_slides", "analyze_slide")
        workflow.add_conditional_edges(
            "analyze_slide",
            self._should_continue_analysis,
            {
                "continue": "analyze_slide",
                "done": "generate_summary",
            },
        )
        workflow.add_edge("generate_summary", "generate_audio")
        workflow.add_edge("generate_audio", END)
        
        return workflow.compile()
    
    def _load_presentation(self, state: AgentState) -> AgentState:
        """
        Load the PowerPoint presentation.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        try:
            processor = SlideProcessor(state["pptx_path"])
            state["total_slides"] = processor.slides_count
            state["current_slide_index"] = 0
            state["slides"] = []
            state["analyses"] = []
            state["summaries"] = []
            state["audio_files"] = []
            state["error"] = None
        except Exception as e:
            state["error"] = f"Error loading presentation: {str(e)}"
        
        return state
    
    def _extract_slides(self, state: AgentState) -> AgentState:
        """
        Extract all slides from the presentation.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        try:
            processor = SlideProcessor(state["pptx_path"])
            state["slides"] = processor.process_all_slides()
        except Exception as e:
            state["error"] = f"Error extracting slides: {str(e)}"
        
        return state
    
    def _analyze_slide(self, state: AgentState) -> AgentState:
        """
        Analyze a single slide.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        idx = state["current_slide_index"]
        
        if idx < len(state["slides"]):
            try:
                slide_content = state["slides"][idx]
                analysis = self.vision_analyzer.analyze_slide_content(slide_content)
                state["analyses"].append(analysis)
                state["current_slide_index"] += 1
            except Exception as e:
                state["error"] = f"Error analyzing slide {idx + 1}: {str(e)}"
        
        return state
    
    def _should_continue_analysis(self, state: AgentState) -> str:
        """
        Determine if more slides need to be analyzed.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" or "done"
        """
        if state.get("error"):
            return "done"
        if state["current_slide_index"] < len(state["slides"]):
            return "continue"
        return "done"
    
    def _generate_summary(self, state: AgentState) -> AgentState:
        """
        Generate summaries for all analyzed slides.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        try:
            for slide_content in state["slides"]:
                summary = self.vision_analyzer.generate_slide_summary(slide_content)
                state["summaries"].append(summary)
        except Exception as e:
            state["error"] = f"Error generating summaries: {str(e)}"
        
        return state
    
    def _generate_audio(self, state: AgentState) -> AgentState:
        """
        Generate audio files from summaries.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        try:
            pptx_name = Path(state["pptx_path"]).stem
            audio_dir = self.output_dir / pptx_name / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            audio_files = self.tts_engine.generate_speech_batch(
                state["summaries"],
                audio_dir,
                filename_prefix="slide",
            )
            state["audio_files"] = audio_files
        except Exception as e:
            state["error"] = f"Error generating audio: {str(e)}"
        
        return state
    
    def process_presentation(self, pptx_path: str) -> ProcessingResult:
        """
        Process a PowerPoint presentation end-to-end.
        
        Args:
            pptx_path: Path to the PowerPoint file
            
        Returns:
            ProcessingResult with all outputs
        """
        # Initialize state
        initial_state: AgentState = {
            "pptx_path": pptx_path,
            "slides": [],
            "analyses": [],
            "summaries": [],
            "audio_files": [],
            "current_slide_index": 0,
            "total_slides": 0,
            "error": None,
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Create result
        pptx_path_obj = Path(pptx_path)
        output_dir = self.output_dir / pptx_path_obj.stem
        
        result = ProcessingResult(
            pptx_path=pptx_path_obj,
            total_slides=final_state["total_slides"],
            slides_content=final_state["slides"],
            analyses=final_state["analyses"],
            summaries=final_state["summaries"],
            audio_files=final_state["audio_files"],
            output_dir=output_dir,
            success=final_state["error"] is None,
            error=final_state["error"],
        )
        
        return result
    
    def process_presentation_streaming(self, pptx_path: str):
        """
        Process a presentation with streaming updates.
        
        Args:
            pptx_path: Path to the PowerPoint file
            
        Yields:
            State updates as processing progresses
        """
        # Initialize state
        initial_state: AgentState = {
            "pptx_path": pptx_path,
            "slides": [],
            "analyses": [],
            "summaries": [],
            "audio_files": [],
            "current_slide_index": 0,
            "total_slides": 0,
            "error": None,
        }
        
        # Stream the workflow
        for state in self.workflow.stream(initial_state):
            yield state
