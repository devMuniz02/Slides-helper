"""
PowerPoint Connector Module

This module provides integration with Microsoft PowerPoint using COM automation.
It allows connecting to active presentations and extracting slide content in real-time.
"""

from .connector import PowerPointConnector

__all__ = ["PowerPointConnector"]