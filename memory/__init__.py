"""
Memory system for the AI consciousness project.

This module provides a comprehensive memory system with vector-based storage,
consolidation mechanisms, and specialized memory types for different content.
"""

# Import from submodules for convenient access
from .models import Memory, Goal, Reflection, Creation
from .repository import VectorMemoryRepository
from .chroma_repository import ChromaDBMemoryRepository
from .manager import MemoryManager

# Version info
__version__ = "0.1.0"