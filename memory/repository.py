from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import numpy as np
from models import Memory, Goal, Reflection, Creation
from datetime import datetime

class VectorMemoryRepository(ABC):
    """Abstract interface for vector memory storage"""
    
    @abstractmethod
    def add(self, memory: Memory) -> str:
        """
        Add a memory to the repository
        
        Args:
            memory: Memory object to add
            
        Returns:
            ID of the added memory
        """
        pass
    
    @abstractmethod
    def get(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory object if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update(self, memory: Memory) -> bool:
        """
        Update an existing memory
        
        Args:
            memory: Memory object with updated fields
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search_by_content(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """
        Search memories by content similarity
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of matching Memory objects
        """
        pass
    
    @abstractmethod
    def search_by_vector(self, vector: List[float], limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """
        Search memories by vector similarity
        
        Args:
            vector: Vector embedding to search for
            limit: Maximum number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of matching Memory objects
        """
        pass
    
    @abstractmethod
    def list(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100, offset: int = 0) -> List[Memory]:
        """
        List memories with optional filtering
        
        Args:
            filters: Optional metadata filters
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            
        Returns:
            List of matching Memory objects
        """
        pass
    
    @abstractmethod
    def exists(self, memory_id: str) -> bool:
        """
        Check if a memory exists
        
        Args:
            memory_id: ID of the memory to check
            
        Returns:
            True if exists, False otherwise
        """
        pass
    
    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count memories with optional filtering
        
        Args:
            filters: Optional metadata filters
            
        Returns:
            Number of matching memories
        """
        pass
        
    def list_by_recency(self, filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Memory]:
        """
        List memories ordered by recency
        
        Args:
            filters: Optional filters to apply
            limit: Maximum number of memories to return
            
        Returns:
            List of memories ordered by creation time (newest first)
        """
        memories = self.list(filters, limit)
        # Sort by created_at timestamp (newest first)
        return sorted(memories, key=lambda m: m.created_at, reverse=True)
        
    def search_by_vector_excluding(self, vector: List[float], exclude_ids: List[str], 
                                limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """
        Search by vector similarity while excluding certain memory IDs
        
        Args:
            vector: The query vector
            exclude_ids: List of memory IDs to exclude
            limit: Maximum number of results
            filters: Optional additional filters
            
        Returns:
            List of relevant memories, excluding the specified IDs
        """
        # Get more results than needed to account for filtering
        results = self.search_by_vector(vector, limit * 2, filters)
        
        # Filter out excluded IDs
        filtered_results = [m for m in results if m.id not in exclude_ids]
        
        # Return only up to the limit
        return filtered_results[:limit]