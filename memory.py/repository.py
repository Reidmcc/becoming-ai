from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import numpy as np
import uuid
from models import Memory
from datetime import datetime
from models import Memory, Goal, Reflection, Creation

# TODO 
class Memory:
    """Data class representing a memory stored in the system"""
    
    def __init__(
        self,
        id: str = None,
        content: str = "",
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        type: str = "thought",
        is_consolidated: bool = False,
        source_ids: Optional[List[str]] = None,
        importance: float = 0.0,
    ):
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.embedding = embedding
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or self.created_at
        self.type = type
        self.is_consolidated = is_consolidated
        self.source_ids = source_ids or []
        self.importance = importance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to a dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "type": self.type,
            "is_consolidated": self.is_consolidated,
            "source_ids": self.source_ids,
            "importance": self.importance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding=None) -> 'Memory':
        """Create a Memory instance from a dictionary"""
        created_at = datetime.fromisoformat(data.get("created_at")) if isinstance(data.get("created_at"), str) else data.get("created_at")
        updated_at = datetime.fromisoformat(data.get("updated_at")) if isinstance(data.get("updated_at"), str) else data.get("updated_at")
        
        return cls(
            id=data.get("id"),
            content=data.get("content", ""),
            embedding=embedding,
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
            type=data.get("type", "thought"),
            is_consolidated=data.get("is_consolidated", False),
            source_ids=data.get("source_ids", []),
            importance=data.get("importance", 0.0)
        )


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