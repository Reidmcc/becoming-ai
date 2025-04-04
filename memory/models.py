# memory/models.py
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
import json

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
        """
        Initialize a memory object
        
        Args:
            id: Unique identifier (uuid if not provided)
            content: Text content of the memory
            embedding: Vector embedding of the content (optional)
            metadata: Additional metadata for the memory (optional)
            created_at: Creation timestamp (defaults to now)
            updated_at: Last update timestamp (defaults to created_at)
            type: Type of memory (thought, reflection, conversation, etc.)
            is_consolidated: Whether this memory is a consolidation of others
            source_ids: IDs of source memories if this is a consolidation
            importance: Importance score (0.0-1.0)
        """
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
        """
        Convert memory to a dictionary for storage
        
        Returns:
            Dictionary representation of the memory
        """
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
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[List[float]] = None) -> 'Memory':
        """
        Create a Memory instance from a dictionary
        
        Args:
            data: Dictionary containing memory data
            embedding: Optional embedding vector (may be stored separately)
            
        Returns:
            Memory instance
        """
        # Parse timestamps if they're strings
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
            
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
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
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"Memory(id={self.id}, type={self.type}, created={self.created_at.isoformat()})"
    
    def update_content(self, new_content: str, embedding_service=None) -> None:
        """
        Update memory content and optionally regenerate embedding
        
        Args:
            new_content: New content for the memory
            embedding_service: Optional service to regenerate embedding
        """
        self.content = new_content
        self.updated_at = datetime.now()
        
        # Regenerate embedding if service provided
        if embedding_service:
            self.embedding = embedding_service.get_embedding(new_content)
    
    def update_metadata(self, new_metadata: Dict[str, Any], merge: bool = True) -> None:
        """
        Update memory metadata
        
        Args:
            new_metadata: New metadata to add
            merge: If True, merge with existing metadata; if False, replace
        """
        if merge:
            self.metadata.update(new_metadata)
        else:
            self.metadata = new_metadata
            
        self.updated_at = datetime.now()
    
    def add_source(self, source_id: str) -> None:
        """
        Add a source memory ID
        
        Args:
            source_id: ID of source memory to add
        """
        if source_id not in self.source_ids:
            self.source_ids.append(source_id)
            self.updated_at = datetime.now()
    
    def calculate_age(self) -> float:
        """
        Calculate age of memory in seconds
        
        Returns:
            Age in seconds
        """
        return (datetime.now() - self.created_at).total_seconds()


class Goal(Memory):
    """
    Special type of memory representing a goal
    Extends Memory with goal-specific attributes
    """
    
    def __init__(
        self,
        id: str = None,
        content: str = "",
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        active: bool = True,
        completed: bool = False,
        completed_at: Optional[datetime] = None,
        priority: float = 0.5
    ):
        """
        Initialize a goal
        
        Args:
            id: Unique identifier
            content: Goal description
            embedding: Vector embedding
            metadata: Additional metadata
            created_at: Creation timestamp
            updated_at: Last update timestamp
            active: Whether goal is active
            completed: Whether goal is completed
            completed_at: When goal was completed
            priority: Goal priority (0.0-1.0)
        """
        # Initialize parent class
        super().__init__(
            id=id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            created_at=created_at,
            updated_at=updated_at,
            type="goal",
            is_consolidated=False,
            source_ids=[],
            importance=priority
        )
        
        # Goal-specific attributes
        self.active = active
        self.completed = completed
        self.completed_at = completed_at
        self.priority = priority
        
        # Add to metadata
        self.metadata["active"] = active
        self.metadata["completed"] = completed
        if completed_at:
            self.metadata["completed_at"] = completed_at.isoformat()
        self.metadata["priority"] = priority
    
    def complete(self) -> None:
        """Mark goal as completed"""
        self.completed = True
        self.completed_at = datetime.now()
        self.active = False
        self.updated_at = datetime.now()
        
        # Update metadata
        self.metadata["completed"] = True
        self.metadata["completed_at"] = self.completed_at.isoformat()
        self.metadata["active"] = False
    
    def deactivate(self) -> None:
        """Deactivate goal without completing"""
        self.active = False
        self.updated_at = datetime.now()
        self.metadata["active"] = False
    
    def reactivate(self) -> None:
        """Reactivate an inactive goal"""
        if not self.completed:
            self.active = True
            self.updated_at = datetime.now()
            self.metadata["active"] = True
    
    @classmethod
    def from_memory(cls, memory: Memory) -> 'Goal':
        """
        Create a Goal from a standard Memory
        
        Args:
            memory: Memory object to convert
        
        Returns:
            Goal object
        """
        metadata = memory.metadata.copy()
        active = metadata.pop("active", True)
        completed = metadata.pop("completed", False)
        priority = metadata.pop("priority", 0.5)
        
        completed_at = None
        if "completed_at" in metadata:
            completed_at_str = metadata.pop("completed_at")
            if isinstance(completed_at_str, str):
                completed_at = datetime.fromisoformat(completed_at_str)
        
        return cls(
            id=memory.id,
            content=memory.content,
            embedding=memory.embedding,
            metadata=metadata,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            active=active,
            completed=completed,
            completed_at=completed_at,
            priority=priority
        )


class Reflection(Memory):
    """
    Special type of memory representing a reflection on another memory
    """
    
    def __init__(
        self,
        id: str = None,
        content: str = "",
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        thought_id: str = None,
        insights: Optional[List[str]] = None
    ):
        """
        Initialize a reflection
        
        Args:
            id: Unique identifier
            content: Reflection content
            embedding: Vector embedding
            metadata: Additional metadata
            created_at: Creation timestamp
            updated_at: Last update timestamp
            thought_id: ID of thought being reflected on
            insights: List of key insights from the reflection
        """
        # Build source IDs list
        source_ids = []
        if thought_id:
            source_ids.append(thought_id)
        
        # Initialize parent class
        super().__init__(
            id=id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            created_at=created_at,
            updated_at=updated_at,
            type="reflection",
            is_consolidated=False,
            source_ids=source_ids,
            importance=0.7  # Reflections tend to be more important
        )
        
        # Reflection-specific attributes
        self.thought_id = thought_id
        self.insights = insights or []
        
        # Add to metadata
        self.metadata["thought_id"] = thought_id
        self.metadata["insights"] = self.insights
    
    def add_insight(self, insight: str) -> None:
        """
        Add a key insight from the reflection
        
        Args:
            insight: Insight text to add
        """
        if insight not in self.insights:
            self.insights.append(insight)
            self.metadata["insights"] = self.insights
            self.updated_at = datetime.now()
    
    @classmethod
    def from_memory(cls, memory: Memory) -> 'Reflection':
        """
        Create a Reflection from a standard Memory
        
        Args:
            memory: Memory object to convert
        
        Returns:
            Reflection object
        """
        metadata = memory.metadata.copy()
        thought_id = metadata.pop("thought_id", None)
        
        # Get thought_id from source_ids if not in metadata
        if not thought_id and memory.source_ids:
            thought_id = memory.source_ids[0]
            
        insights = metadata.pop("insights", [])
        
        return cls(
            id=memory.id,
            content=memory.content,
            embedding=memory.embedding,
            metadata=metadata,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            thought_id=thought_id,
            insights=insights
        )
    
class Creation(Memory):
    """
    Special type of memory representing a creative work
    Extends Memory with creation-specific attributes
    """
    
    def __init__(
        self,
        id: str = None,
        content: str = "",
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        title: str = "Untitled Creation",
        creation_type: str = "general",
        tags: Optional[List[str]] = None,
        version: int = 1,
        inspiration_ids: Optional[List[str]] = None
    ):
        """
        Initialize a creative work memory
        
        Args:
            id: Unique identifier
            content: Creation content/text
            embedding: Vector embedding
            metadata: Additional metadata
            created_at: Creation timestamp
            updated_at: Last update timestamp
            title: Title of the creative work
            creation_type: Type of creation (poem, story, essay, etc.)
            tags: List of descriptive tags
            version: Version number (for tracking revisions)
            inspiration_ids: IDs of memories that inspired this creation
        """
        # Initialize parent class
        super().__init__(
            id=id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            created_at=created_at,
            updated_at=updated_at,
            type="creation",
            is_consolidated=False,
            source_ids=inspiration_ids or [],
            importance=0.8  # Creative works tend to be important memories
        )
        
        # Creation-specific attributes
        self.title = title
        self.creation_type = creation_type
        self.tags = tags or []
        self.version = version
        self.inspiration_ids = inspiration_ids or []
        
        # Add to metadata
        self.metadata["title"] = title
        self.metadata["creation_type"] = creation_type
        self.metadata["tags"] = self.tags
        self.metadata["version"] = version
        
        # If inspiration_ids provided, make sure they're also in source_ids
        for insp_id in self.inspiration_ids:
            if insp_id not in self.source_ids:
                self.source_ids.append(insp_id)
    
    def create_revision(self, new_content: str, embedding_service=None) -> 'Creation':
        """
        Create a new version/revision of this creation
        
        Args:
            new_content: Updated content for the new revision
            embedding_service: Optional service to generate embedding
            
        Returns:
            New Creation object representing the next version
        """
        # Generate new embedding if service provided
        new_embedding = None
        if embedding_service:
            new_embedding = embedding_service.get_embedding(new_content)
        
        # Create new creation with incremented version
        revision = Creation(
            content=new_content,
            embedding=new_embedding,
            metadata=self.metadata.copy(),
            title=self.title,
            creation_type=self.creation_type,
            tags=self.tags.copy(),
            version=self.version + 1,
            inspiration_ids=self.inspiration_ids.copy()
        )
        
        # Add this creation as a source for the revision
        revision.source_ids.append(self.id)
        
        return revision
    
    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the creation
        
        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
            self.metadata["tags"] = self.tags
            self.updated_at = datetime.now()
    
    def add_inspiration(self, memory_id: str) -> None:
        """
        Add a memory as inspiration for this creation
        
        Args:
            memory_id: ID of inspiring memory
        """
        if memory_id not in self.inspiration_ids:
            self.inspiration_ids.append(memory_id)
            self.source_ids.append(memory_id)
            self.updated_at = datetime.now()
    
    @classmethod
    def from_memory(cls, memory: Memory, title: str = "Converted Creation", creation_type: str = "general") -> 'Creation':
        """
        Create a Creation from a standard Memory
        
        Args:
            memory: Memory object to convert
            title: Title for the creation
            creation_type: Type of creation
            
        Returns:
            Creation object
        """
        metadata = memory.metadata.copy()
        
        # Extract creation-specific fields from metadata if present
        title = metadata.pop("title", title)
        creation_type = metadata.pop("creation_type", creation_type)
        tags = metadata.pop("tags", [])
        version = metadata.pop("version", 1)
        
        return cls(
            id=memory.id,
            content=memory.content,
            embedding=memory.embedding,
            metadata=metadata,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            title=title,
            creation_type=creation_type,
            tags=tags,
            version=version,
            inspiration_ids=memory.source_ids.copy()
        )