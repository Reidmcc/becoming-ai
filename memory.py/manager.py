from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
# Also import your Memory class from wherever it's defined
from models import Memory  # Adjust path as needed
from repository import VectorMemoryRepository
import numpy as np

class MemoryManager:
    """Service for managing AI memories with consolidation"""
    
    def __init__(
        self,
        memory_repository: VectorMemoryRepository,
        embedding_service,
        frontier_client,
        consolidation_interval: int = 86400,  # 24 hours in seconds
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize memory manager
        
        Args:
            memory_repository: Vector memory repository implementation
            embedding_service: Service for creating embeddings
            frontier_client: Client for frontier model access
            consolidation_interval: Seconds between consolidation operations
            min_cluster_size: Minimum memories for a cluster to be consolidated
            similarity_threshold: Threshold for clustering similar memories
        """
        self.repository = memory_repository
        self.embedding_service = embedding_service
        self.frontier_client = frontier_client
        self.consolidation_interval = consolidation_interval
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.last_consolidation = datetime.now()
        self.consolidation_queue = []
    
    def add_memory(
        self, 
        content: str, 
        memory_type: str = "thought", 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new memory
        
        Args:
            content: Text content of the memory
            memory_type: Type of memory (thought, reflection, etc.)
            metadata: Additional metadata for the memory
            
        Returns:
            ID of the added memory
        """
        # Create embedding
        embedding = self.embedding_service.get_embedding(content)
        
        # Create memory object
        memory = Memory(
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            type=memory_type,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_consolidated=False
        )
        
        # Add to repository
        memory_id = self.repository.add(memory)
        
        # Add to consolidation queue
        self.consolidation_queue.append(memory_id)
        
        # Check if it's time to consolidate
        self._check_consolidation_needed()
        
        return memory_id
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID"""
        return self.repository.get(memory_id)
    
    def find_relevant_memories(self, query: str, limit: int = 5, include_consolidated: bool = True) -> List[Memory]:
        """
        Find memories relevant to a query
        
        Args:
            query: Text query to find relevant memories for
            limit: Maximum number of memories to return
            include_consolidated: Whether to include consolidated memories
            
        Returns:
            List of relevant memories
        """
        filters = None
        if not include_consolidated:
            filters = {"is_consolidated": False}
        
        return self.repository.search_by_content(query, limit, filters)
    
    def _check_consolidation_needed(self) -> None:
        """Check if consolidation should be triggered"""
        # Check if enough time has passed
        time_since_last = (datetime.now() - self.last_consolidation).total_seconds()
        
        if (time_since_last >= self.consolidation_interval and 
                len(self.consolidation_queue) >= self.min_cluster_size):
            self._run_consolidation()
    
    def _run_consolidation(self) -> None:
        """Run the consolidation process"""
        if not self.consolidation_queue:
            return
        
        # Get memories from queue
        memory_ids = self.consolidation_queue.copy()
        self.consolidation_queue = []
        
        # Fetch full memory objects
        memories = [self.repository.get(memory_id) for memory_id in memory_ids]
        memories = [m for m in memories if m is not None]
        
        if not memories:
            return
        
        # Cluster memories by similarity
        clusters = self._cluster_memories(memories)
        
        # Process each cluster
        for cluster in clusters:
            if len(cluster) >= self.min_cluster_size:
                # Consolidate the cluster
                consolidated_memory = self._consolidate_cluster(cluster)
                if consolidated_memory:
                    # Save consolidated memory
                    self.repository.add(consolidated_memory)
                    
                    # Update original memories to mark as consolidated
                    for memory in cluster:
                        memory.is_consolidated = True
                        memory.metadata["consolidated_into"] = consolidated_memory.id
                        self.repository.update(memory)
        
        self.last_consolidation = datetime.now()
    
    def _cluster_memories(self, memories: List[Memory]) -> List[List[Memory]]:
        """
        Cluster memories by similarity
        
        Args:
            memories: List of memories to cluster
            
        Returns:
            List of clusters (each cluster is a list of memories)
        """
        # Simple clustering implementation - can be enhanced later
        clusters = []
        unprocessed = memories.copy()
        
        while unprocessed:
            # Take the first memory as the seed for a new cluster
            seed = unprocessed.pop(0)
            current_cluster = [seed]
            
            # Find similar memories
            i = 0
            while i < len(unprocessed):
                memory = unprocessed[i]
                similarity = self._calculate_similarity(seed.embedding, memory.embedding)
                
                if similarity >= self.similarity_threshold:
                    current_cluster.append(memory)
                    unprocessed.pop(i)
                else:
                    i += 1
            
            clusters.append(current_cluster)
        
        return clusters
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _consolidate_cluster(self, cluster: List[Memory]) -> Optional[Memory]:
        """
        Consolidate a cluster of memories
        
        Args:
            cluster: List of related memories to consolidate
            
        Returns:
            Consolidated memory or None if consolidation failed
        """
        try:
            # Format memories for the prompt
            memory_texts = [f"Memory {i+1}: {memory.content}" for i, memory in enumerate(cluster)]
            memory_content = "\n\n".join(memory_texts)
            
            # Create consolidation prompt
            prompt = f"""
            The following are related memories. Please:
            1. Identify common patterns and themes
            2. Create a consolidated representation that preserves important details
            3. Extract any key concepts or relationships
            
            Memories:
            {memory_content}
            
            Please respond with a consolidated memory that captures the essence of these related memories.
            """
            
            # Get consolidated memory from frontier model
            response = self.frontier_client.generate(prompt)
            
            # Create embedding for consolidated content
            consolidated_embedding = self.embedding_service.get_embedding(response)
            
            # Create consolidated memory
            source_ids = [memory.id for memory in cluster]
            
            consolidated_memory = Memory(
                content=response,
                embedding=consolidated_embedding,
                metadata={
                    "source_count": len(cluster),
                    "source_types": list(set(memory.type for memory in cluster)),
                    "oldest_source": min(memory.created_at for memory in cluster).isoformat(),
                    "newest_source": max(memory.created_at for memory in cluster).isoformat()
                },
                type="consolidated",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_consolidated=True,
                source_ids=source_ids
            )
            
            return consolidated_memory
            
        except Exception as e:
            print(f"Error consolidating memories: {e}")
            return None
        
    def get_recent_memories(
        self, 
        memory_type: Optional[str] = None,
        limit: int = 10, 
        max_age_seconds: Optional[int] = None
    ) -> List[Memory]:
        """
        Get recent memories based on timestamp
        
        Args:
            memory_type: Filter by memory type (thought, reflection, etc.) or None for all types
            limit: Maximum number of memories to return
            max_age_seconds: Only return memories newer than this many seconds (None for no limit)
            
        Returns:
            List of recent Memory objects
        """
        filters = {}
        
        # Apply type filter if specified
        if memory_type:
            filters["type"] = memory_type
        
        # Apply timestamp filter if specified
        if max_age_seconds:
            cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)
            filters["created_at_after"] = cutoff_time
        
        # Use a dedicated repository method for timestamp-based retrieval
        return self.repository.list_by_recency(
            filters=filters,
            limit=limit
        )
    
    def find_memories_by_similarity(
        self, 
        query: str, 
        limit: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> List[Memory]:
        """
        Find memories by content similarity
        
        Args:
            query: Text query to find similar memories
            limit: Maximum number of memories to return
            filters: Optional additional filters (type, etc.)
            exclude_ids: Optional list of memory IDs to exclude
            
        Returns:
            List of relevant Memory objects
        """
        # Get embedding for query
        query_embedding = self.embedding_service.get_embedding(query)
        
        # Apply additional filters
        combined_filters = filters or {}
        
        # Find similar memories
        return self.repository.search_by_vector(
            vector=query_embedding,
            limit=limit,
            filters=combined_filters,
            exclude_ids=exclude_ids
        )
    
    def get_memories_for_context(
        self, 
        query: Optional[str] = None,
        limit_total: int = 10
    ) -> List[Memory]:
        """
        Get a balanced set of memories for context generation
        Combines recent memories with query-relevant memories
        
        Args:
            query: Optional context query to find relevant memories 
            limit_total: Maximum total memories to return
            
        Returns:
            List of Memory objects for context
        """
        # Allocate limits for recent vs. relevant memories
        limit_recent = limit_total // 2
        limit_relevant = limit_total - limit_recent
        
        # Get recent memories first (from last 24 hours)
        recent_memories = self.get_recent_memories(
            limit=limit_recent,
            max_age_seconds=86400  # Last 24 hours
        )
        
        # If no query, just return recent memories
        if not query or not limit_relevant:
            return recent_memories
        
        # Get memory IDs to exclude (don't duplicate recent memories)
        exclude_ids = [memory.id for memory in recent_memories]
        
        # Get query-relevant memories
        relevant_memories = self.find_memories_by_similarity(
            query=query,
            limit=limit_relevant,
            exclude_ids=exclude_ids
        )
        
        # Combine both types (recent first, then relevant)
        return recent_memories + relevant_memories