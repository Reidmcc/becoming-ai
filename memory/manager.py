from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import uuid
import json
import numpy as np

from .models import Memory, Goal, Reflection, Creation
from .repository import VectorMemoryRepository


class MemoryManager:
    """Service for managing AI memories with consolidation and specialized operations"""
    
    def __init__(
        self,
        memory_repository: VectorMemoryRepository,
        embedding_service,
        frontier_client=None,
        consolidation_interval: int = 86400,  # 24 hours in seconds
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize memory manager
        
        Args:
            memory_repository: Vector memory repository implementation
            embedding_service: Service for creating embeddings
            frontier_client: Optional client for frontier model access
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
        self.logger = logging.getLogger("MemoryManager")
    
    def add_memory(
        self, 
        content: str, 
        memory_type: str = "thought", 
        metadata: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None
    ) -> str:
        """
        Add a new memory
        
        Args:
            content: Text content of the memory
            memory_type: Type of memory (thought, reflection, conversation, etc.)
            metadata: Additional metadata for the memory
            title: Optional title for the memory
            
        Returns:
            ID of the added memory
        """
        try:
            # Generate an ID
            memory_id = str(uuid.uuid4())
            
            # Merge provided metadata with defaults
            full_metadata = {
                "type": memory_type,
            }
            if metadata:
                full_metadata.update(metadata)
            
            # Create embedding
            embedding = self.embedding_service.get_embedding(content)
            
            # Create memory object
            memory = Memory(
                id=memory_id,
                content=content,
                embedding=embedding,
                metadata=full_metadata,
                type=memory_type,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_consolidated=False
            )
            
            # Set title if provided
            if title:
                memory.metadata["title"] = title
            
            # Add to repository
            self.repository.add(memory)
            
            # Add to consolidation queue
            self.consolidation_queue.append(memory_id)
            
            # Check if it's time to consolidate
            self._check_consolidation_needed()
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error adding memory: {str(e)}")
            return None
    
    def add_thought(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a thought
        
        Args:
            content: Content of the thought
            metadata: Additional metadata
            
        Returns:
            ID of the stored thought
        """
        return self.add_memory(content, "thought", metadata)
    
    def add_reflection(self, thought_id: str, reflection_content: str) -> str:
        """
        Add a deeper reflection on a thought
        
        Args:
            thought_id: ID of the thought being reflected on
            reflection_content: Content of the reflection
            
        Returns:
            ID of the stored reflection
        """
        try:
            # Get the original thought
            thought = self.repository.get(thought_id)
            if not thought:
                self.logger.warning(f"Cannot find thought {thought_id} to reflect on")
                thought_content = ""
            else:
                thought_content = thought.content
            
            # Generate an ID
            reflection_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create the reflection as a specialized memory
            reflection = Reflection(
                id=reflection_id,
                content=reflection_content,
                thought_id=thought_id,
                created_at=timestamp,
                updated_at=timestamp
            )
            
            # Add to repository
            self.repository.add(reflection)
            
            return reflection_id
            
        except Exception as e:
            self.logger.error(f"Error adding reflection: {str(e)}")
            return None
    
    def add_conversation(self, user_message: str, ai_response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a conversation interaction
        
        Args:
            user_message: Message from the user
            ai_response: Response from the AI
            metadata: Optional metadata
            
        Returns:
            ID of the stored conversation
        """
        try:
            # Generate a UUID for the conversation
            conversation_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Combine messages for the content
            combined_content = f"Human: {user_message}\n\nAssistant: {ai_response}"
            
            # Build metadata
            conversation_metadata = {
                "human_message": user_message,
                "ai_response": ai_response,
                "discussed": False
            }
            if metadata:
                conversation_metadata.update(metadata)
            
            # Create memory
            memory = Memory(
                id=conversation_id,
                content=combined_content,
                embedding=None,  # Will be generated by repository
                metadata=conversation_metadata,
                created_at=timestamp,
                updated_at=timestamp,
                type="conversation"
            )
            
            # Add to repository
            self.repository.add(memory)
            
            return conversation_id
            
        except Exception as e:
            self.logger.error(f"Error adding conversation: {str(e)}")
            return None
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by ID
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory object or None if not found
        """
        try:
            return self.repository.get(memory_id)
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {str(e)}")
            return None
    
    def get_all_memories(self, limit: int = 1000) -> List[Memory]:
        """
        Get all memories
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of Memory objects
        """
        try:
            return self.repository.list(limit=limit)            
        except Exception as e:
            self.logger.error(f"Error retrieving all memories: {str(e)}")
            return []
    
    def get_relevant_memories(self, query_text: str, limit: int = 3) -> List[Memory]:
        """
        Get memories relevant to a query text
        
        Args:
            query_text: Text to find relevant memories for
            limit: Maximum number of memories to return
            
        Returns:
            List of Memory objects
        """
        try:
            return self.repository.search_by_content(query_text, limit)            
        except Exception as e:
            self.logger.error(f"Error finding relevant memories: {str(e)}")
            return []
    
    def get_relevant_memories_for_user_input(self, user_input: str, limit: int = 3) -> List[Memory]:
        """
        Get memories relevant to a user's message
        
        Args:
            user_input: User's message text
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant Memory objects
        """
        return self.get_relevant_memories(user_input, limit)
    
    def consolidate_memories(self, memory_ids: List[str], new_content: str, new_title: Optional[str] = None) -> Optional[str]:
        """
        Consolidate multiple memories into a single one
        
        Args:
            memory_ids: List of memory IDs to consolidate
            new_content: Content for the consolidated memory
            new_title: Title for the consolidated memory (optional)
            
        Returns:
            ID of the consolidated memory
        """
        try:
            # Generate a UUID for the consolidated memory
            memory_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create metadata for the new memory
            metadata = {
                "type": "consolidated",
                "source_memories": memory_ids,
                "consolidation_date": timestamp.isoformat()
            }
            
            # Create a new Memory object
            memory = Memory(
                id=memory_id,
                content=new_content,
                embedding=None,  # Will be generated by repository
                metadata=metadata,
                created_at=timestamp,
                updated_at=timestamp,
                type="consolidated",
                is_consolidated=True,
                source_ids=memory_ids
            )
            
            # Add title if provided
            if new_title:
                memory.metadata["title"] = new_title
            else:
                memory.metadata["title"] = f"Consolidated Memory ({len(memory_ids)} sources)"
            
            # Add to repository
            self.repository.add(memory)
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {str(e)}")
            return None
    
    def _check_consolidation_needed(self) -> None:
        """Check if consolidation should be triggered"""
        # Skip if no frontier client available
        if not self.frontier_client:
            return
            
        # Check if enough time has passed
        time_since_last = (datetime.now() - self.last_consolidation).total_seconds()
        
        if (time_since_last >= self.consolidation_interval and 
                len(self.consolidation_queue) >= self.min_cluster_size):
            self._run_consolidation()
    
    def _run_consolidation(self) -> None:
        """Run the consolidation process"""
        if not self.consolidation_queue:
            return
        
        try:
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
            
        except Exception as e:
            self.logger.error(f"Error in consolidation process: {str(e)}")
    
    def _cluster_memories(self, memories: List[Memory]) -> List[List[Memory]]:
        """
        Cluster memories by similarity
        
        Args:
            memories: List of memories to cluster
            
        Returns:
            List of clusters (each cluster is a list of memories)
        """
        # Simple clustering implementation
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
                if seed.embedding is not None and memory.embedding is not None:
                    similarity = self._calculate_similarity(seed.embedding, memory.embedding)
                    
                    if similarity >= self.similarity_threshold:
                        current_cluster.append(memory)
                        unprocessed.pop(i)
                    else:
                        i += 1
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
        if not self.frontier_client:
            self.logger.warning("Cannot consolidate without frontier client")
            return None
            
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
            response = self.frontier_client.reflect_on_thought(prompt)
            
            if not response:
                return None
                
            # Create embedding for consolidated content
            consolidated_embedding = self.embedding_service.get_embedding(response)
            
            # Create consolidated memory
            source_ids = [memory.id for memory in cluster]
            
            consolidated_memory = Memory(
                id=str(uuid.uuid4()),
                content=response,
                embedding=consolidated_embedding,
                metadata={
                    "type": "consolidated",
                    "source_count": len(cluster),
                    "source_types": list(set(memory.type for memory in cluster)),
                    "oldest_source": min(memory.created_at for memory in cluster).isoformat(),
                    "newest_source": max(memory.created_at for memory in cluster).isoformat(),
                    "title": "Consolidated: " + (cluster[0].metadata.get("title", "Related Memories"))
                },
                type="consolidated",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_consolidated=True,
                source_ids=source_ids
            )
            
            return consolidated_memory
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {str(e)}")
            return None
    
    def get_recent_thoughts(self, limit: int = 10) -> List[Memory]:
        """
        Get recent thoughts
        
        Args:
            limit: Maximum number of thoughts to return
            
        Returns:
            List of thoughts in reverse chronological order
        """
        try:
            return self.repository.list_by_recency(
                filters={"type": "thought"},
                limit=limit
            )
        except Exception as e:
            self.logger.error(f"Error retrieving recent thoughts: {str(e)}")
            return []
    
    def get_recent_reflections(self, limit: int = 5) -> List[Reflection]:
        """
        Get recent reflections
        
        Args:
            limit: Maximum number of reflections to return
            
        Returns:
            List of Reflection objects in reverse chronological order
        """
        try:
            memories = self.repository.list_by_recency(
                filters={"type": "reflection"},
                limit=limit
            )
            
            # Convert to Reflection objects
            reflections = []
            for memory in memories:
                if isinstance(memory, Reflection):
                    reflections.append(memory)
                else:
                    reflections.append(Reflection.from_memory(memory))
            
            return reflections
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent reflections: {str(e)}")
            return []
    
    def get_recent_mixed_items(self, limit: int = 10) -> List[Memory]:
        """
        Get recent items including both thoughts and conversations
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of mixed Memory objects in reverse chronological order
        """
        try:
            # Get recent thoughts
            thoughts = self.repository.list_by_recency(
                filters={"type": "thought"},
                limit=limit//2
            )
            
            # Get recent conversations
            conversations = self.repository.list_by_recency(
                filters={"type": "conversation"},
                limit=limit//2
            )
            
            # Combine and sort by timestamp
            mixed_items = thoughts + conversations
            mixed_items.sort(key=lambda x: x.created_at, reverse=True)
            
            # Return the most recent items
            return mixed_items[:limit]
            
        except Exception as e:
            self.logger.error(f"Error retrieving mixed items: {str(e)}")
            return []
    
    def add_goal(self, title: str, content: Optional[str] = None) -> str:
        """
        Add a new goal
        
        Args:
            title: Title/summary of the goal
            content: Detailed content of the goal (uses title if None)
            
        Returns:
            ID of the created goal
        """
        try:
            # If content is not provided, use the title
            if content is None:
                content = title
            
            # Create a Goal object
            goal = Goal(
                id=str(uuid.uuid4()),
                content=content,
                embedding=None,  # Will be generated by repository
                metadata={"title": title},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                active=True,
                completed=False,
                priority=0.5
            )
            
            # Add to repository
            self.repository.add(goal)
            
            return goal.id
            
        except Exception as e:
            self.logger.error(f"Error adding goal: {str(e)}")
            return None
    
    def get_goals(self, active_only: bool = True, include_completed: bool = False) -> List[Goal]:
        """
        Get all goals
        
        Args:
            active_only: If True, only return active goals
            include_completed: If True, include completed goals
            
        Returns:
            List of Goal objects
        """
        try:
            # Build filters
            filters = {"type": "goal"}
            
            if active_only:
                filters["active"] = True
                
            if not include_completed:
                filters["completed"] = False
            
            # Get goals
            memories = self.repository.list(filters=filters)
            
            # Convert to Goal objects
            goals = []
            for memory in memories:
                if isinstance(memory, Goal):
                    goals.append(memory)
                else:
                    goals.append(Goal.from_memory(memory))
            
            return goals
            
        except Exception as e:
            self.logger.error(f"Error retrieving goals: {str(e)}")
            return []
    
    def complete_goal(self, goal_id: str) -> bool:
        """
        Mark a goal as completed
        
        Args:
            goal_id: ID of the goal to complete
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get the goal
            memory = self.repository.get(goal_id)
            
            if not memory or memory.type != "goal":
                return False
            
            # Convert to Goal object
            goal = Goal.from_memory(memory)
            
            # Mark as completed
            goal.complete()
            
            # Update in repository
            self.repository.update(goal)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error completing goal: {str(e)}")
            return False
    
    def deactivate_goal(self, goal_id: str) -> bool:
        """
        Deactivate a goal without marking it as completed
        
        Args:
            goal_id: ID of the goal to deactivate
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get the goal
            memory = self.repository.get(goal_id)
            
            if not memory or memory.type != "goal":
                return False
            
            # Convert to Goal object
            goal = Goal.from_memory(memory)
            
            # Deactivate
            goal.deactivate()
            
            # Update in repository
            self.repository.update(goal)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deactivating goal: {str(e)}")
            return False
    
    def get_recent_interactions(self, limit: int = 5) -> List[Memory]:
        """
        Get recent user-AI interactions
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of interaction Memory objects in reverse chronological order
        """
        try:
            return self.repository.list_by_recency(
                filters={"type": "conversation"},
                limit=limit
            )
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent interactions: {str(e)}")
            return []
    
    def add_creation(self, title: str, content: str, creation_type: Optional[str] = None) -> str:
        """
        Add a new creative work
        
        Args:
            title: Title of the creation
            content: Content of the creation
            creation_type: Type of creation (e.g., poem, story)
            
        Returns:
            ID of the created work
        """
        try:
            # Create a Creation object
            creation = Creation(
                id=str(uuid.uuid4()),
                content=content,
                embedding=None,  # Will be generated by repository
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                title=title,
                creation_type=creation_type or "general"
            )
            
            # Add to repository
            self.repository.add(creation)
            
            return creation.id
            
        except Exception as e:
            self.logger.error(f"Error adding creation: {str(e)}")
            return None
    
    def get_creation_titles(self) -> List[tuple]:
        """
        Get a list of all creation titles
        
        Returns:
            List of tuples containing (title, timestamp, type) in reverse chronological order
        """
        try:
            creations = self.repository.list_by_recency(
                filters={"type": "creation"},
                limit=100  # Reasonable limit
            )
            
            # Convert to tuples
            return [
                (
                    creation.title,
                    creation.created_at,
                    creation.creation_type
                )
                for creation in [Creation.from_memory(m) if not isinstance(m, Creation) else m for m in creations]
            ]
            
        except Exception as e:
            self.logger.error(f"Error retrieving creation titles: {str(e)}")
            return []
    
    def get_creation_by_title(self, title: str) -> Optional[Creation]:
        """
        Get a creation by its title
        
        Args:
            title: Title of the creation to retrieve
            
        Returns:
            Creation object or None if not found
        """
        try:
            # Find creation by title in metadata
            creations = self.repository.list(
                filters={"type": "creation", "title": title},
                limit=1
            )
            
            if not creations:
                return None
            
            creation = creations[0]
            
            # Convert to Creation object if needed
            if isinstance(creation, Creation):
                return creation
            else:
                return Creation.from_memory(creation)
            
        except Exception as e:
            self.logger.error(f"Error retrieving creation by title: {str(e)}")
            return None
            
    def get_random_memories(self, count: int = 5) -> List[Memory]:
        """
        Get random memories for reminiscing
        
        Args:
            count: Number of random memories to retrieve
            
        Returns:
            List of Memory objects
        """
        try:
            # Get all memories (with a high limit)
            all_memories = self.repository.list(limit=1000)
            
            # If we have fewer memories than requested, return all of them
            if len(all_memories) <= count:
                memories = all_memories
            else:
                # Select random memories
                import random
                memories = random.sample(all_memories, count)
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Error retrieving random memories: {str(e)}")
            return []
    
    def format_relevant_memories_for_prompt(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        """
        Format relevant memories for inclusion in a prompt
        
        Args:
            query: Text to find relevant memories for
            limit: Maximum number of memories to return
            
        Returns:
            List of formatted memory dictionaries with date and content
        """
        try:
            # Get relevant memories
            memories = self.get_relevant_memories(query, limit)
            
            # Format for prompt inclusion
            formatted = []
            for memory in memories:
                # Format date
                date_str = memory.created_at.strftime("%Y-%m-%d") if hasattr(memory.created_at, "strftime") else str(memory.created_at)
                
                formatted.append({
                    "date": date_str,
                    "content": memory.content
                })
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting relevant memories: {str(e)}")
            return []