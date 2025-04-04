# memory/chroma_repository.py
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Union, Any
import numpy as np
import logging
from datetime import datetime
from repository import VectorMemoryRepository
from models import Memory

class ChromaDBMemoryRepository(VectorMemoryRepository):
    """ChromaDB implementation of VectorMemoryRepository"""
    
    def __init__(
        self, 
        collection_name: str = "memories",
        persist_directory: Optional[str] = None,
        embedding_function = None
    ):
        """
        Initialize ChromaDB repository
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist data to (None for in-memory)
            embedding_function: Function to use for creating embeddings
        """
        # Initialize logger
        self.logger = logging.getLogger("ChromaDBMemoryRepository")
        
        # Initialize ChromaDB client
        if persist_directory:
            self.logger.info(f"Creating persistent ChromaDB client at {persist_directory}")
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.logger.info("Creating in-memory ChromaDB client")
            self.client = chromadb.Client()
        
        # Store embedding function
        self.embedding_function = embedding_function
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            self.logger.info(f"Retrieved existing collection: {collection_name}")
        except ValueError:  # Collection doesn't exist
            self.logger.info(f"Creating new collection: {collection_name}")
            self.collection = self.client.create_collection(name=collection_name)
    
    def add(self, memory: Memory) -> str:
        """Add a memory to ChromaDB"""
        try:
            # Ensure we have an embedding
            embedding = memory.embedding
            if embedding is None and self.embedding_function:
                self.logger.debug(f"Generating embedding for memory: {memory.id}")
                embedding = self.embedding_function(memory.content)
            
            if embedding is None:
                raise ValueError("Memory must have an embedding or repository must have an embedding function")
            
            # Prepare metadata (ChromaDB has restrictions on metadata structure)
            metadata = memory.to_dict()
            # Remove content and ID from metadata since they're stored separately
            metadata.pop("content", None)
            metadata.pop("id", None)
            # Convert non-string/numeric values to strings to ensure ChromaDB compatibility
            for key, value in list(metadata.items()):
                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    metadata[key] = str(value)
            
            # Add to ChromaDB
            self.collection.add(
                ids=[memory.id],
                embeddings=[embedding],
                documents=[memory.content],
                metadatas=[metadata]
            )
            
            self.logger.debug(f"Added memory: {memory.id}")
            return memory.id
            
        except Exception as e:
            self.logger.error(f"Error adding memory: {str(e)}")
            raise
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID from ChromaDB"""
        try:
            result = self.collection.get(ids=[memory_id], include=["embeddings", "documents", "metadatas"])
            
            if not result["ids"]:
                self.logger.debug(f"Memory not found: {memory_id}")
                return None
            
            # Extract data
            content = result["documents"][0]
            embedding = result["embeddings"][0] if "embeddings" in result and result["embeddings"] else None
            metadata = result["metadatas"][0]
            
            # Add ID and content back to metadata for Memory construction
            metadata["id"] = memory_id
            metadata["content"] = content
            
            # Parse dates
            if "created_at" in metadata and isinstance(metadata["created_at"], str):
                metadata["created_at"] = datetime.fromisoformat(metadata["created_at"])
            if "updated_at" in metadata and isinstance(metadata["updated_at"], str):
                metadata["updated_at"] = datetime.fromisoformat(metadata["updated_at"])
            
            return Memory.from_dict(metadata, embedding)
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {str(e)}")
            return None
    
    def update(self, memory: Memory) -> bool:
        """Update an existing memory in ChromaDB"""
        if not self.exists(memory.id):
            self.logger.debug(f"Cannot update - memory does not exist: {memory.id}")
            return False
        
        try:
            # Update the timestamp
            memory.updated_at = datetime.now()
            
            # Prepare metadata
            metadata = memory.to_dict()
            metadata.pop("content", None)
            metadata.pop("id", None)
            
            # Convert non-string/numeric values to strings
            for key, value in list(metadata.items()):
                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    metadata[key] = str(value)
            
            # Update embedding if provided
            embedding = None
            if memory.embedding is not None:
                embedding = memory.embedding
            elif self.embedding_function:
                embedding = self.embedding_function(memory.content)
            
            # Update in ChromaDB
            if embedding:
                self.collection.update(
                    ids=[memory.id],
                    embeddings=[embedding],
                    documents=[memory.content],
                    metadatas=[metadata]
                )
            else:
                # Update without changing embedding
                self.collection.update(
                    ids=[memory.id],
                    documents=[memory.content],
                    metadatas=[metadata]
                )
            
            self.logger.debug(f"Updated memory: {memory.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating memory: {str(e)}")
            return False
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory from ChromaDB"""
        try:
            self.collection.delete(ids=[memory_id])
            self.logger.debug(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting memory: {str(e)}")
            return False
    
    def search_by_content(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """Search memories by content similarity using ChromaDB"""
        # Get embedding for query
        if self.embedding_function is None:
            raise ValueError("Repository must have an embedding function to search by content")
        
        embedding = self.embedding_function(query)
        
        # Use the embedding to search
        return self.search_by_vector(embedding, limit, filters)
    
    def search_by_vector(self, vector: List[float], limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Memory]:
        """Search memories by vector similarity using ChromaDB"""
        try:
            # Prepare where clause if filters are provided
            where = None
            if filters:
                where = {}
                for key, value in filters.items():
                    where[key] = value
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            # Convert results to Memory objects
            memories = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, memory_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    content = results["documents"][0][i]
                    embedding = results["embeddings"][0][i] if "embeddings" in results and results["embeddings"] else None
                    
                    # Add ID and content to metadata for Memory construction
                    metadata["id"] = memory_id
                    metadata["content"] = content
                    
                    # Parse dates
                    if "created_at" in metadata and isinstance(metadata["created_at"], str):
                        metadata["created_at"] = datetime.fromisoformat(metadata["created_at"])
                    if "updated_at" in metadata and isinstance(metadata["updated_at"], str):
                        metadata["updated_at"] = datetime.fromisoformat(metadata["updated_at"])
                    
                    memory = Memory.from_dict(metadata, embedding)
                    memories.append(memory)
            
            self.logger.debug(f"Search returned {len(memories)} results")
            return memories
            
        except Exception as e:
            self.logger.error(f"Error searching by vector: {str(e)}")
            return []
    
    def list(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100, offset: int = 0) -> List[Memory]:
        """List memories with optional filtering using ChromaDB"""
        try:
            # Prepare where clause if filters are provided
            where = None
            if filters:
                where = {}
                for key, value in filters.items():
                    where[key] = value
            
            # Query ChromaDB (note: ChromaDB doesn't natively support offset pagination)
            # For large collections, this could be inefficient
            results = self.collection.get(
                where=where,
                limit=limit + offset,
                include=["documents", "metadatas", "embeddings"]
            )
            
            # Apply offset manually
            if offset > 0 and results["ids"]:
                results["ids"] = results["ids"][offset:]
                results["documents"] = results["documents"][offset:]
                results["metadatas"] = results["metadatas"][offset:]
                if "embeddings" in results and results["embeddings"]:
                    results["embeddings"] = results["embeddings"][offset:]
            
            # Convert results to Memory objects
            memories = []
            if results["ids"]:
                for i, memory_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    content = results["documents"][i]
                    embedding = results["embeddings"][i] if "embeddings" in results and results["embeddings"] else None
                    
                    # Add ID and content to metadata for Memory construction
                    metadata["id"] = memory_id
                    metadata["content"] = content
                    
                    # Parse dates
                    if "created_at" in metadata and isinstance(metadata["created_at"], str):
                        metadata["created_at"] = datetime.fromisoformat(metadata["created_at"])
                    if "updated_at" in metadata and isinstance(metadata["updated_at"], str):
                        metadata["updated_at"] = datetime.fromisoformat(metadata["updated_at"])
                    
                    memory = Memory.from_dict(metadata, embedding)
                    memories.append(memory)
            
            self.logger.debug(f"List returned {len(memories)} memories")
            return memories
            
        except Exception as e:
            self.logger.error(f"Error listing memories: {str(e)}")
            return []
    
    def exists(self, memory_id: str) -> bool:
        """Check if a memory exists in ChromaDB"""
        try:
            result = self.collection.get(ids=[memory_id], include=[])
            return bool(result["ids"])
        except Exception as e:
            self.logger.error(f"Error checking if memory exists: {str(e)}")
            return False
    
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count memories with optional filtering using ChromaDB"""
        try:
            # Prepare where clause if filters are provided
            where = None
            if filters:
                where = {}
                for key, value in filters.items():
                    where[key] = value
            
            # ChromaDB doesn't have a direct count method, so we need to get IDs only
            result = self.collection.get(where=where, include=[])
            return len(result["ids"]) if result["ids"] else 0
        except Exception as e:
            self.logger.error(f"Error counting memories: {str(e)}")
            return 0
            
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
        try:
            # Get more results than needed to account for filtering
            results = self.search_by_vector(vector, limit * 2, filters)
            
            # Filter out excluded IDs
            filtered_results = [m for m in results if m.id not in exclude_ids]
            
            # Return only up to the limit
            return filtered_results[:limit]
        except Exception as e:
            self.logger.error(f"Error in search_by_vector_excluding: {str(e)}")
            return []