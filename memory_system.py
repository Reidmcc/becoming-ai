# memory_system.py
import os
import json
import uuid
import logging
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, String, Integer, Boolean, Text, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

# Define the base class for SQLAlchemy models
Base = declarative_base()

# Define the SQLAlchemy models
class Thought(Base):
    __tablename__ = 'thoughts'
    
    id = Column(String(36), primary_key=True)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    thought_type = Column(String(50), nullable=False)
    
    # Relationships
    reflections = relationship("Reflection", back_populates="thought")

class Memory(Base):
    __tablename__ = 'memories'
    
    id = Column(String(36), primary_key=True)
    content = Column(Text, nullable=False)
    title = Column(String(255))
    timestamp = Column(DateTime, nullable=False)
    source_thoughts = Column(Text)  # JSON array of thought IDs
    metadata = Column(Text)  # JSON object for additional metadata

class Reflection(Base):
    __tablename__ = 'reflections'
    
    id = Column(String(36), primary_key=True)
    thought_id = Column(String(36), ForeignKey('thoughts.id'), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    # Relationships
    thought = relationship("Thought", back_populates="reflections")

class Interaction(Base):
    __tablename__ = 'interactions'
    
    id = Column(String(36), primary_key=True)
    human_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    discussed = Column(Boolean, default=False)
    metadata = Column(Text)  # JSON object for additional metadata

class Creation(Base):
    __tablename__ = 'creations'
    
    id = Column(String(36), primary_key=True)
    title = Column(String(255), nullable=False, unique=True)  # Title is unique for easy lookup
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    creation_type = Column(String(50))  # E.g., "poem", "story", "essay", etc.
    metadata = Column(Text)  # JSON object for additional metadata

class MemorySystem:
    def __init__(self, config=None):
        """Initialize the memory system
        
        Args:
            config: Configuration dict with parameters
        """
        self.config = config or {}
        self.short_term_limit = self.config.get('short_term_limit', 100)
        self.recent_thoughts_cache = []
        self.logger = logging.getLogger("MemorySystem")
        
        # Initialize the database
        self._initialize_db()
        
        # Initialize vector embeddings
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_db(self):
        """Initialize the database using SQLAlchemy"""
        try:
            # Determine the connection string based on configuration
            self.use_remote_db = self.config.get('use_remote_db', False)
            
            if self.use_remote_db:
                self.db_host = self.config.get('db_host', 'localhost')
                self.db_port = self.config.get('db_port', 3306)
                self.db_name = self.config.get('db_name', 'becoming_ai')
                self.db_user = self.config.get('db_user', 'becoming_ai')
                self.db_password = self.config.get('db_password', '')
                
                connection_string = f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            else:
                self.db_path = self.config.get('db_path', 'data/memories.db')
                # Ensure database directory exists
                os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
                
                connection_string = f"sqlite:///{self.db_path}"
            
            # Create the engine
            self.engine = create_engine(connection_string)
            
            # Create all tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            # Create a session factory
            self.Session = sessionmaker(bind=self.engine)
            
            self.logger.info(f"Database initialized with {connection_string}")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            # If remote DB fails, fall back to local SQLite
            if self.use_remote_db:
                self.logger.info("Falling back to local SQLite database")
                self.use_remote_db = False
                return self._initialize_db()
    
    def _initialize_vector_store(self):
        """Initialize the vector store for semantic search"""
        self.logger.info("Initializing vector store for memory retrieval")
        try:
            from sentence_transformers import SentenceTransformer
            
            # This will download the model if not already present
            model = SentenceTransformer('all-mpnet-base-v2')
            
            self.logger.info("Vector embedding model loaded successfully")
            
            # Load existing vectors if available
            vectors = {}
            vector_path = self.config.get('vector_store_path', 'data/vectors.pkl')
            if os.path.exists(vector_path):
                try:
                    with open(vector_path, 'rb') as f:
                        vectors = pickle.load(f)
                    self.logger.info(f"Loaded {len(vectors)} vectors from {vector_path}")
                except Exception as e:
                    self.logger.error(f"Error loading vectors: {str(e)}")
            
            return {
                "model": model,
                "vectors": vectors
            }
        except ImportError:
            self.logger.warning("sentence-transformers not installed. Please install it with:")
            self.logger.warning("pip install sentence-transformers")
            return None
    
    def add_thought(self, thought):
        """Add a new thought to the memory system
        
        Args:
            thought: Dict with id, content, timestamp, type, and sequence
            
        Returns:
            ID of the stored thought
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Create a new Thought object
            db_thought = Thought(
                id=thought["id"],
                content=thought["content"],
                timestamp=thought["timestamp"],
                thought_type=thought["type"],
            )
            
            # Add and commit
            session.add(db_thought)
            session.commit()
            
            # Add to recent thoughts cache
            self.recent_thoughts_cache.append(thought)
            if len(self.recent_thoughts_cache) > self.short_term_limit:
                self.recent_thoughts_cache.pop(0)
            
            # Add to vector store if available
            if self.vector_store:
                vector = self.vector_store["model"].encode(thought["content"])
                self.vector_store["vectors"][thought["id"]] = vector
            
            return thought["id"]
        
        except Exception as e:
            self.logger.error(f"Error adding thought: {str(e)}")
            if session:
                session.rollback()
            return None
        
        finally:
            if session:
                session.close()
    
    def add_reflection(self, thought_id, reflection_content):
        """Add a deeper reflection on a thought
        
        Args:
            thought_id: ID of the thought being reflected on
            reflection_content: Content of the reflection
            
        Returns:
            ID of the stored reflection
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Generate a UUID for the reflection
            reflection_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create a new Reflection object
            reflection = Reflection(
                id=reflection_id,
                thought_id=thought_id,
                content=reflection_content,
                timestamp=timestamp
            )
            
            # Add and commit
            session.add(reflection)
            session.commit()
            
            return reflection_id
        
        except Exception as e:
            self.logger.error(f"Error adding reflection: {str(e)}")
            if session:
                session.rollback()
            return None
        
        finally:
            if session:
                session.close()
    
    def add_conversation(self, user_message, ai_response, metadata=None):
        """Store a conversation interaction in the memory system
        
        Args:
            user_message: Message from the user
            ai_response: Response from the AI
            metadata: Optional metadata
            
        Returns:
            ID of the stored conversation
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Generate a UUID for the conversation
            conversation_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create a new Interaction object
            interaction = Interaction(
                id=conversation_id,
                human_message=user_message,
                ai_response=ai_response,
                timestamp=timestamp,
                discussed=False,
                metadata=json.dumps(metadata) if metadata else None
            )
            
            # Add and commit
            session.add(interaction)
            session.commit()
            
            # Add to vector store if available
            if self.vector_store:
                combined_text = f"Human: {user_message} Me: {ai_response}"
                vector = self.vector_store["model"].encode(combined_text)
                self.vector_store["vectors"][conversation_id] = vector
            
            return conversation_id
        
        except Exception as e:
            self.logger.error(f"Error adding conversation: {str(e)}")
            if session:
                session.rollback()
            return None
        
        finally:
            if session:
                session.close()
    
    def get_recent_thoughts(self, limit=10):
        """Get recent thoughts
        
        Args:
            limit: Maximum number of thoughts to return
            
        Returns:
            List of thought dictionaries in reverse chronological order
        """
        session = None
        try:
            # If we have enough in the cache, return from there
            if len(self.recent_thoughts_cache) >= limit:
                return sorted(self.recent_thoughts_cache, key=lambda x: x["timestamp"], reverse=True)[:limit]
            
            # Create a new session
            session = self.Session()
            
            # Query for recent thoughts
            thoughts_query = session.query(Thought).order_by(Thought.timestamp.desc()).limit(limit)
            
            # Convert to dictionaries
            thoughts = []
            for thought in thoughts_query:
                thoughts.append({
                    "id": thought.id,
                    "content": thought.content,
                    "timestamp": thought.timestamp,
                    "type": thought.type,
                    "sequence": thought.sequence
                })
            
            return thoughts
        
        except Exception as e:
            self.logger.error(f"Error retrieving recent thoughts: {str(e)}")
            return []
        
        finally:
            if session:
                session.close()
    
    def get_recent_reflections(self, limit=5):
        """Get recent reflections
        
        Args:
            limit: Maximum number of reflections to return
            
        Returns:
            List of reflection dictionaries in reverse chronological order
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Query for recent reflections including the associated thought
            reflections_query = session.query(Reflection, Thought).join(Thought).order_by(Reflection.timestamp.desc()).limit(limit)
            
            # Convert to dictionaries
            reflections = []
            for reflection, thought in reflections_query:
                reflections.append({
                    "id": reflection.id,
                    "thought_id": reflection.thought_id,
                    "content": reflection.content,
                    "timestamp": reflection.timestamp,
                    "thought_content": thought.content
                })
            
            return reflections
        
        except Exception as e:
            self.logger.error(f"Error retrieving recent reflections: {str(e)}")
            return []
        
        finally:
            if session:
                session.close()
    
    def get_recent_mixed_items(self, limit=10):
        """Get recent items including both thoughts and conversations
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of mixed items (thoughts and conversations) in chronological order
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Get recent thoughts
            thoughts_query = session.query(Thought).order_by(Thought.timestamp.desc()).limit(limit)
            recent_thoughts = []
            for thought in thoughts_query:
                recent_thoughts.append({
                    "id": thought.id,
                    "type": "thought",
                    "content": thought.content,
                    "timestamp": thought.timestamp,
                    "sequence": thought.sequence
                })
            
            # Get recent conversations
            interactions_query = session.query(Interaction).order_by(Interaction.timestamp.desc()).limit(limit)
            recent_conversations = []
            for interaction in interactions_query:
                recent_conversations.append({
                    "id": interaction.id,
                    "type": "conversation",
                    "user_message": interaction.human_message,
                    "ai_response": interaction.ai_response,
                    "timestamp": interaction.timestamp,
                    "metadata": json.loads(interaction.metadata) if interaction.metadata else {}
                })
            
            # Combine and sort by timestamp
            mixed_items = recent_thoughts + recent_conversations
            mixed_items.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Return the most recent items
            return mixed_items[:limit]
        
        except Exception as e:
            self.logger.error(f"Error retrieving mixed items: {str(e)}")
            return []
        
        finally:
            if session:
                session.close()
    
    def create_memory(self, content, title=None, metadata=None):
        """Create a new memory
        
        Args:
            content: Content of the memory
            title: Title of the memory (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            ID of the created memory
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Generate a UUID for the memory
            memory_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create a new Memory object
            memory = Memory(
                id=memory_id,
                content=content,
                title=title,
                timestamp=timestamp,
                source_thoughts=None,  # No source thoughts by default
                metadata=json.dumps(metadata) if metadata else None
            )
            
            # Add and commit
            session.add(memory)
            session.commit()
            
            # Add to vector store if available
            if self.vector_store:
                vector = self.vector_store["model"].encode(content)
                self.vector_store["vectors"][memory_id] = vector
            
            return memory_id
        
        except Exception as e:
            self.logger.error(f"Error creating memory: {str(e)}")
            if session:
                session.rollback()
            return None
        
        finally:
            if session:
                session.close()
    
    def get_memory(self, memory_id):
        """Get a memory by ID
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory dictionary or None if not found
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Query for the memory
            memory = session.query(Memory).filter(Memory.id == memory_id).first()
            
            if not memory:
                return None
            
            # Convert to dictionary
            return {
                "id": memory.id,
                "content": memory.content,
                "title": memory.title,
                "timestamp": memory.timestamp,
                "source_thoughts": json.loads(memory.source_thoughts) if memory.source_thoughts else [],
                "metadata": json.loads(memory.metadata) if memory.metadata else {}
            }
        
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {str(e)}")
            return None
        
        finally:
            if session:
                session.close()
    
    def _get_all_memories(self):
        """Get all memories
        
        Returns:
            List of memory dictionaries
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Query for all memories
            memories_query = session.query(Memory).order_by(Memory.timestamp.desc())
            
            # Convert to dictionaries
            memories = []
            for memory in memories_query:
                memories.append({
                    "id": memory.id,
                    "content": memory.content,
                    "title": memory.title,
                    "timestamp": memory.timestamp,
                    "source_thoughts": json.loads(memory.source_thoughts) if memory.source_thoughts else [],
                    "metadata": json.loads(memory.metadata) if memory.metadata else {}
                })
            
            return memories
        
        except Exception as e:
            self.logger.error(f"Error retrieving all memories: {str(e)}")
            return []
        
        finally:
            if session:
                session.close()
    
    def get_relevant_memories(self, query_text, limit=3):
        """Get memories relevant to a query text
        
        Args:
            query_text: Text to find relevant memories for
            limit: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries
        """
        if not self.vector_store:
            # Fall back to keyword search if vector store is not available
            return self._get_memories_by_keywords(query_text, limit)
        
        try:
            # Create query vector
            query_vector = self.vector_store["model"].encode(query_text)
            
            # Get all memories
            all_memories = self._get_all_memories()
            
            # Score memories by similarity
            scored_memories = []
            for memory in all_memories:
                memory_id = memory["id"]
                
                # Get or create vector for this memory
                if memory_id in self.vector_store["vectors"]:
                    memory_vector = self.vector_store["vectors"][memory_id]
                else:
                    memory_vector = self.vector_store["model"].encode(memory["content"])
                    self.vector_store["vectors"][memory_id] = memory_vector
                
                # Calculate similarity
                similarity = self._cosine_similarity(query_vector, memory_vector)
                
                scored_memories.append((memory, similarity))
            
            # Sort by similarity score
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            
            # Return top matches
            return [memory for memory, score in scored_memories[:limit]]
        
        except Exception as e:
            self.logger.error(f"Error finding relevant memories: {str(e)}")
            return []
    
    def get_relevant_memories_for_user_input(self, user_input, limit=3):
        """Get memories relevant to a user's message
        
        Args:
            user_input: User's message text
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant memory dictionaries
        """
        # Use the existing get_relevant_memories method but pass the user input
        return self.get_relevant_memories(user_input, limit)
    
    def consolidate_memories(self, memory_ids, new_content, new_title=None):
        """Consolidate multiple memories into a single one
        
        Args:
            memory_ids: List of memory IDs to consolidate
            new_content: Content for the consolidated memory
            new_title: Title for the consolidated memory (optional)
            
        Returns:
            ID of the consolidated memory
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
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
                title=new_title or f"Consolidated Memory ({len(memory_ids)} sources)",
                timestamp=timestamp,
                source_thoughts=json.dumps(memory_ids),
                metadata=json.dumps(metadata)
            )
            
            # Add and commit
            session.add(memory)
            session.commit()
            
            # Add to vector store if available
            if self.vector_store:
                vector = self.vector_store["model"].encode(new_content)
                self.vector_store["vectors"][memory_id] = vector
            
            return memory_id
        
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {str(e)}")
            if session:
                session.rollback()
            return None
        
        finally:
            if session:
                session.close()
    
    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Similarity score between 0 and 1
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def save_vector_store(self, filepath=None):
        """Save vector embeddings to disk
        
        Args:
            filepath: Path to save vectors to (defaults to config setting or 'data/vectors.pkl')
        
        Returns:
            Boolean indicating success
        """
        if not self.vector_store:
            self.logger.warning("No vector store to save")
            return False
            
        try:
            # Use configured path or default
            if filepath is None:
                filepath = self.config.get('vector_store_path', 'data/vectors.pkl')
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Extract just the vectors (not the model) to save
            vectors_to_save = self.vector_store["vectors"]
            
            self.logger.info(f"Saving {len(vectors_to_save)} vectors to {filepath}")
            
            with open(filepath, 'wb') as f:
                pickle.dump(vectors_to_save, f)
                
            self.logger.info("Vector store saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")
            return False
            
    def load_vector_store(self, filepath=None):
        """Load vector embeddings from disk
        
        Args:
            filepath: Path to load vectors from (defaults to config setting or 'data/vectors.pkl')
        
        Returns:
            Boolean indicating success
        """
        if not self.vector_store:
            self.logger.warning("Vector store not initialized, can't load vectors")
            return False
            
        try:
            # Use configured path or default
            if filepath is None:
                filepath = self.config.get('vector_store_path', 'data/vectors.pkl')
                
            # Check if file exists
            if not os.path.exists(filepath):
                self.logger.info(f"No saved vectors found at {filepath}")
                return False
                
            self.logger.info(f"Loading vectors from {filepath}")
            
            with open(filepath, 'rb') as f:
                loaded_vectors = pickle.load(f)
                
            # Merge with existing vectors (if any)
            self.vector_store["vectors"].update(loaded_vectors)
            
            self.logger.info(f"Loaded {len(loaded_vectors)} vectors successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            return False
        
    def add_creation(self, title, content, creation_type=None, metadata=None):
        """Add a new creative work
        
        Args:
            title: Title of the creation
            content: Content of the creation
            creation_type: Type of creation (e.g., poem, story) (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            ID of the created work
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Generate a UUID for the creation
            creation_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create a new Creation object
            creation = Creation(
                id=creation_id,
                title=title,
                content=content,
                timestamp=timestamp,
                type=creation_type,
                metadata=json.dumps(metadata) if metadata else None
            )
            
            # Add and commit
            session.add(creation)
            session.commit()
            
            # Add to vector store if available
            if self.vector_store:
                # Create a combined text for vector embedding
                combined_text = f"{title}: {content}"
                vector = self.vector_store["model"].encode(combined_text)
                self.vector_store["vectors"][creation_id] = vector
            
            return creation_id
        
        except Exception as e:
            self.logger.error(f"Error adding creation: {str(e)}")
            if session:
                session.rollback()
            return None
        
        finally:
            if session:
                session.close()
    
    def get_creation_titles(self):
        """Get a list of all creation titles
        
        Returns:
            List of tuples containing (title, timestamp) in chronological order (newest first)
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Query for all creation titles with timestamps
            titles_query = session.query(Creation.title, Creation.timestamp, Creation.type).order_by(Creation.timestamp.desc())
            
            # Convert to list of title information
            title_info = [(row.title, row.timestamp, row.type) for row in titles_query]
            
            return title_info
        
        except Exception as e:
            self.logger.error(f"Error retrieving creation titles: {str(e)}")
            return []
        
        finally:
            if session:
                session.close()
    
    def get_creation_by_title(self, title):
        """Get a creation by its title
        
        Args:
            title: Title of the creation to retrieve
            
        Returns:
            Creation dictionary or None if not found
        """
        session = None
        try:
            # Create a new session
            session = self.Session()
            
            # Query for the creation by title
            creation = session.query(Creation).filter(Creation.title == title).first()
            
            if not creation:
                return None
            
            # Convert to dictionary
            return {
                "id": creation.id,
                "title": creation.title,
                "content": creation.content,
                "timestamp": creation.timestamp,
                "type": creation.type,
                "metadata": json.loads(creation.metadata) if creation.metadata else {}
            }
        
        except Exception as e:
            self.logger.error(f"Error retrieving creation by title: {str(e)}")
            return None
        
        finally:
            if session:
                session.close()