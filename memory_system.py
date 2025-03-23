# memory_system.py
import sqlite3
import json
import uuid
import logging
import math
from datetime import datetime
import os
import pickle
import numpy as np

class MemorySystem:
    def __init__(self, config=None):
        """Initialize the memory system
        
        Args:
            config: Configuration dict with parameters
        """
        self.config = config or {}
        self.short_term_limit = self.config.get('short_term_limit', 100)
        # load most recent thoughts from the datbase up to short_term_limit
        self.recent_thoughts_cache = []
        self.logger = logging.getLogger("MemorySystem")
        
        # Database configuration
        self.use_remote_db = self.config.get('use_remote_db', False)
        self.db_type = "remote" if self.use_remote_db else "local"
        
        if self.use_remote_db:
            self.db_host = self.config.get('db_host', 'localhost')
            self.db_port = self.config.get('db_port', 3306)
            self.db_name = self.config.get('db_name', 'becoming_ai')
            self.db_user = self.config.get('db_user', 'becoming_ai')
            self.db_password = self.config.get('db_password', '')
        else:
            self.db_path = self.config.get('db_path', 'data/memories.db')
            # Ensure database directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        # Initialize vector embeddings
        self.vector_store = self._initialize_vector_store()
            
        # Initialize database
        self.db = self._initialize_db()
        
    def _initialize_db(self):
        """Initialize the database"""
        if self.use_remote_db:
            return self._initialize_remote_db()
        else:
            return self._initialize_local_db()
    
    def _initialize_local_db(self):
        """Initialize SQLite database"""
        self.logger.info(f"Initializing local SQLite database at {self.db_path}")
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS thoughts (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            sequence INTEGER NOT NULL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            title TEXT,
            timestamp TEXT NOT NULL,
            source_thoughts TEXT,  -- JSON array of thought IDs
            metadata TEXT  -- JSON object for additional metadata
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reflections (
            id TEXT PRIMARY KEY,
            thought_id TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (thought_id) REFERENCES thoughts (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id TEXT PRIMARY KEY,
            human_message TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            discussed BOOLEAN DEFAULT 0,
            metadata TEXT  -- JSON object for additional metadata
        )
        ''')
        
        conn.commit()
        return conn
    
    def _initialize_remote_db(self):
        """Initialize remote MySQL/MariaDB database"""
        try:
            import pymysql
            
            self.logger.info(f"Connecting to remote database at {self.db_host}:{self.db_port}")
            
            # Connect to the database
            conn = pymysql.connect(
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name,
                charset='utf8mb4'
            )
            
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS thoughts (
                id VARCHAR(36) PRIMARY KEY,
                content TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                type VARCHAR(50) NOT NULL,
                sequence INT NOT NULL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id VARCHAR(36) PRIMARY KEY,
                content TEXT NOT NULL,
                title VARCHAR(255),
                timestamp DATETIME NOT NULL,
                source_thoughts TEXT,
                metadata TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reflections (
                id VARCHAR(36) PRIMARY KEY,
                thought_id VARCHAR(36) NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (thought_id) REFERENCES thoughts (id)
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id VARCHAR(36) PRIMARY KEY,
                human_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                discussed BOOLEAN DEFAULT 0,
                metadata TEXT
            )
            ''')
            
            conn.commit()
            self.logger.info("Remote database initialized successfully")
            return conn
            
        except ImportError:
            self.logger.error("PyMySQL not installed. Please install it for remote database support.")
            self.logger.info("Falling back to local SQLite database")
            self.use_remote_db = False
            self.db_type = "local"
            return self._initialize_local_db()
            
        except Exception as e:
            self.logger.error(f"Error connecting to remote database: {str(e)}")
            self.logger.info("Falling back to local SQLite database")
            self.use_remote_db = False
            self.db_type = "local"
            return self._initialize_local_db()
    
    def _initialize_vector_store(self):
        """Initialize the vector store for semantic search"""
        self.logger.info("Initializing vector store for memory retrieval")
        try:
            from sentence_transformers import SentenceTransformer
            
            # This will download the model if not already present
            # The 'all-mpnet-base-v2' model is ~400MB and provides excellent embeddings
            model = SentenceTransformer('all-mpnet-base-v2')
            
            self.logger.info("Vector embedding model loaded successfully")
            return {
                "model": model,
                "vectors": {}  # id -> vector mapping
            }
        except ImportError:
            self.logger.warning("sentence-transformers not installed. Please install it with:")
            self.logger.warning("pip install sentence-transformers")
            # remove if fallback is removed
            self.logger.warning("Falling back to keyword search for memory retrieval")
            return None
    
    def add_thought(self, thought):
        """Add a new thought to the memory system
        
        Args:
            thought: Dict with id, content, timestamp, type, and sequence
        """
        try:
            # Add to database based on type
            if self.db_type == "local":
                with self.db:
                    cursor = self.db.cursor()
                    cursor.execute(
                        "INSERT INTO thoughts VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            thought["id"],
                            thought["content"],
                            thought["timestamp"].isoformat(),
                            thought["type"],
                            thought["sequence"]
                        )
                    )
            else:  # Remote database
                with self.db.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO thoughts VALUES (%s, %s, %s, %s, %s, %s)",
                        (
                            thought["id"],
                            thought["content"],
                            thought["timestamp"],  # MySQL takes datetime objects directly
                            thought["type"],
                            thought["sequence"]
                        )
                    )
                self.db.commit()
            
            # Add to recent thoughts cache
            self.recent_thoughts_cache.append(thought)
            if len(self.recent_thoughts_cache) > self.short_term_limit:
                self.recent_thoughts_cache.pop(0)
            
            vector = self.vector_store["model"].encode(thought["content"])
            self.vector_store["vectors"][thought["id"]] = vector
            
            return thought["id"]
        except Exception as e:
            self.logger.error(f"Error adding thought: {str(e)}")
            return None
    
    def add_conversation(self, user_message, ai_response, metadata=None):
        """Store a conversation interaction in the memory system
        
        Args:
            user_message: Message from the user
            ai_response: Response from the AI
            metadata: Optional metadata
            
        Returns:
            ID of the stored conversation
        """
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Store in database
        if self.db_type == "local":
            with self.db:
                cursor = self.db.cursor()
                cursor.execute(
                    "INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        conversation_id,
                        user_message,
                        ai_response,
                        timestamp.isoformat(),
                        0,  # not discussed yet
                        json.dumps(metadata) if metadata else None
                    )
                )
        else:  # Remote database
            with self.db.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO interactions VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        conversation_id,
                        user_message,
                        ai_response,
                        timestamp,  # MySQL takes datetime objects directly
                        0,  # not discussed yet
                        json.dumps(metadata) if metadata else None
                    )
                )
            self.db.commit()
        
        # Also create a vector embedding for the conversation
        if self.vector_store:
            combined_text = f"Human: {user_message} AI: {ai_response}"
            vector = self.vector_store["model"].encode(combined_text)
            self.vector_store["vectors"][conversation_id] = vector
        
        return conversation_id

    def get_recent_mixed_items(self, limit=10):
        """Get recent items including both thoughts and conversations
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of mixed items (thoughts and conversations) in chronological order
        """
        # Get recent thoughts
        recent_thoughts = self.get_recent_thoughts(limit)
        
        # Get recent conversations
        recent_conversations = []
        if self.db_type == "local":
            with self.db:
                cursor = self.db.cursor()
                cursor.execute(
                    """
                    SELECT id, human_message, ai_response, timestamp, metadata
                    FROM interactions
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,)
                )
                
                for row in cursor.fetchall():
                    recent_conversations.append({
                        "id": row[0],
                        "type": "conversation",
                        "user_message": row[1],
                        "ai_response": row[2],
                        "timestamp": datetime.fromisoformat(row[3]),
                        "metadata": json.loads(row[4]) if row[4] else {}
                    })
        else:  # Remote database
            with self.db.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, human_message, ai_response, timestamp, metadata
                    FROM interactions
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
                
                for row in cursor.fetchall():
                    recent_conversations.append({
                        "id": row[0],
                        "type": "conversation",
                        "user_message": row[1],
                        "ai_response": row[2],
                        "timestamp": row[3],  # Already datetime object
                        "metadata": json.loads(row[4]) if row[4] else {}
                    })
        
        # Combine and sort by timestamp
        mixed_items = recent_thoughts + recent_conversations
        mixed_items.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Return the most recent items
        return mixed_items[:limit]

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

    def find_memories_for_thought(self, thought_content, limit=3):
        """Find memories relevant to an internal thought
        
        Args:
            thought_content: Content of the thought
            limit: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries
        """
        return self.get_relevant_memories(thought_content, limit)

    # Updated memory batch processing methods without keyword dependencies

    def auto_consolidate_memories(self, similarity_threshold=0.75, min_cluster_size=3, max_clusters=5):
        """Automatically find and consolidate similar memories
        
        Args:
            similarity_threshold: Minimum similarity score to consider memories related
            min_cluster_size: Minimum number of memories to form a cluster
            max_clusters: Maximum number of clusters to process
            
        Returns:
            List of new consolidated memory IDs
        """
        try:
            # Find clusters
            clusters = self.find_clusters_in_memories(
                similarity_threshold=similarity_threshold,
                min_cluster_size=min_cluster_size
            )
            
            # Limit the number of clusters to process
            if max_clusters > 0 and len(clusters) > max_clusters:
                clusters = clusters[:max_clusters]
            
            # Process each cluster
            consolidated_ids = []
            
            for cluster in clusters:
                # Get the memories in this cluster
                memories = []
                for memory_id in cluster:
                    memory = self.get_memory(memory_id)
                    if memory:
                        memories.append(memory)
                
                if len(memories) < min_cluster_size:
                    continue
                
                # Sort by timestamp
                memories.sort(key=lambda x: x["timestamp"])
                
                # Generate a title based on common words in titles
                titles = [m.get("title", "") for m in memories if m.get("title")]
                
                # Simple word extraction from titles
                words = []
                for title in titles:
                    # Convert to lowercase and split by non-alphanumeric characters
                    title_words = re.findall(r'\b\w+\b', title.lower())
                    # Filter out common short words
                    title_words = [w for w in title_words if len(w) > 3]
                    words.extend(title_words)
                
                # Count word frequencies
                word_counts = Counter(words)
                common_themes = [word for word, count in word_counts.most_common(3) if count > 1]
                
                if common_themes:
                    cluster_title = f"Consolidated: {', '.join(common_themes)}"
                else:
                    cluster_title = f"Consolidated Memories ({len(memories)})"
                
                # Concatenate content
                content_parts = []
                for memory in memories:
                    timestamp = memory["timestamp"].strftime("%Y-%m-%d") if hasattr(memory["timestamp"], "strftime") else str(memory["timestamp"])
                    title = memory.get("title", "Memory")
                    content_parts.append(f"[{timestamp}] {title}:\n{memory['content']}")
                
                consolidated_content = "\n\n".join(content_parts)
                
                # Create the consolidated memory
                memory_id = self.consolidate_memories(
                    memory_ids=cluster,
                    new_content=consolidated_content,
                    new_title=cluster_title
                )
                
                if memory_id:
                    consolidated_ids.append(memory_id)
            
            return consolidated_ids
            
        except Exception as e:
            self.logger.error(f"Error in auto memory consolidation: {str(e)}")
            return []

    def _check_if_memory_should_be_created(self, thought):
        """Check if a thought is significant enough to be stored as a memory
        
        Args:
            thought: Thought dictionary
            
        Returns:
            Boolean indicating if memory should be created
        """
        # Get recent thoughts
        recent_thoughts = self.memory.get_recent_mixed_items(10)
        
        # Filter out the current thought
        recent_thoughts = [t for t in recent_thoughts if t["id"] != thought["id"]]
        
        # If we don't have enough thoughts yet, don't create a memory
        if len(recent_thoughts) < 5:
            return False
        
        # Check similarity to recent thoughts to identify recurring themes
        recurring_theme = False
        high_similarity_count = 0
        
        # Encode current thought
        thought_vector = self.vector_store["model"].encode(thought["content"])
        
        # Compare to recent thoughts
        for recent in recent_thoughts:
            # Get or create vector for recent thought
            if recent["id"] in self.vector_store["vectors"]:
                recent_vector = self.vector_store["vectors"][recent["id"]]
            else:
                recent_vector = self.vector_store["model"].encode(recent["content"])
                self.vector_store["vectors"][recent["id"]] = recent_vector
            
            # Calculate similarity
            similarity = self._cosine_similarity(thought_vector, recent_vector)
            
            # Count high similarities
            if similarity > 0.7:  # Threshold for high similarity
                high_similarity_count += 1
        
        # If the thought is similar to several recent thoughts, it's a recurring theme
        recurring_theme = high_similarity_count >= 3
        
        # Check for significant keywords
        significant_keywords = [
            "realize", "understand", "discover", "insight", "epiphany", 
            "conclude", "determine", "significant", "important", "critical",
            "breakthrough", "milestone", "achievement", "revelation"
        ]
        
        has_significant_keywords = any(keyword in thought["content"].lower() for keyword in significant_keywords)
        
        # Check for emotional content
        emotional_keywords = [
            "feel", "happy", "sad", "angry", "frustrated", "excited", 
            "worried", "anxious", "curious", "proud", "guilty", "afraid",
            "love", "hate", "hope", "fear", "desire", "regret", "profound",
        ]
        
        has_emotional_content = any(keyword in thought["content"].lower() for keyword in emotional_keywords)
        
        # Decide based on criteria
        return recurring_theme or has_significant_keywords or has_emotional_content


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