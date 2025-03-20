# memory_system.py
import sqlite3
import json
import uuid
import logging
import math
from datetime import datetime
import os

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
    