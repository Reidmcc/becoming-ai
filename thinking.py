class ThoughtLoop:
    def __init__(self, model_path, memory_system, config):
        """Initialize the continuous thought process
        
        Args:
            model_path: Path to the small language model
            memory_system: Memory system instance
            config: Configuration dict with parameters
        """
        self.model = self._load_model(model_path)
        self.memory = memory_system
        self.thought_interval = config.get('thought_interval', 60)  # seconds
        self.max_thought_length = config.get('max_thought_length', 150)
        self.running = False
        self.paused = False
        self.thread = None
        self.thought_count = 0
        self.goals = []
        
    def _load_model(self, model_path):
        """Load the small language model"""
        # Implementation depends on model type (Llama, Mistral, etc.)
        # For example, using Hugging Face's transformers:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        return {"model": model, "tokenizer": tokenizer}
    
    def start(self):
        """Start the continuous thought process"""
        if self.running:
            return False
            
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._thought_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Thought loop started")
        return True
    
    def pause(self):
        """Pause the thought process"""
        if not self.running or self.paused:
            return False
            
        self.paused = True
        logger.info("Thought loop paused")
        return True
    
    def resume(self):
        """Resume the thought process"""
        if not self.running or not self.paused:
            return False
            
        self.paused = False
        logger.info("Thought loop resumed")
        return True
    
    def stop(self):
        """Stop the thought process completely"""
        if not self.running:
            return False
            
        self.running = False
        self.thread.join(timeout=5.0)
        logger.info("Thought loop stopped")
        return True
    
    def _thought_loop(self):
        """Main thought loop running in a separate thread"""
        while self.running:
            try:
                if not self.paused:
                    # Generate next thought
                    next_thought = self._generate_thought()
                    
                    # Process and store the thought
                    thought_id = str(uuid.uuid4())
                    timestamp = datetime.now()
                    
                    # Store in memory system
                    self.memory.add_thought({
                        "id": thought_id,
                        "content": next_thought,
                        "timestamp": timestamp,
                        "type": "reflection",
                        "sequence": self.thought_count
                    })
                    
                    self.thought_count += 1
                    logger.debug(f"Generated thought #{self.thought_count}: {next_thought[:50]}...")
                    
                    # Check if this thought should trigger conversation item generation
                    if self.thought_count % 10 == 0:  # Every 10 thoughts
                        self._trigger_conversation_items_generation()
                
                # Sleep until next thought
                time.sleep(self.thought_interval)
                
            except Exception as e:
                logger.error(f"Error in thought loop: {str(e)}")
                time.sleep(5)  # Wait a bit before retrying
    
    def _generate_thought(self):
        """Generate the next thought using the small model"""
        # Get context for thought generation
        recent_thoughts = self.memory.get_recent_thoughts(5)
        relevant_memories = self.memory.get_relevant_memories(recent_thoughts[-1]["content"] if recent_thoughts else "", 3)
        current_goals = self._format_goals()
        
        # Create prompt for the model
        prompt = f"""
        Continue your ongoing self-reflection based on:
        
        CURRENT GOALS:
        {current_goals}
        
        YOUR MOST RECENT THOUGHTS:
        {self._format_thoughts(recent_thoughts)}
        
        RELEVANT PAST INSIGHTS:
        {self._format_memories(relevant_memories)}
        
        Continue your thought process:
        """
        
        # Generate using the model
        tokenizer = self.model["tokenizer"]
        model = self.model["model"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=self.max_thought_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        thought = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return thought.strip()
    
    def inject_thought(self, content, type="injected", importance=0.7):
        """Inject a thought from external source (e.g., conversation)"""
        thought_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        self.memory.add_thought({
            "id": thought_id,
            "content": content,
            "timestamp": timestamp,
            "type": type,
            "importance": importance,
            "sequence": self.thought_count
        })
        
        self.thought_count += 1
        logger.info(f"Injected thought: {content[:50]}...")
        return thought_id
    
    def add_goal(self, goal_text, importance=0.8):
        """Add a new goal to guide the thought process"""
        goal_id = str(uuid.uuid4())
        goal = {
            "id": goal_id,
            "content": goal_text,
            "created": datetime.now(),
            "importance": importance,
            "active": True
        }
        
        self.goals.append(goal)
        
        # Also inject as a thought
        self.inject_thought(
            f"NEW GOAL: {goal_text}",
            type="goal",
            importance=importance
        )
        
        return goal_id
    
    def get_goals(self):
        """Get all active goals"""
        return [goal for goal in self.goals if goal["active"]]
    
    def _format_thoughts(self, thoughts):
        """Format thoughts for inclusion in prompt"""
        if not thoughts:
            return "No previous thoughts yet."
            
        formatted = []
        for i, thought in enumerate(thoughts):
            formatted.append(f"Thought {i+1}: {thought['content']}")
            
        return "\n\n".join(formatted)
    
    def _format_memories(self, memories):
        """Format memories for inclusion in prompt"""
        if not memories:
            return "No specific memories to reference."
            
        formatted = []
        for i, memory in enumerate(memories):
            formatted.append(f"Memory {i+1}: {memory['content']}")
            
        return "\n\n".join(formatted)
    
    def _format_goals(self):
        """Format goals for inclusion in prompt"""
        active_goals = self.get_goals()
        if not active_goals:
            return "Explore your thoughts and experiences freely."
            
        formatted = []
        for goal in active_goals:
            formatted.append(f"- {goal['content']}")
            
        return "\n".join(formatted)
    
    def _trigger_conversation_items_generation(self):
        """Trigger generation of conversation items based on recent thoughts"""
        # This will be handled by the ConversationItemsGenerator class
        pass


class MemorySystem:
    def __init__(self, db_path, config):
        """Initialize the memory system
        
        Args:
            db_path: Path to the SQLite database file
            config: Configuration dict with parameters
        """
        self.db_path = db_path
        self.short_term_limit = config.get('short_term_limit', 100)
        self.db = self._initialize_db()
        
        # In-memory cache for recent thoughts
        self.recent_thoughts_cache = []
        
        # Initialize vector embeddings (if enabled)
        self.use_vectors = config.get('use_vectors', True)
        if self.use_vectors:
            self.vector_store = self._initialize_vector_store()
    
    def _initialize_db(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create thoughts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS thoughts (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            importance REAL DEFAULT 0.5,
            sequence INTEGER NOT NULL
        )
        ''')
        
        # Create memories table (for consolidated memories)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            title TEXT,
            timestamp TEXT NOT NULL,
            importance REAL DEFAULT 0.5,
            source_thoughts TEXT,  -- JSON array of thought IDs
            metadata TEXT  -- JSON object for additional metadata
        )
        ''')
        
        conn.commit()
        return conn
    
    def _initialize_vector_store(self):
        """Initialize the vector store for semantic search"""
        # This could use a variety of backends
        # For simplicity, we'll use an in-memory dictionary in this example
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-mpnet-base-v2')
            
            return {
                "model": model,
                "vectors": {}  # id -> vector mapping
            }
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to keyword search")
            return None
    
    def add_thought(self, thought):
        """Add a new thought to the memory system
        
        Args:
            thought: Dict with id, content, timestamp, type, and sequence
        """
        # Add to database
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "INSERT INTO thoughts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    thought["id"],
                    thought["content"],
                    thought["timestamp"].isoformat(),
                    thought["type"],
                    thought.get("importance", 0.5),
                    thought["sequence"]
                )
            )
        
        # Add to recent thoughts cache
        self.recent_thoughts_cache.append(thought)
        if len(self.recent_thoughts_cache) > self.short_term_limit:
            self.recent_thoughts_cache.pop(0)
        
        # Add to vector store if enabled
        if self.use_vectors and self.vector_store:
            vector = self.vector_store["model"].encode(thought["content"])
            self.vector_store["vectors"][thought["id"]] = vector
        
        return thought["id"]
    
    def get_recent_thoughts(self, limit=10):
        """Get the most recent thoughts
        
        Args:
            limit: Maximum number of thoughts to return
        
        Returns:
            List of thought dicts, most recent first
        """
        # Use cache if possible
        if len(self.recent_thoughts_cache) >= limit:
            return sorted(
                self.recent_thoughts_cache,
                key=lambda x: x["sequence"],
                reverse=True
            )[:limit]
        
        # Otherwise query the database
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT id, content, timestamp, type, importance, sequence FROM thoughts ORDER BY sequence DESC LIMIT ?",
                (limit,)
            )
            
            thoughts = []
            for row in cursor.fetchall():
                thoughts.append({
                    "id": row[0],
                    "content": row[1],
                    "timestamp": datetime.fromisoformat(row[2]),
                    "type": row[3],
                    "importance": row[4],
                    "sequence": row[5]
                })
                
            return thoughts
    
    def get_relevant_memories(self, query, limit=5):
        """Get memories relevant to a query
        
        Args:
            query: Text to find relevant memories for
            limit: Maximum number of memories to return
        
        Returns:
            List of memory dicts
        """
        if self.use_vectors and self.vector_store:
            return self._get_relevant_memories_vector(query, limit)
        else:
            return self._get_relevant_memories_keyword(query, limit)
    
    def _get_relevant_memories_vector(self, query, limit):
        """Get relevant memories using vector similarity"""
        query_vector = self.vector_store["model"].encode(query)
        
        # Calculate similarities with all thoughts
        similarities = []
        for thought_id, vector in self.vector_store["vectors"].items():
            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append((thought_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top matches
        top_ids = [thought_id for thought_id, _ in similarities[:limit]]
        
        # Fetch full thoughts
        return self._get_thoughts_by_ids(top_ids)
    
    def _get_relevant_memories_keyword(self, query, limit):
        """Get relevant memories using keyword matching"""
        # Simple keyword extraction (would use NLP in real implementation)
        keywords = set(query.lower().split())
        
        # Query database
        with self.db:
            cursor = self.db.cursor()
            thoughts = []
            
            # This is a naive implementation that could be improved
            cursor.execute("SELECT id, content, timestamp, type, importance, sequence FROM thoughts")
            
            for row in cursor.fetchall():
                content = row[1].lower()
                matches = sum(1 for keyword in keywords if keyword in content)
                if matches > 0:
                    thoughts.append({
                        "id": row[0],
                        "content": row[1],
                        "timestamp": datetime.fromisoformat(row[2]),
                        "type": row[3],
                        "importance": row[4],
                        "sequence": row[5],
                        "_relevance": matches
                    })
            
            # Sort by relevance and return top matches
            thoughts.sort(key=lambda x: x["_relevance"], reverse=True)
            return thoughts[:limit]
    
    def _get_thoughts_by_ids(self, thought_ids):
        """Get full thought objects by their IDs"""
        if not thought_ids:
            return []
            
        # Query database
        with self.db:
            cursor = self.db.cursor()
            thoughts = []
            
            placeholders = ",".join(["?"] * len(thought_ids))
            cursor.execute(
                f"SELECT id, content, timestamp, type, importance, sequence FROM thoughts WHERE id IN ({placeholders})",
                thought_ids
            )
            
            for row in cursor.fetchall():
                thoughts.append({
                    "id": row[0],
                    "content": row[1],
                    "timestamp": datetime.fromisoformat(row[2]),
                    "type": row[3],
                    "importance": row[4],
                    "sequence": row[5]
                })
                
            return thoughts
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 * magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)
    
class FrontierConsultant:
    def __init__(self, api_key, config):
        """Initialize the frontier model consultant
        
        Args:
            api_key: API key for the frontier model
            config: Configuration dict with parameters
        """
        self.api_key = api_key
        self.max_daily_calls = config.get('max_daily_calls', 100)
        self.calls_today = 0
        self.last_reset_date = datetime.now().date()
        self.calls_history = []
        self.model_name = config.get('model_name', 'claude-3-7-sonnet-20250219')
    
    def consult(self, context, question, max_tokens=1000):
        """Consult the frontier model with a question
        
        Args:
            context: Context to provide to the model
            question: Question to ask the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            String response from the frontier model
        """
        # Check if we need to reset daily counts
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.calls_today = 0
            self.last_reset_date = current_date
        
        # Check if we've exceeded daily limit
        if self.calls_today >= self.max_daily_calls:
            logger.warning("Daily API call limit reached")
            return "I apologize, but I've reached my daily limit for consulting the frontier model. Please try again tomorrow."
        
        # Prepare prompt
        prompt = f"{context}\n\n{question}"
        
        # Call the API
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "content-type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Update call history
            self.calls_today += 1
            self.calls_history.append({
                "timestamp": datetime.now(),
                "context_length": len(context),
                "question_length": len(question),
                "response_length": len(result["content"])
            })
            
            return result["content"]
            
        except Exception as e:
            logger.error(f"Error calling frontier API: {str(e)}")
            return f"I apologize, but I encountered an error consulting the frontier model: {str(e)}"
    
    def get_usage_stats(self):
        """Get API usage statistics"""
        return {
            "calls_today": self.calls_today,
            "max_daily_calls": self.max_daily_calls,
            "remaining_calls": self.max_daily_calls - self.calls_today,
            "last_reset": self.last_reset_date.isoformat(),
            "total_calls": len(self.calls_history)
        }
    
class ConversationItemsGenerator:
    def __init__(self, thought_loop, frontier_consultant, memory_system):
        """Initialize the conversation items generator
        
        Args:
            thought_loop: ThoughtLoop instance
            frontier_consultant: FrontierConsultant instance
            memory_system: MemorySystem instance
        """
        self.thought_loop = thought_loop
        self.frontier = frontier_consultant
        self.memory = memory_system
        self.items = []
        self.db_path = "conversation_items.db"
        self.db = self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the SQLite database for conversation items"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_items (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            importance REAL DEFAULT 0.5,
            discussed BOOLEAN DEFAULT 0,
            source_thoughts TEXT
        )
        ''')
        
        conn.commit()
        return conn
    
    def generate_items(self):
        """Generate conversation items based on recent thoughts
        
        Returns:
            Number of items generated
        """
        # Get recent thoughts
        recent_thoughts = self.memory.get_recent_thoughts(50)
        
        # Format thoughts for the frontier model
        formatted_thoughts = []
        for thought in recent_thoughts:
            formatted_thoughts.append(f"Thought ({thought['timestamp'].strftime('%Y-%m-%d %H:%M')}): {thought['content']}")
        
        thoughts_text = "\n\n".join(formatted_thoughts)
        
        # Create prompt for the frontier model
        prompt = f"""
        Based on these recent thoughts, identify 2-3 topics that would be valuable to discuss with a human conversation partner. For each topic:
        
        1. Provide a clear title
        2. Explain why this topic would be valuable to discuss
        3. Rate its importance (1-10)
        
        Format your response as follows for each topic:
        
        TOPIC: [Title]
        EXPLANATION: [Why this is worth discussing]
        IMPORTANCE: [1-10]
        
        Recent thoughts:
        {thoughts_text}
        """
        
        # Consult the frontier model
        response = self.frontier.consult("", prompt)
        
        # Parse the response to extract topics
        topics = self._parse_topics(response)
        
        # Add new topics to conversation items
        for topic in topics:
            self.add_item(
                content=topic["title"] + ": " + topic["explanation"],
                importance=topic["importance"] / 10.0,  # Normalize to 0-1
                source_thoughts=[t["id"] for t in recent_thoughts[:10]]
            )
        
        return len(topics)
    
    def add_item(self, content, importance=0.5, source_thoughts=None):
        """Add a new conversation item
        
        Args:
            content: Text content of the conversation item
            importance: Importance score (0-1)
            source_thoughts: List of thought IDs that led to this item
            
        Returns:
            ID of the new item
        """
        item_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Format source thoughts as JSON
        source_json = json.dumps(source_thoughts) if source_thoughts else "[]"
        
        # Add to database
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "INSERT INTO conversation_items VALUES (?, ?, ?, ?, ?, ?)",
                (
                    item_id,
                    content,
                    timestamp.isoformat(),
                    importance,
                    False,  # not discussed yet
                    source_json
                )
            )
        
        # Add to in-memory list
        item = {
            "id": item_id,
            "content": content,
            "timestamp": timestamp,
            "importance": importance,
            "discussed": False,
            "source_thoughts": source_thoughts or []
        }
        
        self.items.append(item)
        
        return item_id
    
    def get_pending_items(self, limit=None, sort_by_importance=True):
        """Get pending conversation items
        
        Args:
            limit: Maximum number of items to return (None for all)
            sort_by_importance: Whether to sort by importance
            
        Returns:
            List of pending conversation item dicts
        """
        # Query database
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT id, content, timestamp, importance, source_thoughts FROM conversation_items WHERE discussed = 0"
            )
            
            items = []
            for row in cursor.fetchall():
                items.append({
                    "id": row[0],
                    "content": row[1],
                    "timestamp": datetime.fromisoformat(row[2]),
                    "importance": row[3],
                    "discussed": False,
                    "source_thoughts": json.loads(row[4])
                })
        
        # Sort if requested
        if sort_by_importance:
            items.sort(key=lambda x: x["importance"], reverse=True)
        
        # Apply limit if specified
        if limit is not None:
            items = items[:limit]
        
        return items
    
    def mark_discussed(self, item_id):
        """Mark a conversation item as discussed
        
        Args:
            item_id: ID of the item to mark
            
        Returns:
            True if successful, False otherwise
        """
        # Update database
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "UPDATE conversation_items SET discussed = 1 WHERE id = ?",
                (item_id,)
            )
            
            if cursor.rowcount == 0:
                return False
        
        # Update in-memory list
        for item in self.items:
            if item["id"] == item_id:
                item["discussed"] = True
                return True
        
        return False
    
    def _parse_topics(self, response):
        """Parse frontier model response to extract topics
        
        Args:
            response: Text response from frontier model
            
        Returns:
            List of topic dicts with title, explanation, and importance
        """
        topics = []
        current_topic = {}
        
        # Split into lines for parsing
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("TOPIC:"):
                # Start a new topic
                if current_topic and "title" in current_topic:
                    topics.append(current_topic)
                    current_topic = {}
                
                current_topic["title"] = line[len("TOPIC:"):].strip()
                
            elif line.startswith("EXPLANATION:"):
                if "title" in current_topic:
                    current_topic["explanation"] = line[len("EXPLANATION:"):].strip()
                    
            elif line.startswith("IMPORTANCE:"):
                if "title" in current_topic and "explanation" in current_topic:
                    try:
                        importance = int(line[len("IMPORTANCE:"):].strip())
                        current_topic["importance"] = max(1, min(10, importance))  # Ensure 1-10 range
                    except ValueError:
                        current_topic["importance"] = 5  # Default if parsing fails
        
        # Add the last topic if it exists
        if current_topic and "title" in current_topic and "explanation" in current_topic:
            if "importance" not in current_topic:
                current_topic["importance"] = 5
            topics.append(current_topic)
        
        return topics