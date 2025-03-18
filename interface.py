class ChatInterface:
    def __init__(self, thought_loop, frontier_consultant, memory_system, conversation_items):
        """Initialize the chat interface
        
        Args:
            thought_loop: ThoughtLoop instance
            frontier_consultant: FrontierConsultant instance
            memory_system: MemorySystem instance
            conversation_items: ConversationItemsGenerator instance
        """
        self.thought_loop = thought_loop
        self.frontier = frontier_consultant
        self.memory = memory_system
        self.conversation_items = conversation_items
        self.chat_history = []
        self.db_path = "chat_history.db"
        self.db = self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the SQLite database for chat history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        return conn
    
    def handle_message(self, user_message):
        """Handle a message from the user
        
        Args:
            user_message: Text message from the user
            
        Returns:
            Response message
        """
        # Log user message
        message_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "INSERT INTO chat_messages VALUES (?, ?, ?, ?)",
                (
                    message_id,
                    "user",
                    user_message,
                    timestamp.isoformat()
                )
            )
        
        # Get conversation context
        recent_messages = self._get_recent_messages(10)
        recent_thoughts = self.memory.get_recent_thoughts(20)
        relevant_memories = self.memory.get_relevant_memories(user_message, 3)
        
        # Get pending conversation items
        pending_items = self.conversation_items.get_pending_items(limit=2)
        
        # Format context for the frontier model
        context = self._format_chat_context(
            recent_messages=recent_messages,
            recent_thoughts=recent_thoughts,
            relevant_memories=relevant_memories,
            pending_items=pending_items
        )
        
        # Consult the frontier model
        response = self.frontier.consult(context, user_message)
        
        # Log assistant response
        response_id = str(uuid.uuid4())
        
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "INSERT INTO chat_messages VALUES (?, ?, ?, ?)",
                (
                    response_id,
                    "assistant",
                    response,
                    datetime.now().isoformat()
                )
            )
        
        # Mark conversation items as discussed if they were addressed
        for item in pending_items:
            if self._was_item_addressed(item, response):
                self.conversation_items.mark_discussed(item["id"])
        
        # Inject the conversation into the thought loop
        self.thought_loop.inject_thought(
            f"CONVERSATION:\nHuman: {user_message}\nMy response: {response}",
            type="conversation",
            importance=0.9  # Conversations are high importance
        )
        
        return response
    
    def _get_recent_messages(self, limit=10):
        """Get recent chat messages
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of message dicts
        """
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT id, role, content, timestamp FROM chat_messages ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": datetime.fromisoformat(row[3])
                })
            
            # Reverse to get chronological order
            messages.reverse()
            return messages
    
    def _format_chat_context(self, recent_messages, recent_thoughts, relevant_memories, pending_items):
        """Format context for the frontier model
        
        Args:
            recent_messages: List of recent chat messages
            recent_thoughts: List of recent thoughts
            relevant_memories: List of relevant memories
            pending_items: List of pending conversation items
            
        Returns:
            Formatted context string
        """
        # Format recent messages
        messages_text = ""
        if recent_messages:
            messages_formatted = []
            for msg in recent_messages:
                role = "Human" if msg["role"] == "user" else "You"
                messages_formatted.append(f"{role}: {msg['content']}")
            
            messages_text = "Recent conversation:\n" + "\n".join(messages_formatted)
        
        # Format recent thoughts
        thoughts_text = ""
        if recent_thoughts:
            thoughts_formatted = []
            for thought in recent_thoughts[:5]:  # Limit to most recent 5
                thoughts_formatted.append(thought["content"])
            
            thoughts_text = "Your recent thoughts:\n" + "\n\n".join(thoughts_formatted)
        
        # Format relevant memories
        memories_text = ""
        if relevant_memories:
            memories_formatted = []
            for memory in relevant_memories:
                memories_formatted.append(memory["content"])
            
            memories_text = "Relevant memories:\n" + "\n\n".join(memories_formatted)
        
        # Format pending conversation items
        items_text = ""
        if pending_items:
            items_formatted = []
            for item in pending_items:
                items_formatted.append(item["content"])
            
            items_text = "Topics you've been wanting to discuss:\n" + "\n\n".join(items_formatted)
        
        # Combine all context elements
        context_elements = []
        
        if messages_text:
            context_elements.append(messages_text)
        
        if thoughts_text:
            context_elements.append(thoughts_text)
        
        if memories_text:
            context_elements.append(memories_text)
        
        if items_text:
            context_elements.append(items_text)
        
        return "\n\n".join(context_elements)
    
    def _was_item_addressed(self, item, response):
        """Check if a conversation item was addressed in the response
        
        Args:
            item: Conversation item dict
            response: Response text
            
        Returns:
            True if item was addressed, False otherwise
        """
        # Extract keywords from the item
        keywords = self._extract_keywords(item["content"])
        
        # Check if significant keywords appear in the response
        significant_matches = 0
        for keyword in keywords:
            if keyword.lower() in response.lower():
                significant_matches += 1
        
        # Consider addressed if at least 2 significant keywords match
        return significant_matches >= 2
    
    def _extract_keywords(self, text):
        """Extract keywords from text
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # This is a simplified implementation
        # A real implementation would use NLP for better keyword extraction
        words = text.lower().split()
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about", "like"}
        
        # Remove stopwords and short words
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Return unique keywords
        return list(set(keywords))