# interface.py
import sqlite3
import uuid
import logging
from datetime import datetime
import json

class ChatInterface:
    def __init__(self, thought_loop, frontier_client, memory_system, conversation_items=None):
        """Initialize the chat interface
        
        Args:
            thought_loop: ThoughtLoop instance
            frontier_client: FrontierConsultant instance
            memory_system: MemorySystem instance
            conversation_items: ConversationItemsGenerator instance (optional)
        """
        self.thought_loop = thought_loop
        self.frontier = frontier_client
        self.memory = memory_system
        self.conversation_items = conversation_items
        self.logger = logging.getLogger("ChatInterface")
    
    def handle_message(self, user_message):
        """Handle a message from the user
        
        Args:
            user_message: Text message from the user
            
        Returns:
            Response message
        """
        try:
            # Log the incoming message
            self.logger.info(f"Received message: {user_message[:50]}...")
            
            # Get conversation context
            recent_messages = self._get_recent_interactions(5)
            recent_thoughts = self.memory.get_recent_thoughts(10)
            relevant_memories = self.memory.get_relevant_memories(user_message, 3)
            
            # Get pending conversation items if available
            pending_items = []
            if self.conversation_items:
                pending_items = self.conversation_items.get_pending_items(limit=2)
            
            # Format context for the frontier model
            context = self._format_chat_context(
                recent_messages=recent_messages,
                recent_thoughts=recent_thoughts,
                relevant_memories=relevant_memories,
                pending_items=pending_items
            )
            
            # Generate response using frontier model
            self.logger.info("Generating response...")
            response = self.frontier.client.messages.create(
                model=self.frontier.model,
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": f"{context}\n\n{user_message}"}
                ]
            )
            
            # Extract response content
            response_text = response.content[0].text
            
            # Store the interaction in memory
            self._store_interaction(user_message, response_text)
            
            # Inject the conversation into the thought loop as a significant thought
            self.thought_loop.inject_thought(
                f"CONVERSATION:\nHuman: {user_message}\nMy response: {response_text}",
                type="conversation",
                importance=0.85  # High importance for conversations
            )
            
            # Mark conversation items as discussed if they were addressed
            if self.conversation_items:
                for item in pending_items:
                    if self._was_item_addressed(item, response_text):
                        self.conversation_items.mark_discussed(item["id"])
            
            return response_text
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            return f"I'm sorry, I encountered an error processing your message: {str(e)}"
    
    def _get_recent_interactions(self, limit=5):
        """Get recent interactions from memory
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of interaction dictionaries
        """
        # Query from the memory system
        # This is a simplified implementation - would need to be adapted based on
        # how interactions are stored in your memory system
        try:
            # Create query based on database type
            if hasattr(self.memory, 'db_type') and self.memory.db_type == "remote":
                # Remote MySQL database
                with self.memory.db.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT id, human_message, ai_response, timestamp, metadata 
                        FROM interactions 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                        """,
                        (limit,)
                    )
                    
                    interactions = []
                    for row in cursor.fetchall():
                        interactions.append({
                            "id": row[0],
                            "human_message": row[1],
                            "ai_response": row[2],
                            "timestamp": row[3],
                            "metadata": json.loads(row[4]) if row[4] else {}
                        })
            else:
                # Local SQLite database
                with self.memory.db:
                    cursor = self.memory.db.cursor()
                    cursor.execute(
                        """
                        SELECT id, human_message, ai_response, timestamp, metadata 
                        FROM interactions 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                        """,
                        (limit,)
                    )
                    
                    interactions = []
                    for row in cursor.fetchall():
                        interactions.append({
                            "id": row[0],
                            "human_message": row[1],
                            "ai_response": row[2],
                            "timestamp": datetime.fromisoformat(row[3]),
                            "metadata": json.loads(row[4]) if row[4] else {}
                        })
            
            # Sort chronologically for context building
            interactions.sort(key=lambda x: x["timestamp"])
            return interactions
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent interactions: {str(e)}")
            return []
    
    def _store_interaction(self, human_message, ai_response, metadata=None):
        """Store an interaction in the memory system
        
        Args:
            human_message: Message from the human
            ai_response: Response from the AI
            metadata: Optional metadata about the interaction
            
        Returns:
            ID of the stored interaction
        """
        return self.memory.add_interaction(human_message, ai_response, metadata)
    
    def _format_chat_context(self, recent_messages, recent_thoughts, relevant_memories, pending_items):
        """Format context for the frontier model
        
        Args:
            recent_messages: List of recent interactions
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
                messages_formatted.append(f"Human: {msg['human_message']}")
                messages_formatted.append(f"You: {msg['ai_response']}")
            
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
                significan