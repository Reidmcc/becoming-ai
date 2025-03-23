# chat_loader.py
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

class ChatExportLoader:
    """Utility class to load Claude chat exports into the memory system"""
    
    def __init__(self, memory_system):
        """Initialize the chat export loader
        
        Args:
            memory_system: MemorySystem instance
        """
        self.memory = memory_system
        self.logger = logging.getLogger("ChatExportLoader")
    
    def load_chat_exports(self, chat_exports_json):
        """Load all conversations from a Claude chat export JSON
        
        Args:
            chat_exports_json: Parsed JSON data from a chat export file
            
        Returns:
            Dictionary with stats about loaded conversations
        """
        loaded_stats = {
            "conversations": 0,
            "exchanges": 0,
            "memories_created": 0
        }
        
        # Check if input is a list of conversations
        if not isinstance(chat_exports_json, list):
            self.logger.error("Chat exports JSON is not a list")
            return loaded_stats
        
        # Process each conversation
        for conversation in chat_exports_json:
            try:
                # Extract conversation metadata
                conv_name = conversation.get("name", "Unnamed Conversation")
                conv_uuid = conversation.get("uuid", "")
                conv_date = conversation.get("created_at", "")
                
                # Try to parse the creation date
                try:
                    if isinstance(conv_date, str) and conv_date:
                        creation_date = datetime.fromisoformat(conv_date.replace('Z', '+00:00'))
                    else:
                        creation_date = datetime.now()
                except ValueError:
                    creation_date = datetime.now()
                
                # Get conversation messages
                messages = conversation.get("chat_messages", [])
                if not messages:
                    self.logger.warning(f"No messages in conversation: {conv_name}")
                    continue
                
                # Create memories for important exchanges
                memories_created = self._process_conversation(conv_name, messages, creation_date)
                
                # Update stats
                loaded_stats["conversations"] += 1
                loaded_stats["exchanges"] += len(messages) // 2  # Approximate exchanges (human+assistant)
                loaded_stats["memories_created"] += len(memories_created)
                
                self.logger.info(f"Processed conversation '{conv_name}' with {len(messages)} messages")
                
            except Exception as e:
                self.logger.error(f"Error processing conversation: {str(e)}")
        
        return loaded_stats
    
    def _process_conversation(self, conv_name, messages, creation_date):
        """Process messages from a conversation and create memories
        
        Args:
            conv_name: Name of the conversation
            messages: List of message objects
            creation_date: Creation date of the conversation
            
        Returns:
            List of created memory IDs
        """
        memory_ids = []
        
        # Group messages into exchanges (human + assistant pairs)
        exchanges = []
        current_exchange = []
        
        for message in messages:
            sender = message.get("sender", "")
            text = message.get("text", "")
            
            if not text:
                continue
                
            # Add to current exchange
            current_exchange.append({"role": sender, "content": text})
            
            # If this is an assistant message and we have a pair, complete the exchange
            if sender == "assistant" and len(current_exchange) >= 2:
                exchanges.append(current_exchange)
                current_exchange = []
        
        # Add any remaining messages
        if current_exchange:
            exchanges.append(current_exchange)
        
        # Process each exchange
        for i, exchange in enumerate(exchanges):
            # Format the exchange content
            exchange_content = self._format_exchange(exchange)
            
            # Create a title
            exchange_title = f"{conv_name}: Exchange {i+1}"
            
            # Create metadata
            metadata = {
                "type": "conversation",
                "conversation_name": conv_name,
                "exchange_index": i,
                "creation_date": creation_date.isoformat()
            }
            
            # Create memory for significant exchanges (skip simple acknowledgments)
            if len(exchange_content) > 100:  # Only create memories for substantive exchanges
                memory_id = self.memory.create_memory(
                    content=exchange_content,
                    title=exchange_title,
                    metadata=metadata
                )
                
                if memory_id:
                    memory_ids.append(memory_id)
        
        # If the conversation as a whole is significant, create a summary memory
        if len(exchanges) >= 3:
            # Create a conversation summary (first might be entire convo if small)
            summary = self._create_conversation_summary(conv_name, exchanges)
            
            summary_memory_id = self.memory.create_memory(
                content=summary,
                title=f"Summary: {conv_name}",
                metadata={
                    "type": "conversation_summary",
                    "conversation_name": conv_name,
                    "creation_date": creation_date.isoformat(),
                    "exchange_count": len(exchanges)
                }
            )
            
            if summary_memory_id:
                memory_ids.append(summary_memory_id)
        
        return memory_ids
    
    def _format_exchange(self, exchange):
        """Format an exchange for inclusion in a memory
        
        Args:
            exchange: List of message objects in the exchange
            
        Returns:
            Formatted exchange text
        """
        formatted = []
        
        for message in exchange:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            if role == "human":
                formatted.append(f"Human: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")
        
        return "\n\n".join(formatted)
    
    def _create_conversation_summary(self, conv_name, exchanges):
        """Create a summary of a conversation
        
        Args:
            conv_name: Name of the conversation
            exchanges: List of exchanges in the conversation
            
        Returns:
            Summary text
        """
        # For simplicity, just use the conversation name and first few exchanges
        # In a more advanced implementation, you could use the frontier model to generate a summary
        
        exchange_count = len(exchanges)
        if exchange_count > 1:
            last_exchange_text = f"Last exchange:\n{last_exchange}"
        else:
            last_exchange_text = ""
        
        # Extract the first and last exchanges for context
        first_exchange = self._format_exchange(exchanges[0]) if exchanges else ""
        last_exchange = self._format_exchange(exchanges[-1]) if exchange_count > 1 else ""
        
        summary = f"""Conversation: {conv_name}
Number of exchanges: {exchange_count}

First exchange:
{first_exchange}

{last_exchange_text}
"""
        
        return summary


def load_chat_exports_from_file(memory_system, file_path):
    """Utility function to load chat exports from a JSON file
    
    Args:
        memory_system: MemorySystem instance
        file_path: Path to the chat exports JSON file
        
    Returns:
        Dictionary with stats about loaded conversations
    """
    logger = logging.getLogger("ChatExportLoader")
    
    try:
        # Load the JSON file
        logger.info(f"Loading chat exports from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            chat_exports = json.load(f)
        
        # Create loader and process the data
        loader = ChatExportLoader(memory_system)
        stats = loader.load_chat_exports(chat_exports)
        
        logger.info(f"Loaded {stats['conversations']} conversations with {stats['exchanges']} exchanges")
        return stats
        
    except Exception as e:
        logger.error(f"Error loading chat exports: {str(e)}")
        return {
            "conversations": 0,
            "exchanges": 0,
            "memories_created": 0,
            "error": str(e)
        }