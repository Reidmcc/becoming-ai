import json
import logging
import os
from datetime import datetime

from memory_system import MemorySystem
from chat_loader import ChatExportLoader, load_chat_exports_from_file

def initialize_memory_system(config, chat_exports_file=None):
    """Initialize the memory system and load initial data
    
    Args:
        config: Configuration dictionary
        chat_exports_file: Path to chat exports JSON file (optional)
        
    Returns:
        Initialized MemorySystem instance
    """
    logger = logging.getLogger("MemorySystemInit")
    
    # Create memory system
    logger.info("Initializing memory system")
    memory_system = MemorySystem(config=config)

    # Check if we should load conversations from chat exports
    if config.get('load_chat_exports', True) and chat_exports_file:
        if os.path.exists(chat_exports_file):
            logger.info(f"Loading chat exports from {chat_exports_file}")
            try:
                # Load chat exports
                stats = load_chat_exports_from_file(memory_system, chat_exports_file)
                
                logger.info(
                    f"Loaded {stats['conversations']} conversations with "
                    f"{stats['exchanges']} exchanges, creating {stats['memories_created']} memories"
                )
            except Exception as e:
                logger.error(f"Error loading chat exports: {str(e)}")
        else:
            logger.warning(f"Chat exports file not found: {chat_exports_file}")
    
    return memory_system