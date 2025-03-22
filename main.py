import os
import logging
import argparse
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("becoming_ai.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BecomingAI")

# Import custom modules
from config_utils import setup_argparse, load_config, get_flattened_config
from model_loader import LocalModelLoader
from memory_system_integration import initialize_memory_system
from frontier_client import FrontierClient
from thought_loop import ThoughtLoop
from memory_consolidator import MemoryConsolidator

def main():
    """Main entry point for the Becoming AI system"""
    
    # Parse command-line arguments
    parser = setup_argparse()
    parser.add_argument("--config", "-c", help="Path to configuration file (YAML or JSON)")
    parser.add_argument("--chat-exports", help="Path to chat exports JSON file")
    args = parser.parse_args()
    
    # Load configuration (from defaults, file, and CLI args)
    config = load_config(args.config, args)
    
    # Add chat exports path to config if provided
    if args.chat_exports:
        config["chat_exports_path"] = args.chat_exports
    
    # Convert to flat config for compatibility with existing code
    flat_config = get_flattened_config(config)
    
    # Display configuration
    logger.info("Starting Becoming AI with configuration:")
    for key, value in flat_config.items():
        # Don't log password
        if key != "db_password":
            logger.info(f"  {key}: {value}")
    
    # Initialize components
    try:
        # Ensure data directory exists
        if not flat_config.get('use_remote_db', False):
            os.makedirs(os.path.dirname(os.path.abspath(flat_config.get('db_path', 'data/memories.db'))), exist_ok=True)
        
        # Initialize memory system with chat exports
        logger.info("Initializing memory system...")
        chat_exports_path = flat_config.get('chat_exports_path', 'data/chat_exports.json')
        memory_system = initialize_memory_system(flat_config, chat_exports_path)
        
        # Initialize model loader
        logger.info("Initializing model loader...")
        model_loader = LocalModelLoader(
            model_name=flat_config["model_name"],
            quantization=flat_config["quantization"]
        )
        
        # Initialize frontier client
        logger.info("Initializing frontier client...")
        frontier_client = FrontierClient(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=flat_config["frontier_model"],
            max_daily_calls=flat_config["max_daily_calls"]
        )
        
        # Initialize thought loop
        logger.info("Initializing thought loop...")
        thought_loop = ThoughtLoop(
            model_loader=model_loader,
            memory_system=memory_system,
            frontier_client=frontier_client,
            config=flat_config
        )
        
        # Initialize memory consolidator
        logger.info("Initializing memory consolidator...")
        memory_consolidator = MemoryConsolidator(
            memory_system=memory_system,
            config={
                'memory_threshold': flat_config.get("memory_threshold", 250),
                'consolidation_interval': flat_config.get("consolidation_interval", 3600),
                'similarity_threshold': flat_config.get("similarity_threshold", 0.75),
                'min_cluster_size': flat_config.get("min_cluster_size", 3),
                'max_clusters_per_consolidation': flat_config.get("max_clusters_per_consolidation", 5)
            }
        )
        
        # Start the thought loop
        logger.info("Starting thought loop...")
        thought_loop.start()
        
        # Start the memory consolidator
        logger.info("Starting memory consolidator...")
        memory_consolidator.start()
        
        # Keep the main thread running
        logger.info("Becoming AI system running in headless mode. Press Ctrl+C to exit.")
        try:
            # Block main thread until interrupted
            while True:
                # Log system status periodically
                memory_count = len(memory_system._get_all_memories())
                thought_count = thought_loop.thought_count
                time_since_last = (time.time() - memory_consolidator.last_consolidation.timestamp()) / 60
                next_consolidation = max(0, memory_consolidator.consolidation_interval / 60 - time_since_last)
                
                logger.info(
                    f"Status: Thoughts={thought_count}, Memories={memory_count}, "
                    f"Next consolidation in {next_consolidation:.1f} minutes"
                )
                
                time.sleep(300)  # Status update every 5 minutes
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Shutting down...")
            thought_loop.stop()
            memory_consolidator.stop()
            model_loader.unload_model()
            logger.info("Shutdown complete.")
    
    except Exception as e:
        logger.error(f"Error starting Becoming AI: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())