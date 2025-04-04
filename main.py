import os
import logging
import argparse
import time
from typing import Dict, Any
from config_utils import setup_argparse, load_config, get_flattened_config
from model_loader import LocalModelLoader
from frontier_client import FrontierClient
from thought_loop import ThoughtLoop
from memory.repository import ChromaDBMemoryRepository
from memory.embedding import FrontierEmbeddingService
from memory.manager import MemoryManager

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
        
        frontier_client = FrontierClient(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model=config.get("frontier_model", "claude-3-7-sonnet-latest"),
        max_daily_calls=config.get("max_daily_calls", 100)
    )

        embedding_service = FrontierEmbeddingService(frontier_client)

        memory_repository = ChromaDBMemoryRepository(
            collection_name="ai_memories",
            persist_directory=os.path.join(config.get("data_dir", "data"), "memory_vectors"),
            embedding_function=embedding_service.get_embedding
        )
        
        # Initialize model loader
        logger.info("Initializing model loader...")
        model_loader = LocalModelLoader(
            model_name=flat_config["model_name"],
            quantization=flat_config["quantization"]
        )

        # Initialize thought loop
        logger.info("Initializing thought loop...")
        thought_loop = ThoughtLoop(
            model_loader=model_loader,
            memory_system=memory_repository,
            frontier_client=frontier_client,
            config=flat_config
        )
        
        # Start the thought loop
        logger.info("Starting thought loop...")
        thought_loop.start()
        
        # Keep the main thread running
        logger.info("Becoming AI system running in headless mode. Press Ctrl+C to exit.")
        try:
            # Block main thread until interrupted
            while True:
                # Log system status periodically
                thought_count = thought_loop.thought_count             
                time.sleep(300)  # Status update every 5 minutes
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Shutting down...")
            thought_loop.stop()
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