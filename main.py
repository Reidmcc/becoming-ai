# main.py
import os
import logging
import argparse
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
from memory_system import MemorySystem
from frontier_client import FrontierClient
from thought_loop import ThoughtLoop

def main():
    """Main entry point for the Becoming AI system"""
    
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Load configuration (from defaults, file, and CLI args)
    config = load_config(args.config, args)
    
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
        # Initialize memory system
        logger.info("Initializing memory system...")
        memory_system = MemorySystem(config=flat_config)
        
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
        
        # Start the thought loop
        logger.info("Starting thought loop...")
        thought_loop.start()
        
        # Keep the main thread running
        logger.info("Becoming AI system running. Press Ctrl+C to exit.")
        try:
            # Block main thread until interrupted
            import time
            while True:
                time.sleep(1)
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