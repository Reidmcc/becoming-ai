# config_utils.py
import os
import yaml
import logging
import argparse
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("ConfigUtils")

def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up argument parser with configuration options.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Run the Becoming AI system")
    
    # Configuration file
    parser.add_argument("--config", "-c", help="Path to configuration file (YAML or JSON)")
    parser.add_argument("--chat-exports", help="Path to chat exports JSON file")
    
    # Model options
    parser.add_argument("--local-model", help="Model to use for thought generation")
    parser.add_argument("--quantization", choices=["int8", "int4", "fp16"], help="Quantization level for model")
    parser.add_argument("--frontier-model", help="Frontier model to use for reflections")
    
    # Thought process options
    parser = parser.add_argument_group("Thought Process Options")
    parser.add_argument("--thought-interval", type=float, help="Interval between thoughts in seconds")
    parser.add_argument("--max-daily-calls", type=int, help="Maximum API calls to frontier model per day")
    
    # Memory options
    parser.add_argument("--use-remote-db", action="store_true",help="Use remote database instead of local SQLite")
    parser.add_argument("--remote-db-host", help="Remote database host (if using remote DB)")
    parser.add_argument("--remote-db-port", type=int, help="Remote database port (if using remote DB)")
    parser.add_argument("--remote-db-name", help="Remote database name (if using remote DB)")
    parser.add_argument("--remote-db-user", help="Remote database user (if using remote DB)")
    parser.add_argument("--remote-db-password", help="Remote database password (if using remote DB)")
    parser.add_argument("--remote-db-path", help="Path to SQLite database file (if using local DB)")
    parser.add_argument("--vector-store-path", help="Path to store vector embeddings")
    
    # Web server options
    parser.add_argument("--host",help="Host to run the web server on")
    parser.add_argument("--port", type=int, help="Port to run the web server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    return parser

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns:
        Dict containing default configuration values
    """
    return {
        # Model configuration
        "model_name": "deepseek-ai/deepseek-R1-Distill-Llama-8B",
        "quantization": "int8",
        "cache_dir": "models/cache",
        
        # Frontier model configuration
        "frontier_model": "claude-3-7-sonnet-20250219",
        "max_daily_calls": 100,
        
        # Thought process configuration
        "thought_interval": 60.0,  # seconds
        "max_thought_length": 500,
        
        # Memory configuration
        "use_remote_db": False,
        "db_path": "data/memories.db",
        "vector_store_path": "data/vectors.pkl",
        
        # Remote database settings (only used if use_remote_db is True)
        "remote_db_host": "localhost",
        "remote_db_port": 3306,
        "remote_db_name": "becoming_ai",
        "remote_db_user": "becoming_ai",
        "remote_db_password": "",
        
        # Web server configuration
        "server_host": "127.0.0.1",
        "server_port": 5000,
        
        # Additional settings
        "load_chat_exports": False
    }

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file (YAML or JSON)
        
    Returns:
        Dict containing configuration values
    """
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {str(e)}")
        return {}

def load_config(config_path: Optional[str] = None, cli_args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """
    Load configuration from defaults, file, and CLI arguments.
    
    Args:
        config_path: Path to configuration file (optional)
        cli_args: Parsed command-line arguments (optional)
        
    Returns:
        Dict containing merged configuration
    """
    # Get default configuration
    config = get_default_config()
    
    # Load configuration from file if provided
    if config_path:
        file_config = load_config_file(config_path)
        # Update config with file values (that aren't None)
        for key, value in file_config.items():
            if value is not None:
                config[key] = value
    
    # Update with CLI arguments if provided
    if cli_args:
        args_dict = vars(cli_args)
        # Only update with non-None values
        for key, value in args_dict.items():
            if value is not None:
                # Convert hyphen to underscore
                config_key = key.replace('-', '_')
                config[config_key] = value
    
    # Validate critical values
    if not config.get('db_path'):
        config['db_path'] = 'data/memories.db'
        logger.warning("Using default database path: data/memories.db")
    
    if not config.get('vector_store_path'):
        config['vector_store_path'] = 'data/vectors.pkl'
        logger.warning("Using default vector store path: data/vectors.pkl")
    
    return config