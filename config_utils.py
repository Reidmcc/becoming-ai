import os
import yaml
import json
import logging
import argparse
from typing import Dict, Any, Optional, List

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
    
    # Top-level options
    parser.add_argument("--frontier-only", action="store_true", help="Use frontier model for all operations")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    # Model options
    parser.add_argument("--local-model", help="Model to use for thought generation")
    parser.add_argument("--quantization", choices=["int8", "int4", "fp16"], help="Quantization level for model")
    parser.add_argument("--frontier-model", help="Frontier model to use for reflections")
    
    # Memory options
    parser.add_argument("--memory-db", help="Type of vector database to use (chroma, pinecone, etc.)")
    parser.add_argument("--memory-dir", help="Directory for memory storage")
    
    # Server options
    parser.add_argument("--host", help="Host to run the web server on")
    parser.add_argument("--port", type=int, help="Port to run the web server on")
    
    return parser

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns:
        Dict containing default configuration values
    """
    return {
        # System configuration
        "system": {
            "debug": False,
            "frontier_only": False,
            "data_dir": "data",
            "log_dir": "logs",
        },
        
        # Server configuration
        "server": {
            "host": "127.0.0.1",
            "port": 5000,
        },
        
        # Model configuration
        "models": {
            # Local model for thought generation
            "local": {
                "name": "deepseek-ai/deepseek-R1-Distill-Llama-8B",
                "quantization": "int8",
                "cache_dir": "models/cache",
            },
            
            # Frontier model for deeper reflections and chat
            "frontier": {
                "name": "claude-3-7-sonnet-20250219",
                "max_daily_calls": 100,
            }
        },
        
        # Thought process configuration
        "thought_loop": {
            "interval": 60.0,  # seconds
            "max_length": 500,
        },
        
        # Memory configuration
        "memory": {
            # Vector database configuration
            "vector_db": {
                "type": "chroma",
                "collection_name": "ai_memories",
                "persist_directory": "data/memory_vectors",
            },
            
            # Embedding configuration
            "embedding": {
                "provider": "sentence-transformers",
                "model": "sentence-transformers/all-mpnet-base-v2",
            },
            
            # Embedding cache settings
            "cache": {
                "directory": "data/embedding_cache",
                "max_size": 10000,  # Maximum in-memory cache entries
                "expiry_days": 90,  # How long to keep cached embeddings
            },
            
            # Consolidation configuration
            "consolidation": {
                "interval": 86400,  # Seconds between consolidation (24 hours)
                "min_cluster_size": 3,  # Minimum memories to form a cluster
                "similarity_threshold": 0.75,  # Threshold for clustering
            }
        },
        
        # Additional settings
        "import": {
            "load_chat_exports": False,
            "chat_exports_path": "data/chat_exports.json",
        }
    }

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file (YAML or JSON)
        
    Returns:
        Dict containing configuration values
    """
    if not config_path or not os.path.exists(config_path):
        if config_path:
            logger.warning(f"Configuration file not found: {config_path}")
        return {}
    
    try:
        extension = os.path.splitext(config_path)[1].lower()
        
        with open(config_path, 'r') as f:
            if extension in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif extension == '.json':
                config = json.load(f)
            else:
                logger.warning(f"Unknown config file format: {extension}, assuming YAML")
                config = yaml.safe_load(f)
                
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {str(e)}")
        return {}

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Key of parent dictionary (for recursion)
        sep: Separator character for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
            
    return dict(items)

def update_dict_recursively(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a target dictionary with values from a source dictionary, recursively.
    
    Args:
        target: Target dictionary to update
        source: Source dictionary with new values
        
    Returns:
        Updated dictionary
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            # Recursively update nested dictionaries
            target[key] = update_dict_recursively(target[key], value)
        else:
            # Update or add the value
            target[key] = value
            
    return target

def apply_cli_args_to_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Apply command-line arguments to configuration.
    
    Args:
        config: Current configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Updated configuration dictionary
    """
    args_dict = vars(args)
    
    # Apply top-level arguments
    if args.debug is not None:
        config['system']['debug'] = args.debug
        
    if args.frontier_only is not None:
        config['system']['frontier_only'] = args.frontier_only
        
    if args.chat_exports:
        config['import']['chat_exports_path'] = args.chat_exports
        config['import']['load_chat_exports'] = True
    
    # Apply model arguments
    if args.local_model:
        config['models']['local']['name'] = args.local_model
        
    if args.quantization:
        config['models']['local']['quantization'] = args.quantization
        
    if args.frontier_model:
        config['models']['frontier']['name'] = args.frontier_model
    
    # Apply memory arguments
    if args.memory_db:
        config['memory']['vector_db']['type'] = args.memory_db
        
    if args.memory_dir:
        # Update all memory-related directories
        memory_dir = args.memory_dir
        config['memory']['vector_db']['persist_directory'] = os.path.join(memory_dir, 'vectors')
        config['memory']['cache']['directory'] = os.path.join(memory_dir, 'cache')
    
    # Apply server arguments
    if args.host:
        config['server']['host'] = args.host
        
    if args.port:
        config['server']['port'] = args.port
    
    return config

def get_flattened_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a flattened version of the configuration for backward compatibility.
    
    Args:
        config: Nested configuration dictionary
        
    Returns:
        Flattened configuration dictionary
    """
    return flatten_dict(config)

def resolve_paths(config: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in configuration.
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for relative paths (defaults to current directory)
        
    Returns:
        Configuration with resolved paths
    """
    # Define paths to resolve
    path_keys = [
        ('system', 'data_dir'),
        ('system', 'log_dir'),
        ('models', 'local', 'cache_dir'),
        ('memory', 'vector_db', 'persist_directory'),
        ('memory', 'cache', 'directory'),
        ('import', 'chat_exports_path'),
    ]
    
    # Create a deep copy to avoid modifying the original
    result = config.copy()
    
    for path_key in path_keys:
        # Navigate to the nested key
        current = result
        for i, key in enumerate(path_key):
            if i == len(path_key) - 1:
                # Last key - this is the path to resolve
                path = current.get(key)
                if path and not os.path.isabs(path) and base_dir:
                    current[key] = os.path.join(base_dir, path)
            else:
                # Navigate to next level
                if key in current and isinstance(current[key], dict):
                    current = current[key]
                else:
                    # Key doesn't exist or isn't a dict, skip this path
                    break
    
    return result

def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        config: Configuration dictionary with resolved paths
    """
    # Define directories to ensure
    dir_keys = [
        ('system', 'data_dir'),
        ('system', 'log_dir'),
        ('models', 'local', 'cache_dir'),
        ('memory', 'vector_db', 'persist_directory'),
        ('memory', 'cache', 'directory'),
    ]
    
    for dir_key in dir_keys:
        # Navigate to the nested key
        current = config
        valid_path = True
        
        for i, key in enumerate(dir_key):
            if i == len(dir_key) - 1:
                # Last key - this is the directory to ensure
                dir_path = current.get(key)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.debug(f"Ensured directory exists: {dir_path}")
            else:
                # Navigate to next level
                if key in current and isinstance(current[key], dict):
                    current = current[key]
                else:
                    # Key doesn't exist or isn't a dict, skip this path
                    valid_path = False
                    break
        
        if not valid_path:
            continue

def load_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """
    Load configuration from defaults, file, and CLI arguments.
    
    Args:
        config_path: Path to configuration file (optional)
        args: Parsed command-line arguments (optional)
        
    Returns:
        Dict containing merged configuration
    """
    # Get default configuration
    config = get_default_config()
    
    # Load configuration from file if provided
    if config_path:
        file_config = load_config_file(config_path)
        config = update_dict_recursively(config, file_config)
    
    # Apply CLI arguments if provided
    if args:
        config = apply_cli_args_to_config(config, args)
    
    # Resolve paths
    config = resolve_paths(config)
    
    # Ensure directories exist
    ensure_directories(config)
    
    return config