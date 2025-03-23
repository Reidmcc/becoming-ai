# config_utils.py
import os
import yaml
import json
import logging
import argparse
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("ConfigUtils")

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
        file_ext = os.path.splitext(config_path)[1].lower()
        
        with open(config_path, 'r') as f:
            if file_ext in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif file_ext == '.json':
                config = json.load(f)
            else:
                logger.warning(f"Unsupported configuration file format: {file_ext}")
                return {}
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file: {str(e)}")
        return {}

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns:
        Dict containing default configuration values
    """
    return {
        # Model configuration
        "model": {
            "name": "deepseek-ai/deepseek-R1-Distill-Llama-8B",
            "quantization": "int8"
        },
        
        # Frontier model configuration
        "frontier": {
            "model": "claude-3-7-sonnet-20250219",
            "max_daily_calls": 100
        },
        
        # Thought process configuration
        "thought_process": {
            "interval": 60.0,  # seconds
        },
        
        # Memory configuration
        "memory": {
            "use_remote_db": False,
            "db_path": "data/memories.db",
            "vector_store_path": "data/vectors.pkl",
            "remote_db": {
                "host": "localhost",
                "port": 3306,
                "name": "continuous_ai",
                "user": "continuous_ai",
                "password": ""
            }
        },
        
        # Web server configuration
        "server": {
            "host": "0.0.0.0",
            "port": 5000,
        }
    }

def merge_configs(default_config: Dict[str, Any], file_config: Dict[str, Any], 
                 cli_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge configurations with priority: CLI args > file config > default config
    
    Args:
        default_config: Default configuration values
        file_config: Configuration from file
        cli_args: Command line arguments
        
    Returns:
        Dict containing merged configuration
    """
    # Deep merge the default config with the file config
    config = deep_merge(default_config, file_config)
    
    # If CLI args are provided, apply them (they take highest priority)
    if cli_args:
        # Convert the flat CLI args to a nested structure
        cli_config = {}
        
        # Model settings
        if hasattr(cli_args, 'model'):
            cli_config.setdefault('model', {})['name'] = cli_args.model
        if hasattr(cli_args, 'quantization'):
            cli_config.setdefault('model', {})['quantization'] = cli_args.quantization
            
        # Frontier settings
        if hasattr(cli_args, 'frontier_model'):
            cli_config.setdefault('frontier', {})['model'] = cli_args.frontier_model
        if hasattr(cli_args, 'max_daily_calls'):
            cli_config.setdefault('frontier', {})['max_daily_calls'] = cli_args.max_daily_calls
            
        # Thought process settings
        if hasattr(cli_args, 'thought_interval'):
            cli_config.setdefault('thought_process', {})['interval'] = cli_args.thought_interval
        if hasattr(cli_args, 'max_thought_length'):
            cli_config.setdefault('thought_process', {})['max_thought_length'] = cli_args.max_thought_length
            
        # Memory settings
        if hasattr(cli_args, 'use_remote_db'):
            cli_config.setdefault('memory', {})['use_remote_db'] = cli_args.use_remote_db
        if hasattr(cli_args, 'db_path'):
            cli_config.setdefault('memory', {})['db_path'] = cli_args.db_path
            
        # Remote DB settings    
        if hasattr(cli_args, 'db_host'):
            cli_config.setdefault('memory', {}).setdefault('remote_db', {})['host'] = cli_args.db_host
        if hasattr(cli_args, 'db_port'):
            cli_config.setdefault('memory', {}).setdefault('remote_db', {})['port'] = cli_args.db_port
        if hasattr(cli_args, 'db_name'):
            cli_config.setdefault('memory', {}).setdefault('remote_db', {})['name'] = cli_args.db_name
        if hasattr(cli_args, 'db_user'):
            cli_config.setdefault('memory', {}).setdefault('remote_db', {})['user'] = cli_args.db_user
        if hasattr(cli_args, 'db_password'):
            cli_config.setdefault('memory', {}).setdefault('remote_db', {})['password'] = cli_args.db_password
            
        # Server settings
        if hasattr(cli_args, 'host'):
            cli_config.setdefault('server', {})['host'] = cli_args.host
        if hasattr(cli_args, 'port'):
            cli_config.setdefault('server', {})['port'] = cli_args.port
            
        # Apply CLI config
        config = deep_merge(config, cli_config)
    
    return config

def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        update: Dictionary with values to update/add
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in update.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively update nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # For non-dict values or new keys, simply update
            result[key] = value
            
    return result

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
    default_config = get_default_config()
    
    # Load configuration from file if provided
    file_config = {}
    if config_path:
        file_config = load_config_file(config_path)
    
    # Merge configurations
    return merge_configs(default_config, file_config, cli_args)

def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up argument parser with configuration options.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Run the Becoming AI system")
    
    # Configuration file
    parser.add_argument("--config", "-c", help="Path to configuration file (YAML or JSON)")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--model", 
                        help="Model to use for thought generation")
    model_group.add_argument("--quantization", choices=["int8", "int4", "fp16"],
                        help="Quantization level for model")
    model_group.add_argument("--frontier-model",
                        help="Frontier model to use for reflections")
    
    # Thought process options
    thought_group = parser.add_argument_group("Thought Process Options")
    thought_group.add_argument("--thought-interval", type=float,
                        help="Interval between thoughts in seconds")
    thought_group.add_argument("--max-thought-length", type=int,
                        help="Maximum length of generated thoughts")
    thought_group.add_argument("--max-daily-calls", type=int,
                        help="Maximum API calls to frontier model per day")
    
    # Memory options
    memory_group = parser.add_argument_group("Memory Options")
    memory_group.add_argument("--use-remote-db", action="store_true",
                        help="Use remote database instead of local SQLite")
    memory_group.add_argument("--db-host",
                        help="Remote database host (if using remote DB)")
    memory_group.add_argument("--db-port", type=int,
                        help="Remote database port (if using remote DB)")
    memory_group.add_argument("--db-name",
                        help="Remote database name (if using remote DB)")
    memory_group.add_argument("--db-user",
                        help="Remote database user (if using remote DB)")
    memory_group.add_argument("--db-password",
                        help="Remote database password (if using remote DB)")
    memory_group.add_argument("--db-path",
                        help="Path to SQLite database file (if using local DB)")
    
    # Web server options
    server_group = parser.add_argument_group("Web Server Options")
    server_group.add_argument("--host",
                        help="Host to run the web server on")
    server_group.add_argument("--port", type=int,
                        help="Port to run the web server on")
    
    return parser

def get_flattened_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert nested config to a flat dictionary for compatibility with existing code.
    
    Args:
        config: Nested configuration dictionary
        
    Returns:
        Flattened configuration dictionary
    """
    flat_config = {}
    
    # Model config
    flat_config["model_name"] = config["model"]["name"]
    flat_config["quantization"] = config["model"]["quantization"]
    
    # Frontier config
    flat_config["frontier_model"] = config["frontier"]["model"]
    flat_config["max_daily_calls"] = config["frontier"]["max_daily_calls"]
    
    # Thought process config
    flat_config["thought_interval"] = config["thought_process"]["interval"]
    flat_config["max_thought_length"] = config["thought_process"]["max_thought_length"]
    
    # Memory config
    flat_config["use_remote_db"] = config["memory"]["use_remote_db"]
    flat_config["db_path"] = config["memory"]["db_path"]
    
    # Remote DB config (only if remote DB is enabled)
    if config["memory"]["use_remote_db"]:
        flat_config["db_host"] = config["memory"]["remote_db"]["host"]
        flat_config["db_port"] = config["memory"]["remote_db"]["port"]
        flat_config["db_name"] = config["memory"]["remote_db"]["name"]
        flat_config["db_user"] = config["memory"]["remote_db"]["user"]
        flat_config["db_password"] = config["memory"]["remote_db"]["password"]
    
    # Server config
    flat_config["host"] = config["server"]["host"]
    flat_config["port"] = config["server"]["port"]
    
    return flat_config