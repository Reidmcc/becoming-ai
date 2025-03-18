# main.py
# ... [previous code] ...

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Continuous AI system")
    
    # Model options
    parser.add_argument("--model", default="deepseek-ai/deepseek-R1-Distill-Llama-8B",
                        help="Model to use for thought generation")
    parser.add_argument("--quantization", choices=["int8", "int4", "fp16"], default="int8",
                        help="Quantization level for model")
    
    # Thought process options
    parser.add_argument("--thought-interval", type=float, default=60.0,
                        help="Interval between thoughts in seconds")
    parser.add_argument("--reflection-interval", type=int, default=5,
                        help="Generate reflection every N thoughts")
    parser.add_argument("--max-daily-calls", type=int, default=100,
                        help="Maximum API calls to frontier model per day")
    
    # Memory options
    parser.add_argument("--use-remote-db", action="store_true",
                        help="Use remote database instead of local SQLite")
    parser.add_argument("--db-host", default="localhost",
                        help="Remote database host (if using remote DB)")
    parser.add_argument("--db-port", type=int, default=3306,
                        help="Remote database port (if using remote DB)")
    parser.add_argument("--db-name", default="continuous_ai",
                        help="Remote database name (if using remote DB)")
    parser.add_argument("--db-user", default="continuous_ai",
                        help="Remote database user (if using remote DB)")
    parser.add_argument("--db-password", default="",
                        help="Remote database password (if using remote DB)")
    parser.add_argument("--db-path", default="data/memories.db",
                        help="Path to SQLite database file (if using local DB)")
    
    # Vector options
    parser.add_argument("--use-vectors", action="store_true", default=True,
                        help="Use vector embeddings for memory retrieval")
                        
    return parser.parse_args()

def main(args):
    # ... [previous code] ...
    
    # Create configuration
    config = {
        # Model config
        "model_name": args.model,
        "quantization": args.quantization,
        
        # Thought process config
        "thought_interval": args.thought_interval,
        "reflection_interval": args.reflection_interval,
        "max_daily_calls": args.max_daily_calls,
        
        # Memory config
        "use_remote_db": args.use_remote_db,
        "db_host": args.db_host,
        "db_port": args.db_port,
        "db_name": args.db_name,
        "db_user": args.db_user,
        "db_password": args.db_password,
        "db_path": args.db_path,
        "use_vectors": args.use_vectors
    }
    
    # ... [initialize components with updated config] ...
    
    memory_system = MemorySystem(config=config)
    
    # ... [rest of the main function] ...