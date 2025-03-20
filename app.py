# app.py
from flask import Flask, render_template, jsonify, request, redirect, url_for
import os
import logging
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("continuous_ai.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ContinuousAI")

def create_app(config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    
    # Default configuration
    if config is None:
        config = {
            "model_name": "deepseek-ai/deepseek-R1-Distill-Llama-8B",
            "quantization": "int8",
            "thought_interval": 60,  # 1 minute between thoughts
            "max_daily_calls": 100,   # API call limit
            "use_vectors": True,      # Use vector embeddings
            "use_remote_db": False,   # Use local SQLite by default
            "db_path": "data/memories.db"
        }
    
    # Initialize components
    from model_loader import LocalModelLoader
    from memory_system import MemorySystem
    from frontier_client import FrontierClient
    from thought_loop import ThoughtLoop
    from interface import ChatInterface
    
    # Ensure data directory exists
    if not config.get('use_remote_db', False):
        os.makedirs(os.path.dirname(os.path.abspath(config.get('db_path', 'data/memories.db'))), exist_ok=True)
    
    # Initialize components
    logger.info("Initializing system components...")
    
    try:
        memory_system = MemorySystem(config=config)
        logger.info("Memory system initialized")
        
        model_loader = LocalModelLoader(
            model_name=config.get("model_name", "deepseek-ai/deepseek-R1-Distill-Llama-8B"),
            quantization=config.get("quantization", "int8")
        )
        logger.info("Model loader initialized")
        
        frontier_client = FrontierClient(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_daily_calls=config.get("max_daily_calls", 100)
        )
        logger.info("Frontier client initialized")
        
        thought_loop = ThoughtLoop(
            model_loader=model_loader,
            memory_system=memory_system,
            frontier_client=frontier_client,
            config=config
        )
        logger.info("Thought loop initialized")
        
        chat_interface = ChatInterface(
            thought_loop=thought_loop,
            frontier_client=frontier_client,
            memory_system=memory_system
        )
        logger.info("Chat interface initialized")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise
    
    # HTML template for the chat interface
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Chat API endpoint
    @app.route('/api/chat', methods=['POST'])
    def handle_chat():
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        response = chat_interface.handle_message(user_message)
        return jsonify({"response": response})
    
    # Recent thoughts API endpoint
    @app.route('/api/thoughts/recent', methods=['GET'])
    def get_recent_thoughts():
        limit = request.args.get('limit', 20, type=int)
        thoughts = memory_system.get_recent_thoughts(limit)
        
        # Format for JSON response
        formatted = []
        for thought in thoughts:
            formatted.append({
                "id": thought["id"],
                "content": thought["content"],
                "timestamp": thought["timestamp"].isoformat(),
                "type": thought["type"]
            })
        
        return jsonify({"thoughts": formatted})
    
    # Recent reflections API endpoint
    @app.route('/api/reflections/recent', methods=['GET'])
    def get_recent_reflections():
        limit = request.args.get('limit', 5, type=int)
        reflections = memory_system.get_recent_reflections(limit)
        
        # Format for JSON response
        formatted = []
        for reflection in reflections:
            formatted.append({
                "id": reflection["id"],
                "thought_id": reflection["thought_id"],
                "content": reflection["content"],
                "timestamp": reflection["timestamp"].isoformat(),
                "thought_content": reflection["thought_content"]
            })
        
        return jsonify({"reflections": formatted})
    
    # System control API endpoints
    @app.route('/api/system/start', methods=['POST'])
    def start_system():
        success = thought_loop.start()
        return jsonify({"success": success, "status": "running" if success else "unchanged"})
    
    @app.route('/api/system/pause', methods=['POST'])
    def pause_system():
        success = thought_loop.pause()
        return jsonify({"success": success, "status": "paused" if success else "unchanged"})
    
    @app.route('/api/system/resume', methods=['POST'])
    def resume_system():
        success = thought_loop.resume()
        return jsonify({"success": success, "status": "running" if success else "unchanged"})
    
    @app.route('/api/system/status', methods=['GET'])
    def system_status():
        return jsonify({
            "running": thought_loop.running,
            "paused": thought_loop.paused,
            "thought_count": thought_loop.thought_count,
            "api_calls_today": frontier_client.calls_today,
            "api_calls_remaining": frontier_client.max_daily_calls - frontier_client.calls_today
        })
    
    # Goal management API endpoints
    @app.route('/api/goals', methods=['GET'])
    def get_goals():
        goals = thought_loop.get_goals()
        
        # Format for JSON response
        formatted = []
        for goal in goals:
            formatted.append({
                "id": goal["id"],
                "content": goal["content"],
                "created": goal["created"].isoformat(),
            })
            
        return jsonify({"goals": formatted})
    
    @app.route('/api/goals', methods=['POST'])
    def add_goal():
        data = request.json
        goal_text = data.get('text', '')
        
        if not goal_text:
            return jsonify({"error": "No goal text provided"}), 400
        
        goal_id = thought_loop.add_goal(goal_text)
        return jsonify({"id": goal_id, "status": "added"})
    
    # Start the continuous thought process
    @app.before_first_request
    def start_background_tasks():
        # Start the thought loop
        logger.info("Starting thought loop...")
        thought_loop.start()
    
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Continuous AI web interface")
    
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
    parser.add_argument("--thought-interval", type=float, default=60.0,
                      help="Interval between thoughts in seconds")
    parser.add_argument("--max-daily-calls", type=int, default=100,
                      help="Maximum API calls to frontier model per day")
    
    # Vector options
    parser.add_argument("--use-vectors", action="store_true", default=True,
                      help="Use vector embeddings for memory retrieval")
    
    # Web server options
    parser.add_argument("--host", default="0.0.0.0",
                      help="Host to run the web server on")
    parser.add_argument("--port", type=int, default=5000,
                      help="Port to run the web server on")
    parser.add_argument("--debug", action="store_true",
                      help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        # Model config
        "model_name": args.model,
        "quantization": args.quantization,
        
        # Thought process config
        "thought_interval": args.thought_interval,
        "reflection_threshold": args.reflection_threshold,
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
    
    app = create_app(config)
    app.run(host=args.host, port=args.port, debug=args.debug)