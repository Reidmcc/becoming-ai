import os
import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for
import json

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

# Import configuration utilities
from config_utils import setup_argparse, load_config

# Import memory system integration
from memory_system_integration import initialize_memory_system

def create_app(config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    
    # Initialize components
    from model_loader import LocalModelLoader
    from frontier_client import FrontierClient
    from thought_loop import ThoughtLoop
    from interface import ChatInterface
    
    # Ensure data directory exists
    if not config.get('use_remote_db', False):
        os.makedirs(os.path.dirname(os.path.abspath(config.get('db_path', 'data/memories.db'))), exist_ok=True)
    
    # Initialize components
    logger.info("Initializing system components...")
    
    try:
        # Initialize memory system with chat exports if available
        
        if config.get('load_chat_exports'):
            chat_exports_path = config.get('chat_exports_path')
        memory_system = initialize_memory_system(config, chat_exports_path)
        logger.info("Memory system initialized")
        
        model_loader = LocalModelLoader(
            model_name=config.get("model_name", "deepseek-ai/deepseek-R1-Distill-Llama-8B"),
            quantization=config.get("quantization", "int8")
        )
        logger.info("Model loader initialized")
        
        frontier_client = FrontierClient(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=config.get("frontier_model", "claude-3-7-sonnet-20250219"),
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
        app.memory_system = memory_system
        
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
                "type": thought["type"],
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
    
    # Memories API endpoints
    @app.route('/api/memories/recent', methods=['GET'])
    def get_recent_memories():
        limit = request.args.get('limit', 10, type=int)
        # This assumes the memory system has a get_recent_memories method
        memories = memory_system._get_all_memories()[:limit]
        
        # Format for JSON response
        formatted = []
        for memory in memories:
            formatted.append({
                "id": memory["id"],
                "title": memory.get("title", "Untitled Memory"),
                "content": memory["content"][:200] + "..." if len(memory["content"]) > 200 else memory["content"],
                "timestamp": memory["timestamp"].isoformat() if hasattr(memory["timestamp"], "isoformat") else str(memory["timestamp"]),
                "type": memory.get("metadata", {}).get("type", "general")
            })
        
        return jsonify({"memories": formatted})
    
    @app.route('/api/memories/<memory_id>', methods=['GET'])
    def get_memory_detail(memory_id):
        memory = memory_system.get_memory(memory_id)
        
        if not memory:
            return jsonify({"error": "Memory not found"}), 404
        
        # Format for JSON response
        formatted = {
            "id": memory["id"],
            "title": memory.get("title", "Untitled Memory"),
            "content": memory["content"],
            "timestamp": memory["timestamp"].isoformat() if hasattr(memory["timestamp"], "isoformat") else str(memory["timestamp"]),
            "metadata": memory.get("metadata", {}),
            "source_thoughts": memory.get("source_thoughts", [])
        }
        
        return jsonify({"memory": formatted})
    
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
            "api_calls_remaining": frontier_client.max_daily_calls - frontier_client.calls_today,
            "memory_count": len(memory_system._get_all_memories())
        })
    
    # Memory management endpoint
    @app.route('/api/memories/consolidate', methods=['POST'])
    def consolidate_memories():
        # Trigger memory consolidation (optional parameters could be added)
        consolidated_ids = memory_system.auto_consolidate_memories()
        
        return jsonify({
            "success": True,
            "consolidated_count": len(consolidated_ids),
            "consolidated_ids": consolidated_ids
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
    with app.app_context():
        logger.info("Starting thought loop...")
        thought_loop.start()
    
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Save vector store when the application shuts down"""
        if hasattr(app, 'memory_system'):
            logger.info("Saving vector store before shutdown...")
            app.memory_system.save_vector_store()
    
    return app

if __name__ == '__main__':
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args() 
    
    # Load configuration (from defaults, file, and CLI args)
    config = load_config(args.config, args) 
    
    # Display configuration
    logger.info("Starting web server with configuration:")
    for key, value in config.items():
        # Don't log password
        if key != "db_password":
            logger.info(f"  {key}: {value}")
    
    # Create and run the application
    app = create_app(config)
    app.run(
        host=config.get("server_host", "127.0.0.1"),  
        port=config.get("server_port", 5000), 
        debug=config.get("debug", False)  
    )