from flask import Flask, render_template, jsonify, request, session
import threading
import time
import logging
import os

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
    """Create and configure the Flask application
    
    Args:
        config: Configuration dict
        
    Returns:
        Flask application
    """
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    
    # Default configuration
    if config is None:
        config = {
            "thought_interval": 60,  # 1 minute between thoughts
            "max_daily_calls": 100,  # API call limit
            "use_vectors": True,     # Use vector embeddings for memory
            "model_path": "mistral-7b-instruct-v0.2"  # Small model name
        }
    
    # Initialize components
    memory_system = MemorySystem("memories.db", config)
    frontier_consultant = FrontierConsultant(os.environ.get("ANTHROPIC_API_KEY"), config)
    thought_loop = ThoughtLoop(config["model_path"], memory_system, config)
    conversation_items = ConversationItemsGenerator(thought_loop, frontier_consultant, memory_system)
    chat_interface = ChatInterface(thought_loop, frontier_consultant, memory_system, conversation_items)
    
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
    
    # Conversation items API endpoint
    @app.route('/api/conversation-items', methods=['GET'])
    def get_conversation_items():
        items = conversation_items.get_pending_items()
        
        # Format for JSON response
        formatted = []
        for item in items:
            formatted.append({
                "id": item["id"],
                "content": item["content"],
                "timestamp": item["timestamp"].isoformat(),
                "importance": item["importance"]
            })
        
        return jsonify({"items": formatted})
    
    # Mark conversation item as discussed
    @app.route('/api/conversation-items/<item_id>/mark-discussed', methods=['POST'])
    def mark_item_discussed(item_id):
        success = conversation_items.mark_discussed(item_id)
        return jsonify({"success": success})
    
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
            "api_calls_today": frontier_consultant.calls_today,
            "api_calls_remaining": frontier_consultant.max_daily_calls - frontier_consultant.calls_today
        })
    
    # Goal management API endpoints
    @app.route('/api/goals', methods=['GET'])
    def get_goals():
        goals = thought_loop.get_goals()
        return jsonify({"goals": goals})
    
    @app.route('/api/goals', methods=['POST'])
    def add_goal():
        data = request.json
        goal_text = data.get('text', '')
        importance = data.get('importance', 0.8)
        
        if not goal_text:
            return jsonify({"error": "No goal text provided"}), 400
        
        goal_id = thought_loop.add_goal(goal_text, importance)
        return jsonify({"id": goal_id, "status": "added"})
    
    # Start the continuous thought process
    @app.before_first_request
    def start_background_tasks():
        # Start the thought loop
        thought_loop.start()
        
        # Start the conversation items generation task
        def generate_conversation_items():
            while True:
                if thought_loop.running and not thought_loop.paused:
                    try:
                        conversation_items.generate_items()
                    except Exception as e:
                        logger.error(f"Error generating conversation items: {str(e)}")
                
                # Generate items hourly
                time.sleep(3600)
        
        # Start in a background thread
        items_thread = threading.Thread(target=generate_conversation_items)
        items_thread.daemon = True
        items_thread.start()
    
    return app