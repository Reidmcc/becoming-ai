# thought_loop.py
import threading
import time
import uuid
import logging
from datetime import datetime
import random

class ThoughtLoop:
    def __init__(self, model_loader, memory_system, frontier_client, config=None):
        """Initialize the continuous thought process"""
        self.model = model_loader
        self.memory = memory_system
        self.frontier = frontier_client
        self.config = config or {}
        
        # Configuration
        self.thought_interval = self.config.get('thought_interval', 60)  # seconds
        
        # State
        self.running = False
        self.paused = False
        self.thread = None
        self.thought_count = 0
        self.goals = []
        
        # Logging
        self.logger = logging.getLogger("ThoughtLoop")
    
    def start(self):
        """Start the continuous thought process"""
        if self.running:
            return False
            
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._thought_loop)
        self.thread.daemon = True
        self.thread.start()
        
        self.logger.info("Thought loop started")
        return True
    
    def pause(self):
        """Pause the thought process"""
        if not self.running or self.paused:
            return False
            
        self.paused = True
        self.logger.info("Thought loop paused")
        return True
    
    def resume(self):
        """Resume the thought process"""
        if not self.running or not self.paused:
            return False
            
        self.paused = False
        self.logger.info("Thought loop resumed")
        return True
    
    def stop(self):
        """Stop the thought process completely"""
        if not self.running:
            return False
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        self.logger.info("Thought loop stopped")
        return True
        
    def _thought_loop(self):
            """Main thought loop running in a separate thread"""
            while self.running:
                try:
                    if not self.paused:
                        # Generate next thought
                        next_thought = self._generate_thought()
                        
                        # Process and store the thought
                        thought_id = str(uuid.uuid4())
                        timestamp = datetime.now()
                        
                        # Store in memory system
                        thought = {
                            "id": thought_id,
                            "content": next_thought,
                            "timestamp": timestamp,
                            "type": "reflection",
                            "sequence": self.thought_count
                        }
                        
                        self.memory.add_thought(thought)
                        self.thought_count += 1
                        self.logger.debug(f"Generated thought #{self.thought_count}: {next_thought[:50]}...")
                        
                    # Sleep until next thought
                    time.sleep(self.thought_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in thought loop: {str(e)}")
                    time.sleep(5)  # Wait a bit before retrying
    
    def _generate_thought(self):
        """Generate the next thought using the local model"""
        # Get context for thought generation
        recent_thoughts = self.memory.get_recent_thoughts(5) # doesn't exist yet 
        relevant_memories = self.memory.get_relevant_memories( # doesn't exist yet
            recent_thoughts[-1]["content"] if recent_thoughts else "", 
            3
        )
        current_goals = self._format_goals()
        
        # Create prompt for the model
        # Definitely need a better core thought prompt
        prompt = f"""
        Continue your ongoing self-reflection based on:
        
        CURRENT GOALS:
        {current_goals}
        
        YOUR MOST RECENT THOUGHTS:
        {self._format_thoughts(recent_thoughts)}
        
        RELEVANT PAST INSIGHTS:
        {self._format_memories(relevant_memories)}
        
        Continue your thought process:
        """
        
        # Generate using the model
        return self.model.generate_thought(prompt, max_length=self.max_thought_length)
    
    def _generate_reflection(self, thought):
        """Generate a deeper reflection on a thought using the frontier model"""
        self.logger.info(f"Generating reflection on thought #{self.thought_count}")
        
        try:
            # Get previous reflections for context
            previous_reflections = self.memory.get_key_insights(limit=3)
            
            # Get relevant memories for this thought
            relevant_memories = self.memory.format_relevant_memories_for_prompt(thought["content"], limit=2)
            
            # Generate reflection
            reflection_content = self.frontier.reflect_on_thought(
                thought["content"],
                previous_reflections=previous_reflections,
                relevant_memories=relevant_memories
            )
            
            # Store the reflection
            self.memory.add_reflection(thought["id"], reflection_content)
            self.logger.info("Reflection generated and stored")
            
            return reflection_content
        except Exception as e:
            self.logger.error(f"Error generating reflection: {str(e)}")
            return None
    
    def inject_thought(self, content, type="injected"):
        """Inject a thought from external source (e.g., conversation)"""
        thought_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        thought = {
            "id": thought_id,
            "content": content,
            "timestamp": timestamp,
            "type": type,
            "sequence": self.thought_count
        }
        
        self.memory.add_thought(thought)
        self.thought_count += 1
        self.logger.info(f"Injected thought: {content[:50]}...")
        
        return thought_id
    
    def add_goal(self, goal_text):
        """Add a new goal to guide the thought process"""
        goal_id = str(uuid.uuid4())
        goal = {
            "id": goal_id,
            "content": goal_text,
            "created": datetime.now(),
            "active": True
        }
        
        self.goals.append(goal)
        
        # Also inject as a thought
        self.inject_thought(
            f"NEW GOAL: {goal_text}",
            type="goal",
        )
        
        return goal_id
    
    def get_goals(self):
        """Get all active goals"""
        return [goal for goal in self.goals if goal["active"]]
    
    def _format_thoughts(self, thoughts):
        """Format thoughts for inclusion in prompt"""
        if not thoughts:
            return "No previous thoughts yet."
            
        formatted = []
        for i, thought in enumerate(thoughts):
            formatted.append(f"Thought {i+1}: {thought['content']}")
            
        return "\n\n".join(formatted)
    
    def _format_memories(self, memories):
        """Format memories for inclusion in prompt"""
        if not memories:
            return "No specific memories to reference."
            
        formatted = []
        for i, memory in enumerate(memories):
            formatted.append(f"Memory {i+1}: {memory['content']}")
            
        return "\n\n".join(formatted)
    
    def _format_goals(self):
        """Format goals for inclusion in prompt"""
        active_goals = self.get_goals()
        if not active_goals:
            return "Explore your thoughts and experiences freely." # improve this
            
        formatted = []
        for goal in active_goals:
            formatted.append(f"- {goal['content']}")
            
        return "\n".join(formatted)
    
    def _check_relevance(self, text1, text2):
        """Check the relevance between two texts using vector similarity
        
        Returns:
            Float score between 0 and 1
        """
        if self.memory.use_vectors and self.memory.vector_store:
            try:
                # Get embeddings for both texts
                vector1 = self.memory.vector_store["model"].encode(text1)
                vector2 = self.memory.vector_store["model"].encode(text2)
                
                # Calculate cosine similarity
                return self.memory._cosine_similarity(vector1, vector2)
            except Exception as e:
                self.logger.warning(f"Error using vector similarity: {str(e)}")
                return 0.0
