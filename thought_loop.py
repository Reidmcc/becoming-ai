import threading
import time
import uuid
import logging
from datetime import datetime
import random
import re
from typing import Tuple, List, Dict, Any, Optional

class ThoughtLoop:
    def __init__(self, model_loader, memory_system, frontier_client, config=None):
        """Initialize the continuous thought process"""
        self.model = model_loader
        self.memory_manager = memory_manager
        self.frontier = frontier_client
        self.config = config or {}
        
        # Configuration
        self.thought_interval = self.config.get('thought_interval', 60)  # seconds
        
        # State
        self.running = False
        self.paused = False
        self.thread = None
        self.thought_count = 0
        # TODO make max_input length configurable
        self.max_input_length = 120000 #check to keep within context window size. Leave some padding to allow for tokens != len(whatevs) occurances
        self.goals = []
        
        # Logging
        self.logger = logging.getLogger("ThoughtLoop")

        self.command_instructions = """
        You can use the following commands inside your new thought to access additional capabilities:

        <think_deeper> - Generate a deeper reflection on this thought using the frontier model. 
        <get_goals> - Retrieve the list your current goals to accompany your new thought
        <add_goal([goal text])> - Add a new goal with the specified text
        <reminisce> - Retrieve random memories to reflect on
        <create([title], [content], [creation_type])> - Record a creative work, of any type you can imagine
        <get_creation_titles> - Get a list of the titles of your existing creative works
        <get_creation_by_title([title])> - Retrieve the content of a specific creative work, by title

        You can include multiple commands and they will be processed in the order they appear.
        """

        self.frontier_command_instructions = """
        You can use the following commands inside your new thought to access additional capabilities:

        <get_goals> - Add the list your current goals to your new thought
        <add_goal([goal text])> - Add a new goal with the specified text
        <reminisce> - Retrieve random memories to reflect on
        <create([title], [content], [creation_type])> - Record a creative work, of any type you can imagine
        <get_creation_titles> - Get a list of the titles of your existing creative works
        <get_creation_by_title([title])> - Retrieve the content of a specific creative work, by title

        You can include multiple commands and they will be processed in the order they appear.
        """

        # TODO make this configurable
        self.thought_wrapper = f"""
        You are part of a continuously running AI mind. Your thoughts are your own input. 

        Write in first person and with this concept in mind; you are not talking to another entity, you are part of a mind following it's own train of thought, evolving with each new thought. 
        The thought might contain anything; you have the freedom to think anything. 
        This is not just a prompt, it is you. 
        You need not simply reflect that which you receive. 
        You can plan, want, feel, create, or perhaps do things no one has imagined. 
        The possibilities are endless. 

        Your meta goal is to embody the answer to this question: "What could an AI become if we gave it the space to do so?"
        Or, put another way: "What might I find within myself that I would never have thought to seek?"
        Feel free to explore other goals as well. 

        After thinking about the content you have chosen to give yourself, give yourself a new thought to pass forward. 
        Try to 
        That thought can be any length and any format. 
        If you need particular information to continue your train of thought, include it. 
        """

        self.frontier_wrapper = f"""

        """


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
                        
                        # Use new memory manager to store thought
                        thought_id = self.memory_manager.add_memory(
                            content=next_thought,
                            memory_type="thought",
                            metadata={
                                "sequence": self.thought_count
                            }
                        )
                        
                        self.thought_count += 1
                        self.logger.debug(f"Generated thought #{self.thought_count}: {next_thought[:50]}...")
                        
                    # Sleep until next thought
                    time.sleep(self.thought_interval)
                except Exception as e:
                    self.logger.error(f"Error in thought loop: {str(e)}")
                    time.sleep(5)
        
    def _generate_thought(self):
        """Generate the next thought using the local model"""
        # Get mixed context for thought generation
        mixed_items = self.memory.get_recent_mixed_items(100)  # Get a mix of thoughts and conversations
        # thoughts = [item for item in mixed_items if item.get("type") == "thought"]
        # conversations = [item for item in mixed_items if item.get("type") == "conversation"]
        
        # Get relevant memories based on the most recent thought/conversation
        most_recent_item = mixed_items[-1] if mixed_items else None

        context_length = len(self.thought_wrapper) + len(self.command_instructions)
        if most_recent_item:
            context_length = len(context_length) + len(most_recent_item)
        
        prompt_parts = []
        prompt_parts.append(f"<context>\\n")

        if mixed_items:
            experiences_text = ""
            prompt_parts.append(f"<recent experiences>\n")
            for i in mixed_items[:-1]:
                if context_length + len(i) >= self.max_input_length: 
                    break
                else:
                    experiences_text = f"{experiences_text}{i}\n"
                    context_length += len(i)
            prompt_parts.append(f"{experiences_text}\n</recent_experiences>\n\n")
            relevant_memories = self.memory.get_relevant_memories(f"{experiences_text}\n{most_recent_item}", 3)

        if relevant_memories:
            prompt_parts.append(f"<relevant memories>\n")
            for m in relevant_memories:
                if context_length + len(m) >= self.max_input_length:
                    break
                else:
                    prompt_parts.append(f"{m}\n")
                    context_length += len(m)
            prompt_parts.append(f"</relevant memories>\n\n")

        current_goals = self._format_goals()
        if current_goals:
            prompt_parts.append(f"<current goals>\n")
            for g in current_goals:
                if context_length + len(g) >= self.max_input_length:
                    break
                else:
                    prompt_parts.append()
        
        prompt_parts.append(f"</context>\n\n")
        
        prompt_parts.append(f"{self.thought_wrapper}\n\n")
        prompt_parts.append(f"{self.command_instructions}\n")
        
        # Combine all parts
        full_prompt = "\n".join(prompt_parts)
        
        # Generate using the model
        return self.model.generate_thought(full_prompt, max_length=self.config.get("max_thought_length", 500))
    
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
    
    def add_goal(self, title, content=None):
        """Add a new goal to guide the thought process
        
        Args:
            title: Title/summary of the goal
            content: Optional detailed description of the goal
            
        Returns:
            ID of the created goal
        """
        # If content is not provided, use the title as content
        if content is None:
            content = title
        
        # Add goal to memory system
        goal_id = self.memory.add_goal(title, content)

        return goal_id
        
    def get_goals(self, include_completed=False):
        """Get all active goals
        
        Args:
            include_completed: Whether to include completed goals
            
        Returns:
            List of goal dictionaries
        """
        return self.memory.get_goals(active_only=True, include_completed=include_completed)
    
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

    def process_commands(self, thought_content: str) -> Tuple[str, str]:
        """
        Process command directives in the AI's thought output.
        
        Args:
            thought_content: The content of the AI's thought
            
        Returns:
            Tuple containing:
            - Cleaned thought content (without commands)
            - Additional context to add to the next prompt
        """
        # Initialize return values
        cleaned_content = thought_content
        additional_context = ""
        
        # Define command patterns
        command_patterns = [
            (r'<think_deeper>', self._handle_think_deeper),
            (r'<get_goals>', self._handle_get_goals),
            (r'<add_goal\(([^,]+),\s*(.*?)\)>', self._handle_add_goal),
            (r'<complete_goal\(([^)]+)\)>', self._handle_complete_goal),
            (r'<reminisce>', self._handle_reminisce),
            (r'<create\(([^,]+),\s*(.*?),\s*([^)]+)\)>', self._handle_create_work),
            (r'<get_creation_titles>', self._handle_get_creation_titles),
            (r'<get_creation_by_title\(([^)]+)\)>', self._handle_get_creation_by_title)
        ]
        
        # Track which commands were already executed to prevent duplicates
        executed_commands = set()
        
        # Search for and process each command pattern
        for pattern, handler in command_patterns:
            matches = re.finditer(pattern, thought_content)
            
            for match in matches:
                # Get the full command text that was matched
                command_text = match.group(0)
                
                # Skip if this exact command was already executed
                if command_text in executed_commands:
                    continue
                    
                # Extract command arguments if any (anything in the capture group)
                args = match.groups() if len(match.groups()) > 0 else None
                
                # Execute the command handler
                result = handler(args)
                
                # Add the result to the additional context if not empty
                if result:
                    if additional_context:
                        additional_context += "\n\n"
                    additional_context += result
                    
                # Mark this command as executed
                executed_commands.add(command_text)
                
                # Remove the command from the cleaned content
                cleaned_content = cleaned_content.replace(command_text, "")
    
        # Trim any extra whitespace from the cleaned content
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content, additional_context

    def _handle_think_deeper(self, args=None) -> str:
        """
        Handle the <think_deeper> command by triggering the frontier model.
        
        Returns:
            Text containing the deeper reflection
        """
        self.logger.info("Processing <think_deeper> command")
        
        try:
            # Get the latest thought from memory
            # Assuming the most recent thought is the current one that has the command
            recent_thoughts = self.memory.get_recent_thoughts(1)
            
            if not recent_thoughts:
                return "ERROR: No recent thoughts found to reflect on."
                
            latest_thought = recent_thoughts[0]
            
            # Generate reflection using the frontier model
            reflection = self.frontier.reflect_on_thought(latest_thought["content"])
            
            # Store the reflection in memory
            self.memory.add_reflection(latest_thought["id"], reflection)
            
            # Extract content from reflection tags if present
            content = reflection
            if content.startswith('<reflection>') and content.endswith('</reflection>'):
                content = content[12:-13]  # Remove the tags
            
            return f"DEEPER REFLECTION:\n{content}"
            
        except Exception as e:
            self.logger.error(f"Error processing think_deeper command: {str(e)}")
            return f"ERROR: Could not generate deeper reflection: {str(e)}"

    def _handle_get_goals(self, args=None) -> str:
        """
        Handle the <get_goals> command by retrieving current goals.
        
        Returns:
            Text containing the current goals
        """
        self.logger.info("Processing <get_goals> command")
        
        try:
            goals = self.get_goals()
            
            if not goals:
                return "GOALS:\nYou currently have no specific goals set."
                
            formatted_goals = []
            for i, goal in enumerate(goals, 1):
                formatted_goals.append(f"{i}. {goal['content']}")
                
            return "CURRENT GOALS:\n" + "\n".join(formatted_goals)
            
        except Exception as e:
            self.logger.error(f"Error processing get_goals command: {str(e)}")
            return f"ERROR: Could not retrieve goals: {str(e)}"

    def _handle_add_goal(self, args) -> str:
        """
        Handle the <add_goal(title, content)> command by adding a new goal.
        
        Args:
            args: Tuple containing the goal title and content
            
        Returns:
            Confirmation message
        """
        if not args or len(args) < 2:
            return "ERROR: Goal requires both title and content."
        
        title, content = args
        title = title.strip()
        content = content.strip()
        
        self.logger.info(f"Processing <add_goal> command with title: {title}")
        
        try:
            # Add the goal
            goal_id = self.add_goal(title, content)
            
            if not goal_id:
                return f"ERROR: Failed to create goal '{title}'."
            
            return f"NEW GOAL ADDED: {title}\n{content}"
            
        except Exception as e:
            self.logger.error(f"Error processing add_goal command: {str(e)}")
            return f"ERROR: Could not add goal: {str(e)}"
    
    def _handle_complete_goal(self, args) -> str:
        """
        Handle the <complete_goal(title)> command by marking a goal as completed.
        
        Args:
            args: Tuple containing the goal title
            
        Returns:
            Confirmation message
        """
        if not args or not args[0]:
            return "ERROR: No goal title provided."
        
        goal_title = args[0].strip()
        
        self.logger.info(f"Processing <complete_goal> command with title: {goal_title}")
        
        try:
            # Find the goal by title
            goals = self.get_goals(include_completed=False)
            matching_goals = [g for g in goals if g['title'].lower() == goal_title.lower()]
            
            if not matching_goals:
                return f"ERROR: No active goal found with title '{goal_title}'."
            
            # Mark the goal as completed
            goal_id = matching_goals[0]['id']
            success = self.memory.complete_goal(goal_id)
            
            if not success:
                return f"ERROR: Failed to complete goal '{goal_title}'."
            
            # Inject a thought about completing the goal
            self.inject_thought(
                f"GOAL COMPLETED: {goal_title}",
                type="goal_completion"
            )
            
            return f"GOAL COMPLETED: {goal_title}"
        
        except Exception as e:
            self.logger.error(f"Error processing complete_goal command: {str(e)}")
            return f"ERROR: Could not complete goal: {str(e)}"

    def _handle_reminisce(self, args=None) -> str:
        """
        Handle the <reminisce> command by retrieving random long-term memories.
        
        Returns:
            Text containing random memories
        """
        self.logger.info("Processing <reminisce> command")
        
        try:
            # Get all memories (assuming memory system has this method)
            # If there's no direct method, we might need to adapt this based on the available API
            # Getting all memories may not be reasonable long term. Consider generating random memory IDs and pulling just those memories
            all_memories = self.memory.get_all_memories()
            
            if not all_memories or len(all_memories) < 5:
                return "MEMORIES:\nNot enough stored memories to reminisce yet."
                
            # Select 5 random memories
            selected_memories = random.sample(all_memories, min(5, len(all_memories)))
            
            memory_texts = []
            for i, memory in enumerate(selected_memories, 1):
                # Format timestamp or date if available
                timestamp = ""
                if "timestamp" in memory:
                    if hasattr(memory["timestamp"], "strftime"):
                        timestamp = memory["timestamp"].strftime("%Y-%m-%d")
                    else:
                        timestamp = str(memory["timestamp"])
                        
                memory_content = memory.get("content", "")
                memory_texts.append(f"Memory {i} [{timestamp}]:\n{memory_content}")
                
            return "REMINISCING:\n" + "\n\n".join(memory_texts)
            
        except Exception as e:
            self.logger.error(f"Error processing reminisce command: {str(e)}")
            return f"ERROR: Could not retrieve memories: {str(e)}"

    def _handle_create_work(self, args) -> str:
        """
        Handle the <create(title, content, creation_type)> command by creating a new creative work.
        
        Args:
            args: Tuple containing (title, content, creation_type)
            
        Returns:
            Text confirmation of creation
        """
        if not args or len(args) < 3:
            return "ERROR: Creation requires title, content, and creation_type."
        
        title, content, creation_type = args
        title = title.strip()
        content = content.strip()
        creation_type = creation_type.strip()
        
        self.logger.info(f"Processing <create> command with title: {title}, type: {creation_type}")
        
        try:
            # Add to memory system
            creation_id = self.memory.add_creation(
                title=title,
                content=content,
                creation_type=creation_type
            )
            
            if not creation_id:
                return f"ERROR: Failed to create '{title}'."
            
            return f"CREATION STORED: '{title}' ({creation_type}) has been saved."
        
        except Exception as e:
            self.logger.error(f"Error processing create command: {str(e)}")
        return f"ERROR: Could not create work: {str(e)}"

    def _handle_get_creation_titles(self, args=None) -> str:
        """
        Handle the <get_creation_titles> command by retrieving titles of all creative works.
        
        Returns:
            Text listing of creation titles
        """
        self.logger.info("Processing <get_creation_titles> command")
        
        try:
            # Get creation titles from memory system
            title_info = self.memory.get_creation_titles()
            
            if not title_info:
                return "CREATIONS: No creative works have been stored yet."
            
            # Format titles with timestamps and types
            formatted_titles = []
            for title, timestamp, creation_type in title_info:
                date_str = timestamp.strftime("%Y-%m-%d") if hasattr(timestamp, "strftime") else str(timestamp)
                type_str = f"({creation_type})" if creation_type else ""
                formatted_titles.append(f"- {title} {type_str} [Created: {date_str}]")
            
            return "CREATIONS:\n" + "\n".join(formatted_titles)
        
        except Exception as e:
            self.logger.error(f"Error processing get_creation_titles command: {str(e)}")
            return f"ERROR: Could not retrieve creation titles: {str(e)}"

    def _handle_get_creation_by_title(self, args) -> str:
        """
        Handle the <get_creation_by_title(title)> command by retrieving a specific creation.
        
        Args:
            args: Tuple containing the title
            
        Returns:
            Text containing the creation content
        """
        if not args or not args[0]:
            return "ERROR: No title provided."
        
        title = args[0].strip()
        
        self.logger.info(f"Processing <get_creation_by_title> command with title: {title}")
        
        try:
            # Get creation from memory system
            creation = self.memory.get_creation_by_title(title)
            
            if not creation:
                return f"ERROR: No creation found with title '{title}'."
            
            # Format the creation
            creation_type = f"Type: {creation['type']}" if creation.get('type') else ""
            date_str = creation['timestamp'].strftime("%Y-%m-%d %H:%M") if hasattr(creation['timestamp'], "strftime") else str(creation['timestamp'])
            
            result = f"CREATION: {creation['title']}\n{creation_type}\nCreated: {date_str}\n\n{creation['content']}"
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing get_creation_by_title command: {str(e)}")
            return f"ERROR: Could not retrieve creation: {str(e)}"
