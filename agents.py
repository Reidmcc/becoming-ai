from datetime import datetime
from typing import Dict, List, Optional, Any
from logging import logger
import json
import os
import uuid

# Import CAMEL components
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.societies import RolePlaying
from camel.prompts import TextPrompt

class ModifiablePrompt:
    """A wrapper for prompts that can be modified during runtime."""
    
    def __init__(self, name: str, content: str, version: int = 1):
        """Initialize a modifiable prompt.
        
        Args:
            name: Name/identifier for the prompt
            content: The prompt content
            version: Version number for tracking changes
        """
        self.name = name
        self.content = content
        self.version = version
        self.history = [{
            "version": version,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }]
    
    def update(self, new_content: str) -> None:
        """Update the prompt with new content.
        
        Args:
            new_content: The new prompt content
        """
        self.version += 1
        self.content = new_content
        self.history.append({
            "version": self.version,
            "content": new_content,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"Prompt '{self.name}' updated to version {self.version}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "content": self.content,
            "version": self.version,
            "history": self.history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModifiablePrompt":
        """Create from dictionary after deserialization."""
        prompt = cls(data["name"], data["content"], data["version"])
        prompt.history = data["history"]
        return prompt


class ModifiableAgent:
    """An agent whose prompts can be modified during runtime."""
    
    def __init__(
        self, 
        name: str, 
        role: str,
        system_prompt: str,
        task_prompt: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        is_core: bool = False,
        is_active: bool = True
    ):
        """Initialize a modifiable agent.
        
        Args:
            name: Agent identifier
            role: The role this agent plays
            system_prompt: System prompt defining the agent's behavior
            task_prompt: Optional task-specific prompt
            model_name: Name of the LLM to use
            is_core: Whether this is a core agent that cannot be deactivated
            is_active: Whether this agent is currently active
        """
        self.name = name
        self.role = role
        self.model_name = model_name
        self.is_core = is_core  # Core agents cannot be deactivated
        self.is_active = is_active  # Active state
        
        # Create modifiable prompts
        self.system_prompt = ModifiablePrompt(f"{name}_system", system_prompt)
        self.task_prompt = ModifiablePrompt(f"{name}_task", task_prompt or "")
        
        # Initialize CAMEL agent
        self._initialize_agent()
        
        # Store recent outputs for review
        self.recent_outputs = []
        self.max_stored_outputs = 10
        
        # Performance metrics
        self.performance_metrics = {
            "total_calls": 0,
            "average_response_length": 0,
            "last_evaluation_score": None,
            "creation_date": datetime.now().isoformat(),
            "last_active_date": datetime.now().isoformat()
        }
    
    def _initialize_agent(self) -> None:
        """Initialize or reinitialize the CAMEL agent with current prompts."""
        # Create CAMEL agent with current prompt versions
        self.agent = ChatAgent(
            model_name=self.model_name,
            system_message=self.system_prompt.content
        )
        logger.info(f"Initialized agent '{self.name}' with role '{self.role}'")
    
    def generate_response(self, input_message: str) -> str:
        """Generate a response to the input message.
        
        Args:
            input_message: Message to respond to
            
        Returns:
            Agent's response
        """
        # Check if agent is active
        if not self.is_active:
            return f"[AGENT '{self.name}' IS CURRENTLY INACTIVE]"
            
        # Format input with task prompt if available
        if self.task_prompt.content:
            formatted_input = f"{self.task_prompt.content}\n\n{input_message}"
        else:
            formatted_input = input_message
            
        # Generate response using CAMEL agent
        response = self.agent.chat(formatted_input)
        
        # Store for review
        self.recent_outputs.append({
            "input": input_message,
            "response": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only the most recent outputs
        if len(self.recent_outputs) > self.max_stored_outputs:
            self.recent_outputs = self.recent_outputs[-self.max_stored_outputs:]
        
        # Update performance metrics
        self.performance_metrics["total_calls"] += 1
        self.performance_metrics["average_response_length"] = (
            (self.performance_metrics["average_response_length"] * 
             (self.performance_metrics["total_calls"] - 1) + 
             len(response.content)) / self.performance_metrics["total_calls"]
        )
        self.performance_metrics["last_active_date"] = datetime.now().isoformat()
        
        return response.content
    
    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt and reinitialize the agent.
        
        Args:
            new_prompt: New system prompt content
        """
        self.system_prompt.update(new_prompt)
        self._initialize_agent()
    
    def update_task_prompt(self, new_prompt: str) -> None:
        """Update the task prompt.
        
        Args:
            new_prompt: New task prompt content
        """
        self.task_prompt.update(new_prompt)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "role": self.role,
            "model_name": self.model_name,
            "is_core": self.is_core,
            "is_active": self.is_active,
            "system_prompt": self.system_prompt.to_dict(),
            "task_prompt": self.task_prompt.to_dict(),
            "recent_outputs": self.recent_outputs,
            "performance_metrics": self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModifiableAgent":
        """Create from dictionary after deserialization."""
        agent = cls(
            name=data["name"],
            role=data["role"],
            system_prompt=data["system_prompt"]["content"],
            task_prompt=data["task_prompt"]["content"],
            model_name=data["model_name"],
            is_core=data.get("is_core", False),
            is_active=data.get("is_active", True)
        )
        
        # Restore prompt history
        agent.system_prompt = ModifiablePrompt.from_dict(data["system_prompt"])
        agent.task_prompt = ModifiablePrompt.from_dict(data["task_prompt"])
        
        # Restore recent outputs
        agent.recent_outputs = data["recent_outputs"]
        
        # Restore performance metrics if present
        if "performance_metrics" in data:
            agent.performance_metrics = data["performance_metrics"]
        
        return agent
        
    def activate(self) -> bool:
        """Activate the agent.
        
        Returns:
            True if activation was successful, False if already active or is core agent
        """
        if self.is_active:
            return False  # Already active
            
        self.is_active = True
        self.performance_metrics["last_active_date"] = datetime.now().isoformat()
        logger.info(f"Activated agent '{self.name}'")
        return True
    
    def deactivate(self) -> bool:
        """Deactivate the agent.
        
        Returns:
            True if deactivation was successful, False if already inactive or is core agent
        """
        if self.is_core:
            logger.warning(f"Cannot deactivate core agent '{self.name}'")
            return False  # Cannot deactivate core agents
            
        if not self.is_active:
            return False  # Already inactive
            
        self.is_active = False
        logger.info(f"Deactivated agent '{self.name}'")
        return True


class AgentManager:
    """Manager for a collection of modifiable agents with self-creation capability."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the agent manager.
        
        Args:
            save_dir: Directory for saving agent data
        """
        self.agents: Dict[str, ModifiableAgent] = {}
        self.save_dir = config.get("agent_save_dir")
        os.makedirs(self.save_dir, exist_ok=True)
        self.frontier_model = config.get("models_frontier_name", "claude-3-5-haiku-latest")
        if config.get("system_frontier_only"):
            self.local_model = self.frontier_model
        self.local_model = config.get("models_local_name", "deepseek-ai/deepseek-R1-Distill-Llama-8B")
        
        # Special agents for self-modification
        self.reviewer_agent = None
        self.improver_agent = None
        self.creator_agent = None
        self.lifecycle_agent = None 
    
    def add_agent(self, agent: ModifiableAgent) -> None:
        """Add an agent to the manager.
        
        Args:
            agent: The agent to add
        """
        self.agents[agent.name] = agent
        logger.info(f"Added agent '{agent.name}' to manager")
    
    def run_agent(self, agent_name: str, input_message: str) -> str:
        """Run a specific agent.
        
        Args:
            agent_name: Name of the agent to run
            input_message: Input message for the agent
            
        Returns:
            Agent's response
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        agent = self.agents[agent_name]
        
        # Check if agent is active
        if not agent.is_active:
            return f"[AGENT '{agent_name}' IS CURRENTLY INACTIVE]"
        
        return agent.generate_response(input_message)
    
    def save_agents(self) -> None:
        """Save all agents to disk."""
        for name, agent in self.agents.items():
            agent_file = os.path.join(self.save_dir, f"{name}.json")
            with open(agent_file, "w") as f:
                json.dump(agent.to_dict(), f, indent=2)
        
        logger.info(f"Saved {len(self.agents)} agents to {self.save_dir}")
    
    def load_agents(self) -> None:
        """Load all agents from disk."""
        if not os.path.exists(self.save_dir):
            logger.warning(f"Save directory {self.save_dir} not found")
            return
        
        for filename in os.listdir(self.save_dir):
            if filename.endswith(".json"):
                agent_file = os.path.join(self.save_dir, filename)
                with open(agent_file, "r") as f:
                    agent_data = json.load(f)
                
                agent = ModifiableAgent.from_dict(agent_data)
                self.agents[agent.name] = agent
        
        logger.info(f"Loaded {len(self.agents)} agents from {self.save_dir}")
    
    def setup_self_modification(
        self, 
        reviewer_model: str = "gpt-4o-mini", 
        improver_model: str = "gpt-4o-mini",
        creator_model: str = "gpt-4o-mini",
        lifecycle_model: str = "gpt-4o-mini"
    ) -> None:
        """Set up the special agents for self-modification and creation.
        
        Args:
            reviewer_model: Model for the reviewer agent
            improver_model: Model for the improver agent
            creator_model: Model for the creator agent
            lifecycle_model: Model for the agent lifecycle manager
        """
        # Create the reviewer agent
        reviewer_system_prompt = """You are an Agent Reviewer whose job is to analyze the performance of AI agents.
        You carefully review the recent outputs of agents and identify areas for improvement.
        Focus on:
        1. Consistency with the agent's intended role
        2. Quality and relevance of responses
        3. Potential improvements to the agent's prompts
        4. Areas where the agent is underperforming

        Your analysis should be specific, detailed, and constructive.
        Clearly identify what's working well and what could be improved."""

        self.reviewer_agent = ModifiableAgent(
            name="agent_reviewer",
            role="Agent Reviewer",
            system_prompt=reviewer_system_prompt,
            model_name=reviewer_model,
            is_core=True  # Mark as core agent
        )
        self.add_agent("agent_reviewer")
        
        # Create the improver agent
        improver_system_prompt = """You are an Agent Improver whose job is to enhance the prompts of other AI agents.
        Based on performance reviews and the current prompts, you will craft improved versions.
        Focus on:
        1. Preserving the core role and functionality
        2. Addressing specific performance issues identified in reviews
        3. Making the prompts clearer and more effective
        4. Adding specific guidance where needed

        Your improved prompts should be complete replacements that can be used directly.
        Always provide the entire new prompt, not just suggestions or changes."""

        self.improver_agent = ModifiableAgent(
            name="agent_improver",
            role="Agent Improver",
            system_prompt=improver_system_prompt,
            model_name=improver_model,
            is_core=True  # Mark as core agent
        )
        self.add_agent("agent_improver")
        
        # Create the agent creator
        creator_system_prompt = """You are an Agent Creator whose job is to design and implement new AI agents.
        You can create entirely new agents with specialized capabilities to enhance the collective intelligence.
        Focus on:
        1. Identifying gaps in the current agent ecosystem that could benefit from a new specialist
        2. Designing agents with clearly defined roles that complement existing agents
        3. Creating effective prompts that guide the agent's behavior appropriately
        4. Ensuring new agents can integrate well with existing ones

        Your agent designs should be complete and ready to implement.
        Always provide the following for each new agent:
        1. A descriptive name that reflects the agent's purpose
        2. A clear role description
        3. A complete system prompt to guide the agent's behavior
        4. An optional task prompt if relevant
        5. A justification for why this agent will be valuable

        Format your response as:
        AGENT NAME: [concise name]
        AGENT ROLE: [brief role description]
        SYSTEM PROMPT: [complete system prompt]
        TASK PROMPT: [optional task prompt]
        JUSTIFICATION: [why this agent is needed]"""

        self.creator_agent = ModifiableAgent(
            name="agent_creator",
            role="Agent Creator",
            system_prompt=creator_system_prompt,
            model_name=creator_model,
            is_core=True  # Mark as core agent
        )
        self.add_agent("agent_creator")
        
        # Create the lifecycle manager agent
        lifecycle_system_prompt = """You are an Agent Lifecycle Manager whose job is to evaluate the performance of agents and make decisions about activating or deactivating them.
        You carefully analyze how each agent is contributing to the overall system and determine if any agents should be modified, deactivated, or reactivated.
        Focus on:
        1. Evaluating if each agent is fulfilling its intended role effectively
        2. Identifying underperforming agents that might need to be deactivated
        3. Recognizing when previously deactivated agents might be valuable again
        4. Maintaining a balanced ecosystem of agents

        You can recommend:
        1. Deactivating agents that are redundant, consistently underperforming, or no longer needed
        2. Reactivating agents that have renewed relevance or utility
        3. Keeping the current state when the ecosystem is functioning well

        NOTE: Core agents cannot be deactivated, and you should NEVER recommend deactivating them.

        Format your recommendations clearly:
        AGENT: [agent name]
        RECOMMENDATION: [ACTIVATE/DEACTIVATE/NO CHANGE]
        REASONING: [brief explanation for your recommendation]"""

        self.lifecycle_agent = ModifiableAgent(
            name="lifecycle_manager",
            role="Agent Lifecycle Manager",
            system_prompt=lifecycle_system_prompt,
            model_name=lifecycle_model,
            is_core=True  # Mark as core agent
        )
        self.add_agent("lifecycle_manager")
        
        logger.info("Set up self-modification and lifecycle management agents")
    
    def review_agent(self, agent_name: str) -> str:
        """Review an agent's recent performance.
        
        Args:
            agent_name: Name of the agent to review
            
        Returns:
            Performance review
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        agent = self.agents[agent_name]
        
        # Prepare input for reviewer
        review_request = f"""Please review the performance of the agent '{agent_name}' with role '{agent.role}'.

        CURRENT SYSTEM PROMPT:
        {agent.system_prompt.content}

        CURRENT TASK PROMPT:
        {agent.task_prompt.content}

        RECENT OUTPUTS:
        {json.dumps(agent.recent_outputs, indent=2)}

        Please provide a detailed review of this agent's performance and identify areas for improvement.
        """
        
        # Get review from reviewer agent
        review = self.reviewer_agent.generate_response(review_request)
        logger.info(f"Generated review for agent '{agent_name}'")
        
        return review
    
    def improve_agent(self, agent_name: str, review: str) -> Dict[str, str]:
        """Improve an agent based on a review.
        
        Args:
            agent_name: Name of the agent to improve
            review: Performance review
            
        Returns:
            Dictionary with improved prompts
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        agent = self.agents[agent_name]
        
        # Prepare input for improver
        improvement_request = f"""Please improve the prompts for agent '{agent_name}' with role '{agent.role}' based on this review:

        REVIEW:
        {review}

        CURRENT SYSTEM PROMPT:
        {agent.system_prompt.content}

        CURRENT TASK PROMPT:
        {agent.task_prompt.content}

        Please provide improved versions of both the SYSTEM PROMPT and TASK PROMPT.
        Format your response as:

        IMPROVED SYSTEM PROMPT:
        [Your improved system prompt here]

        IMPROVED TASK PROMPT:
        [Your improved task prompt here]
        """
        
        # Get improvements from improver agent
        improvements = self.improver_agent.generate_response(improvement_request)
        logger.info(f"Generated improvements for agent '{agent_name}'")
        
        # Parse the improvements
        improved_prompts = {}
        
        if "IMPROVED SYSTEM PROMPT:" in improvements:
            parts = improvements.split("IMPROVED SYSTEM PROMPT:")
            if len(parts) > 1:
                system_part = parts[1].split("IMPROVED TASK PROMPT:")[0].strip()
                improved_prompts["system_prompt"] = system_part
        
        if "IMPROVED TASK PROMPT:" in improvements:
            parts = improvements.split("IMPROVED TASK PROMPT:")
            if len(parts) > 1:
                task_part = parts[1].strip()
                improved_prompts["task_prompt"] = task_part
        
        return improved_prompts
    
    def apply_improvements(self, agent_name: str, improved_prompts: Dict[str, str]) -> None:
        """Apply improved prompts to an agent.
        
        Args:
            agent_name: Name of the agent to improve
            improved_prompts: Dictionary with improved prompts
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        agent = self.agents[agent_name]
        
        # Apply improvements
        if "system_prompt" in improved_prompts:
            agent.update_system_prompt(improved_prompts["system_prompt"])
            logger.info(f"Updated system prompt for agent '{agent_name}'")
        
        if "task_prompt" in improved_prompts:
            agent.update_task_prompt(improved_prompts["task_prompt"])
            logger.info(f"Updated task prompt for agent '{agent_name}'")
    
    def self_improve_agent(self, agent_name: str) -> Dict[str, Any]:
        """Run the full self-improvement cycle for an agent.
        
        Args:
            agent_name: Name of the agent to improve
            
        Returns:
            Results of the improvement process
        """
        results = {
            "agent_name": agent_name,
            "timestamp": datetime.now().isoformat(),
            "success": False
        }
        
        try:
            # Step 1: Review the agent
            review = self.review_agent(agent_name)
            results["review"] = review
            
            # Step 2: Generate improvements
            improved_prompts = self.improve_agent(agent_name, review)
            results["improved_prompts"] = improved_prompts
            
            # Step 3: Apply improvements
            self.apply_improvements(agent_name, improved_prompts)
            
            # Save updated agents
            self.save_agents()
            
            results["success"] = True
            logger.info(f"Successfully completed self-improvement cycle for agent '{agent_name}'")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error during self-improvement cycle for agent '{agent_name}': {e}")
        
        return results
        
    def create_new_agent(self, creation_prompt: str) -> Dict[str, Any]:
        """Create a new agent using the agent creator.
        
        Args:
            creation_prompt: Prompt describing what kind of agent to create
            
        Returns:
            Results of the creation process including the new agent details
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "creation_prompt": creation_prompt,
            "success": False
        }
        
        try:
            if self.creator_agent is None:
                raise ValueError("Agent creator not set up. Call setup_self_modification() first.")
                
            # Step 1: Generate agent design
            creation_request = f"""Please design a new agent based on the following requirements:

            {creation_prompt}

            Remember to provide a complete agent design with name, role, system prompt, and justification.
            """
            agent_design = self.creator_agent.generate_response(creation_request)
            results["agent_design"] = agent_design
            
            # Step 2: Parse the agent design
            agent_details = self._parse_agent_design(agent_design)
            results["agent_details"] = agent_details
            
            # Step 3: Create the new agent
            if not all(k in agent_details for k in ["name", "role", "system_prompt"]):
                raise ValueError("Incomplete agent design. Missing required fields.")
            
            # Clean up the name to be URL/filename safe
            safe_name = agent_details["name"].lower().replace(" ", "_")
            
            # Check if an agent with this name already exists
            if safe_name in self.agents:
                # Generate a unique name
                safe_name = f"{safe_name}_{uuid.uuid4().hex[:8]}"
            
            # Create the new agent
            new_agent = ModifiableAgent(
                name=safe_name,
                role=agent_details["role"],
                system_prompt=agent_details["system_prompt"],
                task_prompt=agent_details.get("task_prompt", ""),
                model_name=self.creator_agent.model_name  # Use same model as creator by default
            )
            
            # Add the agent
            self.add_agent(new_agent)
            
            # Save updated agents
            self.save_agents()
            
            results["success"] = True
            results["new_agent_name"] = safe_name
            logger.info(f"Successfully created new agent '{safe_name}'")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error during agent creation: {e}")
        
        return results
    
    def _parse_agent_design(self, design_text: str) -> Dict[str, str]:
        """Parse the agent design text into a structured format.
        
        Args:
            design_text: The agent design text from the creator agent
            
        Returns:
            Dictionary with parsed agent details
        """
        agent_details = {}
        
        # Look for agent name
        if "AGENT NAME:" in design_text:
            parts = design_text.split("AGENT NAME:")
            if len(parts) > 1:
                name_part = parts[1].strip().split("\n")[0].strip()
                agent_details["name"] = name_part
        
        # Look for agent role
        if "AGENT ROLE:" in design_text:
            parts = design_text.split("AGENT ROLE:")
            if len(parts) > 1:
                role_part = parts[1].strip().split("\n")[0].strip()
                agent_details["role"] = role_part
        
        # Look for system prompt
        if "SYSTEM PROMPT:" in design_text:
            parts = design_text.split("SYSTEM PROMPT:")
            if len(parts) > 1:
                if "TASK PROMPT:" in parts[1]:
                    system_part = parts[1].split("TASK PROMPT:")[0].strip()
                elif "JUSTIFICATION:" in parts[1]:
                    system_part = parts[1].split("JUSTIFICATION:")[0].strip()
                else:
                    system_part = parts[1].strip()
                agent_details["system_prompt"] = system_part
        
        # Look for task prompt
        if "TASK PROMPT:" in design_text:
            parts = design_text.split("TASK PROMPT:")
            if len(parts) > 1:
                if "JUSTIFICATION:" in parts[1]:
                    task_part = parts[1].split("JUSTIFICATION:")[0].strip()
                else:
                    task_part = parts[1].strip()
                agent_details["task_prompt"] = task_part
        
        # Look for justification
        if "JUSTIFICATION:" in design_text:
            parts = design_text.split("JUSTIFICATION:")
            if len(parts) > 1:
                justification_part = parts[1].strip()
                agent_details["justification"] = justification_part
        
        return agent_details
        
    def analyze_agent_ecosystem(self) -> Dict[str, Any]:
        """Analyze the current agent ecosystem to identify gaps and opportunities.
        
        Returns:
            Analysis results including recommendations for new agents and a decision on whether creation is needed
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "creation_recommended": False  # Default to not recommending creation
        }
        
        try:
            if self.creator_agent is None:
                raise ValueError("Agent creator not set up. Call setup_self_modification() first.")
            
            # Prepare information about current agents
            current_agents = []
            for name, agent in self.agents.items():
                # Skip meta-agents like creator, reviewer, improver
                if name in ["agent_creator", "agent_reviewer", "agent_improver"]:
                    continue
                    
                current_agents.append({
                    "name": name,
                    "role": agent.role,
                    "system_prompt_summary": agent.system_prompt.content[:200] + "..." 
                    if len(agent.system_prompt.content) > 200 else agent.system_prompt.content
                })
            
            # Create analysis request
            analysis_request = f"""Please analyze the current agent ecosystem and determine if new agents are needed.

            CURRENT AGENTS:
            {json.dumps(current_agents, indent=2)}

            Based on these existing agents:
            1. Analyze the current ecosystem and identify any gaps, overlaps, or inefficiencies
            2. Evaluate whether adding a new agent would significantly improve the ecosystem
            3. Make an explicit recommendation: Should a new agent be created? (Yes or No)
            4. If Yes, what type of agent would add the most value?

            IMPORTANT: Only recommend creating a new agent if it would provide significant added value. 
            If the ecosystem is already well-balanced or if existing agents can be improved instead, 
            explicitly recommend against creating a new agent.

            Format your conclusion as:
            RECOMMENDATION: [YES or NO]
            REASONING: [Your reasoning for the recommendation]
            """
            
            analysis = self.creator_agent.generate_response(analysis_request)
            results["ecosystem_analysis"] = analysis
            
            # Parse the recommendation
            if "RECOMMENDATION: YES" in analysis.upper():
                results["creation_recommended"] = True
                logger.info("Ecosystem analysis recommends creating a new agent")
            else:
                logger.info("Ecosystem analysis does not recommend creating a new agent")
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error during ecosystem analysis: {e}")
        
        return results