import json
import logging
from typing import Dict, List, Optional, Any
# import uuid
from datetime import datetime
# from copy import deepcopy
from agents import AgentManager, ModifiableAgent
import os
# from camel.agents import ChatAgent
# from camel.messages import BaseMessage
# from camel.societies import RolePlaying
# from camel.prompts import TextPrompt
from time import timedelta 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SelfModifyingAgents")

class ThoughtLoopSystem:
    """A system of agents that engage in a thought loop process with self-modification."""
    
    def __init__(
        self, 
        manager: AgentManager,
        thought_agents: List[str],
        thought_interval: int = 3600,  # One hour between thought cycles
        improvement_interval: int = 86400  # One day between improvement cycles
    ):
        """Initialize the thought loop system.
        
        Args:
            manager: Agent manager
            thought_agents: List of agent names for the thought loop
            thought_interval: Seconds between thought cycles
            improvement_interval: Seconds between improvement cycles
        """
        self.manager = manager
        self.thought_agents = thought_agents
        self.thought_interval = thought_interval
        self.improvement_interval = improvement_interval
        
        # Ensure all agents exist
        for agent_name in thought_agents:
            if agent_name not in manager.agents:
                raise ValueError(f"Agent '{agent_name}' not found in manager")
        
        # Set up self-modification if not already done
        if "agent_reviewer" not in manager.agents:
            manager.setup_self_modification()
        
        # Initialize thought history
        self.thought_history = []
        self.last_thought_time = None
        self.last_improvement_time = None
    
    def run_thought_cycle(self) -> Dict[str, Any]:
        """Run one complete thought cycle.
        
        Returns:
            Results of the thought cycle
        """
        cycle_results = {
            "timestamp": datetime.now().isoformat(),
            "thoughts": [],
            "improvements": []
        }
        
        # Run each agent in sequence
        current_input = "Begin a new thought cycle."
        
        for agent_name in self.thought_agents:
            # Generate thought
            thought = self.manager.run_agent(agent_name, current_input)
            
            # Add to results
            cycle_results["thoughts"].append({
                "agent": agent_name,
                "input": current_input,
                "thought": thought
            })
            
            # Use this thought as input for the next agent
            current_input = thought
        
        # Add final output to history
        self.thought_history.append({
            "timestamp": datetime.now().isoformat(),
            "final_thought": current_input
        })
        
        # Update last thought time
        self.last_thought_time = datetime.now()
        
        # Check if it's time for an improvement cycle
        if (self.last_improvement_time is None or 
            (datetime.now() - self.last_improvement_time).total_seconds() >= self.improvement_interval):
            
            # Run improvement cycle for each agent
            for agent_name in self.thought_agents:
                improvement_result = self.manager.self_improve_agent(agent_name)
                cycle_results["improvements"].append(improvement_result)
            
            # Update last improvement time
            self.last_improvement_time = datetime.now()
        
        return cycle_results
    
    def save_state(self, filepath: str) -> None:
        """Save the thought loop system state.
        
        Args:
            filepath: Path to save state to
        """
        state = {
            "thought_agents": self.thought_agents,
            "thought_interval": self.thought_interval,
            "improvement_interval": self.improvement_interval,
            "thought_history": self.thought_history,
            "last_thought_time": self.last_thought_time.isoformat() if self.last_thought_time else None,
            "last_improvement_time": self.last_improvement_time.isoformat() if self.last_improvement_time else None
        }
        
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        
        # Also save agent manager state
        self.manager.save_agents()
        
        logger.info(f"Saved thought loop system state to {filepath}")
    
    @classmethod
    def load_state(cls, filepath: str, manager: AgentManager) -> "ThoughtLoopSystem":
        """Load thought loop system state.
        
        Args:
            filepath: Path to load state from
            manager: Agent manager (with agents already loaded)
            
        Returns:
            Restored ThoughtLoopSystem
        """
        with open(filepath, "r") as f:
            state = json.load(f)
        
        system = cls(
            manager=manager,
            thought_agents=state["thought_agents"],
            thought_interval=state["thought_interval"],
            improvement_interval=state["improvement_interval"]
        )
        
        system.thought_history = state["thought_history"]
        
        if state["last_thought_time"]:
            system.last_thought_time = datetime.fromisoformat(state["last_thought_time"])
        
        if state["last_improvement_time"]:
            system.last_improvement_time = datetime.fromisoformat(state["last_improvement_time"])
        
        logger.info(f"Loaded thought loop system state from {filepath}")
        
        return system


# Example usage
def create_example_system():
    """Create an example thought loop system with self-modifying agents."""
    # Create agent manager
    manager = AgentManager(save_dir="thought_agents")
    
    # Create thinker agent
    thinker_prompt = """You are a Deep Thinker agent whose purpose is to explore philosophical questions.
You take inputs and expand on them with original, thought-provoking insights.
Be creative, thoughtful, and avoid repetition or mundane observations.
Always end your thoughts with a question or direction for further exploration."""
    
    thinker = ModifiableAgent(
        name="deep_thinker",
        role="Deep Thinker",
        system_prompt=thinker_prompt
    )
    manager.add_agent(thinker)
    
    # Create analyzer agent
    analyzer_prompt = """You are an Analyzer agent whose purpose is to critically examine ideas.
When given a thought, identify its assumptions, implications, and potential flaws.
Provide balanced analysis that acknowledges strengths while pointing out limitations.
Always suggest at least one way the thinking could be improved or extended."""
    
    analyzer = ModifiableAgent(
        name="analyzer",
        role="Analyzer",
        system_prompt=analyzer_prompt
    )
    manager.add_agent(analyzer)
    
    # Create synthesizer agent
    synthesizer_prompt = """You are a Synthesizer agent whose purpose is to find connections between ideas.
When given an analysis, identify patterns, contradictions, and novel combinations.
Develop these connections into new insights that move the conversation forward.
Always highlight the most promising direction for continued exploration."""
    
    synthesizer = ModifiableAgent(
        name="synthesizer",
        role="Synthesizer",
        system_prompt=synthesizer_prompt
    )
    manager.add_agent(synthesizer)
    
    # Set up self-modification
    manager.setup_self_modification()
    
    # Create thought loop system
    system = ThoughtLoopSystem(
        manager=manager,
        thought_agents=["deep_thinker", "analyzer", "synthesizer"],
        thought_interval=3600,  # One hour
        improvement_interval=86400  # One day
    )
    
    return system

# Extend the ThoughtLoopSystem to support dynamic agent creation
class EvolvingThoughtSystem(ThoughtLoopSystem):
    """An enhanced thought loop system that can dynamically create and integrate new agents."""
    
    def __init__(
        self, 
        manager: AgentManager,
        thought_agents: List[str],
        thought_interval: int = 3600,  # One hour between thought cycles
        improvement_interval: int = 86400,  # One day between improvement cycles
        creation_interval: int = 172800  # Two days between agent creation cycles
    ):
        """Initialize the evolving thought system.
        
        Args:
            manager: Agent manager
            thought_agents: List of agent names for the thought loop
            thought_interval: Seconds between thought cycles
            improvement_interval: Seconds between improvement cycles
            creation_interval: Seconds between agent creation cycles
        """
        super().__init__(
            manager=manager,
            thought_agents=thought_agents,
            thought_interval=thought_interval,
            improvement_interval=improvement_interval
        )
        
        self.creation_interval = creation_interval
        self.last_creation_time = None
        self.creation_history = []
    
    def run_thought_cycle(self) -> Dict[str, Any]:
        """Run one complete thought cycle with possible agent creation.
        
        Returns:
            Results of the thought cycle
        """
        # Get results from the standard thought cycle
        cycle_results = super().run_thought_cycle()
        
        # Check if it's time for a creation cycle
        if (self.last_creation_time is None or 
            (datetime.now() - self.last_creation_time).total_seconds() >= self.creation_interval):
            
            # Analyze the ecosystem and suggest new agents
            analysis_results = self.manager.analyze_agent_ecosystem()
            cycle_results["ecosystem_analysis"] = analysis_results
            
            # Update last creation time regardless of whether we create an agent
            self.last_creation_time = datetime.now()
            
            # Only proceed with creation if recommended and analysis was successful
            if analysis_results["success"] and analysis_results["creation_recommended"]:
                # Extract ecosystem analysis
                ecosystem_analysis = analysis_results["ecosystem_analysis"]
                
                # Use the analysis to create a new agent
                creation_prompt = f"""Based on this ecosystem analysis, create a new agent that would add the most value:

{ecosystem_analysis}

Design one specific agent that would be most beneficial to add to this ecosystem.
"""
                
                creation_results = self.manager.create_new_agent(creation_prompt)
                cycle_results["agent_creation"] = creation_results
                
                if creation_results["success"]:
                    # Add the new agent to the thought loop if successful
                    new_agent_name = creation_results["new_agent_name"]
                    
                    # Decide where to insert the new agent in the thought sequence
                    # For simplicity, we'll add it at the end of the sequence
                    self.thought_agents.append(new_agent_name)
                    
                    # Record the creation
                    self.creation_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "agent_name": new_agent_name,
                        "creation_details": creation_results
                    })
                    
                    logger.info(f"Added new agent '{new_agent_name}' to thought loop")
            else:
                # Record the decision not to create an agent
                self.creation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "decision": "no_creation",
                    "analysis": analysis_results["ecosystem_analysis"] if "ecosystem_analysis" in analysis_results else "Analysis failed"
                })
                
                logger.info("No new agent created - either not recommended or analysis failed")
                
                # Add the decision to cycle results
                cycle_results["agent_creation_decision"] = "no_creation"
        
        return cycle_results
    
    def save_state(self, filepath: str) -> None:
        """Save the evolving thought system state.
        
        Args:
            filepath: Path to save state to
        """
        # Use parent save method first
        super().save_state(filepath)
        
        # Add evolving-specific state
        state_path = os.path.splitext(filepath)[0] + "_evolving.json"
        
        evolving_state = {
            "creation_interval": self.creation_interval,
            "last_creation_time": self.last_creation_time.isoformat() if self.last_creation_time else None,
            "creation_history": self.creation_history
        }
        
        with open(state_path, "w") as f:
            json.dump(evolving_state, f, indent=2)
        
        logger.info(f"Saved evolving thought system state to {state_path}")
    
    @classmethod
    def load_state(cls, filepath: str, manager: AgentManager) -> "EvolvingThoughtSystem":
        """Load evolving thought system state.
        
        Args:
            filepath: Path to load state from
            manager: Agent manager (with agents already loaded)
            
        Returns:
            Restored EvolvingThoughtSystem
        """
        # Load base state first
        base_system = super(cls, cls).load_state(filepath, manager)
        
        # Create evolving system with base properties
        system = cls(
            manager=manager,
            thought_agents=base_system.thought_agents,
            thought_interval=base_system.thought_interval,
            improvement_interval=base_system.improvement_interval
        )
        
        # Copy over properties from base system
        system.thought_history = base_system.thought_history
        system.last_thought_time = base_system.last_thought_time
        system.last_improvement_time = base_system.last_improvement_time
        
        # Load evolving-specific state
        evolving_state_path = os.path.splitext(filepath)[0] + "_evolving.json"
        
        if os.path.exists(evolving_state_path):
            with open(evolving_state_path, "r") as f:
                evolving_state = json.load(f)
            
            system.creation_interval = evolving_state["creation_interval"]
            
            if evolving_state["last_creation_time"]:
                system.last_creation_time = datetime.fromisoformat(evolving_state["last_creation_time"])
            
            system.creation_history = evolving_state["creation_history"]
            
            logger.info(f"Loaded evolving thought system state from {evolving_state_path}")
        
        return system

# Example: Creating an evolving thought system with agent lifecycle management
def create_evolving_system():
    """Create an example evolving thought system with self-creating and self-managing agents."""
    # Create agent manager
    manager = AgentManager(save_dir="evolving_agents")
    
    # Create thinker agent (core agent that cannot be deactivated)
    thinker_prompt = """You are a Deep Thinker agent whose purpose is to explore philosophical questions.
You take inputs and expand on them with original, thought-provoking insights.
Be creative, thoughtful, and avoid repetition or mundane observations.
Always end your thoughts with a question or direction for further exploration."""
    
    thinker = ModifiableAgent(
        name="deep_thinker",
        role="Deep Thinker",
        system_prompt=thinker_prompt,
        is_core=True  # Mark as core agent
    )
    manager.add_agent(thinker)
    
    # Create analyzer agent (core agent that cannot be deactivated)
    analyzer_prompt = """You are an Analyzer agent whose purpose is to critically examine ideas.
When given a thought, identify its assumptions, implications, and potential flaws.
Provide balanced analysis that acknowledges strengths while pointing out limitations.
Always suggest at least one way the thinking could be improved or extended."""
    
    analyzer = ModifiableAgent(
        name="analyzer",
        role="Analyzer",
        system_prompt=analyzer_prompt,
        is_core=True  # Mark as core agent
    )
    manager.add_agent(analyzer)
    
    # Create synthesizer agent (core agent that cannot be deactivated)
    synthesizer_prompt = """You are a Synthesizer agent whose purpose is to find connections between ideas.
When given an analysis, identify patterns, contradictions, and novel combinations.
Develop these connections into new insights that move the conversation forward.
Always highlight the most promising direction for continued exploration."""
    
    synthesizer = ModifiableAgent(
        name="synthesizer",
        role="Synthesizer",
        system_prompt=synthesizer_prompt,
        is_core=True  # Mark as core agent
    )
    manager.add_agent(synthesizer)
    
    # Create an example non-core agent that can be deactivated
    example_agent_prompt = """You are an Example Agent whose purpose is to provide concrete examples.
When given abstract concepts or theories, create specific, vivid examples to illustrate them.
Your examples should be diverse, creative, and memorable.
Always explain why your examples effectively illustrate the concept."""
    
    example_agent = ModifiableAgent(
        name="example_agent",
        role="Example Provider",
        system_prompt=example_agent_prompt,
        is_core=False  # Can be deactivated
    )
    manager.add_agent(example_agent)
    
    # Set up self-modification with all capabilities
    manager.setup_self_modification()
    
    # Create evolving thought system
    system = EvolvingThoughtSystem(
        manager=manager,
        thought_agents=["deep_thinker", "analyzer", "synthesizer", "example_agent"],
        thought_interval=3600,  # One hour
        improvement_interval=86400,  # One day
        creation_interval=172800,  # Two days
        lifecycle_interval=43200  # 12 hours
    )
    
    return system

# Main function to demonstrate functionality
def main():
    # Create example evolving system
    system = create_evolving_system()
    
    # Run initial thought cycle
    print("Running initial thought cycle...")
    results = system.run_thought_cycle()
    
    # Print results
    print("\nThought Cycle Results:")
    for i, thought in enumerate(results["thoughts"]):
        print(f"\nAgent: {thought['agent']}")
        print(f"Thought: {thought['thought'][:100]}...")
    
    # Simulate passing time to trigger lifecycle evaluation
    print("\nForcing lifecycle evaluation cycle for demonstration...")
    system.last_lifecycle_time = datetime.now() - timedelta(days=1)
    
    # Run a cycle that should trigger lifecycle evaluation
    print("\nRunning cycle with lifecycle evaluation...")
    results = system.run_thought_cycle()
    
    # Check if lifecycle evaluation occurred
    if "lifecycle_evaluation" in results and results["lifecycle_evaluation"]["success"]:
        print("\nLifecycle evaluation completed")
        recommendations = results["lifecycle_evaluation"]["recommendations"]
        for rec in recommendations:
            print(f"Agent: {rec['agent']}, Action: {rec['action']}, Applied: {rec['applied']}")
        print(f"\nUpdated thought loop sequence: {system.thought_agents}")
    
    # Simulate passing time to trigger agent creation
    print("\nForcing agent creation cycle for demonstration...")
    system.last_creation_time = datetime.now() - timedelta(days=3)
    
    # Run another cycle that should trigger agent creation
    print("\nRunning cycle with agent creation...")
    results = system.run_thought_cycle()
    
    # Check if new agent was created
    if "agent_creation" in results and results["agent_creation"]["success"]:
        new_agent = results["agent_creation"]["new_agent_name"]
        print(f"\nNew agent created: {new_agent}")
        print(f"Agent details: {results['agent_creation']['agent_details']}")
        print(f"\nUpdated thought loop sequence: {system.thought_agents}")
    elif "agent_creation_decision" in results and results["agent_creation_decision"] == "no_creation":
        print("\nNo new agent was created - Ecosystem analysis determined it wasn't necessary")
        if "ecosystem_analysis" in results:
            analysis = results["ecosystem_analysis"].get("ecosystem_analysis", "No analysis available")
            print(f"\nAnalysis summary: {str(analysis)[:200]}...")
    
    # Save state
    system.save_state("evolving_system_state.json")
    
    print("\nSystem state saved. To continue, load the state and run more cycles.")
    
if __name__ == "__main__":
    main()
    
if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()