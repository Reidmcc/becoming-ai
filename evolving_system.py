from thought_loop import ThoughtLoopSystem
from agents import AgentManager
from typing import Dict, List, Optional, Any
from logging import logger
from datetime import datetime
import os
import json
import regex as re

class EvolvingThoughtSystem(ThoughtLoopSystem):
    """An enhanced thought loop system that can dynamically create and integrate new agents."""
    
    def __init__(
        self, 
        manager: AgentManager,
        thought_agents: List[str],
        thought_interval: int = 3600,  # One hour between thought cycles
        improvement_interval: int = 86400,  # One day between improvement cycles
        creation_interval: int = 172800,  # Two days between agent creation cycles
        lifecycle_interval: int = 86400  # One day between lifecycle evaluations
    ):
        """Initialize the evolving thought system.
        
        Args:
            manager: Agent manager
            thought_agents: List of agent names for the thought loop
            thought_interval: Seconds between thought cycles
            improvement_interval: Seconds between improvement cycles
            creation_interval: Seconds between agent creation cycles
            lifecycle_interval: Seconds between agent lifecycle evaluations
        """
        super().__init__(
            manager=manager,
            thought_agents=thought_agents,
            thought_interval=thought_interval,
            improvement_interval=improvement_interval
        )
        
        self.creation_interval = creation_interval
        self.lifecycle_interval = lifecycle_interval
        self.last_creation_time = None
        self.last_lifecycle_time = None
        self.creation_history = []
        self.lifecycle_history = []
    
    def run_thought_cycle(self) -> Dict[str, Any]:
        """Run one complete thought cycle with possible agent creation and lifecycle management.
        
        Returns:
            Results of the thought cycle
        """
        # Get results from the standard thought cycle
        cycle_results = super().run_thought_cycle()
        
        # Check if it's time for a lifecycle evaluation
        if (self.last_lifecycle_time is None or 
            (datetime.now() - self.last_lifecycle_time).total_seconds() >= self.lifecycle_interval):
            
            # Evaluate agent lifecycle
            lifecycle_results = self.manager.evaluate_agent_lifecycle()
            cycle_results["lifecycle_evaluation"] = lifecycle_results
            
            if lifecycle_results["success"]:
                # Record the lifecycle evaluation
                self.lifecycle_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "recommendations": lifecycle_results["recommendations"]
                })
                
                # Update active thought agents list
                self._update_thought_agents_from_lifecycle(lifecycle_results["recommendations"])
                
                logger.info(f"Completed lifecycle evaluation with {len(lifecycle_results['recommendations'])} recommendations")
            
            # Update last lifecycle time
            self.last_lifecycle_time = datetime.now()
        
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
    
    def _update_thought_agents_from_lifecycle(self, recommendations: List[Dict[str, Any]]) -> None:
        """Update the thought agents list based on lifecycle recommendations.
        
        Args:
            recommendations: List of lifecycle recommendations
        """
        # Track which agents were deactivated or activated
        deactivated = []
        activated = []
        
        for rec in recommendations:
            agent_name = rec["agent"]
            action = rec["action"]
            
            if action == "DEACTIVATE" and rec["applied"]:
                # Remove from thought sequence if present
                if agent_name in self.thought_agents:
                    self.thought_agents.remove(agent_name)
                    deactivated.append(agent_name)
                    logger.info(f"Removed deactivated agent '{agent_name}' from thought sequence")
            
            elif action == "ACTIVATE" and rec["applied"]:
                # Add to thought sequence if not already present
                if agent_name not in self.thought_agents:
                    self.thought_agents.append(agent_name)
                    activated.append(agent_name)
                    logger.info(f"Added activated agent '{agent_name}' to thought sequence")
        
        if deactivated or activated:
            logger.info(f"Updated thought agents: deactivated {len(deactivated)}, activated {len(activated)}")
    
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
            "lifecycle_interval": self.lifecycle_interval,
            "last_creation_time": self.last_creation_time.isoformat() if self.last_creation_time else None,
            "last_lifecycle_time": self.last_lifecycle_time.isoformat() if self.last_lifecycle_time else None,
            "creation_history": self.creation_history,
            "lifecycle_history": self.lifecycle_history
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
            system.lifecycle_interval = evolving_state.get("lifecycle_interval", 86400)  # Default if not present
            
            if evolving_state["last_creation_time"]:
                system.last_creation_time = datetime.fromisoformat(evolving_state["last_creation_time"])
            
            if "last_lifecycle_time" in evolving_state and evolving_state["last_lifecycle_time"]:
                system.last_lifecycle_time = datetime.fromisoformat(evolving_state["last_lifecycle_time"])
            
            system.creation_history = evolving_state["creation_history"]
            system.lifecycle_history = evolving_state.get("lifecycle_history", [])  # Default if not present
            
            logger.info(f"Loaded evolving thought system state from {evolving_state_path}")
        
        return system    
    
    def evaluate_agent_lifecycle(self) -> Dict[str, Any]:
        """Evaluate which agents should be activated or deactivated.
        
        Returns:
            Evaluation results including recommendations for each agent
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "recommendations": []
        }
        
        try:
            if self.lifecycle_agent is None:
                raise ValueError("Lifecycle manager not set up. Call setup_self_modification() first.")
            
            # Prepare information about current agents
            agent_data = []
            for name, agent in self.agents.items():
                # Skip core agents - they cannot be deactivated
                if agent.is_core:
                    continue
                    
                agent_data.append({
                    "name": name,
                    "role": agent.role,
                    "is_active": agent.is_active,
                    "recent_output_count": len(agent.recent_outputs),
                    "performance_metrics": agent.performance_metrics,
                    "system_prompt_summary": agent.system_prompt.content[:200] + "..." 
                    if len(agent.system_prompt.content) > 200 else agent.system_prompt.content
                })
            
            # If there are no non-core agents, nothing to evaluate
            if not agent_data:
                results["message"] = "No non-core agents to evaluate"
                results["success"] = True
                return results
                
            # Create evaluation request
            evaluation_request = f"""Please evaluate the following agents and determine which should be activated or deactivated.

            CURRENT AGENTS:
            {json.dumps(agent_data, indent=2)}

            For each agent, decide whether it should be:
            1. ACTIVATED (if currently inactive but would be useful)
            2. DEACTIVATED (if currently active but redundant or underperforming)
            3. NO CHANGE (if current state is appropriate)

            Remember:
            - Only recommend activating an agent if it would provide significant value in the current ecosystem
            - Only recommend deactivating an agent if it's clearly underperforming or redundant
            - When in doubt, recommend NO CHANGE

            For each agent, provide your recommendation in this format:
            AGENT: [agent name]
            RECOMMENDATION: [ACTIVATE/DEACTIVATE/NO CHANGE]
            REASONING: [brief explanation for your recommendation]
            """
            
            evaluation = self.lifecycle_agent.generate_response(evaluation_request)
            results["evaluation"] = evaluation
            
            # Parse recommendations
            recommendations = self._parse_lifecycle_recommendations(evaluation)
            results["recommendations"] = recommendations
            
            # Apply recommendations
            for rec in recommendations:
                agent_name = rec["agent"]
                action = rec["action"]
                
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    
                    # Skip core agents
                    if agent.is_core:
                        continue
                        
                    if action == "ACTIVATE" and not agent.is_active:
                        agent.activate()
                        rec["applied"] = True
                    elif action == "DEACTIVATE" and agent.is_active:
                        agent.deactivate()
                        rec["applied"] = True
                    else:
                        rec["applied"] = False
                else:
                    rec["applied"] = False
                    rec["error"] = f"Agent '{agent_name}' not found"
            
            results["success"] = True
            logger.info(f"Successfully evaluated agent lifecycle with {len(recommendations)} recommendations")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Error during agent lifecycle evaluation: {e}")
        
        return results
    
    def _parse_lifecycle_recommendations(self, evaluation_text: str) -> List[Dict[str, Any]]:
        """Parse the lifecycle recommendations from the evaluation text.
        
        Args:
            evaluation_text: The evaluation text from the lifecycle agent
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Extract recommendations using regex pattern
        pattern = r"AGENT:\s*([^\n]+)\s*\nRECOMMENDATION:\s*(ACTIVATE|DEACTIVATE|NO CHANGE)\s*\nREASONING:\s*([^\n]+(?:\n[^\n]+)*?)(?=\nAGENT:|$)"
        matches = re.finditer(pattern, evaluation_text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            agent_name = match.group(1).strip()
            action = match.group(2).strip().upper()
            reasoning = match.group(3).strip()
            
            recommendations.append({
                "agent": agent_name,
                "action": action,
                "reasoning": reasoning,
                "applied": False
            })
        
        return recommendations