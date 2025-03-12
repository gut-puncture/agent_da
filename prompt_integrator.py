"""
Prompt Integrator Module
-----------------------
This module connects the agent implementations with their respective prompts
and provides utilities for integrating prompts into the agent execution flow.
"""

import os
import logging
import json
from typing import Optional, Dict, Any

from master_planner_prompt import MASTER_PLANNER_PROMPT
from data_explorer_prompt import DATA_EXPLORER_PROMPT
from hypothesis_generator_prompt import HYPOTHESIS_GENERATOR_PROMPT
from hypothesis_validator_prompt import HYPOTHESIS_VALIDATOR_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PromptIntegrator")

class PromptIntegrator:
    """
    Integrates prompt-based guidance with agent execution.
    
    This class manages the prompts for each agent type, handles prompt retrieval,
    and provides utilities for agent-prompt integration.
    """
    
    def __init__(self):
        """Initialize the PromptIntegrator with predefined prompts."""
        self.prompts = {
            "master_planner": MASTER_PLANNER_PROMPT,
            "data_explorer": DATA_EXPLORER_PROMPT,
            "hypothesis_generator": HYPOTHESIS_GENERATOR_PROMPT,
            "hypothesis_validator": HYPOTHESIS_VALIDATOR_PROMPT
        }
        
        # Initialize cache for any dynamically loaded prompts
        self.prompt_cache = {}
        
        # Default generation configuration for Gemini models
        self.gemini_config = {
            "gemini-2.0-pro-exp-02-05": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 8192  # Increased for the newer model
            },
            "default": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 4096
            }
        }
        
        # Check for environment-based prompt overrides
        self._load_environment_prompts()
    
    def _load_environment_prompts(self):
        """
        Load prompts from environment variables if they exist.
        
        This allows for dynamic prompt updates through environment
        configuration without code changes.
        """
        for agent_type in self.prompts.keys():
            env_var_name = f"{agent_type.upper()}_PROMPT_PATH"
            prompt_path = os.environ.get(env_var_name)
            
            if prompt_path and os.path.exists(prompt_path):
                try:
                    with open(prompt_path, 'r') as f:
                        self.prompts[agent_type] = f.read()
                    logger.info(f"Loaded {agent_type} prompt from {prompt_path}")
                except Exception as e:
                    logger.error(f"Error loading prompt from {prompt_path}: {str(e)}")
    
    def get_prompt(self, agent_type: str) -> str:
        """
        Get the prompt for a specific agent type.
        
        Args:
            agent_type: The type of agent (e.g., "master_planner", "data_explorer")
            
        Returns:
            The prompt text for the specified agent type
            
        Raises:
            ValueError: If the agent type is unknown
        """
        if agent_type not in self.prompts:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return self.prompts[agent_type]
    
    def extend_prompt(self, 
                     agent_type: str, 
                     additional_content: str, 
                     section: Optional[str] = None) -> str:
        """
        Extend an existing prompt with additional content.
        
        Args:
            agent_type: The type of agent whose prompt should be extended
            additional_content: Content to add to the prompt
            section: Optional section name where content should be added
            
        Returns:
            The extended prompt text
        """
        prompt = self.get_prompt(agent_type)
        
        if section:
            # Try to find the section and add content there
            section_marker = f"## {section}"
            if section_marker in prompt:
                parts = prompt.split(section_marker, 1)
                # Add after the section header
                prompt = f"{parts[0]}{section_marker}\n\n{additional_content}\n\n{parts[1]}"
            else:
                # If section not found, add to the end
                prompt = f"{prompt}\n\n## {section}\n\n{additional_content}"
        else:
            # Add to the end
            prompt = f"{prompt}\n\n{additional_content}"
        
        # Update prompt in memory
        self.prompts[agent_type] = prompt
        
        return prompt
    
    def create_gemini_prompt(self, 
                           agent_type: str, 
                           context: Dict[str, Any] = None,
                           model: str = "gemini-2.0-pro-exp-02-05") -> Dict[str, Any]:
        """
        Create a formatted prompt for Gemini API with relevant context.
        
        Args:
            agent_type: The type of agent
            context: Additional context to include in the prompt
            model: The model to use (defaults to gemini-2.0-pro-exp-02-05)
            
        Returns:
            A dictionary containing the prompt ready for Gemini API
        """
        prompt = self.get_prompt(agent_type)
        
        # Add context if provided
        if context:
            context_str = json.dumps(context, indent=2)
            prompt = f"{prompt}\n\n## Current Context\n\n```json\n{context_str}\n```\n\n"
        
        # Get the appropriate generation config for the model
        generation_config = self.gemini_config.get(model, self.gemini_config["default"])
        
        # Log which model and configuration is being used
        logger.info(f"Creating prompt for model {model} with agent {agent_type}")
        
        return {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": generation_config
        }
    
    def register_with_agent(self, agent):
        """
        Register the appropriate prompt with an agent instance.
        
        This method adds the prompt to the agent instance based on its type.
        
        Args:
            agent: An agent instance
        """
        agent_type = self._determine_agent_type(agent)
        if agent_type:
            prompt = self.get_prompt(agent_type)
            agent.prompt = prompt
            logger.info(f"Registered {agent_type} prompt with agent {agent.agent_id}")
    
    def _determine_agent_type(self, agent) -> Optional[str]:
        """
        Determine the type of an agent instance.
        
        Args:
            agent: An agent instance
            
        Returns:
            The agent type or None if not recognized
        """
        agent_class = agent.__class__.__name__.lower()
        
        if "masterplanner" in agent_class:
            return "master_planner"
        elif "dataexplorer" in agent_class:
            return "data_explorer"
        elif "hypothesisgenerator" in agent_class:
            return "hypothesis_generator"
        elif "hypothesisvalidator" in agent_class:
            return "hypothesis_validator"
        
        return None

# Create a singleton instance for easy import
prompt_integrator = PromptIntegrator()

def get_prompt(agent_type: str) -> str:
    """
    Convenience function to get a prompt for a specific agent type.
    
    Args:
        agent_type: The type of agent
        
    Returns:
        The prompt text
    """
    return prompt_integrator.get_prompt(agent_type) 