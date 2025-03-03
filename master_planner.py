import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field

from agent_framework_core import (
    BaseAgent, 
    AgentMemory, 
    AgentSystem, 
    Message, 
    AgentState,
    DatasetInfo
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class WorkflowStep(Enum):
    """Represents the different steps in the data analysis workflow."""
    DATA_INGESTION = "data_ingestion"
    DATA_EXPLORATION = "data_exploration"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    HYPOTHESIS_VALIDATION = "hypothesis_validation"
    INSIGHT_SYNTHESIS = "insight_synthesis"
    COMMUNICATION = "communication"

@dataclass
class WorkflowConfig:
    """Configuration for a workflow."""
    name: str
    steps: List[Dict[str, Any]]
    max_retries: int = 3
    timeout_seconds: int = 300  # 5 minutes default timeout
    continue_on_error: bool = False
    notification_on_completion: bool = True
    notification_on_error: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class MasterPlanner(BaseAgent):
    """
    Master Planner agent that orchestrates the entire data analysis workflow.
    
    This agent is responsible for registering, triggering, and coordinating the various
    specialized agents, ensuring that data flows correctly through the pipeline from
    ingestion to final insights.
    """
    
    def __init__(self, agent_id: str, memory: AgentMemory, agent_system: AgentSystem):
        """
        Initialize the Master Planner.
        
        Args:
            agent_id: Unique identifier for this agent
            memory: Shared memory system for agent communication
            agent_system: Reference to the agent system for managing agents
        """
        super().__init__(agent_id, memory)
        self.agent_system = agent_system
        self.workflows = {}
        self.current_workflow = None
        self.current_step = None
        self.logger = logging.getLogger(f"MasterPlanner_{agent_id}")
        self.step_results = {}
        self.step_timeouts = {}
        self.step_retries = {}
    
    def _execute(self, action: str = "execute_workflow", **kwargs) -> Any:
        """
        Execute the specified action.
        
        Args:
            action: The action to execute
            kwargs: Additional arguments specific to the action
            
        Returns:
            The result of the action
        """
        if action == "register_workflow":
            return self.register_workflow(**kwargs)
        elif action == "execute_workflow":
            return self.execute_workflow(**kwargs)
        elif action == "get_workflow_status":
            return self.get_workflow_status(**kwargs)
        elif action == "abort_workflow":
            return self.abort_workflow(**kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def register_workflow(self, workflow_config: WorkflowConfig) -> str:
        """
        Register a new workflow with the Master Planner.
        
        Args:
            workflow_config: Configuration for the workflow
            
        Returns:
            The workflow ID
        """
        workflow_id = f"workflow_{len(self.workflows) + 1}"
        self.workflows[workflow_id] = workflow_config
        self.logger.info(f"Registered workflow '{workflow_config.name}' with ID '{workflow_id}'")
        return workflow_id
    
    def execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a registered workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
            input_data: Initial data to pass to the workflow
            
        Returns:
            The results of the workflow execution
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow '{workflow_id}' not registered")
        
        workflow = self.workflows[workflow_id]
        self.current_workflow = workflow_id
        self.step_results = {}
        self.step_timeouts = {}
        self.step_retries = {}
        
        self.logger.info(f"Starting workflow '{workflow.name}'")
        self.state = AgentState.RUNNING
        
        # Initialize the context with input data
        context = input_data or {}
        
        # Execute each step in the workflow
        for step_idx, step in enumerate(workflow.steps):
            step_id = step.get("id", f"step_{step_idx}")
            self.current_step = step_id
            agent_id = step.get("agent_id")
            action = step.get("action", "run")
            args = step.get("args", {})
            
            # Update the context with step-specific info
            step_context = context.copy()
            step_context.update(args)
            
            # Initialize retry counter for this step
            self.step_retries[step_id] = 0
            
            # Execute the step with retries
            result = self._execute_step_with_retries(
                step_id, agent_id, action, step_context, workflow.max_retries, workflow.timeout_seconds
            )
            
            # Store the result
            self.step_results[step_id] = result
            
            # Update the context with the step result
            if isinstance(result, dict):
                context.update(result)
            else:
                context[step_id + "_result"] = result
            
            # Check for conditions to proceed
            if not result and not workflow.continue_on_error:
                self.logger.error(f"Step '{step_id}' failed and continue_on_error is False. Aborting workflow.")
                self.state = AgentState.FAILED
                
                # Send a notification if needed
                if workflow.notification_on_error:
                    self._send_error_notification(workflow_id, step_id, "Step failed")
                
                return {"status": "failed", "step": step_id, "results": self.step_results}
        
        # Successfully completed all steps
        self.state = AgentState.COMPLETED
        self.logger.info(f"Workflow '{workflow.name}' completed successfully")
        
        # Send a completion notification if needed
        if workflow.notification_on_completion:
            self._send_completion_notification(workflow_id)
        
        return {"status": "completed", "results": self.step_results}
    
    def _execute_step_with_retries(
        self, step_id: str, agent_id: str, action: str, 
        args: Dict[str, Any], max_retries: int, timeout_seconds: int
    ) -> Any:
        """
        Execute a step with retry logic and timeout.
        
        Args:
            step_id: ID of the step
            agent_id: ID of the agent to execute
            action: Action to execute
            args: Arguments for the action
            max_retries: Maximum number of retries
            timeout_seconds: Timeout in seconds
            
        Returns:
            The result of the step execution
        """
        while self.step_retries[step_id] <= max_retries:
            try:
                self.logger.info(f"Executing step '{step_id}' with agent '{agent_id}', action '{action}'")
                
                # Set a timeout for this step
                start_time = time.time()
                self.step_timeouts[step_id] = start_time + timeout_seconds
                
                # Execute the step
                agent = self.agent_system.get_agent(agent_id)
                if not agent:
                    raise ValueError(f"Agent '{agent_id}' not found")
                
                # Execute the agent with the specified action and arguments
                result = self.agent_system.execute_agent(agent_id, action=action, **args)
                
                # Check if we've hit the timeout
                if time.time() > self.step_timeouts[step_id]:
                    raise TimeoutError(f"Step '{step_id}' timed out after {timeout_seconds} seconds")
                
                self.logger.info(f"Step '{step_id}' completed successfully")
                return result
                
            except TimeoutError as e:
                self.logger.warning(f"Step '{step_id}' timed out: {str(e)}")
                self.step_retries[step_id] += 1
                
            except Exception as e:
                self.logger.error(f"Error executing step '{step_id}': {str(e)}")
                self.step_retries[step_id] += 1
            
            # Check if we've hit max retries
            if self.step_retries[step_id] > max_retries:
                self.logger.error(f"Step '{step_id}' failed after {max_retries} retries")
                return None
            
            # Log retry attempt
            self.logger.info(f"Retrying step '{step_id}', attempt {self.step_retries[step_id]} of {max_retries}")
            
            # Wait before retrying (with exponential backoff)
            backoff_time = 2 ** self.step_retries[step_id]
            time.sleep(min(backoff_time, 30))  # Cap at 30 seconds
        
        return None
    
    def get_workflow_status(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the status of a workflow.
        
        Args:
            workflow_id: ID of the workflow to check status for
                        If None, returns status for the current workflow
            
        Returns:
            The workflow status information
        """
        if workflow_id is None:
            workflow_id = self.current_workflow
        
        if not workflow_id or workflow_id not in self.workflows:
            return {"status": "unknown", "error": "Workflow not found"}
        
        workflow = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "state": self.state.value if self.current_workflow == workflow_id else "unknown",
            "current_step": self.current_step if self.current_workflow == workflow_id else None,
            "steps_completed": len(self.step_results) if self.current_workflow == workflow_id else 0,
            "total_steps": len(workflow.steps),
            "step_results": self.step_results if self.current_workflow == workflow_id else {}
        }
    
    def abort_workflow(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Abort a running workflow.
        
        Args:
            workflow_id: ID of the workflow to abort
                        If None, aborts the current workflow
            
        Returns:
            Status of the abort operation
        """
        if workflow_id is None:
            workflow_id = self.current_workflow
        
        if not workflow_id or workflow_id not in self.workflows:
            return {"status": "error", "message": "Workflow not found"}
        
        if self.current_workflow != workflow_id or self.state != AgentState.RUNNING:
            return {"status": "error", "message": "Workflow is not currently running"}
        
        self.logger.info(f"Aborting workflow '{workflow_id}'")
        self.state = AgentState.FAILED
        
        # Send an abort notification
        self._send_error_notification(workflow_id, self.current_step, "Workflow aborted")
        
        return {"status": "aborted", "workflow_id": workflow_id, "step": self.current_step}
    
    def _send_completion_notification(self, workflow_id: str) -> None:
        """
        Send a notification when a workflow completes successfully.
        
        Args:
            workflow_id: ID of the completed workflow
        """
        workflow = self.workflows[workflow_id]
        message = f"Workflow '{workflow.name}' completed successfully"
        self.send_message("communication_agent", "notification", {
            "level": "info",
            "message": message,
            "workflow_id": workflow_id,
            "results": self.step_results
        })
        self.logger.info(message)
    
    def _send_error_notification(self, workflow_id: str, step_id: str, message: str) -> None:
        """
        Send a notification when a workflow encounters an error.
        
        Args:
            workflow_id: ID of the workflow with an error
            step_id: ID of the step where the error occurred
            message: Error message
        """
        workflow = self.workflows[workflow_id]
        full_message = f"Error in workflow '{workflow.name}', step '{step_id}': {message}"
        self.send_message("communication_agent", "notification", {
            "level": "error",
            "message": full_message,
            "workflow_id": workflow_id,
            "step_id": step_id
        })
        self.logger.error(full_message)

# Example usage
def create_sample_workflow() -> WorkflowConfig:
    """
    Create a sample data analysis workflow configuration.
    
    Returns:
        A sample WorkflowConfig
    """
    return WorkflowConfig(
        name="Data Analysis Workflow",
        steps=[
            {
                "id": "data_ingestion",
                "agent_id": "google_sheets_connector",
                "action": "get_sheet_data",
                "args": {
                    "spreadsheet_id": "your_spreadsheet_id",
                    "sheet_name": "Sheet1"
                }
            },
            {
                "id": "data_exploration",
                "agent_id": "data_explorer",
                "action": "explore_data",
                "args": {
                    "include_correlations": True,
                    "include_histograms": True,
                    "top_n_values": 5,
                    "outlier_detection_method": "iqr",
                    "outlier_threshold": 1.5
                }
            },
            {
                "id": "hypothesis_generation",
                "agent_id": "hypothesis_generator",
                "action": "generate_hypotheses",
                "args": {}  # Will be populated with results from previous step
            },
            {
                "id": "hypothesis_validation",
                "agent_id": "hypothesis_validator",
                "action": "validate_hypotheses",
                "args": {}  # Will be populated with results from previous step
            },
            {
                "id": "insight_synthesis",
                "agent_id": "insight_synthesizer",
                "action": "synthesize_insights",
                "args": {}  # Will be populated with results from previous step
            },
            {
                "id": "communication",
                "agent_id": "communication_agent",
                "action": "generate_report",
                "args": {}  # Will be populated with results from previous step
            }
        ],
        max_retries=2,
        timeout_seconds=180,
        continue_on_error=False,
        notification_on_completion=True,
        notification_on_error=True,
        metadata={
            "description": "Sample workflow for data analysis",
            "owner": "user@example.com"
        }
    ) 