"""
Google Sheets Data Analysis Agent Framework
-----------------------------------------
A modular system for connecting to Google Sheets and generating deep, non-obvious insights.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
import time
import re
from enum import Enum
import traceback
import uuid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataAnalysisAgent")

class AgentState(Enum):
    """Represents the current state of an agent's execution."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"

@dataclass
class Message:
    """Represents a message passed between agent components."""
    sender: str
    recipient: str
    message_type: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to a dictionary."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from a dictionary."""
        return cls(
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=data["message_type"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id", str(uuid.uuid4()))
        )

class AgentMemory:
    """Shared memory system for agents to store and retrieve information."""
    
    def __init__(self):
        self._memory: Dict[str, Any] = {}
        self._message_history: List[Message] = []
        
    def store(self, key: str, value: Any) -> None:
        """Store a value in memory."""
        self._memory[key] = value
        logger.debug(f"Stored in memory: {key}")
        
    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from memory."""
        return self._memory.get(key, default)
    
    def all_keys(self) -> List[str]:
        """Get all keys in memory."""
        return list(self._memory.keys())
    
    def clear(self) -> None:
        """Clear all memory."""
        self._memory = {}
        
    def add_message(self, message: Message) -> None:
        """Add a message to the history."""
        self._message_history.append(message)
        
    def get_messages(self, 
                    sender: Optional[str] = None, 
                    recipient: Optional[str] = None, 
                    message_type: Optional[str] = None,
                    limit: Optional[int] = None) -> List[Message]:
        """Retrieve messages from history with optional filtering."""
        filtered = self._message_history
        
        if sender:
            filtered = [m for m in filtered if m.sender == sender]
        if recipient:
            filtered = [m for m in filtered if m.recipient == recipient]
        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]
        
        # Sort by timestamp (newest first)
        filtered = sorted(filtered, key=lambda m: m.timestamp, reverse=True)
        
        if limit:
            filtered = filtered[:limit]
            
        return filtered

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, agent_id: str, memory: AgentMemory):
        self.agent_id = agent_id
        self.memory = memory
        self.state = AgentState.IDLE
        logger.info(f"Initialized agent: {agent_id}")
        
    def send_message(self, recipient: str, message_type: str, content: Any) -> Message:
        """Send a message to another agent."""
        message = Message(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content
        )
        self.memory.add_message(message)
        logger.debug(f"Message sent from {self.agent_id} to {recipient}: {message_type}")
        return message
    
    def get_messages(self, 
                     sender: Optional[str] = None, 
                     message_type: Optional[str] = None,
                     limit: Optional[int] = None) -> List[Message]:
        """Get messages sent to this agent."""
        return self.memory.get_messages(
            sender=sender,
            recipient=self.agent_id,
            message_type=message_type,
            limit=limit
        )
    
    def run(self, *args, **kwargs) -> Any:
        """Main execution method for the agent."""
        self.state = AgentState.RUNNING
        try:
            result = self._execute(*args, **kwargs)
            self.state = AgentState.COMPLETED
            return result
        except Exception as e:
            logger.error(f"Error in agent {self.agent_id}: {str(e)}")
            logger.error(traceback.format_exc())
            self.state = AgentState.FAILED
            raise
    
    def _execute(self, *args, **kwargs) -> Any:
        """Implementation of agent's main functionality. To be overridden by subclasses."""
        raise NotImplementedError("Agent must implement _execute method")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, state={self.state.value})"

class AgentSystem:
    """Orchestrates the execution of multiple agents."""
    
    def __init__(self):
        self.memory = AgentMemory()
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the system."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
        
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def execute_agent(self, agent_id: str, *args, **kwargs) -> Any:
        """Execute a specific agent."""
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        
        start_time = time.time()
        log_entry = {
            "agent_id": agent_id,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "args": args,
            "kwargs": kwargs
        }
        
        try:
            result = agent.run(*args, **kwargs)
            log_entry["status"] = "success"
            log_entry["result"] = str(result)
            return result
        except Exception as e:
            log_entry["status"] = "error"
            log_entry["error"] = str(e)
            log_entry["traceback"] = traceback.format_exc()
            raise
        finally:
            end_time = time.time()
            log_entry["end_time"] = datetime.fromtimestamp(end_time).isoformat()
            log_entry["duration"] = end_time - start_time
            self.execution_log.append(log_entry)
    
    def execute_workflow(self, workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a sequence of agents according to a workflow specification."""
        results = {}
        
        for step in workflow:
            agent_id = step["agent_id"]
            args = step.get("args", [])
            kwargs = step.get("kwargs", {})
            
            # Replace any placeholders with previous results
            for key, value in kwargs.items():
                if isinstance(value, str) and value.startswith("$result."):
                    result_key = value[8:]
                    if result_key in results:
                        kwargs[key] = results[result_key]
            
            try:
                result = self.execute_agent(agent_id, *args, **kwargs)
                if "output_key" in step:
                    results[step["output_key"]] = result
            except Exception as e:
                logger.error(f"Workflow failed at step {agent_id}: {str(e)}")
                if step.get("critical", True):
                    raise
                
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the agent system."""
        return {
            "agents": {agent_id: agent.state.value for agent_id, agent in self.agents.items()},
            "memory_keys": self.memory.all_keys(),
            "execution_log_count": len(self.execution_log)
        }

@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    source: str
    rows: int
    columns: int
    column_types: Dict[str, str]
    column_descriptions: Dict[str, str] = field(default_factory=dict)
    missing_values: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    @classmethod
    def from_dataframe(cls, name: str, source: str, df: pd.DataFrame) -> 'DatasetInfo':
        """Create DatasetInfo from a pandas DataFrame."""
        column_types = {col: str(df[col].dtype) for col in df.columns}
        missing_values = {col: df[col].isna().sum() for col in df.columns}
        
        return cls(
            name=name,
            source=source,
            rows=len(df),
            columns=len(df.columns),
            column_types=column_types,
            missing_values=missing_values
        )

@dataclass
class Hypothesis:
    """Represents a hypothesis about the data."""
    id: str
    statement: str
    confidence: float  # 0.0 to 1.0
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    validation_status: str = "unvalidated"  # unvalidated, validated, rejected
    p_value: Optional[float] = None
    validation_method: Optional[str] = None
    related_columns: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_timestamp: float = field(default_factory=time.time)
    updated_timestamp: float = field(default_factory=time.time)
    
    def update_evidence(self, evidence: str, supports: bool = True) -> None:
        """Add evidence supporting or contradicting the hypothesis."""
        if supports:
            self.supporting_evidence.append(evidence)
        else:
            self.contradicting_evidence.append(evidence)
        self.updated_timestamp = time.time()
    
    def validate(self, validation_status: str, p_value: Optional[float] = None, 
                validation_method: Optional[str] = None) -> None:
        """Update the validation status of the hypothesis."""
        self.validation_status = validation_status
        self.p_value = p_value
        self.validation_method = validation_method
        self.updated_timestamp = time.time()
    
    @property
    def strength(self) -> float:
        """Calculate the overall strength of the hypothesis."""
        evidence_ratio = len(self.supporting_evidence) / max(1, len(self.supporting_evidence) + len(self.contradicting_evidence))
        if self.validation_status == "validated":
            return (self.confidence + evidence_ratio + 1) / 3
        elif self.validation_status == "rejected":
            return (self.confidence + evidence_ratio) / 3
        else:
            return (self.confidence + evidence_ratio) / 2

@dataclass
class Insight:
    """Represents an insight derived from the data."""
    id: str
    title: str
    description: str
    importance: float  # 0.0 to 1.0
    action_items: List[str] = field(default_factory=list)
    source_hypotheses: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_timestamp: float = field(default_factory=time.time)
    
    @property
    def has_actions(self) -> bool:
        """Check if the insight has actionable items."""
        return len(self.action_items) > 0