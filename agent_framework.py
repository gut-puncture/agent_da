"""
Alias module to provide compatibility for agent_framework imports.
Re-exports core agent framework classes.
"""
from agent_framework_core import BaseAgent, AgentMemory, AgentState, DatasetInfo

__all__ = ['BaseAgent', 'AgentMemory', 'AgentState', 'DatasetInfo']