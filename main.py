#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application file for the agent-based data analysis system.

This file demonstrates how to set up the system, register agents, and execute workflows.
"""

import time
import logging
from typing import Dict, Any

from agent_framework_core import AgentMemory, AgentSystem
from master_planner import MasterPlanner, WorkflowConfig, create_sample_workflow
from google_sheets_connector import GoogleSheetsConnector, MultipleSheetConnector
from data_explorer import DataExplorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

def initialize_system() -> Dict[str, Any]:
    """
    Initialize the agent system and register all agents.
    
    Returns:
        Dict containing the agent system and master planner
    """
    logger.info("Initializing agent system...")
    
    # Create shared memory and agent system
    memory = AgentMemory()
    agent_system = AgentSystem()
    
    # Create the Master Planner
    master_planner = MasterPlanner("master_planner", memory, agent_system)
    agent_system.register_agent(master_planner)
    
    # Create and register Google Sheets Connector
    google_sheets_connector = GoogleSheetsConnector("google_sheets_connector", memory)
    agent_system.register_agent(google_sheets_connector)
    
    # Create and register Multiple Sheet Connector
    multiple_sheet_connector = MultipleSheetConnector(
        "multiple_sheet_connector", memory, google_sheets_connector
    )
    agent_system.register_agent(multiple_sheet_connector)
    
    # Create and register Data Explorer
    data_explorer = DataExplorer("data_explorer", memory)
    agent_system.register_agent(data_explorer)
    
    # Here you would register other agents like:
    # - HypothesisGenerator
    # - HypothesisValidator
    # - InsightSynthesizer
    # - CommunicationAgent
    
    logger.info("Agent system initialized successfully")
    
    return {
        "agent_system": agent_system,
        "master_planner": master_planner
    }

def run_sample_workflow(master_planner: MasterPlanner) -> Dict[str, Any]:
    """
    Register and run a sample workflow.
    
    Args:
        master_planner: The master planner agent
        
    Returns:
        The workflow results
    """
    logger.info("Setting up sample workflow...")
    
    # Create a sample workflow configuration
    workflow_config = create_sample_workflow()
    
    # Register the workflow with the master planner
    workflow_id = master_planner.register_workflow(workflow_config=workflow_config)
    
    # Execute the workflow
    logger.info(f"Executing workflow with ID: {workflow_id}")
    
    initial_data = {
        "spreadsheet_id": "your_actual_spreadsheet_id",
        "sheet_name": "Sheet1"
    }
    
    return master_planner.execute_workflow(workflow_id=workflow_id, input_data=initial_data)

def main():
    """Main entry point for the application."""
    try:
        # Initialize the system
        system = initialize_system()
        master_planner = system["master_planner"]
        
        # Run a sample workflow
        results = run_sample_workflow(master_planner)
        
        # Display the results
        logger.info(f"Workflow execution completed with status: {results['status']}")
        
        # If successful, print the final insights
        if results['status'] == 'completed' and 'insight_synthesis' in results['results']:
            insights = results['results']['insight_synthesis']
            logger.info(f"Generated {len(insights)} insights from the data analysis")
            
            # Here you might want to format and display the insights
            # or save them to a file, database, etc.
        
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        raise

if __name__ == "__main__":
    main() 