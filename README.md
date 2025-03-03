# Agent-Based Data Analysis Framework

This framework provides a modular, extensible system for data analysis using a multi-agent architecture. At its core is the Master Planner, which orchestrates the entire workflow from data ingestion to insight generation.

## Architecture Overview

The system implements a hierarchical agent architecture with the following components:

1. **Agent Framework Core** - The foundation of the system that provides:
   - Agent state management
   - Inter-agent messaging
   - Memory management
   - Base agent class for extension

2. **Master Planner** - The central orchestrator that:
   - Manages workflow definition and execution
   - Coordinates agents based on workflow steps
   - Handles retries and error conditions
   - Provides workflow status tracking

3. **Specialized Agents** - Purpose-built agents for specific tasks:
   - Google Sheets Connector - Retrieves data from Google Sheets
   - Data Explorer - Profiles and explores datasets
   - Hypothesis Generator - Generates potential insights as hypotheses
   - Hypothesis Validator - Tests hypotheses against data
   - Insight Synthesizer - Consolidates validated hypotheses into actionable insights
   - Communication Agent - Formats and delivers results

## Master Planner

The Master Planner is the core orchestration component responsible for:

- **Workflow Management**: Defining, registering, and executing workflows that specify the sequence of data analysis steps.
- **Agent Coordination**: Triggering agents in the right order, passing data between them, and managing their lifecycle.
- **Error Handling**: Implementing retries, timeouts, and fallback mechanisms when agents fail or take too long.
- **Status Tracking**: Providing real-time status updates on workflow execution and results.

### Workflow Configuration

Workflows are defined as a sequence of steps, each specifying:
- An agent to execute
- The action to perform
- Arguments for the action
- Error handling behavior

Example workflow configuration:
```python
workflow_config = WorkflowConfig(
    name="Data Analysis Workflow",
    steps=[
        {
            "id": "data_ingestion",
            "agent_id": "google_sheets_connector",
            "action": "get_sheet_data",
            "args": { "spreadsheet_id": "your_spreadsheet_id" }
        },
        # Additional steps...
    ],
    max_retries=3,
    timeout_seconds=300,
    continue_on_error=False
)
```

## Getting Started

### Prerequisites

- Python 3.8+
- Google API credentials (for Google Sheets integration)
- Required packages (see `requirements.txt`)

### Basic Usage

1. **Initialize the agent system**:
   ```python
   memory = AgentMemory()
   agent_system = AgentSystem()
   master_planner = MasterPlanner("master_planner", memory, agent_system)
   agent_system.register_agent(master_planner)
   
   # Register other necessary agents
   ```

2. **Define and register a workflow**:
   ```python
   workflow_config = create_sample_workflow()  # Or define your custom workflow
   workflow_id = master_planner.register_workflow(workflow_config)
   ```

3. **Execute the workflow**:
   ```python
   results = master_planner.execute_workflow(workflow_id, input_data)
   ```

## Error Handling and Resilience

The system is designed to be resilient through:
- Configurable retry mechanisms with exponential backoff
- Timeouts to prevent stuck processes
- Ability to continue on error for non-critical steps
- Comprehensive logging for debugging and monitoring

## Extending the Framework

To add a new agent type:
1. Create a new class extending `BaseAgent`
2. Implement the `_execute` method for agent-specific logic
3. Register the agent with the agent system

Example:
```python
class MyCustomAgent(BaseAgent):
    def _execute(self, action: str = "default_action", **kwargs) -> Any:
        # Implement agent-specific logic
        return results
```

## License

[MIT License](LICENSE) 