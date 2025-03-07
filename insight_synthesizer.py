import logging
import numpy as np
import pandas as pd
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import requests
import os

from agent_framework_core import (
    BaseAgent, 
    AgentMemory, 
    Message, 
    AgentState,
    Insight,
    Hypothesis
)

from insight_synthesizer_prompt import (
    INSIGHT_SYNTHESIZER_PROMPT,
    INSIGHT_SYNTHESIS_SYSTEM_PROMPT,
    INSIGHT_SYNTHESIS_JSON_FORMAT,
    SYNTHESIS_PROMPT_TEMPLATE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("InsightSynthesizer")

@dataclass
class InsightGroup:
    """
    Represents a group of related hypotheses that can be synthesized into an insight.
    
    This is an internal representation used during the synthesis process.
    """
    hypotheses: List[Dict[str, Any]]
    primary_columns: List[str]
    relationship_types: List[str]
    strength: float  # Average confidence/strength of the hypotheses
    
    @property
    def hypothesis_ids(self) -> List[str]:
        """Get the IDs of all hypotheses in this group."""
        return [h["id"] for h in self.hypotheses]
    
    @property
    def tags(self) -> List[str]:
        """Extract and merge unique tags from all hypotheses."""
        all_tags = []
        for h in self.hypotheses:
            if "tags" in h and h["tags"]:
                all_tags.extend(h["tags"])
        return list(set(all_tags))

class InsightSynthesizer(BaseAgent):
    """
    Agent responsible for synthesizing validated hypotheses into actionable insights.
    
    This agent transforms statistical findings into business-relevant insights by:
    1. Grouping related hypotheses
    2. Generating cohesive narratives
    3. Identifying actionable recommendations
    4. Prioritizing insights by importance
    
    The synthesizer uses both rule-based methods and Gemini API for advanced synthesis.
    """
    
    def __init__(self, agent_id: str, memory: AgentMemory):
        """
        Initialize the InsightSynthesizer agent.
        
        Args:
            agent_id: Unique identifier for this agent
            memory: Shared memory instance for storing/retrieving data
        """
        super().__init__(agent_id, memory)
        
        # LLM API configuration (from environment or defaulted)
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_url = os.environ.get(
            "GEMINI_API_URL", 
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        )
        
        # Configure parameters
        self.min_hypotheses_per_insight = 1
        self.max_hypotheses_per_insight = 5
        self.similarity_threshold = 0.5  # For grouping related hypotheses
        
        logger.info(f"InsightSynthesizer initialized with ID: {agent_id}")
    
    def _execute(self, action: str = "generate_insights", **kwargs) -> Any:
        """
        Execute the agent's main functionality based on the requested action.
        
        Args:
            action: The action to perform (generate_insights, get_insight, etc.)
            **kwargs: Additional arguments specific to the action
            
        Returns:
            Results of the requested action
        """
        logger.info(f"Executing action: {action} with args: {kwargs}")
        
        # Set agent to running state
        self._state = AgentState.RUNNING
        
        try:
            # Route to appropriate method based on action
            if action == "generate_insights":
                result = self.generate_insights(**kwargs)
            elif action == "get_insight_by_id":
                result = self.get_insight_by_id(**kwargs)
            elif action == "get_all_insights":
                result = self.get_all_insights(**kwargs)
            elif action == "filter_insights":
                result = self.filter_insights(**kwargs)
            else:
                logger.warning(f"Unknown action: {action}")
                result = {"error": f"Unknown action: {action}"}
                self._state = AgentState.FAILED
                return result
            
            # Set agent to completed state
            self._state = AgentState.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error executing action {action}: {str(e)}", exc_info=True)
            self._state = AgentState.FAILED
            return {"error": str(e)}
    
    def generate_insights(self, 
                         dataset_name: str,
                         hypothesis_filter: Optional[Dict[str, Any]] = None,
                         max_insights: int = 5,
                         min_importance: float = 0.5) -> List[Dict[str, Any]]:
        """
        Generate insights by synthesizing validated hypotheses.
        
        Args:
            dataset_name: Name of the dataset to generate insights for
            hypothesis_filter: Optional filters to apply when selecting hypotheses
                (e.g., only validated ones, specific columns, etc.)
            max_insights: Maximum number of insights to generate
            min_importance: Minimum importance threshold for returned insights
            
        Returns:
            List of generated insights as dictionaries
        """
        logger.info(f"Generating insights for dataset: {dataset_name}")
        
        # 1. Retrieve validated hypotheses
        hypotheses = self._get_filtered_hypotheses(dataset_name, hypothesis_filter)
        logger.info(f"Retrieved {len(hypotheses)} validated hypotheses")
        
        if not hypotheses:
            logger.warning(f"No validated hypotheses found for dataset: {dataset_name}")
            return []
        
        # 2. Group related hypotheses
        hypothesis_groups = self._group_related_hypotheses(hypotheses)
        logger.info(f"Grouped hypotheses into {len(hypothesis_groups)} clusters")
        
        # 3. Generate insights from groups
        insights = []
        
        # First try with LLM-based synthesis
        try:
            insights = self._generate_gemini_insights(
                dataset_name, 
                hypotheses, 
                hypothesis_groups
            )
            logger.info(f"Generated {len(insights)} insights using Gemini")
        except Exception as e:
            logger.error(f"Error generating insights with Gemini: {str(e)}", exc_info=True)
            # Fall back to rule-based synthesis
            insights = self._generate_rule_based_insights(hypothesis_groups)
            logger.info(f"Generated {len(insights)} insights using rule-based approach")
        
        # 4. Filter by importance and limit
        insights = [i for i in insights if i["importance"] >= min_importance]
        insights.sort(key=lambda x: x["importance"], reverse=True)
        insights = insights[:max_insights]
        
        # 5. Store the insights in memory
        memory_key = f"insights:{dataset_name}"
        all_existing_insights = self.memory.retrieve(memory_key, [])
        
        # Create lookup of existing IDs to avoid duplicates
        existing_ids = {i["id"]: True for i in all_existing_insights}
        
        # Add only new insights
        for insight in insights:
            if insight["id"] not in existing_ids:
                all_existing_insights.append(insight)
        
        # Update memory
        self.memory.store(memory_key, all_existing_insights)
        logger.info(f"Stored {len(insights)} new insights in memory")
        
        # 6. Notify that insights have been generated
        self.send_message(
            recipient="master_planner", 
            message_type="insights_generated",
            content={
                "dataset_name": dataset_name,
                "num_insights": len(insights),
                "insight_ids": [i["id"] for i in insights]
            }
        )
        
        return insights
    
    def _get_filtered_hypotheses(self, 
                             dataset_name: str, 
                             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve hypotheses from memory with optional filtering.
        
        Args:
            dataset_name: Name of the dataset
            filters: Optional filtering criteria
            
        Returns:
            List of hypotheses matching the criteria
        """
        # Default to retrieving validated hypotheses
        if filters is None:
            filters = {"validation_status": "confirmed"}
        
        # Get all hypotheses for this dataset
        memory_key = f"hypotheses:{dataset_name}"
        all_hypotheses = self.memory.retrieve(memory_key, [])
        
        if not all_hypotheses:
            return []
        
        # Apply filters if specified
        filtered_hypotheses = []
        for hyp in all_hypotheses:
            match = True
            
            # Check each filter criterion
            for key, value in filters.items():
                if key not in hyp:
                    match = False
                    break
                    
                if isinstance(value, list):
                    if hyp[key] not in value:
                        match = False
                        break
                else:
                    if hyp[key] != value:
                        match = False
                        break
            
            if match:
                filtered_hypotheses.append(hyp)
        
        return filtered_hypotheses
    
    def _group_related_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> List[InsightGroup]:
        """
        Group related hypotheses based on shared columns, relationship types, etc.
        
        Args:
            hypotheses: List of hypothesis dictionaries
            
        Returns:
            List of InsightGroup objects containing related hypotheses
        """
        # Simple clustering based on shared columns
        groups = []
        
        # Sort hypotheses by strength (confidence) to process strongest first
        sorted_hypotheses = sorted(
            hypotheses, 
            key=lambda h: h.get("confidence_level", 0.5), 
            reverse=True
        )
        
        # Track which hypotheses have been assigned to groups
        assigned = {h["id"]: False for h in hypotheses}
        
        # First pass: Create initial groups
        for hyp in sorted_hypotheses:
            # Skip if already assigned
            if assigned[hyp["id"]]:
                continue
                
            # Create a new group
            columns = hyp.get("columns_involved", [])
            relationship = hyp.get("relationship_type", "unknown")
            
            group = InsightGroup(
                hypotheses=[hyp],
                primary_columns=columns,
                relationship_types=[relationship],
                strength=hyp.get("confidence_level", 0.5)
            )
            
            groups.append(group)
            assigned[hyp["id"]] = True
        
        # Second pass: Try to merge groups with shared columns
        merged = True
        while merged:
            merged = False
            for i in range(len(groups)):
                if i >= len(groups):  # Account for shifting indexes during merging
                    break
                    
                for j in range(i + 1, len(groups)):
                    if j >= len(groups):  # Account for shifting indexes during merging
                        break
                        
                    # Check for column overlap
                    cols_i = set(groups[i].primary_columns)
                    cols_j = set(groups[j].primary_columns)
                    
                    overlap = cols_i.intersection(cols_j)
                    
                    # If sufficient overlap, merge groups
                    if len(overlap) > 0 and len(overlap) / min(len(cols_i), len(cols_j)) >= self.similarity_threshold:
                        # Merge j into i
                        groups[i].hypotheses.extend(groups[j].hypotheses)
                        groups[i].primary_columns = list(set(groups[i].primary_columns + groups[j].primary_columns))
                        groups[i].relationship_types = list(set(groups[i].relationship_types + groups[j].relationship_types))
                        
                        # Update strength (weighted average)
                        total_hyps = len(groups[i].hypotheses)
                        groups[i].strength = (groups[i].strength * (total_hyps - len(groups[j].hypotheses)) + 
                                            groups[j].strength * len(groups[j].hypotheses)) / total_hyps
                        
                        # Remove the merged group
                        groups.pop(j)
                        merged = True
                        break
                        
                if merged:
                    break
        
        # Filter groups to ensure they don't exceed max hypotheses per insight
        for group in groups:
            if len(group.hypotheses) > self.max_hypotheses_per_insight:
                # Sort by confidence and keep only the strongest
                group.hypotheses.sort(key=lambda h: h.get("confidence_level", 0.5), reverse=True)
                group.hypotheses = group.hypotheses[:self.max_hypotheses_per_insight]
        
        return groups
    
    def _generate_rule_based_insights(self, hypothesis_groups: List[InsightGroup]) -> List[Dict[str, Any]]:
        """
        Generate insights using rule-based methods (fallback if LLM fails).
        
        Args:
            hypothesis_groups: Grouped hypotheses
            
        Returns:
            List of generated insights as dictionaries
        """
        insights = []
        
        for group in hypothesis_groups:
            if len(group.hypotheses) < self.min_hypotheses_per_insight:
                continue
                
            # Create a new insight ID
            insight_id = f"insight_{uuid.uuid4()}"
            
            # Get the primary hypothesis (highest confidence)
            primary_hyp = max(group.hypotheses, key=lambda h: h.get("confidence_level", 0.5))
            
            # Generate a title based on primary hypothesis
            relationship_type = primary_hyp.get("relationship_type", "relationship")
            columns = primary_hyp.get("columns_involved", [])
            column_text = " and ".join(columns[:2])  # Keep it concise with just first two
            
            title = f"Strong {relationship_type} in {column_text}"
            if len(title) > 60:  # Title too long, simplify
                title = f"{relationship_type.capitalize()} detected in dataset"
            
            # Generate description
            description_parts = []
            for hyp in group.hypotheses[:3]:  # Include up to 3 hypotheses in description
                desc = hyp.get("description", "")
                if desc:
                    description_parts.append(desc)
            
            description = " ".join(description_parts)
            if len(description) > 300:
                description = description[:297] + "..."
            
            # Calculate importance based on hypothesis strength and count
            importance = min(0.95, group.strength * (0.6 + 0.1 * min(4, len(group.hypotheses))))
            
            # Generate basic action items
            action_items = [
                f"Investigate the relationship between {' and '.join(columns[:2])} further",
                f"Consider how this {relationship_type} affects business decisions"
            ]
            
            # Build supporting data
            supporting_data = {}
            for hyp in group.hypotheses:
                if "validation_result" in hyp and hyp["validation_result"]:
                    for key, value in hyp["validation_result"].items():
                        if isinstance(value, (int, float, str)):  # Only include simple values
                            supporting_data[f"{hyp.get('relationship_type', 'stat')}_{key}"] = value
            
            # Create the insight
            insight = {
                "id": insight_id,
                "title": title,
                "description": description,
                "importance": importance,
                "action_items": action_items,
                "source_hypotheses": group.hypothesis_ids,
                "supporting_data": supporting_data,
                "tags": group.tags,
                "created_timestamp": time.time()
            }
            
            insights.append(insight)
        
        return insights
    
    def _generate_gemini_insights(self, 
                             dataset_name: str, 
                             hypotheses: List[Dict[str, Any]],
                             hypothesis_groups: List[InsightGroup]) -> List[Dict[str, Any]]:
        """
        Generate insights using the Gemini API for advanced synthesis.
        
        Args:
            dataset_name: Name of the dataset
            hypotheses: All hypotheses (for context)
            hypothesis_groups: Grouped hypotheses
            
        Returns:
            List of generated insights as dictionaries
        """
        insights = []
        
        # Get dataset info for context
        dataset_info = self.memory.retrieve(f"dataset_info:{dataset_name}")
        
        # Process each group to generate an insight
        for group in hypothesis_groups:
            if len(group.hypotheses) < self.min_hypotheses_per_insight:
                continue
            
            # Prepare the dataset context
            dataset_context = self._prepare_dataset_context(dataset_info)
            
            # Prepare the hypotheses for the prompt
            hypothesis_text = self._format_hypotheses_for_prompt(group.hypotheses)
            
            # Construct the full prompt
            prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                dataset_context=dataset_context,
                validated_hypotheses=hypothesis_text,
                json_format=INSIGHT_SYNTHESIS_JSON_FORMAT
            )
            
            # Call Gemini API
            response_text = self._call_gemini_api(prompt)
            
            # Parse response into insights
            try:
                generated_insights = self._parse_gemini_response(response_text)
                
                # Add source information to each insight
                for insight in generated_insights:
                    # Generate a new UUID if not provided
                    if "id" not in insight or not insight["id"]:
                        insight["id"] = f"insight_{uuid.uuid4()}"
                    
                    # Set source hypotheses if not provided
                    if "source_hypotheses" not in insight or not insight["source_hypotheses"]:
                        insight["source_hypotheses"] = group.hypothesis_ids
                    
                    # Add timestamp if not present
                    if "created_timestamp" not in insight:
                        insight["created_timestamp"] = time.time()
                
                insights.extend(generated_insights)
                
            except Exception as e:
                logger.error(f"Error parsing Gemini response: {str(e)}", exc_info=True)
                # Fall back to rule-based for this group
                rule_based = self._generate_rule_based_insights([group])
                insights.extend(rule_based)
        
        return insights
    
    def _prepare_dataset_context(self, dataset_info: Optional[Dict[str, Any]]) -> str:
        """
        Prepare a textual description of the dataset for context.
        
        Args:
            dataset_info: Dataset metadata (if available)
            
        Returns:
            Formatted dataset context string
        """
        if not dataset_info:
            return "No dataset information available."
        
        context = [
            f"Dataset: {dataset_info.get('name', 'Unknown')}",
            f"Source: {dataset_info.get('source', 'Unknown')}",
            f"Size: {dataset_info.get('rows', 0)} rows x {dataset_info.get('columns', 0)} columns"
        ]
        
        # Add column information if available
        if "column_types" in dataset_info and dataset_info["column_types"]:
            context.append("\nColumns:")
            for col, col_type in dataset_info["column_types"].items():
                # Add description if available
                description = ""
                if "column_descriptions" in dataset_info and col in dataset_info["column_descriptions"]:
                    description = f" - {dataset_info['column_descriptions'][col]}"
                
                context.append(f"- {col} ({col_type}){description}")
        
        return "\n".join(context)
    
    def _format_hypotheses_for_prompt(self, hypotheses: List[Dict[str, Any]]) -> str:
        """
        Format hypotheses into a readable text format for the prompt.
        
        Args:
            hypotheses: List of hypotheses to format
            
        Returns:
            Formatted hypothesis text
        """
        formatted = []
        
        for i, hyp in enumerate(hypotheses, 1):
            formatted.append(f"Hypothesis {i} [ID: {hyp.get('id', 'unknown')}]:")
            formatted.append(f"Title: {hyp.get('title', 'Untitled')}")
            formatted.append(f"Description: {hyp.get('description', 'No description')}")
            formatted.append(f"Relationship Type: {hyp.get('relationship_type', 'unknown')}")
            formatted.append(f"Columns Involved: {', '.join(hyp.get('columns_involved', []))}")
            
            # Add validation information
            status = hyp.get("validation_status", "unvalidated")
            formatted.append(f"Validation Status: {status}")
            
            # Add statistical results if validated
            if status == "confirmed" and "validation_result" in hyp and hyp["validation_result"]:
                formatted.append("Validation Results:")
                for key, value in hyp["validation_result"].items():
                    if isinstance(value, (str, int, float)):  # Only include simple values
                        formatted.append(f"- {key}: {value}")
            
            # Add evidence
            evidence = hyp.get("evidence", "")
            if evidence:
                formatted.append(f"Evidence: {evidence}")
            
            # Add confidence
            confidence = hyp.get("confidence_level", 0.0)
            formatted.append(f"Confidence Level: {confidence:.2f}")
            
            formatted.append("")  # Empty line between hypotheses
        
        return "\n".join(formatted)
    
    def _call_gemini_api(self, prompt: str) -> str:
        """
        Call the Gemini API with the given prompt.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            The response text from Gemini
            
        Raises:
            Exception: If the API call fails
        """
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not set. Set GEMINI_API_KEY environment variable.")
        
        url = f"{self.gemini_url}?key={self.gemini_api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Prepare the request payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
            },
            "systemInstruction": {"parts": [{"text": INSIGHT_SYNTHESIS_SYSTEM_PROMPT}]}
        }
        
        # Log that we're making the API call (but not the full prompt for privacy/security)
        logger.info(f"Calling Gemini API with prompt length: {len(prompt)}")
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            response_json = response.json()
            
            # Extract the generated text
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                candidate = response_json["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            
            # If we can't extract text using the expected path, return the raw response
            logger.warning(f"Unexpected Gemini API response structure: {response_json}")
            return str(response_json)
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}", exc_info=True)
            raise
    
    def _parse_gemini_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the Gemini API response into a list of insights.
        
        Args:
            response_text: The text response from Gemini
            
        Returns:
            List of insight dictionaries
            
        Raises:
            ValueError: If the response cannot be parsed
        """
        # Extract JSON from the text response (in case there's additional text)
        json_str = self._extract_json_from_text(response_text)
        
        try:
            # Parse the JSON
            response_data = json.loads(json_str)
            
            # Ensure it has the expected structure
            if "insights" not in response_data or not isinstance(response_data["insights"], list):
                raise ValueError("Gemini response missing 'insights' array")
            
            # Validate and ensure required fields
            insights = []
            for i, insight_data in enumerate(response_data["insights"]):
                # Ensure it has all required fields
                for field in ["title", "description", "importance"]:
                    if field not in insight_data:
                        raise ValueError(f"Insight {i} missing required field: {field}")
                
                # Ensure importance is a float between 0 and 1
                if not isinstance(insight_data["importance"], (int, float)):
                    insight_data["importance"] = 0.5  # Default if invalid
                else:
                    insight_data["importance"] = max(0.0, min(1.0, float(insight_data["importance"])))
                
                # If it's missing action_items, add an empty list
                if "action_items" not in insight_data or not isinstance(insight_data["action_items"], list):
                    insight_data["action_items"] = []
                
                # If it's missing tags, add an empty list
                if "tags" not in insight_data or not isinstance(insight_data["tags"], list):
                    insight_data["tags"] = []
                
                # If it's missing supporting_data, add an empty dict
                if "supporting_data" not in insight_data or not isinstance(insight_data["supporting_data"], dict):
                    insight_data["supporting_data"] = {}
                
                # Add the insight to our validated list
                insights.append(insight_data)
            
            return insights
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from Gemini response: {str(e)}")
            logger.debug(f"Response text: {response_text}")
            logger.debug(f"Extracted JSON: {json_str}")
            raise ValueError(f"Invalid JSON in Gemini response: {str(e)}")
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON from a text that might contain other content.
        
        Args:
            text: Text that should contain JSON
            
        Returns:
            Extracted JSON string
            
        Raises:
            ValueError: If no JSON is found
        """
        # Try to find the start and end of JSON
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            # Check for array format
            json_start = text.find('[')
            json_end = text.rfind(']') + 1
            
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in the response")
        
        # Extract the potential JSON string
        json_str = text[json_start:json_end]
        
        # Validate that it's parseable
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # If it fails, try to find a smaller valid JSON subset
            while json_start < json_end:
                try:
                    test_str = text[json_start:json_end]
                    json.loads(test_str)
                    return test_str
                except json.JSONDecodeError:
                    # Try shortening from the end
                    json_end = text.rfind('}', json_start, json_end - 1) + 1
                    if json_end <= json_start:
                        break
                    
            # If we're here, we couldn't find a valid JSON
            raise ValueError("Could not extract valid JSON from response")
    
    def get_insight_by_id(self, dataset_name: str, insight_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific insight by ID.
        
        Args:
            dataset_name: Name of the dataset
            insight_id: ID of the insight to retrieve
            
        Returns:
            The insight dictionary or an error message
        """
        memory_key = f"insights:{dataset_name}"
        all_insights = self.memory.retrieve(memory_key, [])
        
        for insight in all_insights:
            if insight["id"] == insight_id:
                return insight
        
        return {"error": f"Insight with ID {insight_id} not found"}
    
    def get_all_insights(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve all insights for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of all insights
        """
        memory_key = f"insights:{dataset_name}"
        return self.memory.retrieve(memory_key, [])
    
    def filter_insights(self, 
                    dataset_name: str,
                    min_importance: float = 0.0,
                    tags: Optional[List[str]] = None,
                    hypotheses_ids: Optional[List[str]] = None,
                    has_actions: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Filter insights based on various criteria.
        
        Args:
            dataset_name: Name of the dataset
            min_importance: Minimum importance score to include
            tags: Filter to insights with these tags (any match)
            hypotheses_ids: Filter to insights derived from these hypotheses
            has_actions: If True, only include insights with action items
            
        Returns:
            Filtered list of insights
        """
        all_insights = self.get_all_insights(dataset_name)
        filtered_insights = []
        
        for insight in all_insights:
            # Check importance threshold
            if insight["importance"] < min_importance:
                continue
            
            # Check for tag match (if specified)
            if tags and not any(tag in insight.get("tags", []) for tag in tags):
                continue
            
            # Check for hypothesis match (if specified)
            if hypotheses_ids:
                source_hyps = insight.get("source_hypotheses", [])
                if not any(h_id in source_hyps for h_id in hypotheses_ids):
                    continue
            
            # Check for action items (if specified)
            if has_actions is not None:
                has_action_items = bool(insight.get("action_items", []))
                if has_action_items != has_actions:
                    continue
            
            # All filters passed, include this insight
            filtered_insights.append(insight)
        
        return filtered_insights 