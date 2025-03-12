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
    DatasetInfo
)

# Import the Gemini API client
from gemini_api import gemini_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HypothesisGenerator")

@dataclass
class Hypothesis:
    """
    Represents a data-driven hypothesis generated from exploratory analysis.
    
    This class encapsulates a single hypothesis about relationships or patterns in data,
    including evidence, confidence levels, and validation status.
    """
    id: str
    title: str
    description: str
    columns_involved: List[str]
    relationship_type: str  # "correlation", "trend", "seasonality", "outlier", "cluster", etc.
    evidence: str
    confidence_level: float  # 0.0 to 1.0
    generation_timestamp: float = field(default_factory=time.time)
    
    # Fields that will be populated during validation
    validation_status: Optional[str] = None  # "pending", "confirmed", "rejected", "indeterminate"
    validation_method: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None
    validation_timestamp: Optional[float] = None
    
    # Additional context
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hypothesis to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "columns_involved": self.columns_involved,
            "relationship_type": self.relationship_type,
            "evidence": self.evidence,
            "confidence_level": self.confidence_level,
            "generation_timestamp": self.generation_timestamp,
            "validation_status": self.validation_status,
            "validation_method": self.validation_method,
            "validation_result": self.validation_result,
            "validation_timestamp": self.validation_timestamp,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypothesis':
        """Create a hypothesis from dictionary data."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data["title"],
            description=data["description"],
            columns_involved=data["columns_involved"],
            relationship_type=data["relationship_type"],
            evidence=data["evidence"],
            confidence_level=data["confidence_level"],
            generation_timestamp=data.get("generation_timestamp", time.time()),
            validation_status=data.get("validation_status"),
            validation_method=data.get("validation_method"),
            validation_result=data.get("validation_result"),
            validation_timestamp=data.get("validation_timestamp"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )

    def validate(self, 
                status: str, 
                method: str, 
                result: Dict[str, Any]) -> None:
        """
        Update the hypothesis with validation results.
        
        Args:
            status: Validation status ("confirmed", "rejected", or "indeterminate")
            method: The statistical method used for validation
            result: Detailed results of the validation
        """
        self.validation_status = status
        self.validation_method = method
        self.validation_result = result
        self.validation_timestamp = time.time()

class HypothesisGenerator(BaseAgent):
    """
    Agent for generating hypotheses from data patterns and exploratory analysis.
    
    This agent analyzes data profiles and creates potential hypotheses about
    relationships, trends, or anomalies in the data that can be validated.
    It uses a combination of statistical techniques and natural language
    generation (via the Gemini API) to create well-formed hypotheses.
    """
    
    def __init__(self, agent_id: str, memory: AgentMemory):
        """
        Initialize the Hypothesis Generator agent.
        
        Args:
            agent_id: Unique identifier for this agent
            memory: Shared memory system for agent communication
        """
        super().__init__(agent_id, memory)
        self.logger = logging.getLogger(f"HypothesisGenerator_{agent_id}")
        
        # Note: We no longer need direct API configuration here as we're using the gemini_api module
    
    def _execute(self, action: str = "generate_hypotheses", **kwargs) -> Any:
        """
        Execute the specified action.
        
        Args:
            action: The action to execute
            kwargs: Additional arguments specific to the action
            
        Returns:
            The result of the action
        """
        if action == "generate_hypotheses":
            return self.generate_hypotheses(**kwargs)
        elif action == "get_hypothesis_by_id":
            return self.get_hypothesis_by_id(**kwargs)
        elif action == "get_all_hypotheses":
            return self.get_all_hypotheses(**kwargs)
        elif action == "filter_hypotheses":
            return self.filter_hypotheses(**kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def generate_hypotheses(self, 
                           dataset_name: str,
                           data_profile_key: Optional[str] = None,
                           max_hypotheses: int = 5,
                           min_confidence: float = 0.6,
                           relationship_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on data profiles and exploratory analysis.
        
        Args:
            dataset_name: Name of the dataset to analyze
            data_profile_key: Key for retrieving data profile from memory (optional)
            max_hypotheses: Maximum number of hypotheses to generate
            min_confidence: Minimum confidence level for hypotheses
            relationship_types: Specific relationship types to focus on
            
        Returns:
            List of generated hypothesis dictionaries
        """
        self.logger.info(f"Generating hypotheses for {dataset_name}")
        
        # Retrieve the data profile from memory
        profile_key = data_profile_key or f"data_profile:{dataset_name}"
        data_profile = self.memory.retrieve(profile_key)
        
        if not data_profile:
            error_msg = f"Data profile not found for {dataset_name}"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        # Get the relationship types to focus on
        if not relationship_types:
            relationship_types = [
                "correlation", "trend", "outlier", "distribution", 
                "cluster", "seasonality", "anomaly"
            ]
        
        # Generate hypotheses using different strategies
        hypotheses = []
        
        # Correlation-based hypotheses
        if "correlation" in relationship_types and data_profile.get("correlations"):
            correlation_hypotheses = self._generate_correlation_hypotheses(
                data_profile, min_confidence, max_hypotheses // 2)
            hypotheses.extend(correlation_hypotheses)
        
        # Outlier-based hypotheses
        if "outlier" in relationship_types:
            outlier_hypotheses = self._generate_outlier_hypotheses(
                data_profile, min_confidence)
            hypotheses.extend(outlier_hypotheses)
        
        # Distribution-based hypotheses
        if "distribution" in relationship_types:
            distribution_hypotheses = self._generate_distribution_hypotheses(
                data_profile, min_confidence)
            hypotheses.extend(distribution_hypotheses)
        
        # Generate additional hypotheses using Gemini if needed
        remaining_slots = max_hypotheses - len(hypotheses)
        if remaining_slots > 0:
            gemini_hypotheses = self._generate_gemini_hypotheses(
                data_profile, remaining_slots, min_confidence, relationship_types)
            hypotheses.extend(gemini_hypotheses)
        
        # Store the hypotheses in memory
        memory_key = f"hypotheses:{dataset_name}"
        self.memory.store(memory_key, hypotheses)
        
        # Send a message that hypotheses generation is complete
        self.send_message(
            recipient="master_planner",
            message_type="hypotheses_generated",
            content={
                "dataset_name": dataset_name,
                "count": len(hypotheses),
                "memory_key": memory_key
            }
        )
        
        return [h.to_dict() for h in hypotheses]
    
    def _generate_correlation_hypotheses(self, 
                                        data_profile: Dict[str, Any], 
                                        min_confidence: float,
                                        max_count: int) -> List[Hypothesis]:
        """
        Generate hypotheses based on correlations between variables.
        
        Args:
            data_profile: Data profile containing correlation information
            min_confidence: Minimum confidence level for hypotheses
            max_count: Maximum number of correlation hypotheses to generate
            
        Returns:
            List of correlation-based Hypothesis objects
        """
        hypotheses = []
        
        # Extract correlations from data profile
        correlations = data_profile.get("correlations", {})
        if not correlations:
            return hypotheses
        
        # Create a list of (column1, column2, correlation_value) tuples
        correlation_pairs = []
        for col1, values in correlations.items():
            for col2, value in values.items():
                if col1 != col2 and abs(value) >= 0.5:  # Adjust threshold as needed
                    correlation_pairs.append((col1, col2, value))
        
        # Sort by absolute correlation value (descending)
        correlation_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Generate hypotheses for top correlations
        for col1, col2, value in correlation_pairs[:max_count]:
            correlation_type = "positive" if value > 0 else "negative"
            confidence = min(abs(value), 1.0)  # Convert correlation to confidence level
            
            if confidence < min_confidence:
                continue
            
            hypothesis = Hypothesis(
                id=str(uuid.uuid4()),
                title=f"{correlation_type.capitalize()} correlation between {col1} and {col2}",
                description=f"There appears to be a {correlation_type} correlation (r={value:.2f}) between {col1} and {col2}.",
                columns_involved=[col1, col2],
                relationship_type="correlation",
                evidence=f"Pearson correlation coefficient: {value:.2f}",
                confidence_level=confidence,
                tags=["correlation", correlation_type]
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_outlier_hypotheses(self, 
                                    data_profile: Dict[str, Any], 
                                    min_confidence: float) -> List[Hypothesis]:
        """
        Generate hypotheses based on outliers in the data.
        
        Args:
            data_profile: Data profile containing outlier information
            min_confidence: Minimum confidence level for hypotheses
            
        Returns:
            List of outlier-based Hypothesis objects
        """
        hypotheses = []
        
        # Extract columns with outliers
        column_profiles = data_profile.get("column_profiles", {})
        for col_name, profile in column_profiles.items():
            outliers_count = profile.get("outliers_count", 0)
            if outliers_count <= 0:
                continue
            
            total_count = profile.get("count", 0)
            if total_count <= 0:
                continue
            
            # Calculate confidence based on proportion of outliers
            outlier_ratio = outliers_count / total_count
            confidence = min(0.5 + outlier_ratio * 2, 1.0)  # Convert to confidence level
            
            if confidence < min_confidence:
                continue
            
            hypothesis = Hypothesis(
                id=str(uuid.uuid4()),
                title=f"Outliers in {col_name}",
                description=f"The column {col_name} contains {outliers_count} outliers, which may indicate unusual patterns or data quality issues.",
                columns_involved=[col_name],
                relationship_type="outlier",
                evidence=f"Detected {outliers_count} outliers out of {total_count} values ({outlier_ratio:.1%}).",
                confidence_level=confidence,
                tags=["outlier", "data quality"]
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_distribution_hypotheses(self, 
                                        data_profile: Dict[str, Any], 
                                        min_confidence: float) -> List[Hypothesis]:
        """
        Generate hypotheses based on data distributions.
        
        Args:
            data_profile: Data profile containing distribution information
            min_confidence: Minimum confidence level for hypotheses
            
        Returns:
            List of distribution-based Hypothesis objects
        """
        hypotheses = []
        
        # Extract columns with distribution information
        column_profiles = data_profile.get("column_profiles", {})
        for col_name, profile in column_profiles.items():
            # Skip non-numeric columns
            if profile.get("dtype") not in ["int64", "float64", "int32", "float32"]:
                continue
            
            # Check for skewness based on mean vs median
            mean = profile.get("mean")
            median = profile.get("median")
            
            if mean is None or median is None:
                continue
            
            # Calculate skewness indicator
            skewness = (mean - median) / profile.get("std_dev", 1)
            
            if abs(skewness) > 0.5:  # Adjust threshold as needed
                skew_type = "right" if skewness > 0 else "left"
                confidence = min(0.6 + abs(skewness) * 0.2, 1.0)  # Convert to confidence
                
                if confidence < min_confidence:
                    continue
                
                hypothesis = Hypothesis(
                    id=str(uuid.uuid4()),
                    title=f"{col_name} has a {skew_type}-skewed distribution",
                    description=f"The distribution of {col_name} is {skew_type}-skewed, which may affect statistical analyses.",
                    columns_involved=[col_name],
                    relationship_type="distribution",
                    evidence=f"Mean ({mean:.2f}) is {'greater' if skewness > 0 else 'less'} than median ({median:.2f}), indicating {skew_type} skewness.",
                    confidence_level=confidence,
                    tags=["distribution", "skewness"]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_gemini_hypotheses(self, 
                                  data_profile: Dict[str, Any], 
                                  max_count: int,
                                  min_confidence: float,
                                  relationship_types: List[str]) -> List[Hypothesis]:
        """
        Generate hypotheses using the Gemini API.
        
        Args:
            data_profile: The data profile dictionary
            max_count: Maximum number of hypotheses to generate
            min_confidence: Minimum confidence level for hypotheses
            relationship_types: List of relationship types to focus on
            
        Returns:
            List of generated hypotheses
        """
        self.logger.info("Generating hypotheses using Gemini API")
        
        # Simplify the data profile for use in the prompt
        simplified_profile = self._simplify_profile_for_prompt(data_profile)
        
        # Create the prompt
        prompt = self._construct_gemini_prompt(
            simplified_profile, relationship_types, max_count
        )
        
        try:
            # Call the Gemini API using the gemini_api module
            response = gemini_api.generate_content(prompt)
            text = gemini_api.extract_text(response)
            
            if not text:
                self.logger.warning("Empty response from Gemini API")
                return []
            
            # Parse the response into hypotheses
            hypotheses = self._parse_gemini_response(text, min_confidence)
            self.logger.info(f"Generated {len(hypotheses)} hypotheses using Gemini API")
            
            return hypotheses
        
        except Exception as e:
            self.logger.error(f"Error generating hypotheses with Gemini: {str(e)}")
            return []
    
    def _simplify_profile_for_prompt(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a simplified version of the data profile for use in prompts.
        
        Args:
            data_profile: Complete data profile with all details
            
        Returns:
            Simplified profile with key information for prompt
        """
        simplified = {
            "dataset_name": data_profile.get("dataset_name", "Unknown Dataset"),
            "row_count": data_profile.get("row_count", 0),
            "column_count": data_profile.get("column_count", 0),
            "columns": {}
        }
        
        # Extract key column information
        for col_name, profile in data_profile.get("column_profiles", {}).items():
            col_info = {
                "dtype": profile.get("dtype", "unknown"),
                "missing_percentage": profile.get("missing_percentage", 0),
            }
            
            # Add type-specific statistics
            if col_info["dtype"] in ["int64", "float64", "int32", "float32"]:
                col_info.update({
                    "min": profile.get("min_value"),
                    "max": profile.get("max_value"),
                    "mean": profile.get("mean"),
                    "median": profile.get("median"),
                    "std_dev": profile.get("std_dev"),
                    "has_outliers": (profile.get("outliers_count", 0) > 0)
                })
            elif col_info["dtype"] in ["object", "category", "bool"]:
                col_info.update({
                    "unique_count": profile.get("unique_count", 0),
                    "top_values": profile.get("top_values", [])
                })
            
            simplified["columns"][col_name] = col_info
        
        # Add correlation information (simplified)
        correlations = data_profile.get("correlations", {})
        if correlations:
            simplified["strong_correlations"] = []
            for col1, values in correlations.items():
                for col2, value in values.items():
                    if col1 != col2 and abs(value) >= 0.7:  # Only include strong correlations
                        simplified["strong_correlations"].append({
                            "column1": col1,
                            "column2": col2,
                            "correlation": value
                        })
        
        return simplified
    
    def _construct_gemini_prompt(self, 
                               profile: Dict[str, Any], 
                               relationship_types: List[str],
                               max_count: int) -> str:
        """
        Construct a prompt for the Gemini API to generate hypotheses.
        
        Args:
            profile: Simplified data profile
            relationship_types: Types of relationships to focus on
            max_count: Maximum number of hypotheses to request
            
        Returns:
            Formatted prompt for the Gemini API
        """
        # Build a comprehensive prompt for a data analyst
        prompt = f"""
        # Data Analysis Task: Generate Insightful Hypotheses

        ## Context
        You are a world-class data analyst in the top 1 percentile of your field. You're examining a dataset with the following profile:

        Dataset: {profile["dataset_name"]}
        Rows: {profile["row_count"]}
        Columns: {profile["column_count"]}

        ## Column Information
        """
        
        # Add column details
        for col_name, info in profile["columns"].items():
            prompt += f"\n### {col_name} ({info['dtype']})\n"
            
            if info["dtype"] in ["int64", "float64", "int32", "float32"]:
                prompt += f"- Range: {info.get('min')} to {info.get('max')}\n"
                prompt += f"- Mean: {info.get('mean')}, Median: {info.get('median')}\n"
                prompt += f"- Standard Deviation: {info.get('std_dev')}\n"
                prompt += f"- Missing: {info['missing_percentage']:.1%}\n"
                prompt += f"- Contains outliers: {'Yes' if info.get('has_outliers') else 'No'}\n"
            else:
                prompt += f"- Unique values: {info.get('unique_count', 0)}\n"
                prompt += f"- Missing: {info['missing_percentage']:.1%}\n"
                if info.get('top_values'):
                    top_values = info.get('top_values', [])[:3]  # Limit to top 3
                    prompt += f"- Top values: {', '.join(str(v[0]) for v in top_values)}\n"
        
        # Add correlation information if available
        if "strong_correlations" in profile and profile["strong_correlations"]:
            prompt += "\n## Strong Correlations\n"
            for corr in profile["strong_correlations"]:
                prompt += f"- {corr['column1']} and {corr['column2']}: {corr['correlation']:.2f}\n"
        
        # Add specific instructions for generating hypotheses
        prompt += f"""
        ## Task
        Based on this data profile, generate {max_count} insightful hypotheses that a top data analyst would investigate.

        Focus on these relationship types: {', '.join(relationship_types)}

        For each hypothesis:
        1. Provide a clear, concise title
        2. Write a detailed description of the hypothesis
        3. List the columns involved
        4. Specify the relationship type
        5. Include specific evidence from the data profile that supports this hypothesis
        6. Assign a confidence level (0.0-1.0) based on how strongly the evidence supports the hypothesis

        ## Output Format
        Return your response as a JSON array of hypothesis objects with these fields:
        - title: A concise title for the hypothesis
        - description: A detailed explanation of the hypothesis
        - columns_involved: An array of column names involved in the hypothesis
        - relationship_type: The type of relationship being hypothesized
        - evidence: Supporting evidence from the data profile
        - confidence_level: A float between 0.0 and 1.0 representing confidence

        Only include hypotheses with reasonable evidence and explanation. Quality is more important than quantity.
        """
        
        return prompt
    
    def _call_gemini_api(self, prompt: str) -> str:
        """
        Call the Gemini API with the given prompt.
        
        Args:
            prompt: The text prompt for hypothesis generation
            
        Returns:
            The response text from the API
        """
        # Use the gemini_api module instead of direct API calls
        try:
            response = gemini_api.generate_content(prompt)
            text = gemini_api.extract_text(response)
            
            if not text:
                raise ValueError("Empty response from Gemini API")
                
            return text
            
        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {str(e)}")
            raise
    
    def _parse_gemini_response(self, 
                             response_text: str, 
                             min_confidence: float) -> List[Hypothesis]:
        """
        Parse the Gemini API response into hypothesis objects.
        
        Args:
            response_text: Text response from Gemini API
            min_confidence: Minimum confidence level for hypotheses
            
        Returns:
            List of parsed Hypothesis objects
        """
        hypotheses = []
        
        try:
            # Extract JSON from response (if embedded in markdown or other text)
            json_text = self._extract_json_from_text(response_text)
            
            # Parse JSON into a list of dictionaries
            hypothesis_dicts = json.loads(json_text)
            
            # Convert dictionaries to Hypothesis objects
            for h_dict in hypothesis_dicts:
                # Validate required fields
                if not all(k in h_dict for k in ["title", "description", "columns_involved", 
                                               "relationship_type", "evidence", "confidence_level"]):
                    continue
                
                # Check confidence level
                confidence = float(h_dict["confidence_level"])
                if confidence < min_confidence:
                    continue
                
                # Create a Hypothesis object with a new UUID
                h_dict["id"] = str(uuid.uuid4())
                hypothesis = Hypothesis.from_dict(h_dict)
                hypotheses.append(hypothesis)
            
        except Exception as e:
            self.logger.error(f"Error parsing Gemini response: {str(e)}\nResponse: {response_text[:500]}...")
        
        return hypotheses
    
    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON content from text that might include markdown or other formatting.
        
        Args:
            text: The raw text to extract JSON from
            
        Returns:
            Extracted JSON string
        """
        # Try to find JSON in code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        
        if json_match:
            return json_match.group(1).strip()
        
        # If no code blocks, look for arrays directly
        if text.strip().startswith('[') and text.strip().endswith(']'):
            return text.strip()
        
        # As a fallback, try to find any content that looks like a JSON array
        array_match = re.search(r'\[\s*{[\s\S]*}\s*\]', text)
        if array_match:
            return array_match.group(0)
        
        # If all else fails, return the original text and hope it's valid JSON
        return text.strip()
    
    def get_hypothesis_by_id(self, dataset_name: str, hypothesis_id: str) -> Dict[str, Any]:
        """
        Retrieve a single hypothesis by its ID.
        
        Args:
            dataset_name: Name of the dataset
            hypothesis_id: Unique identifier for the hypothesis
            
        Returns:
            The hypothesis dictionary or error message
        """
        memory_key = f"hypotheses:{dataset_name}"
        hypotheses = self.memory.retrieve(memory_key, [])
        
        for hypothesis in hypotheses:
            if hypothesis.id == hypothesis_id:
                return hypothesis.to_dict()
        
        return {"error": f"Hypothesis not found with ID: {hypothesis_id}"}
    
    def get_all_hypotheses(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve all hypotheses for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of all hypothesis dictionaries
        """
        memory_key = f"hypotheses:{dataset_name}"
        hypotheses = self.memory.retrieve(memory_key, [])
        
        return [h.to_dict() for h in hypotheses]
    
    def filter_hypotheses(self, 
                        dataset_name: str,
                        relationship_types: Optional[List[str]] = None,
                        columns: Optional[List[str]] = None,
                        min_confidence: float = 0.0,
                        validation_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Filter hypotheses based on specified criteria.
        
        Args:
            dataset_name: Name of the dataset
            relationship_types: Filter by these relationship types
            columns: Filter by hypotheses involving these columns
            min_confidence: Minimum confidence level
            validation_status: Filter by validation status
            
        Returns:
            List of filtered hypothesis dictionaries
        """
        memory_key = f"hypotheses:{dataset_name}"
        hypotheses = self.memory.retrieve(memory_key, [])
        
        filtered = []
        for hypothesis in hypotheses:
            # Filter by relationship type
            if relationship_types and hypothesis.relationship_type not in relationship_types:
                continue
                
            # Filter by columns
            if columns and not any(col in hypothesis.columns_involved for col in columns):
                continue
                
            # Filter by minimum confidence
            if hypothesis.confidence_level < min_confidence:
                continue
                
            # Filter by validation status
            if validation_status and hypothesis.validation_status != validation_status:
                continue
                
            filtered.append(hypothesis)
        
        return [h.to_dict() for h in filtered] 