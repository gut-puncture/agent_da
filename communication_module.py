import logging
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import requests
import os
import re
from enum import Enum

from agent_framework_core import (
    BaseAgent, 
    AgentMemory, 
    Message, 
    AgentState,
    Insight
)

from communication_module_prompt import (
    COMMUNICATION_MODULE_PROMPT,
    COMMUNICATION_SYSTEM_PROMPT,
    COMMUNICATION_TEMPLATE,
    AUDIENCE_TEMPLATE,
    VISUALIZATION_RECOMMENDATIONS,
    INSIGHT_TO_COMMUNICATION_TEMPLATE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CommunicationModule")

class AudienceType(Enum):
    """Represents different types of audiences for communication."""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    BUSINESS = "business"
    GENERAL = "general"

@dataclass
class AudienceProfile:
    """
    Represents the characteristics and needs of a target audience.
    Used to tailor communication style and content.
    """
    audience_type: AudienceType
    technical_expertise: int  # Scale of 1-5
    domain_knowledge: int  # Scale of 1-5
    primary_questions: List[str]
    key_decisions: List[str]
    time_constraints: str  # e.g., "brief", "detailed", "comprehensive"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary format."""
        return {
            "audience_type": self.audience_type.value,
            "technical_expertise": self.technical_expertise,
            "domain_knowledge": self.domain_knowledge,
            "primary_questions": self.primary_questions,
            "key_decisions": self.key_decisions,
            "time_constraints": self.time_constraints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudienceProfile':
        """Create profile from dictionary."""
        return cls(
            audience_type=AudienceType(data["audience_type"]),
            technical_expertise=data["technical_expertise"],
            domain_knowledge=data["domain_knowledge"],
            primary_questions=data["primary_questions"],
            key_decisions=data["key_decisions"],
            time_constraints=data["time_constraints"]
        )

@dataclass
class CommunicationOutput:
    """
    Represents a structured communication output tailored for a specific audience.
    """
    id: str
    dataset_name: str
    audience_profile: AudienceProfile
    executive_summary: str
    key_findings: List[Dict[str, Any]]
    detailed_insights: List[Dict[str, Any]]
    recommended_actions: List[Dict[str, Any]]
    methodology: str
    technical_details: Optional[Dict[str, Any]] = None
    visualization_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    created_timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary format."""
        return {
            "id": self.id,
            "dataset_name": self.dataset_name,
            "audience_profile": self.audience_profile.to_dict(),
            "executive_summary": self.executive_summary,
            "key_findings": self.key_findings,
            "detailed_insights": self.detailed_insights,
            "recommended_actions": self.recommended_actions,
            "methodology": self.methodology,
            "technical_details": self.technical_details,
            "visualization_suggestions": self.visualization_suggestions,
            "created_timestamp": self.created_timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunicationOutput':
        """Create output from dictionary."""
        return cls(
            id=data["id"],
            dataset_name=data["dataset_name"],
            audience_profile=AudienceProfile.from_dict(data["audience_profile"]),
            executive_summary=data["executive_summary"],
            key_findings=data["key_findings"],
            detailed_insights=data["detailed_insights"],
            recommended_actions=data["recommended_actions"],
            methodology=data["methodology"],
            technical_details=data.get("technical_details"),
            visualization_suggestions=data.get("visualization_suggestions", []),
            created_timestamp=data.get("created_timestamp", time.time())
        )

class CommunicationModule(BaseAgent):
    """
    Agent responsible for transforming insights into audience-appropriate communications.
    
    This agent:
    1. Analyzes audience characteristics
    2. Prioritizes and structures insights
    3. Generates clear narratives
    4. Suggests appropriate visualizations
    5. Ensures actionability of communications
    
    The module uses both rule-based methods and Gemini API for advanced communication.
    """
    
    def __init__(self, agent_id: str, memory: AgentMemory):
        """Initialize the CommunicationModule agent."""
        super().__init__(agent_id, memory)
        
        # LLM API configuration
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_url = os.environ.get(
            "GEMINI_API_URL", 
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        )
        
        # Load visualization recommendations
        self.visualization_recommendations = json.loads(VISUALIZATION_RECOMMENDATIONS)
        
        logger.info(f"CommunicationModule initialized with ID: {agent_id}")
    
    def _execute(self, action: str = "generate_communication", **kwargs) -> Any:
        """Execute the agent's main functionality."""
        logger.info(f"Executing action: {action} with args: {kwargs}")
        
        self._state = AgentState.RUNNING
        
        try:
            if action == "generate_communication":
                result = self.generate_communication(**kwargs)
            elif action == "get_communication_by_id":
                result = self.get_communication_by_id(**kwargs)
            elif action == "get_all_communications":
                result = self.get_all_communications(**kwargs)
            else:
                logger.warning(f"Unknown action: {action}")
                result = {"error": f"Unknown action: {action}"}
                self._state = AgentState.FAILED
                return result
            
            self._state = AgentState.COMPLETED
            return result
            
        except Exception as e:
            logger.error(f"Error executing action {action}: {str(e)}", exc_info=True)
            self._state = AgentState.FAILED
            return {"error": str(e)}
    
    def generate_communication(self,
                             dataset_name: str,
                             audience_profile: Union[Dict[str, Any], AudienceProfile],
                             insight_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a communication tailored to the specified audience.
        
        Args:
            dataset_name: Name of the dataset
            audience_profile: Profile of the target audience
            insight_filter: Optional filters for selecting insights
            
        Returns:
            Generated communication as a dictionary
        """
        # Convert audience profile if needed
        if isinstance(audience_profile, dict):
            audience_profile = AudienceProfile.from_dict(audience_profile)
        
        # Get insights from memory
        insights = self._get_filtered_insights(dataset_name, insight_filter)
        
        if not insights:
            logger.warning(f"No insights found for dataset: {dataset_name}")
            return {"error": "No insights available for communication"}
        
        # First try with LLM-based communication
        try:
            communication = self._generate_gemini_communication(
                dataset_name,
                audience_profile,
                insights
            )
            logger.info("Generated communication using Gemini")
        except Exception as e:
            logger.error(f"Error generating communication with Gemini: {str(e)}", exc_info=True)
            # Fall back to rule-based communication
            communication = self._generate_rule_based_communication(
                dataset_name,
                audience_profile,
                insights
            )
            logger.info("Generated communication using rule-based approach")
        
        # Store the communication
        memory_key = f"communications:{dataset_name}"
        all_communications = self.memory.retrieve(memory_key, [])
        all_communications.append(communication.to_dict())
        self.memory.store(memory_key, all_communications)
        
        # Notify that communication has been generated
        self.send_message(
            recipient="master_planner",
            message_type="communication_generated",
            content={
                "dataset_name": dataset_name,
                "communication_id": communication.id,
                "audience_type": audience_profile.audience_type.value
            }
        )
        
        return communication.to_dict()
    
    def _get_filtered_insights(self,
                             dataset_name: str,
                             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get insights from memory with optional filtering."""
        memory_key = f"insights:{dataset_name}"
        all_insights = self.memory.retrieve(memory_key, [])
        
        if not filters:
            return all_insights
        
        filtered_insights = []
        for insight in all_insights:
            match = True
            
            for key, value in filters.items():
                if key not in insight:
                    match = False
                    break
                
                if isinstance(value, list):
                    if insight[key] not in value:
                        match = False
                        break
                else:
                    if insight[key] != value:
                        match = False
                        break
            
            if match:
                filtered_insights.append(insight)
        
        return filtered_insights
    
    def _generate_rule_based_communication(self,
                                         dataset_name: str,
                                         audience: AudienceProfile,
                                         insights: List[Dict[str, Any]]) -> CommunicationOutput:
        """Generate communication using rule-based methods."""
        # Sort insights by importance
        insights.sort(key=lambda x: x.get("importance", 0), reverse=True)
        
        # Generate executive summary based on audience type
        if audience.audience_type == AudienceType.EXECUTIVE:
            summary = self._generate_executive_summary(insights[:3])
        else:
            summary = self._generate_detailed_summary(insights[:5])
        
        # Prepare key findings
        key_findings = []
        for insight in insights[:5]:  # Top 5 insights
            finding = {
                "title": insight["title"],
                "summary": self._adapt_text_to_audience(
                    insight["description"],
                    audience
                ),
                "importance": insight["importance"],
                "source_insight_id": insight["id"]
            }
            key_findings.append(finding)
        
        # Prepare detailed insights
        detailed_insights = []
        for insight in insights:
            detail = {
                "title": insight["title"],
                "description": self._adapt_text_to_audience(
                    insight["description"],
                    audience
                ),
                "supporting_data": insight.get("supporting_data", {}),
                "visualization": self._suggest_visualization(insight),
                "source_insight_id": insight["id"]
            }
            detailed_insights.append(detail)
        
        # Prepare actions
        actions = []
        for insight in insights:
            for action in insight.get("action_items", []):
                actions.append({
                    "description": self._adapt_text_to_audience(action, audience),
                    "priority": insight["importance"],
                    "source_insight_id": insight["id"]
                })
        
        # Sort actions by priority
        actions.sort(key=lambda x: x["priority"], reverse=True)
        
        # Generate methodology section
        methodology = self._generate_methodology_section(audience)
        
        # Generate technical details if appropriate
        technical_details = None
        if audience.technical_expertise >= 4:
            technical_details = self._generate_technical_details(insights)
        
        # Create communication output
        return CommunicationOutput(
            id=f"comm_{uuid.uuid4()}",
            dataset_name=dataset_name,
            audience_profile=audience,
            executive_summary=summary,
            key_findings=key_findings,
            detailed_insights=detailed_insights,
            recommended_actions=actions,
            methodology=methodology,
            technical_details=technical_details,
            visualization_suggestions=self._generate_visualization_suggestions(insights)
        )
    
    def _generate_gemini_communication(self,
                                     dataset_name: str,
                                     audience: AudienceProfile,
                                     insights: List[Dict[str, Any]]) -> CommunicationOutput:
        """Generate communication using the Gemini API."""
        # Prepare the prompt
        audience_info = AUDIENCE_TEMPLATE.format(
            audience_type=audience.audience_type.value,
            technical_expertise=f"Level {audience.technical_expertise}/5",
            domain_knowledge=f"Level {audience.domain_knowledge}/5",
            primary_questions="\n".join(f"- {q}" for q in audience.primary_questions),
            key_decisions="\n".join(f"- {d}" for d in audience.key_decisions),
            time_constraints=audience.time_constraints
        )
        
        insights_data = json.dumps(insights, indent=2)
        
        prompt = INSIGHT_TO_COMMUNICATION_TEMPLATE.format(
            audience_information=audience_info,
            insights_data=insights_data,
            communication_template=COMMUNICATION_TEMPLATE
        )
        
        # Call Gemini API
        response_text = self._call_gemini_api(prompt)
        
        # Parse the response
        try:
            sections = self._parse_communication_sections(response_text)
            
            # Create visualization suggestions
            visualization_suggestions = (
                self._generate_visualization_suggestions(insights)
                if audience.technical_expertise >= 3
                else []
            )
            
            # Create communication output
            return CommunicationOutput(
                id=f"comm_{uuid.uuid4()}",
                dataset_name=dataset_name,
                audience_profile=audience,
                executive_summary=sections.get("executive_summary", ""),
                key_findings=self._parse_key_findings(sections.get("key_findings", "")),
                detailed_insights=self._parse_detailed_insights(sections.get("detailed_insights", "")),
                recommended_actions=self._parse_actions(sections.get("recommended_actions", "")),
                methodology=sections.get("methodology", ""),
                technical_details=self._parse_technical_details(sections.get("technical_details", "")),
                visualization_suggestions=visualization_suggestions
            )
            
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}", exc_info=True)
            # Fall back to rule-based
            return self._generate_rule_based_communication(
                dataset_name,
                audience,
                insights
            )
    
    def _adapt_text_to_audience(self, text: str, audience: AudienceProfile) -> str:
        """Adapt text based on audience characteristics."""
        if audience.technical_expertise <= 2:
            # Simplify technical terms
            text = re.sub(r'\b(correlation|coefficient|statistical|significance)\b',
                         lambda m: {
                             'correlation': 'relationship',
                             'coefficient': 'value',
                             'statistical': 'measured',
                             'significance': 'importance'
                         }.get(m.group(1), m.group(1)),
                         text)
        
        if audience.audience_type == AudienceType.EXECUTIVE:
            # Focus on business impact
            if not any(term in text.lower() for term in ['impact', 'roi', 'revenue', 'cost']):
                text += " This finding has potential business impact."
        
        return text
    
    def _suggest_visualization(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest appropriate visualization type for an insight."""
        # Determine the type of data relationship
        relationship_type = None
        
        # Check for time-based patterns
        if any(term in insight.get("tags", []) for term in ["trend", "time_series", "seasonal"]):
            relationship_type = "trend_over_time"
        
        # Check for comparisons
        elif any(term in insight.get("tags", []) for term in ["comparison", "ranking"]):
            relationship_type = "comparison"
        
        # Check for distributions
        elif any(term in insight.get("tags", []) for term in ["distribution", "spread"]):
            relationship_type = "distribution"
        
        # Check for relationships
        elif any(term in insight.get("tags", []) for term in ["correlation", "relationship"]):
            relationship_type = "relationship"
        
        # Check for composition
        elif any(term in insight.get("tags", []) for term in ["composition", "breakdown"]):
            relationship_type = "composition"
        
        # Check for geographic data
        elif any(term in insight.get("tags", []) for term in ["geographic", "spatial"]):
            relationship_type = "geographical"
        
        # Get visualization recommendation
        if relationship_type and relationship_type in self.visualization_recommendations:
            recommendation = self.visualization_recommendations[relationship_type]
            return {
                "type": recommendation["chart_types"][0],  # Primary recommendation
                "alternatives": recommendation["chart_types"][1:],  # Alternative options
                "reason": recommendation["best_for"]
            }
        
        # Default to bar chart if no clear match
        return {
            "type": "Bar Chart",
            "alternatives": ["Column Chart", "Dot Plot"],
            "reason": "General purpose visualization for comparing values"
        }
    
    def _generate_executive_summary(self, top_insights: List[Dict[str, Any]]) -> str:
        """Generate a concise executive summary."""
        summary_parts = []
        
        # Add high-level overview
        summary_parts.append("Key Business Insights:")
        
        # Add top findings
        for insight in top_insights:
            importance = insight.get("importance", 0)
            if importance >= 0.8:
                prefix = "Critical Finding"
            elif importance >= 0.6:
                prefix = "Important Finding"
            else:
                prefix = "Notable Finding"
            
            summary_parts.append(f"\n{prefix}: {insight['title']}")
        
        # Add action-oriented conclusion
        if any(i.get("action_items") for i in top_insights):
            summary_parts.append("\nImmediate Actions Required:")
            for insight in top_insights:
                for action in insight.get("action_items", [])[:1]:  # Just the first action
                    summary_parts.append(f"- {action}")
        
        return "\n".join(summary_parts)
    
    def _generate_detailed_summary(self, insights: List[Dict[str, Any]]) -> str:
        """Generate a more detailed summary."""
        summary_parts = ["Analysis Overview"]
        
        # Group insights by theme
        themes = {}
        for insight in insights:
            for tag in insight.get("tags", []):
                if tag not in themes:
                    themes[tag] = []
                themes[tag].append(insight)
        
        # Add thematic summaries
        for theme, theme_insights in themes.items():
            if len(theme_insights) > 1:
                summary_parts.append(f"\n{theme.title()} Analysis:")
                for insight in theme_insights:
                    summary_parts.append(f"- {insight['title']}")
        
        # Add methodology note
        summary_parts.append("\nMethodology Note:")
        summary_parts.append("This analysis combines statistical validation with domain expertise to ensure reliable insights.")
        
        return "\n".join(summary_parts)
    
    def _generate_methodology_section(self, audience: AudienceProfile) -> str:
        """Generate methodology explanation appropriate for the audience."""
        if audience.technical_expertise >= 4:
            return """
            Methodology:
            1. Data Exploration: Comprehensive statistical profiling and quality assessment
            2. Hypothesis Generation: Advanced pattern detection and relationship analysis
            3. Statistical Validation: Rigorous testing using appropriate statistical methods
            4. Insight Synthesis: Integration of validated findings into actionable insights
            5. Impact Assessment: Evaluation of business implications and prioritization
            """
        else:
            return """
            Our Analysis Process:
            1. We thoroughly examined the data to understand its characteristics
            2. We identified potential patterns and relationships
            3. We tested these patterns to confirm their validity
            4. We combined related findings into meaningful insights
            5. We assessed which findings matter most for the business
            """
    
    def _generate_technical_details(self, insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate technical details for technical audiences."""
        details = {
            "statistical_methods": [],
            "data_quality": {},
            "validation_metrics": {}
        }
        
        for insight in insights:
            # Extract statistical methods used
            if "validation_method" in insight:
                details["statistical_methods"].append({
                    "method": insight["validation_method"],
                    "insight_id": insight["id"]
                })
            
            # Extract data quality information
            if "data_quality" in insight.get("supporting_data", {}):
                details["data_quality"].update(insight["supporting_data"]["data_quality"])
            
            # Extract validation metrics
            if "validation_result" in insight:
                details["validation_metrics"][insight["id"]] = insight["validation_result"]
        
        return details
    
    def _generate_visualization_suggestions(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate visualization suggestions for the insights."""
        suggestions = []
        
        for insight in insights:
            visualization = self._suggest_visualization(insight)
            
            suggestion = {
                "insight_id": insight["id"],
                "title": insight["title"],
                "primary_visualization": visualization["type"],
                "alternative_visualizations": visualization["alternatives"],
                "rationale": visualization["reason"],
                "data_requirements": {
                    "required_columns": insight.get("columns_involved", []),
                    "data_type": "time_series" if "trend" in insight.get("tags", []) else "static"
                }
            }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call the Gemini API with the given prompt."""
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not set")
        
        url = f"{self.gemini_url}?key={self.gemini_api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
            },
            "systemInstruction": {"parts": [{"text": COMMUNICATION_SYSTEM_PROMPT}]}
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            response_json = response.json()
            
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                candidate = response_json["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            
            logger.warning(f"Unexpected Gemini API response structure: {response_json}")
            return str(response_json)
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}", exc_info=True)
            raise
    
    def _parse_communication_sections(self, text: str) -> Dict[str, str]:
        """Parse the communication sections from the Gemini response."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in text.split("\n"):
            if line.startswith("## "):
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                
                # Start new section
                current_section = line[3:].lower().replace(" ", "_")
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()
        
        return sections
    
    def _parse_key_findings(self, text: str) -> List[Dict[str, Any]]:
        """Parse key findings from text."""
        findings = []
        current_finding = None
        
        for line in text.split("\n"):
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                if current_finding:
                    findings.append(current_finding)
                current_finding = {
                    "title": line.strip()[2:],
                    "details": []
                }
            elif current_finding and line.strip():
                current_finding["details"].append(line.strip())
        
        if current_finding:
            findings.append(current_finding)
        
        return findings
    
    def _parse_detailed_insights(self, text: str) -> List[Dict[str, Any]]:
        """Parse detailed insights from text."""
        insights = []
        current_insight = None
        
        for line in text.split("\n"):
            if line.startswith("### "):
                if current_insight:
                    insights.append(current_insight)
                current_insight = {
                    "title": line[4:].strip(),
                    "content": []
                }
            elif current_insight and line.strip():
                current_insight["content"].append(line.strip())
        
        if current_insight:
            insights.append(current_insight)
        
        return insights
    
    def _parse_actions(self, text: str) -> List[Dict[str, Any]]:
        """Parse recommended actions from text."""
        actions = []
        
        for line in text.split("\n"):
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                action_text = line.strip()[2:]
                
                # Try to extract priority/timeline if specified in parentheses
                priority_match = re.search(r"\((.*?)\)", action_text)
                if priority_match:
                    priority_text = priority_match.group(1)
                    action_text = action_text.replace(f"({priority_text})", "").strip()
                else:
                    priority_text = "medium"
                
                actions.append({
                    "description": action_text,
                    "priority": priority_text
                })
        
        return actions
    
    def _parse_technical_details(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse technical details from text."""
        if not text.strip():
            return None
        
        details = {
            "methods": [],
            "metrics": {},
            "notes": []
        }
        
        current_section = None
        
        for line in text.split("\n"):
            if line.startswith("### "):
                current_section = line[4:].lower().replace(" ", "_")
            elif line.strip().startswith("- ") or line.strip().startswith("* "):
                content = line.strip()[2:]
                if current_section == "methods":
                    details["methods"].append(content)
                elif current_section == "metrics":
                    try:
                        key, value = content.split(":", 1)
                        details["metrics"][key.strip()] = value.strip()
                    except ValueError:
                        details["metrics"]["note"] = content
                else:
                    details["notes"].append(content)
        
        return details
    
    def get_communication_by_id(self, dataset_name: str, communication_id: str) -> Dict[str, Any]:
        """Retrieve a specific communication by ID."""
        memory_key = f"communications:{dataset_name}"
        all_communications = self.memory.retrieve(memory_key, [])
        
        for comm in all_communications:
            if comm["id"] == communication_id:
                return comm
        
        return {"error": f"Communication with ID {communication_id} not found"}
    
    def get_all_communications(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Retrieve all communications for a dataset."""
        memory_key = f"communications:{dataset_name}"
        return self.memory.retrieve(memory_key, []) 