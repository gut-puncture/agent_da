"""
Gemini API Integration Module
---------------------------
This module provides utilities for interacting with the Gemini API
in a serverless-friendly way, with proper error handling and retries.
"""

import os
import logging
import json
import time
import requests
from typing import Dict, Any, Optional, List, Union
import backoff

from prompt_integrator import prompt_integrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GeminiAPI")

class GeminiAPI:
    """
    Client for interacting with Google's Gemini API.
    
    This class handles authentication, request formatting, error handling,
    and response parsing for Gemini API interactions.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini API client.
        
        Args:
            api_key: API key for Gemini (defaults to GEMINI_API_KEY environment variable)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable.")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1/models"
        self.default_model = "gemini-pro"
        
        # Default generation config
        self.default_generation_config = {
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 40,
            "maxOutputTokens": 4096
        }
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.HTTPError),
        max_tries=5,
        max_time=30
    )
    def generate_content(self, 
                        prompt: Union[str, Dict[str, Any]], 
                        model: Optional[str] = None,
                        generation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate content using the Gemini API.
        
        Args:
            prompt: Text prompt or formatted prompt dictionary
            model: Model name (defaults to gemini-pro)
            generation_config: Configuration for generation
            
        Returns:
            API response as a dictionary
            
        Raises:
            ValueError: For invalid inputs
            requests.exceptions.HTTPError: For API errors after retries
        """
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        model_name = model or self.default_model
        url = f"{self.base_url}/{model_name}:generateContent?key={self.api_key}"
        
        # Prepare request payload
        if isinstance(prompt, str):
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": generation_config or self.default_generation_config
            }
        elif isinstance(prompt, dict):
            payload = prompt
        else:
            raise ValueError("Prompt must be a string or dictionary")
        
        # Make the API request
        try:
            start_time = time.time()
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            response.raise_for_status()
            
            duration = time.time() - start_time
            logger.info(f"Gemini API request completed in {duration:.2f}s")
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Gemini API error: {str(e)}")
            logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
            raise
    
    def extract_text(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from a Gemini API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If response format is unexpected
        """
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                return ""
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                return ""
            
            return parts[0].get("text", "")
            
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting text from response: {str(e)}")
            logger.error(f"Response structure: {json.dumps(response)[:500]}...")
            return ""
    
    def extract_structured_data(self, 
                              response: Dict[str, Any], 
                              expected_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract and parse structured data (JSON) from a Gemini API response.
        
        Args:
            response: API response dictionary
            expected_format: Optional schema for validation
            
        Returns:
            Parsed structured data
            
        Raises:
            ValueError: If parsing fails or validation fails
        """
        text = self.extract_text(response)
        
        # Try to extract JSON from the text
        try:
            # Look for JSON blocks in markdown
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no code blocks, use the whole text
                json_str = text
            
            # Parse the JSON
            data = json.loads(json_str)
            
            # Validate against expected format if provided
            if expected_format:
                # Simple validation - check for required keys
                missing_keys = [k for k in expected_format if k not in data]
                if missing_keys:
                    logger.warning(f"Missing expected keys in response: {missing_keys}")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {str(e)}")
            logger.error(f"Text content: {text[:500]}...")
            return {}
    
    def generate_with_agent_prompt(self,
                                 agent_type: str,
                                 context: Dict[str, Any] = None,
                                 model: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate content using an agent's prompt.
        
        Args:
            agent_type: Type of agent (master_planner, data_explorer, etc.)
            context: Context to include in the prompt
            model: Model name (defaults to gemini-pro)
            
        Returns:
            API response
        """
        prompt_payload = prompt_integrator.create_gemini_prompt(agent_type, context)
        return self.generate_content(prompt_payload, model)

# Create a singleton instance for easy import
gemini_api = GeminiAPI()

def generate_with_prompt(agent_type: str, context: Dict[str, Any] = None) -> str:
    """
    Convenience function to generate text with an agent prompt.
    
    Args:
        agent_type: Type of agent
        context: Context to include
        
    Returns:
        Generated text
    """
    response = gemini_api.generate_with_agent_prompt(agent_type, context)
    return gemini_api.extract_text(response) 