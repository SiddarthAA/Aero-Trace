"""
Gemini client for structured JSON output
"""
import os
import json
import time
from google import genai
from typing import Optional, Dict, Any
from rich.console import Console
from . import config

console = Console()


class GeminiClient:
    """Client for Google Gemini API for structured output generation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        """
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        self.model = config.GEMINI_MODEL
        self.last_request_time = 0
        self.min_request_interval = config.GEMINI_MIN_REQUEST_INTERVAL
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            console.print(f"[dim]  ⏱ Rate limiting: waiting {sleep_time:.1f}s...[/dim]")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def generate_structured_output(
        self, 
        prompt: str, 
        schema_description: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from Gemini with retry logic
        
        Args:
            prompt: The generation prompt
            schema_description: Description of expected JSON schema
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed JSON dictionary
        """
        for attempt in range(max_retries):
            try:
                # Rate limit before request
                self._rate_limit()
                
                full_prompt = f"""{prompt}

{schema_description}

Output ONLY valid JSON, no markdown formatting, no explanation."""
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt
                )
                
                # Extract and parse JSON
                response_text = response.text.strip()
                
                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                response_text = response_text.strip()
                
                return json.loads(response_text)
            
            except json.JSONDecodeError as e:
                console.print(f"[yellow]⚠ JSON parse error (attempt {attempt+1}/{max_retries}): {e}[/yellow]")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    console.print(f"[red]✗ Failed to parse JSON after {max_retries} attempts[/red]")
                    return {}
            
            except Exception as e:
                error_msg = str(e)
                
                # Check for rate limit errors
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    # Extract retry delay if available
                    retry_delay = 30  # Default 30 seconds
                    if "retry" in error_msg.lower():
                        try:
                            import re
                            match = re.search(r'retry.*?(\d+(?:\.\d+)?)\s*s', error_msg, re.IGNORECASE)
                            if match:
                                retry_delay = float(match.group(1))
                        except:
                            pass
                    
                    console.print(f"[yellow]⚠ Rate limit hit (attempt {attempt+1}/{max_retries})[/yellow]")
                    if attempt < max_retries - 1:
                        console.print(f"[dim]  Waiting {retry_delay:.1f}s before retry...[/dim]")
                        time.sleep(retry_delay)
                        continue
                    else:
                        console.print(f"[red]✗ Rate limit exceeded after {max_retries} attempts[/red]")
                        return {}
                else:
                    console.print(f"[red]✗ Gemini API error: {e}[/red]")
                    if attempt < max_retries - 1:
                        console.print(f"[dim]  Retrying in 5s...[/dim]")
                        time.sleep(5)
                        continue
                    return {}
        
        return {}
