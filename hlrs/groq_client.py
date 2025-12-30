"""
Groq client for reasoning with multiple models
"""
import os
import time
import json
from groq import Groq
from typing import Optional, Dict, Any
from rich.console import Console
from . import config

console = Console()


class GroqClient:
    """Client for Groq API using multiple models for different tasks"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
        """
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.reasoning_model = config.GROQ_REASONING_MODEL
        self.structured_model = config.GROQ_STRUCTURED_MODEL
        self.fallback_model = config.GROQ_FALLBACK_MODEL
        self.last_request_time = 0
        self.min_request_interval = config.GROQ_MIN_REQUEST_INTERVAL
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def reason(self, prompt: str, temperature: float = 0.2, max_retries: int = 3) -> str:
        """
        Send reasoning prompt to Groq with retry logic
        Uses the reasoning model (gpt-oss-120b)
        
        Args:
            prompt: The reasoning prompt
            temperature: Temperature for generation (lower = more deterministic)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Model response
        """
        return self._call_model(
            model=self.reasoning_model,
            prompt=prompt,
            temperature=temperature,
            max_retries=max_retries,
            system_message="You are an expert requirements engineer analyzing traceability between high-level and low-level requirements. Provide clear, logical reasoning."
        )
    
    def generate_structured_output(
        self, 
        prompt: str, 
        schema_description: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using Groq's fast model
        Uses llama-3.1-8b-instant for quick JSON formatting
        
        Args:
            prompt: The generation prompt
            schema_description: Description of expected JSON schema
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed JSON dictionary
        """
        full_prompt = f"""{prompt}

{schema_description}

Output ONLY valid JSON, no markdown formatting, no explanation. Respond with pure JSON only."""
        
        for attempt in range(max_retries):
            try:
                # Rate limit before request
                self._rate_limit()
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a JSON formatting assistant. Output only valid JSON without any markdown or explanation."
                        },
                        {
                            "role": "user",
                            "content": full_prompt,
                        }
                    ],
                    model=self.structured_model,
                    temperature=0.1,  # Very low for structured output
                    max_tokens=2000,
                )
                
                response_text = chat_completion.choices[0].message.content.strip()
                
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
                console.print(f"[yellow]⚠ JSON parse error (attempt {attempt+1}/{max_retries})[/yellow]")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    console.print(f"[red]✗ Failed to parse JSON after {max_retries} attempts[/red]")
                    return {}
            
            except Exception as e:
                error_msg = str(e)
                
                if "429" in error_msg or "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    retry_delay = 10
                    console.print(f"[yellow]⚠ Groq rate limit (attempt {attempt+1}/{max_retries})[/yellow]")
                    if attempt < max_retries - 1:
                        console.print(f"[dim]  Waiting {retry_delay}s before retry...[/dim]")
                        time.sleep(retry_delay)
                        continue
                    else:
                        console.print(f"[red]✗ Rate limit exceeded[/red]")
                        return {}
                else:
                    console.print(f"[red]✗ Groq API error: {e}[/red]")
                    if attempt < max_retries - 1:
                        console.print(f"[dim]  Retrying in 3s...[/dim]")
                        time.sleep(3)
                        continue
                    return {}
        
        return {}
    
    def _call_model(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_retries: int,
        system_message: str
    ) -> str:
        """
        Internal method to call Groq API with any model
        
        Args:
            model: Model name to use
            prompt: User prompt
            temperature: Temperature setting
            max_retries: Max retry attempts
            system_message: System message
            
        Returns:
            Model response text
        """
        for attempt in range(max_retries):
            try:
                # Rate limit before request
                self._rate_limit()
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model,
                    temperature=temperature,
                    max_tokens=1000,
                )
                return chat_completion.choices[0].message.content
            
            except Exception as e:
                error_msg = str(e)
                
                # Check for rate limit errors
                if "429" in error_msg or "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    retry_delay = 10
                    console.print(f"[yellow]⚠ Groq rate limit on {model} (attempt {attempt+1}/{max_retries})[/yellow]")
                    if attempt < max_retries - 1:
                        console.print(f"[dim]  Waiting {retry_delay}s before retry...[/dim]")
                        time.sleep(retry_delay)
                        continue
                    else:
                        console.print(f"[red]✗ Groq rate limit exceeded[/red]")
                        return ""
                else:
                    console.print(f"[red]✗ Groq API error on {model}: {e}[/red]")
                    if attempt < max_retries - 1:
                        console.print(f"[dim]  Retrying in 3s...[/dim]")
                        time.sleep(3)
                        continue
                    return ""
        
        return ""
        for attempt in range(max_retries):
            try:
                # Rate limit before request
                self._rate_limit()
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert requirements engineer analyzing traceability between high-level and low-level requirements. Provide clear, logical reasoning."
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=self.model,
                    temperature=temperature,
                    max_tokens=1000,
                )
                return chat_completion.choices[0].message.content
            
            except Exception as e:
                error_msg = str(e)
                
                # Check for rate limit errors
                if "429" in error_msg or "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    retry_delay = 10  # Default 10 seconds for rate limit
                    console.print(f"[yellow]⚠ Groq rate limit (attempt {attempt+1}/{max_retries})[/yellow]")
                    if attempt < max_retries - 1:
                        console.print(f"[dim]  Waiting {retry_delay}s before retry...[/dim]")
                        time.sleep(retry_delay)
                        continue
                    else:
                        console.print(f"[red]✗ Groq rate limit exceeded[/red]")
                        return ""
                else:
                    console.print(f"[red]✗ Groq API error: {e}[/red]")
                    if attempt < max_retries - 1:
                        console.print(f"[dim]  Retrying in 3s...[/dim]")
                        time.sleep(3)
                        continue
                    return ""
        
        return ""
    
    def batch_reason(self, prompts: list[str], temperature: float = 0.2) -> list[str]:
        """
        Process multiple reasoning prompts with rate limiting
        
        Args:
            prompts: List of reasoning prompts
            temperature: Temperature for generation
            
        Returns:
            List of model responses
        """
        responses = []
        total = len(prompts)
        
        for i, prompt in enumerate(prompts, 1):
            if i > 1:  # Add extra delay between batch items
                time.sleep(0.2)
            
            response = self.reason(prompt, temperature)
            responses.append(response)
            
            # Progress indicator for large batches
            if total > 5 and i % 5 == 0:
                console.print(f"[dim]  Processed {i}/{total} prompts...[/dim]")
        
        return responses
