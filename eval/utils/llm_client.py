"""LLM client supporting OpenAI and Gemini APIs"""

import re
import json
import logging

from google import genai
import openai
from typing import Optional, Tuple, TypeVar, Type
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception_message,
    before_sleep_log,
)

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """Client for calling LLM APIs (OpenAI and Gemini)"""
    
    def __init__(self, provider: str, model: str, temperature: float, max_tokens: int, api_key: str):
        """
        Initialize LLM client.
        
        Args:
            provider: "openai" or "gemini" (required, no auto-detection)
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            api_key: API key for the provider
        
        Raises:
            ValueError: If provider is not "openai" or "gemini"
        """
        if provider not in ("openai", "gemini"):
            raise ValueError(f"Provider must be 'openai' or 'gemini', got '{provider}'")
        
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        # Initialize provider-specific client
        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
            self.api_uri = "https://api.openai.com/v1/chat/completions"

        elif provider == "gemini":
            self.client = genai.Client(api_key=api_key)
            self.api_uri = "https://generativelanguage.googleapis.com/v1beta/models"
    
    def _parse_prompt(self, prompt: str) -> Tuple[Optional[str], str]:
        """
        Parse prompt to extract system and user parts.
        
        Args:
            prompt: Full prompt string with SYSTEM: and USER: markers
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Extract SYSTEM section
        system_match = re.search(r'SYSTEM:\s*(.*?)(?=USER:|$)', prompt, re.DOTALL)
        system_prompt = system_match.group(1).strip() if system_match else None
        
        # Extract USER section
        user_match = re.search(r'USER:\s*(.*?)$', prompt, re.DOTALL)
        user_prompt = user_match.group(1).strip() if user_match else prompt.strip()
        
        return system_prompt, user_prompt
    
    def _extract_json_from_markdown(self, text: str) -> str:
        """
        Extract JSON from markdown code blocks if present.
        
        Args:
            text: Text that may contain JSON wrapped in markdown code blocks
            
        Returns:
            Extracted JSON string, or original text if no code blocks found
        """
        # Try to extract JSON from markdown code blocks (```json ... ```)
        json_match = re.search(r"```(?:json|JSON)?\s*\n([\s\S]*?)\n```", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # If no code blocks, return original text
        return text.strip()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=(
            retry_if_exception_type(openai.RateLimitError) |
            retry_if_exception_type(openai.APITimeoutError) |
            retry_if_exception_message(match=r".*(timeout|connection|network|dns|rate limit|rate_limit|429|quota|resource exhausted|503|502|500|internal server error|service unavailable).*")
        ),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.INFO),
        reraise=True,
    )
    def call(self, prompt: str) -> str:
        """
        Call LLM with prompt and retry logic.
        
        Args:
            prompt: Full prompt string (will be parsed for SYSTEM/USER sections)
        
        Returns:
            Raw response text from LLM
        
        Raises:
            Exception: If API call fails after all retries
        """
        system_prompt, user_prompt = self._parse_prompt(prompt)
        
        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        
        elif self.provider == "gemini":
            # Gemini handles system instructions differently
            # Combine system and user prompts
            full_prompt = ""
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                full_prompt = user_prompt
            
            # Use the new google.genai API
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            return response.text or ""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=(
            retry_if_exception_type(openai.RateLimitError) |
            retry_if_exception_type(openai.APITimeoutError) |
            retry_if_exception_message(match=r".*(timeout|connection|network|dns|rate limit|rate_limit|429|quota|resource exhausted|503|502|500|internal server error|service unavailable).*")
        ),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.INFO),
        reraise=True,
    )
    def call_structured(self, prompt: str, response_model: Type[T]) -> T:
        """
        Call LLM with structured output using Gemini's structured output API.
        
        Args:
            prompt: Full prompt string (will be parsed for SYSTEM/USER sections)
            response_model: Pydantic model class for response validation
        
        Returns:
            Parsed Pydantic model instance
        
        Raises:
            ValueError: If provider is not "gemini" (structured output only supported for Gemini)
            Exception: If API call fails after all retries
        """
        if self.provider != "gemini":
            raise ValueError(f"Structured output only supported for Gemini, got {self.provider}")
        
        system_prompt, user_prompt = self._parse_prompt(prompt)
        
        # Combine system and user prompts
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt
        
        # Use Gemini's structured output API
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": response_model.model_json_schema(),
            }
        )
        
        # Parse JSON response
        data = json.loads(response.text)
        
        # Validate and return Pydantic model
        return response_model.model_validate(data)
    
    def call_structured_raw(self, prompt: str, response_model: Type[T]) -> str:
        """
        Call LLM with structured output and return raw text response.
        
        Args:
            prompt: Full prompt string (will be parsed for SYSTEM/USER sections)
            response_model: Pydantic model class for response schema (used for API call only)
        
        Returns:
            Raw text response from LLM (may contain JSON in markdown code blocks or raw JSON)
        
        Raises:
            ValueError: If provider is not "gemini" (structured output only supported for Gemini)
            Exception: If API call fails after all retries
        """
        if self.provider != "gemini":
            raise ValueError(f"Structured output only supported for Gemini, got {self.provider}")
        
        # Use Gemini's structured output API
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": response_model.model_json_schema(),
            }
        )
        
        # Return raw text response (JSON extraction will be handled by caller)
        return response.text or ""
    
    def get_api_args(self) -> dict:
        """Get API arguments for logging"""
        args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return args

