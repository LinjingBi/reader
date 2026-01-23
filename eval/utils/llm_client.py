"""LLM client supporting OpenAI and Gemini APIs"""

import json
import re
from typing import Optional, Tuple


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
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                self.api_uri = "https://api.openai.com/v1/chat/completions"
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        elif provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.client = genai
                self.api_uri = "https://generativelanguage.googleapis.com/v1beta/models"
            except ImportError:
                raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
    
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
    
    def call(self, prompt: str) -> str:
        """
        Call LLM with prompt.
        
        Args:
            prompt: Full prompt string (will be parsed for SYSTEM/USER sections)
        
        Returns:
            Raw response text from LLM
        
        Raises:
            Exception: If API call fails
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
            
            model = self.client.GenerativeModel(self.model)
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
            
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            return response.text
    
    def get_api_args(self) -> dict:
        """Get API arguments for logging"""
        args = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return args

