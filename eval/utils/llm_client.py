"""LLM client supporting OpenAI and Gemini APIs"""

import re
import json
import logging
import time
import threading

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
    retry_if_exception,
)

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger(__name__)


def _should_retry_exception(exception: Exception) -> bool:
    """
    Check if exception should trigger a retry.
    Only retries on HTTP 5xx server errors (500-599).
    Does not retry on:
    - ValueError (programming errors)
    - HTTP 4xx errors (client errors like bad requests)
    - Other exceptions that don't represent server-side issues
    """
    # Never retry on ValueError (programming errors)
    if isinstance(exception, ValueError):
        return False
    
    # Check if exception has a status_code attribute (common for HTTP exceptions)
    status_code = getattr(exception, 'status_code', None)
    if status_code is not None:
        # Only retry on 5xx server errors
        return 500 <= status_code < 600
    
    # Check if exception has a code attribute (Google API exceptions often use 'code')
    code = getattr(exception, 'code', None)
    if code is not None:
        # Only retry on 5xx server errors
        return 500 <= code < 600
    
    # Check if exception message contains HTTP status code
    error_str = str(exception).lower()
    for status in range(500, 600):
        if f"{status}" in error_str or f" {status} " in error_str:
            return True
    
    # Default: don't retry if we can't determine it's a 5xx error
    return False


def estimate_tokens_from_prompt(prompt: str, expected_output_tokens: int = 512) -> int:
    """
    Estimate tokens from prompt.
    
    Args:
        prompt: Prompt string
        expected_output_tokens: usually max_output_tokens in the api call
        
    Returns:
        Estimated tokens
    """
    # very common heuristic
    input_tokens = max(1, len(prompt) // 4)
    return input_tokens + expected_output_tokens


class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float, name: str):
        """
        capacity: max tokens
        refill_rate: tokens per second
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.lock = threading.Lock()
        self.last_ts = time.monotonic()
        self.name = name

    def _refill(self):
        now = time.monotonic()
        dt = now - self.last_ts
        self.last_ts = now
        self.tokens = min(self.capacity, self.tokens + dt * self.refill_rate)

    def consume(self, amount: float):
        while True:
            with self.lock:
                self._refill()

                if self.tokens >= amount:
                    self.tokens -= amount
                    logger.debug(
                        "[%s] consumed=%.1f remaining=%.1f",
                        self.name, amount, self.tokens
                    )
                    return

                wait_time = (amount - self.tokens) / self.refill_rate

            time.sleep(wait_time)


class LLMClient:
    """Client for calling LLM APIs (OpenAI and Gemini)"""
    
    def __init__(self, provider: str, model: str, temperature: float, max_tokens: int, api_key: str, rpm_bucket: TokenBucket, tpm_bucket: TokenBucket):
        """
        Initialize LLM client.
        
        Args:
            provider: "openai" or "gemini" (required, no auto-detection)
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            api_key: API key for the provider
            rpm_bucket: TokenBucket for requests per minute rate limiting (required)
            tpm_bucket: TokenBucket for tokens per minute rate limiting (required)
        
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
        self.rpm_bucket = rpm_bucket
        self.tpm_bucket = tpm_bucket
        
        # Initialize provider-specific client
        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
            self.api_uri = "https://api.openai.com/v1/chat/completions"

        elif provider == "gemini":
            self.client = genai.Client(api_key=api_key)
            self.api_uri = "https://generativelanguage.googleapis.com/v1beta/models"
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception(_should_retry_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
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
        
        # Estimate tokens and consume from rate limit buckets
        # Token consumption happens on each retry attempt to respect rate limits
        estimated_tokens = estimate_tokens_from_prompt(prompt, self.max_tokens)
        self.rpm_bucket.consume(1)
        self.tpm_bucket.consume(estimated_tokens)
        
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

