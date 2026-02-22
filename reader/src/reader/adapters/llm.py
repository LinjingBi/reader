"""LLM adapter for Gemini API"""

import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, TypeVar, Type

from google import genai
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_exception,
)

from reader.logging.logging_setup import get_logger

T = TypeVar('T', bound=BaseModel)

logger = get_logger()

class LLMGenerationError(Exception):
    """
    Exception raised when LLM generation API call fails.
    
    This exception wraps errors from the Gemini API and preserves
    status_code/code attributes for retry logic.
    """
    def __init__(self, message: str, original_error: Exception):
        """
        Initialize LLM generation error.
        
        Args:
            message: Error message indicating this is an LLM generation API error
            original_error: The original exception from the API call
        """
        super().__init__(message)
        self.original_error = original_error
        
        # Preserve status_code and code attributes from original error for retry logic
        self.status_code = getattr(original_error, 'status_code', None)
        self.code = getattr(original_error, 'code', None)
        
        # If original error doesn't have status_code but has code, use code as status_code
        if self.status_code is None and self.code is not None:
            self.status_code = self.code



def _should_retry_exception(exception: Exception) -> bool:
    """
    Check if exception should trigger a retry.
    Retries on:
    - HTTP 5xx server errors (500-599)
    - ValidationError (Pydantic validation errors - may be transient LLM formatting issues)
    Does not retry on:
    - ValueError (programming errors)
    - HTTP 4xx errors (client errors like bad requests)
    - Other exceptions that don't represent server-side issues
    """
    # Retry on ValidationError (may be transient LLM formatting issues)
    if isinstance(exception, ValidationError):
        return True
    
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


class TokenBucket:
    """Thread-safe token bucket rate limiter"""
    
    def __init__(self, capacity: float, refill_rate: float, name: str):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens per second refill rate
            name: Name for logging
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.lock = threading.Lock()
        self.last_ts = time.monotonic()
        self.name = name

    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.monotonic()
        dt = now - self.last_ts
        self.last_ts = now
        self.tokens = min(self.capacity, self.tokens + dt * self.refill_rate)

    def consume(self, amount: float):
        """
        Consume tokens from bucket, blocking if necessary.
        
        Args:
            amount: Number of tokens to consume
        """
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
    """Client for calling Gemini LLM API"""
    
    def __init__(self, model: str, api_key: str, rpm_bucket: TokenBucket, tpm_bucket: TokenBucket, executor: Optional[ThreadPoolExecutor] = None):
        """
        Initialize Gemini LLM client.
        
        Args:
            model: Gemini model name
            api_key: Gemini API key
            rpm_bucket: TokenBucket for requests per minute rate limiting
            tpm_bucket: TokenBucket for tokens per minute rate limiting
            executor: Optional thread pool executor for async calls. If None, calls are synchronous.
        """
        self.model = model
        self.rpm_bucket = rpm_bucket
        self.tpm_bucket = tpm_bucket
        self.api_key = api_key
        self.executor = executor
        self.client = genai.Client(api_key=api_key)
            
    def estimate_tokens(self, prompt: str, expected_output_tokens: int) -> int:
        """
        Estimate tokens from prompt.
        
        Args:
            prompt: Prompt string
            expected_output_tokens: Expected output tokens
            
        Returns:
            Estimated total tokens (input + output)
        """
        # Very common heuristic: ~4 characters per token
        input_tokens = max(1, len(prompt) // 4)
        return input_tokens + expected_output_tokens
    
    def _raise_for_status(self, response):
        """
        Check response for HTTP errors and raise if found.
        
        Args:
            response: Gemini API response object
            
        Raises:
            Exception: If response indicates an HTTP error
        """
        # Gemini API responses typically don't have explicit status codes in the response object
        # Errors are usually raised as exceptions. This method is called after successful
        # API call, so we mainly check for any error indicators in the response.
        # If there are issues, they would have been raised as exceptions already.
        
        # Check if response has any error indicators
        if hasattr(response, 'error'):
            error = response.error
            if error:
                status_code = getattr(error, 'code', None) or getattr(error, 'status_code', None)
                error_message = str(error)
                
                if status_code:
                    logger.error(
                        "Gemini API HTTP error: status_code=%s, message=%s",
                        status_code, error_message
                    )
                    # Create an exception with status code info
                    http_error = Exception(f"Gemini API error: {status_code} - {error_message}")
                    http_error.status_code = status_code
                    raise http_error
                else:
                    logger.error("Gemini API error: %s", error_message)
                    raise Exception(f"Gemini API error: {error_message}")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception(_should_retry_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def call_structured_raw(self, prompt: str, response_model: Type[T], temperature: float, max_tokens: int) -> T:
        """
        Call Gemini LLM with structured output and return parsed Pydantic model instance.
        
        Args:
            prompt: Full prompt string
            response_model: Pydantic model class for response schema
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed Pydantic model instance of type T
            
        Raises:
            ValidationError: If JSON parsing/validation fails after all retries
            Exception: If API call fails after all retries
        """
        logger.debug("Sending request to Gemini API...")
        
        
        # Estimate tokens and consume from rate limit buckets
        # Token consumption happens on each retry attempt to respect rate limits
        estimated_tokens = self.estimate_tokens(prompt, max_tokens)
        self.rpm_bucket.consume(1)
        self.tpm_bucket.consume(estimated_tokens)
        
        try:
            # Use Gemini's structured output API
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json",
                    "response_json_schema": response_model.model_json_schema(),
                }
            )
            
            # Check for HTTP errors in response
            self._raise_for_status(response)
            
            # Parse JSON response into Pydantic model
            json_text = response.text or ""
            if not json_text:
                raise ValueError("Empty response from LLM")
            
            parsed_model = response_model.model_validate_json(json_text)
            return parsed_model
            
        except ValidationError as e:
            # Log validation errors
            logger.error("Pydantic validation error: %s", str(e))
            # Re-raise to allow retry logic to handle it
            raise
        except Exception as e:
            # Extract status code from exception for logging
            status_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            
            if status_code:
                logger.error(
                    "Gemini API HTTP error: status_code=%s, error=%s",
                    status_code, str(e)
                )
            else:
                # Log non-HTTP errors at error level too
                logger.error("Gemini API error: %s", str(e))
            
            # Re-raise to allow retry logic to handle it
            # Wrap the exception in LLMGenerationError with clear error message
            error_message = f"LLM generation API error: {str(e)}"
            raise LLMGenerationError(error_message, e) from e
    
    async def call_structured_raw_async(self, prompt: str, response_model: Type[T], temperature: float, max_tokens: int) -> T:
        """
        Async wrapper for call_structured_raw that uses executor if provided.
        
        Args:
            prompt: Full prompt string
            response_model: Pydantic model class for response schema
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed Pydantic model instance of type T
            
        Raises:
            ValidationError: If JSON parsing/validation fails after all retries
            Exception: If API call fails after all retries
        """
        if self.executor is not None:
            # Use executor to run synchronous call in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self.call_structured_raw,
                prompt,
                response_model,
                temperature,
                max_tokens
            )
        else:
            # No executor provided, use default thread pool to avoid blocking event loop
            return await asyncio.to_thread(
                self.call_structured_raw,
                prompt,
                response_model,
                temperature,
                max_tokens
            )
