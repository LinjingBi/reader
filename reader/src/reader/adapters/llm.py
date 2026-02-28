"""LLM adapter for Gemini API"""

import asyncio
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, TypeVar, Type, Callable

from google import genai
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_exception,
    RetryError,
)

from reader.logging.logging_setup import get_logger

T = TypeVar('T', bound=BaseModel)

logger = get_logger()

def _extract_retry_delay(error: Exception) -> Optional[float]:
    """
    Extract retryDelay from Google API error response.
    
    The error response may contain a retryDelay in the details, e.g.:
    {'error': {'details': [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '26s'}]}}
    
    Args:
        error: The exception from the API call
        
    Returns:
        Retry delay in seconds, or None if not found
    """
    # Try to get error response from exception attributes
    # Google API exceptions may have error attribute or response attribute
    error_dict = getattr(error, 'error', None) or getattr(error, 'response', None)
    
    # If error_dict is a dict-like object, try to access details
    if error_dict is not None:
        if isinstance(error_dict, dict):
            details = error_dict.get('details', [])
            for detail in details:
                if isinstance(detail, dict):
                    retry_delay = detail.get('retryDelay')
                    if retry_delay:
                        # Parse "26s" format to float
                        if isinstance(retry_delay, str):
                            match = re.match(r'(\d+(?:\.\d+)?)s', retry_delay)
                            if match:
                                try:
                                    return float(match.group(1))
                                except ValueError:
                                    pass
                        elif isinstance(retry_delay, (int, float)):
                            return float(retry_delay)
        
        # Also check if error_dict has details attribute
        if hasattr(error_dict, 'details'):
            details = getattr(error_dict, 'details', [])
            for detail in details:
                if hasattr(detail, 'retryDelay'):
                    retry_delay = getattr(detail, 'retryDelay')
                    if retry_delay:
                        if isinstance(retry_delay, str):
                            match = re.match(r'(\d+(?:\.\d+)?)s', retry_delay)
                            if match:
                                try:
                                    return float(match.group(1))
                                except ValueError:
                                    pass
                        elif isinstance(retry_delay, (int, float)):
                            return float(retry_delay)
    
    # Try to parse from string representation (fallback)
    # The error might be stringified, e.g., "{'error': {'details': [{'retryDelay': '26s'}]}}"
    error_str = str(error)
    # Look for retryDelay pattern like "retryDelay': '26s'" or "retryDelay": "26s" or 'retryDelay': '26s'
    # Handle both single and double quotes, and various spacing
    patterns = [
        r"retryDelay['\"]?\s*[:=]\s*['\"](\d+(?:\.\d+)?)s['\"]",  # 'retryDelay': '26s'
        r"['\"]retryDelay['\"]\s*[:=]\s*['\"](\d+(?:\.\d+)?)s['\"]",  # "retryDelay": "26s"
        r"retryDelay\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s",  # retryDelay: '26s' or retryDelay='26s'
    ]
    for pattern in patterns:
        match = re.search(pattern, error_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None


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
        
        # Also check if error has nested error dict with code (Google API structure)
        if self.code is None:
            error_dict = getattr(original_error, 'error', None)
            if isinstance(error_dict, dict):
                self.code = error_dict.get('code')
            elif hasattr(error_dict, 'code'):
                self.code = getattr(error_dict, 'code', None)
        
        # If original error doesn't have status_code but has code, use code as status_code
        if self.status_code is None and self.code is not None:
            self.status_code = self.code
        
        # Extract retryDelay from error response if available
        self.retry_delay = _extract_retry_delay(original_error)



def _should_retry_exception(exception: Exception) -> bool:
    """
    Check if exception should trigger a retry.
    Retries on:
    - HTTP 5xx server errors (500-599)
    - HTTP 429 rate limit/quota errors (with retryDelay)
    - ValidationError (Pydantic validation errors - may be transient LLM formatting issues)
    Does not retry on:
    - ValueError (programming errors)
    - LLMGenerationError (final wrapped error after retries)
    - Other HTTP 4xx errors (client errors like bad requests)
    - Other exceptions that don't represent server-side issues
    
    Note: This function works directly with original exceptions (ValidationError, Google API exceptions).
    Exceptions are only wrapped in LLMGenerationError after retries are exhausted.
    """
    # Retry on ValidationError (may be transient LLM formatting issues)
    if isinstance(exception, ValidationError):
        return True
    
    # Never retry on ValueError (programming errors)
    if isinstance(exception, ValueError):
        return False
    
    # Never retry on LLMGenerationError (this is the final wrapped error after retries)
    if isinstance(exception, LLMGenerationError):
        return False
    
    # Check if exception has a status_code attribute (common for HTTP exceptions)
    status_code = getattr(exception, 'status_code', None)
    if status_code is not None:
        # Retry on 5xx server errors and 429 rate limit errors
        return status_code == 429 or (500 <= status_code < 600)
    
    # Check if exception has a code attribute (Google API exceptions often use 'code')
    code = getattr(exception, 'code', None)
    if code is not None:
        # Retry on 5xx server errors and 429 rate limit errors
        return code == 429 or (500 <= code < 600)
    
    # Check nested error dict structure (Google API may nest error info)
    error_dict = getattr(exception, 'error', None)
    if isinstance(error_dict, dict):
        nested_code = error_dict.get('code')
        if nested_code is not None:
            return nested_code == 429 or (500 <= nested_code < 600)
    
    # Check if exception message contains HTTP status code
    error_str = str(exception).lower()
    # Check for 429 specifically
    if '429' in error_str or 'resource_exhausted' in error_str:
        return True
    # Check for 5xx errors
    for status in range(500, 600):
        if f"{status}" in error_str or f" {status} " in error_str:
            return True
    
    # Default: don't retry if we can't determine it's a retryable error
    return False


class WaitWithRetryDelay:
    """
    Custom wait strategy that uses retryDelay from API error response when available,
    otherwise falls back to exponential backoff.
    
    This is a callable class that matches tenacity's wait strategy interface.
    Tenacity accepts any callable that takes a RetryCallState and returns a float.
    """
    def __init__(self, exponential_wait: Callable):
        """
        Initialize wait strategy.
        
        Args:
            exponential_wait: Exponential backoff wait function to use as fallback
        """
        self.exponential_wait = exponential_wait
    
    def __call__(self, retry_state):
        """
        Calculate wait time for retry.
        
        Args:
            retry_state: Tenacity retry state containing exception info
            
        Returns:
            Wait time in seconds
        """
        # Extract retryDelay directly from the original exception
        # Since we work directly with original exceptions during retry, extract from the exception itself
        exception = retry_state.outcome.exception()
        if exception:
            # Extract retryDelay directly from the original exception
            retry_delay = _extract_retry_delay(exception)
            
            if retry_delay is not None and retry_delay > 0:
                logger.info(
                    "Using API-provided retryDelay: %.2fs (attempt %d/%d)",
                    retry_delay, retry_state.attempt_number, 5
                )
                return retry_delay
        
        # Fall back to exponential backoff
        return self.exponential_wait(retry_state)


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
        wait=WaitWithRetryDelay(wait_exponential(multiplier=1, min=2, max=60)),
        retry=retry_if_exception(_should_retry_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=False,  # Don't reraise - we'll wrap in LLMGenerationError after retries
    )
    def _call_structured_raw_inner(self, prompt: str, response_model: Type[T], temperature: float, max_tokens: int) -> T:
        """
        Inner function that performs the actual API call and raises original exceptions.
        This is wrapped by the retry decorator to handle retries with original exceptions.
        
        Args:
            prompt: Full prompt string
            response_model: Pydantic model class for response schema
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed Pydantic model instance of type T
            
        Raises:
            ValidationError: If JSON parsing/validation fails
            Exception: Original Google API exceptions (not wrapped)
        """
        logger.debug("Sending request to Gemini API...")
        
        # Estimate tokens and consume from rate limit buckets
        # Token consumption happens on each retry attempt to respect rate limits
        estimated_tokens = self.estimate_tokens(prompt, max_tokens)
        self.rpm_bucket.consume(1)
        self.tpm_bucket.consume(estimated_tokens)
        
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
    
    def call_structured_raw(self, prompt: str, response_model: Type[T], temperature: float, max_tokens: int) -> T:
        """
        Call Gemini LLM with structured output and return parsed Pydantic model instance.
        
        This method wraps the inner function with retry logic that works directly with
        original exceptions. After retries are exhausted (or if retry shouldn't happen),
        exceptions are wrapped in LLMGenerationError for consistent error handling.
        
        Args:
            prompt: Full prompt string
            response_model: Pydantic model class for response schema
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed Pydantic model instance of type T
            
        Raises:
            LLMGenerationError: If API call fails after all retries or if retry is not applicable
        """
        try:
            return self._call_structured_raw_inner(prompt, response_model, temperature, max_tokens)
        except ValidationError as e:
            # Log validation errors
            logger.error("Pydantic validation error: %s", str(e))
            # Wrap in LLMGenerationError after retries are exhausted
            error_message = f"LLM generation API error: {str(e)}"
            raise LLMGenerationError(error_message, e) from e
        except RetryError as e:
            # RetryError is raised when all retries are exhausted
            # Extract the underlying exception from the last attempt
            last_attempt = getattr(e, 'last_attempt', None)
            underlying_exception = None
            attempt_number = None
            
            if last_attempt:
                attempt_number = getattr(last_attempt, 'attempt_number', None)
                outcome = getattr(last_attempt, 'outcome', None)
                if outcome:
                    underlying_exception = outcome.exception()
            
            # Extract status code from underlying exception if available
            status_code = None
            if underlying_exception:
                status_code = getattr(underlying_exception, 'status_code', None) or getattr(underlying_exception, 'code', None)
            
            # Create a clearer error message
            if status_code:
                if status_code == 429:
                    error_message = (
                        f"LLM API call failed after 5 retry attempts: "
                        f"Rate limit/quota exceeded (HTTP {status_code}). "
                        f"Last error: {str(underlying_exception) if underlying_exception else str(e)}"
                    )
                else:
                    error_message = (
                        f"LLM API call failed after 5 retry attempts: "
                        f"HTTP {status_code} error. "
                        f"Last error: {str(underlying_exception) if underlying_exception else str(e)}"
                    )
                logger.error(
                    "Gemini API retries exhausted: status_code=%s, attempts=%s, error=%s",
                    status_code, attempt_number or "unknown", str(underlying_exception) if underlying_exception else str(e)
                )
            else:
                error_message = (
                    f"LLM API call failed after 5 retry attempts. "
                    f"Last error: {str(underlying_exception) if underlying_exception else str(e)}"
                )
                logger.error(
                    "Gemini API retries exhausted: attempts=%s, error=%s",
                    attempt_number or "unknown", str(underlying_exception) if underlying_exception else str(e)
                )
            
            # Wrap in LLMGenerationError with clearer message
            raise LLMGenerationError(error_message, underlying_exception or e) from e
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
            
            # Wrap the exception in LLMGenerationError after retries are exhausted
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
            loop = asyncio.get_running_loop()
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
