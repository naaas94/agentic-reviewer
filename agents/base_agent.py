import requests
import json
import time
import hashlib
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Union
from jinja2 import Template
import yaml
import os
from dataclasses import dataclass
from enum import Enum

from core.exceptions import LLMConnectionError, LLMResponseError, ConfigurationError, PromptTemplateError, SecurityError
from core.config import config
from core.validators import validate_text, validate_label, validate_confidence
from core.cache import get_cache
from core.security import get_security_manager


# Configure logging
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Enum for different agent tasks."""
    EVALUATE = "evaluate"
    PROPOSE = "propose"
    REASON = "reason"


@dataclass
class AgentTask:
    """Represents a task for the multi-task agent."""
    task_type: TaskType
    context: Dict[str, Any]
    expected_fields: List[str]
    prompt_template: str


class BaseAgent:
    """Base class for all LLM agents with shared functionality and multi-task support."""
    
    def __init__(self, model_name: Optional[str] = None, ollama_url: Optional[str] = None):
        self.model_name = model_name or config.llm.model_name
        self.ollama_url = ollama_url or config.llm.ollama_url
        self.labels = self._load_labels()
        self.cache = get_cache()
        
        # Initialize security manager
        ground_truth_file = os.path.join("data", "ground_truth.json")
        self.security_manager = get_security_manager(ground_truth_file)
        
        # Load all prompt templates
        self.prompt_templates = {
            TaskType.EVALUATE: self._load_prompt_template("evaluator_prompt.txt"),
            TaskType.PROPOSE: self._load_prompt_template("proposer_prompt.txt"),
            TaskType.REASON: self._load_prompt_template("reasoner_prompt.txt")
        }
        
    def _load_labels(self) -> Dict[str, Any]:
        """Load label definitions from YAML config."""
        try:
            config_path = os.path.join("configs", "labels.yaml")
            with open(config_path, 'r', encoding='utf-8') as f:
                labels = yaml.safe_load(f)
            
            if not labels or "labels" not in labels:
                raise ConfigurationError("Invalid labels configuration: missing 'labels' key")
            
            logger.debug(f"Loaded {len(labels['labels'])} label definitions")
            return labels
            
        except FileNotFoundError:
            raise ConfigurationError(f"Labels configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in labels configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load labels configuration: {e}")
    
    def _load_prompt_template(self, prompt_file: str) -> Template:
        """Load and return a Jinja2 template from the prompts directory."""
        try:
            prompt_path = os.path.join("prompts", prompt_file)
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            if not template_content.strip():
                raise PromptTemplateError(f"Prompt template file is empty: {prompt_path}")
            
            template = Template(template_content)
            logger.debug(f"Loaded prompt template: {prompt_file}")
            return template
            
        except FileNotFoundError:
            raise PromptTemplateError(f"Prompt template file not found: {prompt_path}")
        except Exception as e:
            raise PromptTemplateError(f"Failed to load prompt template {prompt_file}: {e}")
    
    def _validate_input_security(self, text: str, predicted_label: str, confidence: float) -> Dict[str, Any]:
        """
        Validate input for security threats including prompt injection and adversarial attacks.
        
        Args:
            text: Input text to validate
            predicted_label: Predicted label
            confidence: Confidence score
            
        Returns:
            Security validation results
            
        Raises:
            SecurityError: If input is deemed unsafe
        """
        security_result = self.security_manager.validate_input(text, predicted_label, confidence)
        
        if not security_result["is_safe"]:
            logger.warning(f"Security validation failed: {security_result}")
            
            # Log security violations
            for violation in security_result["violations"]:
                logger.warning(f"Security violation: {violation.violation_type} - {violation.description}")
            
            # For critical violations, raise exception
            critical_violations = [v for v in security_result["violations"] if v.severity == "critical"]
            if critical_violations:
                raise SecurityError(f"Critical security violation detected: {critical_violations[0].description}")
        
        return security_result
    
    def _call_ollama(self, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Make a call to Ollama API with retry logic and circuit breaker (synchronous)."""
        max_tokens = max_tokens or config.llm.max_tokens
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": config.llm.temperature
            }
        }
        
        # Circuit breaker implementation with proper state machine
        cache_key = f"circuit_breaker:{self.ollama_url}"
        circuit_state = self.cache.get(cache_key)
        
        if circuit_state:
            state = circuit_state.get("state", "closed")
            last_failure = circuit_state.get("last_failure", 0)
            failure_count = circuit_state.get("failure_count", 0)
            success_count = circuit_state.get("success_count", 0)
            current_time = time.time()
            
            if state == "open":
                # Exponential backoff: 1min, 2min, 4min, max 5min
                timeout_seconds = min(60 * (2 ** min(failure_count, 3)), 300)
                if current_time - last_failure > timeout_seconds:
                    # Move to half-open
                    self.cache.set(cache_key, {
                        "state": "half_open",
                        "last_failure": last_failure,
                        "failure_count": failure_count,
                        "success_count": 0
                    }, ttl=600)
                    logger.info(f"Circuit breaker moved to half-open state for {self.ollama_url}")
                else:
                    raise LLMConnectionError(f"Circuit breaker is open - service temporarily unavailable (retry in {int(timeout_seconds - (current_time - last_failure))}s)")
            elif state == "half_open":
                # Allow one request to test if service is back
                logger.debug(f"Circuit breaker in half-open state, testing service for {self.ollama_url}")
                pass

        for attempt in range(config.llm.max_retries):
            try:
                logger.debug(f"Calling Ollama (attempt {attempt + 1}/{config.llm.max_retries})")
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=config.llm.timeout_seconds
                )
                response.raise_for_status()
                result = response.json()
                end_time = time.time()
                if "response" not in result:
                    raise LLMResponseError("Invalid response from Ollama: missing 'response' field")
                response_data = {
                    "response": result.get("response", "").strip(),
                    "tokens_used": result.get("eval_count", 0),
                    "latency_ms": int((end_time - start_time) * 1000),
                    "success": True
                }
                # Handle circuit breaker state on success
                if circuit_state and circuit_state.get("state") == "half_open":
                    # If in half-open, increment success count and close if threshold reached
                    success_count = circuit_state.get("success_count", 0) + 1
                    if success_count >= 3:  # Require 3 successful calls to close
                        self.cache.delete(cache_key)
                        logger.info(f"Circuit breaker closed after {success_count} successful calls for {self.ollama_url}")
                    else:
                        self.cache.set(cache_key, {
                            "state": "half_open",
                            "last_failure": circuit_state.get("last_failure", 0),
                            "failure_count": circuit_state.get("failure_count", 0),
                            "success_count": success_count
                        }, ttl=600)
                else:
                    # Normal success, close circuit breaker
                    self.cache.delete(cache_key)
                
                logger.debug(f"Ollama call successful: {response_data['tokens_used']} tokens, {response_data['latency_ms']}ms")
                return response_data
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, 
                    requests.exceptions.RequestException) as e:
                logger.warning(f"Ollama request error (attempt {attempt + 1}): {e}")
                if attempt == config.llm.max_retries - 1:
                    # Open circuit breaker on final failure
                    if circuit_state and circuit_state.get("state") == "half_open":
                        # If in half-open, increment failure count and go back to open
                        self.cache.set(cache_key, {
                            "state": "open",
                            "last_failure": time.time(),
                            "failure_count": failure_count + 1
                        }, ttl=300)
                    else:
                        self.cache.set(cache_key, {
                            "state": "open",
                            "last_failure": time.time(),
                            "failure_count": failure_count + 1 if circuit_state else 1
                        }, ttl=300)
                    raise LLMConnectionError(f"Ollama request failed after {config.llm.max_retries} attempts: {e}")
            # Exponential backoff
            if attempt < config.llm.max_retries - 1:
                delay = config.llm.retry_delay_seconds * (2 ** attempt)
                time.sleep(delay)
        
        # This should never be reached due to exceptions above
        raise LLMConnectionError("Unexpected error in Ollama call")
    
    async def _call_ollama_async(self, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Make an async call to Ollama API with retry logic and circuit breaker."""
        max_tokens = max_tokens or config.llm.max_tokens
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": config.llm.temperature
            }
        }
        
        # Circuit breaker implementation
        cache_key = f"circuit_breaker:{self.ollama_url}"
        circuit_state = self.cache.get(cache_key)
        
        if circuit_state and circuit_state.get("state") == "open":
            if time.time() - circuit_state.get("last_failure", 0) < 60:  # 1 minute timeout
                raise LLMConnectionError("Circuit breaker is open - service temporarily unavailable")
            else:
                # Reset circuit breaker
                self.cache.delete(cache_key)
        
        for attempt in range(config.llm.max_retries):
            try:
                logger.debug(f"Calling Ollama async (attempt {attempt + 1}/{config.llm.max_retries})")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.ollama_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=config.llm.timeout_seconds)
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                
                end_time = time.time()
                
                # Validate response structure
                if "response" not in result:
                    raise LLMResponseError("Invalid response from Ollama: missing 'response' field")
                
                response_data = {
                    "response": result.get("response", "").strip(),
                    "tokens_used": result.get("eval_count", 0),
                    "latency_ms": int((end_time - start_time) * 1000),
                    "success": True
                }
                
                # Close circuit breaker on success
                self.cache.delete(cache_key)
                
                logger.debug(f"Ollama async call successful: {response_data['tokens_used']} tokens, {response_data['latency_ms']}ms")
                return response_data
                
            except (asyncio.TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientError) as e:
                logger.warning(f"Ollama async request error (attempt {attempt + 1}): {e}")
                
                if attempt == config.llm.max_retries - 1:
                    # Open circuit breaker on final failure
                    self.cache.set(cache_key, {
                        "state": "open",
                        "last_failure": time.time(),
                        "failure_count": circuit_state.get("failure_count", 0) + 1 if circuit_state else 1
                    }, ttl=300)  # 5 minutes
                    
                    raise LLMConnectionError(f"Ollama async request failed after {config.llm.max_retries} attempts: {e}")
            
            # Exponential backoff
            if attempt < config.llm.max_retries - 1:
                delay = config.llm.retry_delay_seconds * (2 ** attempt)
                await asyncio.sleep(delay)
        
        # This should never be reached due to exceptions above
        raise LLMConnectionError("Unexpected error in Ollama async call")
    
    def _parse_response(self, response: str, expected_fields: List[str]) -> Dict[str, Any]:
        """Parse structured response from LLM with improved error handling and validation."""
        if not response or not isinstance(response, str):
            raise LLMResponseError("Invalid response: must be a non-empty string")
        
        # Sanitize response to prevent injection
        response = self._sanitize_response(response)
        
        parsed = {}
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key in expected_fields:
                    parsed[key] = value
        
        # Log parsing results
        found_fields = list(parsed.keys())
        missing_fields = [field for field in expected_fields if field not in parsed]
        
        if missing_fields:
            logger.warning(f"Missing fields in LLM response: {missing_fields}")
        
        logger.debug(f"Parsed {len(found_fields)}/{len(expected_fields)} fields: {found_fields}")
        return parsed
    
    def _sanitize_response(self, response: str) -> str:
        """Sanitize LLM response to prevent injection attacks."""
        # Remove potential script tags
        response = response.replace('<script>', '').replace('</script>', '')
        
        # Remove potential HTML tags
        import re
        response = re.sub(r'<[^>]+>', '', response)
        
        # Limit response length
        if len(response) > 10000:
            response = response[:10000] + "..."
        
        return response
    
    def _get_prompt_hash(self, prompt: str) -> str:
        """Generate a hash of the prompt for versioning."""
        if not prompt:
            return "empty"
        return hashlib.md5(prompt.encode()).hexdigest()[:8]
    
    def _process_llm_response(self, result: Dict[str, Any], expected_fields: List[str]) -> Dict[str, Any]:
        """Common processing logic for LLM responses to eliminate duplication."""
        # Parse response if expected fields provided
        if expected_fields and result["success"]:
            try:
                parsed = self._parse_response(result["response"], expected_fields)
                result["parsed"] = parsed
            except Exception as e:
                logger.warning(f"Failed to parse structured response: {e}")
                result["parsed"] = {}
        
        # Add prompt hash
        result["prompt_hash"] = result.get("prompt_hash", "unknown")
        
        return result

    def _prepare_llm_call(self, prompt_template: str, context: Dict[str, Any], 
                         expected_fields: Optional[List[str]] = None) -> tuple:
        """Common preparation logic for LLM calls to eliminate duplication."""
        # Validate inputs
        if not prompt_template or not isinstance(prompt_template, str):
            raise PromptTemplateError("Prompt template must be a non-empty string")
        
        if not context or not isinstance(context, dict):
            raise PromptTemplateError("Context must be a non-empty dictionary")
        
        # Render the prompt template
        template = Template(prompt_template)
        prompt = template.render(**context, labels=self.labels["labels"])
        
        # Generate cache key
        prompt_hash = self._get_prompt_hash(prompt)
        cache_key = f"llm_response:{prompt_hash}"
        
        return prompt, cache_key, prompt_hash
    
    def _call_llm_common(self, prompt_template: str, context: Dict[str, Any], 
                        expected_fields: Optional[List[str]] = None, 
                        is_async: bool = False) -> Dict[str, Any]:
        """Common logic for both sync and async LLM calls to eliminate duplication."""
        try:
            # Prepare LLM call
            prompt, cache_key, prompt_hash = self._prepare_llm_call(prompt_template, context, expected_fields)
            
            # Check cache first
            if config.performance.enable_caching:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for prompt hash: {prompt_hash}")
                    return cached_result
            
            # Call LLM (sync or async)
            if is_async:
                import asyncio
                result = asyncio.run(self._call_ollama_async(prompt))
            else:
                result = self._call_ollama(prompt)
            
            # Process response
            result = self._process_llm_response(result, expected_fields or [])
            result["prompt_hash"] = prompt_hash
            
            # Cache result
            if config.performance.enable_caching:
                self.cache.set(cache_key, result, ttl=config.performance.cache_ttl_seconds)
            
            return result
            
        except Exception as e:
            logger.error(f"{'Async ' if is_async else ''}LLM call failed: {e}")
            raise

    def call_llm(self, prompt_template: str, context: Dict[str, Any], 
                 expected_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main method to call LLM with templated prompt and validation (synchronous)."""
        return self._call_llm_common(prompt_template, context, expected_fields, is_async=False)
    
    async def call_llm_async(self, prompt_template: str, context: Dict[str, Any], 
                           expected_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main method to call LLM with templated prompt and validation (async)."""
        try:
            # Prepare LLM call
            prompt, cache_key, prompt_hash = self._prepare_llm_call(prompt_template, context, expected_fields)
            
            # Check cache first
            if config.performance.enable_caching:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for prompt hash: {prompt_hash}")
                    return cached_result
            
            # Call LLM
            result = await self._call_ollama_async(prompt)
            
            # Process response
            result = self._process_llm_response(result, expected_fields or [])
            result["prompt_hash"] = prompt_hash
            
            # Cache result
            if config.performance.enable_caching:
                self.cache.set(cache_key, result, ttl=config.performance.cache_ttl_seconds)
            
            return result
            
        except Exception as e:
            logger.error(f"Async LLM call failed: {e}")
            raise
    
    async def process_multi_task(self, tasks: List[AgentTask]) -> List[Dict[str, Any]]:
        """Process multiple tasks in a single LLM call for efficiency."""
        if not tasks:
            return []
        
        # Create combined prompt for all tasks
        combined_prompt = self._create_multi_task_prompt(tasks)
        
        # Create a combined context from all tasks
        combined_context = {}
        for task in tasks:
            combined_context.update(task.context)
        
        # Call LLM once with combined context
        result = await self.call_llm_async(combined_prompt, combined_context, [])
        
        # Parse results for each task
        return self._parse_multi_task_response(result["response"], tasks)
    
    def _create_multi_task_prompt(self, tasks: List[AgentTask]) -> str:
        """Create a combined prompt for multiple tasks."""
        prompt_parts = []
        
        for i, task in enumerate(tasks):
            # Render individual task prompt
            template = Template(task.prompt_template)
            task_prompt = template.render(**task.context, labels=self.labels["labels"])
            
            prompt_parts.append(f"=== TASK {i+1}: {task.task_type.value.upper()} ===\n{task_prompt}\n")
        
        return "\n".join(prompt_parts)
    
    def _parse_multi_task_response(self, response: str, tasks: List[AgentTask]) -> List[Dict[str, Any]]:
        """Parse response for multiple tasks."""
        results = []
        
        # Split response by task markers
        task_responses = response.split("=== TASK")
        
        for i, task in enumerate(tasks):
            if i + 1 < len(task_responses):
                task_response = task_responses[i + 1].split("===")[0].strip()
                
                try:
                    parsed = self._parse_response(task_response, task.expected_fields)
                    results.append({
                        "task_type": task.task_type,
                        "parsed": parsed,
                        "raw_response": task_response,
                        "success": True
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse task {i+1} response: {e}")
                    results.append({
                        "task_type": task.task_type,
                        "parsed": {},
                        "raw_response": task_response,
                        "success": False,
                        "error": str(e)
                    })
            else:
                results.append({
                    "task_type": task.task_type,
                    "parsed": {},
                    "raw_response": "",
                    "success": False,
                    "error": "No response for this task"
                })
        
        return results
    
    def _extract_field_fallback(self, response: str, field_name: str, 
                               valid_values: Optional[List[str]] = None) -> str:
        """Extract field value from response with fallback logic."""
        if not response:
            return "Unknown"
        
        # Try to find field in response
        lines = response.lower().split('\n')
        for line in lines:
            if field_name.lower() in line and ':' in line:
                value = line.split(':', 1)[1].strip()
                if valid_values:
                    # Find best match
                    for valid_value in valid_values:
                        if valid_value.lower() in value:
                            return valid_value
                return value
        
        # Fallback based on field name
        if field_name == "verdict":
            return self._extract_verdict_fallback(response)
        elif field_name == "confidence":
            return self._extract_confidence_fallback(response)
        elif field_name == "label":
            return self._extract_label_fallback(response)
        elif field_name == "reasoning":
            return self._extract_reasoning_fallback(response)
        elif field_name == "explanation":
            return self._extract_explanation_fallback(response)
        
        return "Unknown"
    
    def _extract_verdict_fallback(self, response: str) -> str:
        """Fallback method to extract verdict from response."""
        response_lower = response.lower()
        
        # Check for explicit incorrect patterns first (more specific)
        incorrect_patterns = [
            "incorrect", "wrong", "inaccurate", "doesn't match", "does not match", 
            "doesn't fit", "does not fit", "not correct", "not right", "not accurate"
        ]
        
        correct_patterns = [
            "correct", "right", "accurate", "matches", "fits", "is correct", 
            "is right", "is accurate"
        ]
        
        # Check for incorrect first (more specific patterns)
        for pattern in incorrect_patterns:
            if pattern in response_lower:
                return "Incorrect"
        
        # Then check for correct patterns
        for pattern in correct_patterns:
            if pattern in response_lower:
                return "Correct"
        
        return "Uncertain"
    
    def _extract_confidence_fallback(self, response: str) -> str:
        """Fallback method to extract confidence from response."""
        response_lower = response.lower()
        if "high" in response_lower:
            return "High"
        elif "low" in response_lower:
            return "Low"
        else:
            return "Medium"
    
    def _extract_label_fallback(self, response: str) -> str:
        """Fallback method to extract label from response."""
        # Check against available labels
        available_labels = [label["name"] for label in self.labels["labels"]]
        
        for label in available_labels:
            if label.lower() in response.lower():
                return label
        
        return "No suitable label found"
    
    def _extract_reasoning_fallback(self, response: str) -> str:
        """Fallback method to extract reasoning from response."""
        return response.strip() if response else "No reasoning provided"
    
    def _extract_explanation_fallback(self, response: str) -> str:
        """Fallback method to extract explanation from response."""
        return response.strip() if response else "No explanation provided" 