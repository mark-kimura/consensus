#!/usr/bin/env python3
"""
Generic AI provider script that queries multiple AI services concurrently and consolidates responses.
Supports OpenAI, Gemini, and Perplexity APIs with configurable web search modes.

This script replaces the original askgpt.py with a more flexible, multi-provider approach.

Features:
- Concurrent querying of multiple AI providers
- Configurable API endpoints and models via consensus_config.json
- Support for web search, deep research, and no-web modes
- Consolidated response analysis
- Environment variable support for API keys

Setup:
1. Set API keys in environment variables:
   export OPENAI_API_KEY="your-openai-key"
   export GEMINI_API_KEY="your-gemini-key" 
   export OPENROUTER_API_KEY="your-openrouter-key"  # For Perplexity models via OpenRouter

2. Or configure them in consensus_config.json (not recommended for security)

3. Install dependencies: pip install -r requirements.txt

Usage: python consensus.py <markdown_file_path> [--search-mode {web|deep|none}] [--output-dir DIR] [--config CONFIG]
"""

import sys
import os
import argparse
import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import re

def load_config(config_path: str = "consensus_config.json") -> Dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Override with environment variables if they exist
        if os.getenv("OPENAI_API_KEY"):
            config["api_keys"]["openai"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("GEMINI_API_KEY"):
            config["api_keys"]["gemini"] = os.getenv("GEMINI_API_KEY")
        if os.getenv("OPENROUTER_API_KEY"):
            config["api_keys"]["openrouter"] = os.getenv("OPENROUTER_API_KEY")
        # Keep backward compatibility with PERPLEXITY_API_KEY
        if os.getenv("PERPLEXITY_API_KEY") and "openrouter" not in config["api_keys"]:
            config["api_keys"]["openrouter"] = os.getenv("PERPLEXITY_API_KEY")
        
        return config
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found. Please create it or use environment variables.")
        return {
            "api_keys": {
                "openai": os.getenv("OPENAI_API_KEY"),
                "gemini": os.getenv("GEMINI_API_KEY"),
                "openrouter": os.getenv("OPENROUTER_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
            },
            "settings": {
                "default_output_dir": "consensus_docs",
                "default_search_mode": "none",
                "max_tokens": 4000,
                "temperature": 0.7
            },
            "models": {
                "openai": "gpt-5",
                "gemini": "gemini-2.5-pro",
                "perplexity_sonar": "perplexity/sonar",
                "perplexity_sonar_reasoning": "perplexity/sonar-reasoning",
                "perplexity_sonar_deep": "perplexity/sonar-deep-research"
            },
            "endpoints": {
                "openai": "https://api.openai.com/v1",
                "gemini": "https://generativelanguage.googleapis.com/v1beta",
                "openrouter": "https://openrouter.ai/api/v1"
            }
        }

class AIProvider:
    """Base class for AI providers"""
    
    def __init__(self, name: str, api_key: str, config: Dict):
        self.name = name
        self.api_key = api_key
        self.available = bool(api_key)
        self.config = config
    
    async def query(self, session: aiohttp.ClientSession, prompt: str, search_mode: str) -> Optional[str]:
        """Query the AI provider with the given prompt"""
        raise NotImplementedError

class OpenAIProvider(AIProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: str, config: Dict):
        super().__init__("OpenAI", api_key, config)
        self.base_url = config["endpoints"]["openai"]
    
    async def query(self, session: aiohttp.ClientSession, prompt: str, search_mode: str) -> Optional[str]:
        if not self.available:
            return None
            
        # Modify prompt based on search mode
        enhanced_prompt = self._enhance_prompt(prompt, search_mode)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use Responses API for GPT-5, Chat Completions for older models
        if self.config["providers"]["openai"]["model"].startswith("gpt-5"):
            # GPT-5 uses the new Responses API
            reasoning_effort = "low" if search_mode == "none" else "medium"
            data = {
                "model": self.config["providers"]["openai"]["model"],
                "input": enhanced_prompt,
                "reasoning": {"effort": reasoning_effort},
                "text": {"verbosity": "medium"}
            }
            endpoint = f"{self.base_url}/responses"
        else:
            # Older models use Chat Completions API
            data = {
                "model": self.config["providers"]["openai"]["model"],
                "messages": [{"role": "user", "content": enhanced_prompt}],
                "max_tokens": self.config["settings"]["max_tokens"]
            }
            endpoint = f"{self.base_url}/chat/completions"
        
        try:
            async with session.post(endpoint, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if self.config["providers"]["openai"]["model"].startswith("gpt-5"):
                        # GPT-5 Responses API structure - extract text from the message content
                        try:
                            # The response has a 'text' field which is a list of items
                            # Look for the message item with output_text content
                            if "text" in result and isinstance(result["text"], list):
                                for item in result["text"]:
                                    if (item.get("type") == "message" and 
                                        item.get("status") == "completed" and 
                                        "content" in item):
                                        for content_item in item["content"]:
                                            if content_item.get("type") == "output_text":
                                                return content_item.get("text", "")
                            
                            # Try the output field instead - it contains the full response structure
                            if "output" in result and isinstance(result["output"], list):
                                for item in result["output"]:
                                    if (item.get("type") == "message" and 
                                        item.get("status") == "completed" and 
                                        "content" in item):
                                        for content_item in item["content"]:
                                            if content_item.get("type") == "output_text":
                                                return content_item.get("text", "")
                            elif "output" in result:
                                return str(result["output"])
                            
                            print(f"Could not extract text from GPT-5 response structure")
                            print(f"Response keys: {list(result.keys())}")
                            if "output" in result:
                                print(f"Output field structure: {result['output']}")
                            return None
                        except Exception as e:
                            print(f"Error parsing GPT-5 response: {e}")
                            return None
                    else:
                        return result["choices"][0]["message"]["content"]
                else:
                    print(f"OpenAI API error: {response.status}")
                    return None
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return None
    
    def _enhance_prompt(self, prompt: str, search_mode: str) -> str:
        if search_mode == "web":
            return f"Please provide a comprehensive answer using current web information when relevant:\n\n{prompt}"
        elif search_mode == "deep":
            return f"Please provide a deep, analytical response with thorough research and multiple perspectives:\n\n{prompt}"
        else:
            return prompt

class GeminiProvider(AIProvider):
    """Google Gemini provider"""
    
    def __init__(self, api_key: str, config: Dict):
        super().__init__("Gemini", api_key, config)
        self.base_url = config["endpoints"]["gemini"]
    
    async def query(self, session: aiohttp.ClientSession, prompt: str, search_mode: str) -> Optional[str]:
        if not self.available:
            return None
            
        enhanced_prompt = self._enhance_prompt(prompt, search_mode)
        
        params = {"key": self.api_key}
        # Use Gemini-specific token limit if available, otherwise fall back to default
        max_tokens = self.config["settings"].get("max_tokens_gemini", self.config["settings"]["max_tokens"])
        
        data = {
            "contents": [{
                "parts": [{"text": enhanced_prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": self.config["settings"]["temperature"]
            }
        }
        
        # Retry logic for handling temporary server errors
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                async with session.post(f"{self.base_url}/models/{self.config['providers']['gemini']['model']}:generateContent", 
                                      params=params, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "candidates" in result and len(result["candidates"]) > 0:
                            candidate = result["candidates"][0]
                            # Handle different response structures
                            if "content" in candidate and "parts" in candidate["content"]:
                                text = candidate["content"]["parts"][0]["text"]
                                # Check if response was truncated but still has content
                                if candidate.get("finishReason") == "MAX_TOKENS":
                                    print(f"  Warning: Gemini response truncated due to MAX_TOKENS but returning partial content")
                                return text
                            elif "finishReason" in candidate and candidate["finishReason"] == "MAX_TOKENS":
                                print(f"  Warning: Gemini response truncated due to MAX_TOKENS and no content found")
                                return None
                        return None
                    elif response.status >= 500 and attempt < max_retries:
                        # Server error - retry with exponential backoff
                        delay = base_delay * (2 ** attempt)
                        print(f"  Gemini server error {response.status}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        print(f"Gemini API error: {response.status}")
                        return None
            except Exception as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(f"  Gemini connection error, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"Error querying Gemini after {max_retries} retries: {e}")
                    return None
        
        return None
    
    def _enhance_prompt(self, prompt: str, search_mode: str) -> str:
        if search_mode == "web":
            return f"Please search for current information and provide a comprehensive answer:\n\n{prompt}"
        elif search_mode == "deep":
            return f"Please provide a thorough, in-depth analysis with multiple angles and detailed research:\n\n{prompt}"
        else:
            return prompt

class PerplexityProvider(AIProvider):
    """Perplexity AI provider via OpenRouter"""
    
    def __init__(self, api_key: str, config: Dict):
        super().__init__("Perplexity", api_key, config)
        self.base_url = config["endpoints"]["openrouter"]
    
    async def query(self, session: aiohttp.ClientSession, prompt: str, search_mode: str) -> Optional[str]:
        if not self.available:
            return None
            
        # Map search mode to appropriate Sonar model
        model = self._get_model(search_mode)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/consensus",  # Optional but recommended
            "X-Title": "AskAIs Multi-Provider Query Tool"  # Optional but recommended
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config["settings"]["max_tokens"]
        }
        
        try:
            async with session.post(f"{self.base_url}/chat/completions", 
                                  headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    print(f"Perplexity (OpenRouter) API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"Error querying Perplexity via OpenRouter: {e}")
            return None
    
    def _get_model(self, search_mode: str) -> str:
        """Map search mode to appropriate Perplexity Sonar model"""
        if search_mode == "web":
            return "perplexity/sonar-reasoning"
        else:  # search_mode == "none"
            return "perplexity/sonar"

class OpenRouterProvider(AIProvider):
    """Generic OpenRouter provider for any model"""
    
    def __init__(self, api_key: str, model_config, provider_name: str, config: Dict):
        super().__init__(provider_name, api_key, config)
        self.model_config = model_config  # Can be string or dict
        self.base_url = config["endpoints"]["openrouter"]
    
    def get_model_for_mode(self, search_mode: str) -> str:
        """Get the model name for a specific search mode"""
        if isinstance(self.model_config, dict):
            return self.model_config.get(search_mode, self.model_config.get("none", ""))
        else:
            return self.model_config
    
    async def query(self, session: aiohttp.ClientSession, prompt: str, search_mode: str) -> Optional[str]:
        if not self.available:
            return None
            
        # Get the model based on search mode
        if isinstance(self.model_config, dict):
            model = self.model_config.get(search_mode, self.model_config.get("none", ""))
        else:
            model = self.model_config
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/consensus",
            "X-Title": "Consensus Multi-Provider Query Tool"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config["settings"]["max_tokens"]
        }
        
        # Add web search plugins for web mode (when not using :online models)
        if search_mode == "web" and not model.endswith(":online"):
            data["plugins"] = [{
                "id": "web", 
                "max_results": 5,
                "search_prompt": "Relevant web results:"
            }]
        
        try:
            async with session.post(f"{self.base_url}/chat/completions", 
                                  headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    print(f"{self.name} (OpenRouter) API error: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"Error querying {self.name} via OpenRouter: {e}")
            return None

async def query_all_providers(providers: List[AIProvider], prompt: str, search_mode: str) -> Dict[str, str]:
    """Query all available providers concurrently"""
    responses = {}
    
    async def query_with_progress(provider: AIProvider, session: aiohttp.ClientSession, prompt: str, search_mode: str):
        """Wrapper to add progress reporting for each provider"""
        provider_name = provider.name
        model_name = ""
        
        # Get the model name for display
        if hasattr(provider, 'get_model_for_mode'):
            model_name = provider.get_model_for_mode(search_mode)
        elif hasattr(provider, 'model'):
            model_name = provider.model
        elif provider_name == "OpenAI":
            model_name = provider.config["providers"]["openai"]["model"]
        elif provider_name == "Gemini":
            model_name = provider.config["providers"]["gemini"]["model"]
        elif provider_name == "Perplexity":
            # Show which Sonar model based on search mode
            if search_mode == "web":
                model_name = "perplexity/sonar-reasoning"
            else:
                model_name = "perplexity/sonar"
        
        print(f"  • {provider_name} ({model_name}): Starting...", flush=True)
        try:
            start_time = time.time()
            response = await provider.query(session, prompt, search_mode)
            elapsed = time.time() - start_time
            if response:
                print(f"  ✓ {provider_name} ({model_name}): Completed in {elapsed:.1f}s", flush=True)
                return provider_name, response
            else:
                print(f"  ✗ {provider_name} ({model_name}): No response received", flush=True)
                return provider_name, None
        except Exception as e:
            print(f"  ✗ {provider_name} ({model_name}): Error - {e}", flush=True)
            return provider_name, None
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for provider in providers:
            if provider.available:
                task = asyncio.create_task(query_with_progress(provider, session, prompt, search_mode))
                tasks.append(task)
        
        # Gather results as they complete
        for completed_task in asyncio.as_completed(tasks):
            provider_name, response = await completed_task
            if response:
                responses[provider_name] = response
    
    return responses

def consolidate_responses(responses: Dict[str, str], prompt: str) -> str:
    """Consolidate responses from multiple providers into a final document"""
    if not responses:
        return "No responses received from any provider."
    
    consolidated = []
    
    # Header
    consolidated.append(f"# Consolidated AI Response - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    consolidated.append("")
    consolidated.append(f"**Query:** {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    consolidated.append("")
    consolidated.append(f"**Providers Used:** {', '.join(responses.keys())}")
    consolidated.append("")
    
    # Individual responses
    for provider_name, response in responses.items():
        consolidated.append(f"## {provider_name} Response")
        consolidated.append("")
        # Handle both string and dict responses
        if isinstance(response, str):
            consolidated.append(response.strip())
        else:
            consolidated.append(str(response))
        consolidated.append("")
        consolidated.append("---")
        consolidated.append("")
    
    
    return "\n".join(consolidated)

def setup_providers(config: Dict) -> List[AIProvider]:
    """Initialize AI providers based on flexible configuration"""
    providers = []
    
    for provider_name, provider_config in config["providers"].items():
        if not provider_config["enabled"]:
            continue
            
        # Determine which API to use and which key
        if provider_config["use_openrouter"]:
            # Use OpenRouter for this provider
            openrouter_key = config["api_keys"].get("openrouter")
            if not openrouter_key:
                print(f"Warning: {provider_name} configured for OpenRouter but no OpenRouter API key found, skipping...", flush=True)
                continue
                
            if provider_name == "perplexity":
                providers.append(PerplexityProvider(openrouter_key, config))
            else:
                # Use generic OpenRouter provider for other models
                providers.append(OpenRouterProvider(
                    openrouter_key, 
                    provider_config["openrouter_model"],
                    provider_name,
                    config
                ))
        else:
            # Use direct API if key is available
            api_key = config["api_keys"].get(provider_name)
            if not api_key:
                print(f"Warning: {provider_name} direct API enabled but no API key found, skipping...", flush=True)
                continue
                
            # Create appropriate provider class
            if provider_name == "openai":
                providers.append(OpenAIProvider(api_key, config))
            elif provider_name == "gemini":
                providers.append(GeminiProvider(api_key, config))
    
    return providers

async def main():
    parser = argparse.ArgumentParser(description="Query multiple AI providers concurrently")
    parser.add_argument("markdown_file", help="Path to the markdown file containing the prompt")
    parser.add_argument("--search-mode", choices=["web", "none"],
                      help="Search mode: web (web search), none (no web search)")
    parser.add_argument("--output-dir",
                      help="Output directory for responses")
    parser.add_argument("--config", default="consensus_config.json",
                      help="Path to configuration file (default: consensus_config.json)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set defaults from config if not provided
    if not args.search_mode:
        args.search_mode = config["settings"]["default_search_mode"]
    if not args.output_dir:
        args.output_dir = config["settings"]["default_output_dir"]
    
    # Check if input file exists
    if not os.path.exists(args.markdown_file):
        print(f"Error: File '{args.markdown_file}' not found")
        sys.exit(1)
    
    # Read the prompt
    try:
        with open(args.markdown_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Setup providers
    providers = setup_providers(config)
    available_providers = [p for p in providers if p.available]
    
    if not available_providers:
        print("Error: No API keys found. Please set environment variables:")
        print("- OPENAI_API_KEY for OpenAI")
        print("- GEMINI_API_KEY for Gemini")
        print("- OPENROUTER_API_KEY for Perplexity (via OpenRouter)")
        print("  (or PERPLEXITY_API_KEY for backward compatibility)")
        sys.exit(1)
    
    print(f"Available providers: {', '.join(p.name for p in available_providers)}")
    print(f"Search mode: {args.search_mode}")
    
    # Query all providers
    print("Querying providers...", flush=True)
    responses = await query_all_providers(available_providers, prompt, args.search_mode)
    
    if not responses:
        print("Error: No responses received from any provider")
        sys.exit(1)
    
    # Consolidate responses
    consolidated = consolidate_responses(responses, prompt)
    
    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"consolidated-{timestamp}.md")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(consolidated)
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())