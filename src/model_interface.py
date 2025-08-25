import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class BaseModelInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        pass

class OllamaInterface(BaseModelInterface):
    def __init__(self):
        import ollama
        self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.model = os.getenv("OLLAMA_MODEL", "gemma3:12b")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens)
                }
            )
            return response['response']
        except Exception as e:
            print(f"Error generating with Ollama: {e}")
            return ""
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

class OpenAIInterface(BaseModelInterface):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-5")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating with OpenAI: {e}")
            return ""
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

class DeepSeekInterface(BaseModelInterface):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v3")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating with DeepSeek: {e}")
            return ""
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

class GeminiInterface(BaseModelInterface):
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            generation_config = genai.GenerationConfig(
                temperature=kwargs.get("temperature", self.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            print(f"Error generating with Gemini: {e}")
            return ""
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

class AnthropicInterface(BaseModelInterface):
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-1-20250805")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error generating with Anthropic: {e}")
            return ""
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

class VLLMInterface(BaseModelInterface):
    def __init__(self):
        from openai import OpenAI
        self.base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8001")
        self.api_key = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM doesn't require API key
        self.model_name = os.getenv("VLLM_MODEL_NAME", "meta-llama/Llama-2-7b-hf")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Initialize OpenAI client for vLLM's OpenAI-compatible API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.base_url}/v1"
        )
        
        # Test connection
        try:
            models = self.client.models.list()
            available_models = [model.id for model in models.data]
            if available_models:
                print(f"✓ Connected to vLLM server. Available models: {available_models}")
                # Use the first available model if not explicitly set
                if self.model_name not in available_models and available_models:
                    self.model_name = available_models[0]
                    print(f"  Using model: {self.model_name}")
            else:
                print(f"⚠️  vLLM server has no models loaded")
        except Exception as e:
            print(f"⚠️  Could not connect to vLLM server at {self.base_url}: {e}")
            print("  Make sure to start the server with: python scripts/start_vllm_server.py --model <model_name>")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", 0.9),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
                stop=kwargs.get("stop", None)
            )
            return response.choices[0].text
        except Exception as e:
            # Try chat completion format as fallback
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=kwargs.get("top_p", 0.9),
                    frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                    presence_penalty=kwargs.get("presence_penalty", 0.0),
                    stop=kwargs.get("stop", None)
                )
                return response.choices[0].message.content
            except Exception as e2:
                print(f"Error generating with vLLM: {e}, {e2}")
                return ""
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        # vLLM supports async batch processing, but for simplicity we'll use sequential calls
        # For production, consider using async/await with vLLM's batch API
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

class TransformersInterface(BaseModelInterface):
    def __init__(self):
        import requests
        self.session = requests.Session()
        
        # Get model name and determine endpoint
        self.model_name = os.getenv("TRANSFORMERS_MODEL_NAME", "kimi-vl")
        
        # Map model names to their deployment ports
        model_endpoints = {
            "kimi-vl": "http://localhost:8010",
            "openvla": "http://localhost:8011", 
            "deepseek-vl": "http://localhost:8012"
        }
        
        # Use TRANSFORMERS_ENDPOINT if set, otherwise use model-specific endpoint
        if os.getenv("TRANSFORMERS_ENDPOINT"):
            self.base_url = os.getenv("TRANSFORMERS_ENDPOINT")
        elif self.model_name.lower() in model_endpoints:
            self.base_url = model_endpoints[self.model_name.lower()]
        else:
            # Default to port 8000 for generic transformers server
            self.base_url = os.getenv("TRANSFORMERS_BASE_URL", "http://localhost:8000")
        
        self.max_tokens = int(os.getenv("MAX_TOKENS", "512"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Test connection using OpenAI-compatible endpoint
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                print(f"✓ Connected to Transformers server at {self.base_url}")
                if "data" in models_data and models_data["data"]:
                    available_models = [model["id"] for model in models_data["data"]]
                    print(f"  Available models: {available_models}")
                    # Use the first available model if our specified model isn't available
                    if self.model_name not in available_models and available_models:
                        print(f"  Using available model: {available_models[0]}")
                        self.model_name = available_models[0]
                else:
                    print("  No models available")
            else:
                print(f"⚠️  Transformers server returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Could not connect to Transformers server at {self.base_url}: {e}")
            print(f"  Make sure the model server is running with OpenAI-compatible API.")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            # Use OpenAI-compatible chat completions format (like LlamaFactory)
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": kwargs.get("model", self.model_name) or "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                    "top_p": kwargs.get("top_p", 0.9),
                    "stream": False
                },
                timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                # Handle OpenAI-compatible response format
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    print(f"Unexpected response format: {result}")
                    return ""
            else:
                print(f"Error from Transformers server: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            print(f"Error generating with Transformers: {e}")
            return ""
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        # Most deployed models don't have batch endpoints, so we'll do sequential generation
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results

class ModelFactory:
    @staticmethod
    def create_model(provider: Optional[str] = None) -> BaseModelInterface:
        if provider is None:
            provider = os.getenv("MODEL_PROVIDER", "ollama")
        
        provider = provider.lower()
        
        if provider == "ollama":
            return OllamaInterface()
        elif provider == "openai":
            return OpenAIInterface()
        elif provider == "deepseek":
            return DeepSeekInterface()
        elif provider == "gemini":
            return GeminiInterface()
        elif provider == "anthropic":
            return AnthropicInterface()
        elif provider == "transformers":
            return TransformersInterface()
        elif provider == "vllm":
            return VLLMInterface()
        else:
            raise ValueError(f"Unknown model provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        return ["ollama", "openai", "deepseek", "gemini", "anthropic", "transformers", "vllm"]

if __name__ == "__main__":
    print("Testing Model Interface...")
    print(f"Current provider: {os.getenv('MODEL_PROVIDER', 'ollama')}")
    
    try:
        model = ModelFactory.create_model()
        
        test_prompt = "What is 2 + 2? Answer with just the number."
        print(f"\nTest prompt: {test_prompt}")
        
        response = model.generate(test_prompt, temperature=0.1)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure your model is running and API keys are configured in .env")
