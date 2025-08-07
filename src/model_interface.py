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
        else:
            raise ValueError(f"Unknown model provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        return ["ollama", "openai", "deepseek", "gemini", "anthropic"]

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