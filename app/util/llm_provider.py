from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.llms import Ollama


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str):
        self.llm = Ollama(base_url=base_url, model=model)

    async def generate(self, prompt: str) -> str:
        return self.llm.predict(prompt)


class LLMFactory:
    @staticmethod
    def create_provider(provider_type: str, base_url: str, model: str) -> LLMProvider:
        if provider_type.lower() == "ollama":
            return OllamaProvider(base_url, model)
        raise ValueError(f"Unsupported LLM provider: {provider_type}")
