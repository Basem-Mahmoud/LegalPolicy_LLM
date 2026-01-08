"""
LLM Client for interfacing with different language model providers.
Supports OpenAI, Anthropic, and local HuggingFace models.
"""

import os
from typing import List, Dict, Optional, Any
from enum import Enum
import yaml


class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


class LLMClient:
    """
    Unified client for interacting with different LLM providers.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize LLM client with configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.provider = ModelProvider(self.config["model"]["provider"])
        self.model_name = self.config["model"]["name"]
        self.temperature = self.config["model"]["temperature"]
        self.max_tokens = self.config["model"]["max_tokens"]

        self.client = self._initialize_client()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider."""
        if self.provider == ModelProvider.OPENAI:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            return OpenAI(api_key=api_key)

        elif self.provider == ModelProvider.ANTHROPIC:
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            return Anthropic(api_key=api_key)

        elif self.provider == ModelProvider.HUGGINGFACE:
            from transformers import pipeline
            return pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto"
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            tools: Optional list of tool definitions for function calling

        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == ModelProvider.OPENAI:
            return self._generate_openai(messages, temp, max_tok, tools)
        elif self.provider == ModelProvider.ANTHROPIC:
            return self._generate_anthropic(messages, temp, max_tok, tools)
        elif self.provider == ModelProvider.HUGGINGFACE:
            return self._generate_huggingface(messages, temp, max_tok)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """Generate response using OpenAI API."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)

        # Check if tool calls were made
        if tools and response.choices[0].message.tool_calls:
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls
            }

        return response.choices[0].message.content

    def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict]] = None
    ) -> str:
        """Generate response using Anthropic API."""
        # Separate system message from conversation
        system_message = ""
        conversation_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                conversation_messages.append(msg)

        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation_messages
        }

        if system_message:
            kwargs["system"] = system_message

        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        # Check if tool calls were made
        if tools and any(block.type == "tool_use" for block in response.content):
            tool_calls = [block for block in response.content if block.type == "tool_use"]
            text_content = " ".join([block.text for block in response.content if hasattr(block, "text")])
            return {
                "content": text_content,
                "tool_calls": tool_calls
            }

        return response.content[0].text

    def _generate_huggingface(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using HuggingFace model."""
        # Convert messages to prompt string
        prompt = self._messages_to_prompt(messages)

        response = self.client(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            return_full_text=False
        )

        return response[0]["generated_text"]

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert message format to prompt string for HuggingFace models."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Convenience method to generate with system prompt and user message.

        Args:
            system_prompt: System instruction
            user_message: User query
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.generate(messages, temperature, max_tokens)


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()

    client = LLMClient()
    response = client.generate_with_system_prompt(
        system_prompt="You are a helpful legal assistant.",
        user_message="What is a contract?"
    )
    print(response)
