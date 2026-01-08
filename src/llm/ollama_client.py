"""
Ollama LLM Client with Instructor Integration
Provides both standard text generation and structured output generation.
"""

import logging
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
import ollama
from openai import OpenAI
import instructor

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Wrapper for Ollama LLM with support for:
    - Standard text generation
    - Structured output via Instructor
    - Function calling
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 60,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Ollama model name (e.g., "llama3.1:8b", "mistral:7b")
            base_url: Ollama server URL
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Initialize Ollama client
        self.ollama_client = ollama.Client(host=base_url)

        # Initialize Instructor client for structured outputs
        openai_client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",  # Ollama doesn't need a real API key
        )
        self.instructor_client = instructor.from_openai(
            openai_client, mode=instructor.Mode.JSON
        )

        logger.info(f"Initialized OllamaClient with model: {model}")

    def test_connection(self) -> bool:
        """
        Test if Ollama server is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.ollama_client.list()
            logger.info("Successfully connected to Ollama server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate text response from Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stream: Whether to stream the response

        Returns:
            Generated text response

        Example:
            >>> client = OllamaClient()
            >>> messages = [{"role": "user", "content": "What is a contract?"}]
            >>> response = client.generate(messages)
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temp,
                    "num_predict": max_tok,
                },
                stream=stream,
            )

            if stream:
                # Return iterator for streaming
                return response
            else:
                return response["message"]["content"]

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> BaseModel:
        """
        Generate structured output using Instructor.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Pydantic model class for structured output
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Instance of response_model with parsed data

        Example:
            >>> from pydantic import BaseModel
            >>> class QueryClassification(BaseModel):
            ...     complexity: str
            ...     reasoning: str
            >>> client = OllamaClient()
            >>> messages = [{"role": "user", "content": "Classify this query"}]
            >>> result = client.generate_structured(messages, QueryClassification)
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = self.instructor_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=response_model,
                temperature=temp,
                max_tokens=max_tok,
            )
            return response

        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            raise

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate response with function/tool calling support.

        Args:
            messages: List of message dicts
            tools: List of tool definitions (OpenAI format)
            temperature: Override default temperature

        Returns:
            Response dict with potential tool calls

        Example:
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "search_documents",
            ...         "description": "Search legal documents",
            ...         "parameters": {...}
            ...     }
            ... }]
            >>> response = client.generate_with_tools(messages, tools)
        """
        temp = temperature if temperature is not None else self.temperature

        try:
            # Note: Ollama's tool calling support varies by model
            # llama3.1 and later models support tool calling
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                options={"temperature": temp},
            )
            return response

        except Exception as e:
            logger.error(f"Error in tool calling: {e}")
            raise

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text using Ollama's embedding models.
        Note: Most chat models don't support embeddings. Use separate embedding models.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Note:
            For production, use sentence-transformers instead (more reliable).
        """
        try:
            response = self.ollama_client.embeddings(
                model=self.model,  # Some models support this
                prompt=text,
            )
            return response["embedding"]

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def list_models(self) -> List[str]:
        """
        List available Ollama models.

        Returns:
            List of model names
        """
        try:
            response = self.ollama_client.list()
            # Supports latest ollama sdk
            if hasattr(response, "models"):
                return [m.model for m in response.models]
            # Fallback for older dictionary validation
            if isinstance(response, dict) and "models" in response:
                return [model["name"] for model in response["models"]]
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama registry.

        Args:
            model_name: Name of model to pull (e.g., "llama3.1:8b")

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model: {model_name}")
            self.ollama_client.pull(model_name)
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    def __repr__(self) -> str:
        return f"OllamaClient(model='{self.model}', base_url='{self.base_url}')"


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize client
    client = OllamaClient(model="llama3.1:8b")

    # Test connection
    if client.test_connection():
        print("✓ Ollama connection successful")

        # List available models
        models = client.list_models()
        print(f"Available models: {models}")

        # Test text generation
        messages = [
            {"role": "system", "content": "You are a helpful legal assistant."},
            {"role": "user", "content": "What is a contract in one sentence?"},
        ]

        try:
            response = client.generate(messages)
            print(f"\nText generation test:\n{response}")
        except Exception as e:
            print(f"Text generation failed: {e}")

        # Test structured output
        from pydantic import BaseModel, Field

        class LegalDefinition(BaseModel):
            term: str = Field(description="Legal term")
            definition: str = Field(description="Simple definition")
            confidence: float = Field(description="Confidence score 0-1")

        try:
            structured_messages = [
                {
                    "role": "user",
                    "content": "Define 'contract' as a legal term with confidence score.",
                }
            ]
            result = client.generate_structured(structured_messages, LegalDefinition)
            print(f"\nStructured output test:\n{result}")
        except Exception as e:
            print(f"Structured output failed: {e}")

    else:
        print("✗ Failed to connect to Ollama")
        print("Make sure Ollama is running: ollama serve")
