import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama")
    )
    # Ollama
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    # OpenAI
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    # Anthropic
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    )
    # Agente
    db_path: str = field(
        default_factory=lambda: os.getenv("DB_PATH", "memory.db")
    )
    recursion_limit: int = field(
        default_factory=lambda: int(os.getenv("RECURSION_LIMIT", "10"))
    )

    def __post_init__(self) -> None:
        valid_providers = {"ollama", "openai", "anthropic"}
        if self.llm_provider not in valid_providers:
            raise ValueError(
                f"LLM_PROVIDER inválido: '{self.llm_provider}'. "
                f"Opciones: {valid_providers}"
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY es requerido cuando LLM_PROVIDER=openai")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY es requerido cuando LLM_PROVIDER=anthropic"
            )
