"""Componentes de IA baseados em LangChain.

Esta camada concentra a integração com o modelo local Ollama.
"""

from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


MODELO_OLLAMA = "phi3"


class ClassificacaoPergunta(BaseModel):
    """Classificação estruturada de uma pergunta de Ciência da Computação."""

    categoria: Literal[
        "algoritmos",
        "dados",
        "programacao",
        "sistemas",
        "redes",
        "software",
        "ia",
        "seguranca",
        "compiladores",
        "outros",
    ] = Field(description="Categoria principal da pergunta.")

    palavras_chave: list[str] = Field(
        description="Principais conceitos presentes na pergunta."
    )

    nivel: Literal["basico", "intermediario", "avancado"] = Field(
        description="Nível estimado da pergunta."
    )


def criar_modelo() -> ChatOllama:
    """Cria o modelo conversacional local usado pelo assistente."""
    return ChatOllama(
        model=MODELO_OLLAMA,
        temperature=0,
    )


def criar_classificador():
    """Cria um modelo LangChain com saída estruturada via Pydantic."""
    return criar_modelo().with_structured_output(ClassificacaoPergunta)
