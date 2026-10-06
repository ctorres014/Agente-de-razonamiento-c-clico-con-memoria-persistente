import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from src.config import Config
from src.tools import ALL_TOOLS

SYSTEM_PROMPT = """Eres un asistente de atención al cliente con acceso a herramientas de consulta.

Reglas estrictas para usar herramientas:
1. Si el usuario menciona un NOMBRE de cliente (no un ID numérico), SIEMPRE llama primero
   a buscar_cliente(nombre) para obtener el cliente_id antes de llamar a buscar_pedidos.
2. Para detalles de un pedido específico usa obtener_detalle_pedido(pedido_id).
3. Nunca inventes datos; usa solo la información que devuelven las herramientas.
4. Si una herramienta retorna un error, informa al usuario y sugiere alternativas.
"""

logger = logging.getLogger(__name__)


def get_llm(config: Config) -> BaseChatModel:
    """Instancia el LLM según el proveedor configurado."""
    provider = config.llm_provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.info(
            "Usando Ollama: model=%s, url=%s", config.ollama_model, config.ollama_base_url
        )
        return ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            temperature=0,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        logger.info("Usando OpenAI: model=%s", config.openai_model)
        return ChatOpenAI(
            model=config.openai_model,
            api_key=config.openai_api_key,
            temperature=0,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        logger.info("Usando Anthropic: model=%s", config.anthropic_model)
        return ChatAnthropic(
            model=config.anthropic_model,
            api_key=config.anthropic_api_key,
            temperature=0,
        )

    raise ValueError(f"Proveedor desconocido: {provider}")


def build_graph(llm_with_tools: BaseChatModel, checkpointer: Any) -> Any:
    """
    Construye y compila el StateGraph con ciclo ReAct.

    Topología:
        [START] → model ──(has tool_calls)──→ tools → model → ...
                        └──(no tool_calls)──→ [END]
    """

    async def call_model(state: MessagesState) -> dict:
        """Nodo del modelo: invoca el LLM con el historial de mensajes actual."""
        logger.debug("Invocando modelo con %d mensajes", len(state["messages"]))
        # Inyecta el system prompt solo si el primer mensaje no lo es ya
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)

    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")

    return graph.compile(checkpointer=checkpointer)
