"""
Punto de entrada del agente de razonamiento cíclico.

Ejecutar:
    python main.py                    # demo predefinida + guarda traza
    python main.py --interactive      # modo interactivo por consola
    python main.py --thread mi-sesion # demo con thread_id personalizado
"""

import asyncio
import json
import logging
import sys
import uuid
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.agent import build_graph, get_llm
from src.config import Config
from src.tools import ALL_TOOLS

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRACES_DIR = Path("traces")


# ──────────────────────────────────────────────
# Extracción de traza ReAct
# ──────────────────────────────────────────────


def _parse_json_safe(text: str) -> Any:
    """Intenta parsear JSON; si falla devuelve el string original."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def extract_steps_from_chunks(chunks: list[dict]) -> list[dict]:
    """
    Convierte los chunks de astream (stream_mode='updates') en pasos de traza.

    Cada chunk es {node_name: {messages: [...]}}. Acumula tool_calls del nodo
    'model' y los empareja con los ToolMessages del nodo 'tools' via tool_call_id.
    """
    steps: list[dict] = []
    pending_calls: dict[str, dict] = {}

    for chunk in chunks:
        for node_name, state_update in chunk.items():
            for msg in state_update.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        step: dict = {
                            "type": "tool_call",
                            "tool_name": tc["name"],
                            "tool_args": tc["args"],
                        }
                        steps.append(step)
                        pending_calls[tc["id"]] = step

                elif isinstance(msg, ToolMessage):
                    matched = pending_calls.pop(msg.tool_call_id, None)
                    if matched is not None:
                        matched["tool_result"] = _parse_json_safe(msg.content)

                elif isinstance(msg, AIMessage) and msg.content:
                    steps.append({"type": "ai_response", "content": msg.content})

    return steps


# ──────────────────────────────────────────────
# Runner de interacción
# ──────────────────────────────────────────────


async def run_interaction(
    graph: Any,
    message: str,
    thread_id: str,
    recursion_limit: int = 10,
) -> tuple[str, list[dict]]:
    """
    Ejecuta una interacción con el agente y devuelve (respuesta, pasos_de_traza).

    Usa stream_mode='updates' para capturar cada paso del ciclo ReAct.
    """
    run_config: dict = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    input_data = {"messages": [HumanMessage(content=message)]}

    chunks: list[dict] = []
    final_response = ""

    logger.info("── Turno: %s", message[:80])

    async for chunk in graph.astream(input_data, config=run_config, stream_mode="updates"):
        chunks.append(chunk)
        for node_name, state_update in chunk.items():
            for msg in state_update.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        logger.info(
                            "  → Herramienta: %s(%s)",
                            tc["name"],
                            json.dumps(tc["args"], ensure_ascii=False),
                        )
                elif isinstance(msg, ToolMessage):
                    logger.info("  ← Resultado: %s", msg.content[:120])
                elif isinstance(msg, AIMessage) and msg.content:
                    final_response = msg.content
                    logger.info("  ✓ Respuesta: %s", msg.content[:120])

    steps = extract_steps_from_chunks(chunks)
    return final_response, steps


# ──────────────────────────────────────────────
# Demo predefinida (razonamiento multi-paso)
# ──────────────────────────────────────────────

DEMO_INTERACTIONS = [
    # Turno 1: dos herramientas en un solo ciclo (buscar_pedidos + obtener_detalle_pedido)
    "Dame el resumen de pedidos del cliente 102 y también el detalle de su pedido más reciente.",
    # Turno 2: mismo thread — recuerda el contexto, usa buscar_cliente
    "¿Qué datos tiene registrados la cliente llamada Ana García?",
    # Turno 3: pura memoria, sin herramientas
    "¿Está entregado el pedido que mencionaste antes?",
]


async def run_demo(thread_id: str, config: Config) -> None:
    """
    Ejecuta la demo predefinida de razonamiento multi-paso y persiste la traza.

    Turno 1: invoca buscar_pedidos + obtener_detalle_pedido (>=2 herramientas).
    Turno 2: invoca buscar_cliente (usa memoria del thread para contexto).
    Turno 3: responde desde la memoria de sesión sin llamar herramientas.
    """
    llm = get_llm(config)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    async with AsyncSqliteSaver.from_conn_string(config.db_path) as checkpointer:
        graph = build_graph(llm_with_tools, checkpointer)

        trace: dict = {
            "metadata": {
                "thread_id": thread_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "llm_provider": config.llm_provider,
                "model": getattr(config, f"{config.llm_provider}_model", "N/A"),
                "recursion_limit": config.recursion_limit,
                "tools": [t.name for t in ALL_TOOLS],
            },
            "interactions": [],
        }

        for turn_num, user_input in enumerate(DEMO_INTERACTIONS, start=1):
            print(f"\n{'='*60}")
            print(f"Turno {turn_num} — Usuario: {user_input}")
            print("=" * 60)

            response, steps = await run_interaction(
                graph, user_input, thread_id, config.recursion_limit
            )

            tool_calls_count = sum(1 for s in steps if s["type"] == "tool_call")

            trace["interactions"].append(
                {
                    "turn": turn_num,
                    "input": user_input,
                    "tool_calls_count": tool_calls_count,
                    "reasoning_steps": steps,
                    "final_response": response,
                }
            )

            print(f"\nAgente: {response}")

        _save_trace(trace, thread_id)


def _save_trace(trace: dict, thread_id: str) -> None:
    TRACES_DIR.mkdir(exist_ok=True)
    filename = TRACES_DIR / f"trace_{thread_id}.json"
    filename.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Traza guardada en: %s", filename)


# ──────────────────────────────────────────────
# Modo interactivo
# ──────────────────────────────────────────────


async def run_interactive(thread_id: str, config: Config) -> None:
    """Modo conversacional por consola usando el mismo thread_id para memoria persistente."""
    llm = get_llm(config)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    print(f"\nModo interactivo | thread_id: {thread_id}")
    print("Escribe tu mensaje o 'salir' para terminar.\n")

    async with AsyncSqliteSaver.from_conn_string(config.db_path) as checkpointer:
        graph = build_graph(llm_with_tools, checkpointer)

        while True:
            try:
                user_input = input("Usuario: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nHasta luego.")
                break

            if user_input.lower() in {"salir", "exit", "quit"}:
                break
            if not user_input:
                continue

            response, _ = await run_interaction(
                graph, user_input, thread_id, config.recursion_limit
            )
            print(f"\nAgente: {response}\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def cli_entry() -> None:
    asyncio.run(_main())


async def _main() -> None:
    parser = ArgumentParser(description="Agente de razonamiento cíclico con LangGraph")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Iniciar en modo conversacional por consola",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="thread_id de sesión (default: UUID aleatorio para demo, 'interactivo-01' para modo interactivo)",
    )
    args = parser.parse_args()

    config = Config()

    if args.interactive:
        thread_id = args.thread or "interactivo-01"
        await run_interactive(thread_id, config)
    else:
        thread_id = args.thread or f"demo-{uuid.uuid4().hex[:8]}"
        await run_demo(thread_id, config)


if __name__ == "__main__":
    asyncio.run(_main())
