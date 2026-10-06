# Agente de Razonamiento Cíclico con Memoria Persistente

Agente ReAct construido con **LangGraph** que usa ciclos razonamiento → herramienta → razonamiento, con **memoria persistente por sesión** via `SqliteSaver`. El proveedor por defecto es **Ollama** (modelo local, sin costo de API).

## Arquitectura

```
[START]
   │
   ▼
┌──────┐   tool_calls?   ┌───────┐
│model │ ─────── sí ──▶  │ tools │
│      │ ◀─────────────  │       │
└──────┘                 └───────┘
   │
   │ no tool_calls
   ▼
 [END]
```

- **`model`**: invoca el LLM con el historial de mensajes (`MessagesState`)
- **`tools`**: ejecuta las herramientas con `ToolNode`
- **Arista condicional** (`tools_condition`): decide si continuar con herramientas o terminar
- **`SqliteSaver`** (`memory.db`): persiste el estado por `thread_id`

## Herramientas

| Herramienta | Descripción |
|---|---|
| `buscar_cliente(nombre)` | Busca cliente por nombre (parcial, insensible a mayúsculas) |
| `buscar_pedidos(cliente_id)` | Lista todos los pedidos y el monto total del cliente |
| `obtener_detalle_pedido(pedido_id)` | Devuelve todos los campos de un pedido específico |

## Requisitos previos

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- [Ollama](https://ollama.com/) *(si usas el proveedor por defecto)*

### Instalar Ollama y descargar el modelo

```bash
# macOS
brew install ollama

# Iniciar el servidor Ollama
ollama serve

# Descargar el modelo (en otra terminal)
ollama pull llama3.1
```

> Otros modelos compatibles con tool-calling: `llama3.2`, `qwen2.5`, `mistral-nemo`

## Instalación

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPO>
cd agente-razonamiento-ciclico

# 2. Instalar dependencias con Poetry
poetry install

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env si querés cambiar el modelo o proveedor
```

### Usar OpenAI o Anthropic en lugar de Ollama

Editar `.env`:

```env
# Para OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Para Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Instalar las dependencias opcionales:

```bash
poetry install --extras openai       # solo OpenAI
poetry install --extras anthropic    # solo Anthropic
poetry install --extras all          # ambos
```

## Ejecución

### Demo predefinida (razonamiento multi-paso)

```bash
poetry run python main.py
```

Ejecuta 3 turnos conversacionales con el mismo `thread_id`:
1. El agente llama **≥2 herramientas** para responder "¿Cuántos pedidos tuvo Carlos Ruiz?"
2. Recuerda el contexto y consulta el detalle del último pedido
3. Responde sin herramientas usando la memoria de la sesión

La traza completa se guarda en `traces/trace_<thread_id>.json`.

### Demo con thread_id personalizado

```bash
poetry run python main.py --thread mi-sesion-test
```

### Modo interactivo

```bash
poetry run python main.py --interactive
```

Conversación libre por consola. Usa `--thread mi-sesion` para continuar una sesión previa.

## Ejemplo de traza ReAct

Ver [`traces/example_trace.json`](traces/example_trace.json) para la traza completa.

Resumen del flujo:

```
Usuario: "¿Cuántos pedidos tuvo el cliente Carlos Ruiz y cuál fue el total?"

  → Herramienta: buscar_cliente({"nombre": "Carlos Ruiz"})
  ← { cliente_id: 102, nombre: "Carlos Ruiz", ciudad: "Córdoba" }

  → Herramienta: buscar_pedidos({"cliente_id": 102})
  ← { total_pedidos: 3, monto_total: 14500.0, pedidos: [...] }

Agente: "Carlos Ruiz tuvo 3 pedidos por un total de $14.500."

─────────────────────────── (mismo thread_id) ───────────────────────────

Usuario: "¿Y cuál fue el último pedido?"

  → Herramienta: obtener_detalle_pedido({"pedido_id": 5})
  ← { producto: "Teclado Mecánico Keychron K2", monto: 3000, estado: "pendiente" }

Agente: "El último pedido fue el #5: Teclado Mecánico Keychron K2 por $3.000, estado pendiente."

─────────────────────────────────────────────────────────────────────────

Usuario: "¿Está entregado ese pedido?"

  (Sin herramientas — el agente usa el contexto de la sesión)

Agente: "No, su estado actual es pendiente."
```

## Estructura del proyecto

```
.
├── src/
│   ├── __init__.py
│   ├── config.py       # Configuración desde variables de entorno
│   ├── tools.py        # Herramientas con @tool y docstrings descriptivos
│   └── agent.py        # StateGraph, get_llm, build_graph
├── traces/
│   └── example_trace.json   # Traza de ejemplo incluida en el repo
├── main.py             # Punto de entrada: demo y modo interactivo
├── pyproject.toml      # Dependencias (Poetry)
├── .env.example        # Plantilla de variables de entorno
└── .gitignore          # Excluye .env, *.db y trazas generadas
```

## Criterios de aceptación cumplidos

| Criterio | Implementación |
|---|---|
| Autonomía (sin if/else) | `tools_condition` decide automáticamente |
| Ciclo de retorno | Arista `tools → model` permite reintentos |
| Resiliencia de estado | `AsyncSqliteSaver` + `thread_id` |
| Python 3.12, type hints, asyncio | `pyproject.toml`, `async def`, `dict[...]` |
| `@tool` con docstrings | `src/tools.py` |
| `StateGraph(MessagesState)` | `src/agent.py` |
| `llm.bind_tools()` | `main.py` |
| `SqliteSaver` + `thread_id` | `main.py` (AsyncSqliteSaver) |
| `recursion_limit` definido | Config + `run_config` en `main.py` |
| ≥2 llamadas a herramientas | Turno 1: `buscar_cliente` + `buscar_pedidos` |
| Traza `.json` incluida | `traces/example_trace.json` |
| Sin API keys en el repo | Variables de entorno, `.gitignore` |
