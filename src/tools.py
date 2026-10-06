import json

from langchain_core.tools import tool

# ──────────────────────────────────────────────
# Base de datos simulada (in-memory)
# ──────────────────────────────────────────────

_CLIENTES: dict[int, dict] = {
    101: {"nombre": "Ana García", "email": "ana.garcia@example.com", "ciudad": "Buenos Aires"},
    102: {"nombre": "Carlos Ruiz", "email": "carlos.ruiz@example.com", "ciudad": "Córdoba"},
    103: {"nombre": "María López", "email": "maria.lopez@example.com", "ciudad": "Rosario"},
    104: {"nombre": "Juan Pérez", "email": "juan.perez@example.com", "ciudad": "Mendoza"},
}

_PEDIDOS: list[dict] = [
    {
        "pedido_id": 1,
        "cliente_id": 101,
        "producto": "Auriculares Bluetooth Sony",
        "monto": 2500.0,
        "fecha": "2024-01-10",
        "estado": "entregado",
    },
    {
        "pedido_id": 2,
        "cliente_id": 102,
        "producto": "Laptop Dell XPS 15",
        "monto": 7000.0,
        "fecha": "2024-01-15",
        "estado": "entregado",
    },
    {
        "pedido_id": 3,
        "cliente_id": 102,
        "producto": 'Monitor LG UltraWide 27"',
        "monto": 4500.0,
        "fecha": "2024-02-20",
        "estado": "entregado",
    },
    {
        "pedido_id": 4,
        "cliente_id": 103,
        "producto": "Mouse Logitech MX Master 3",
        "monto": 1800.0,
        "fecha": "2024-02-25",
        "estado": "entregado",
    },
    {
        "pedido_id": 5,
        "cliente_id": 102,
        "producto": "Teclado Mecánico Keychron K2",
        "monto": 3000.0,
        "fecha": "2024-03-10",
        "estado": "pendiente",
    },
    {
        "pedido_id": 6,
        "cliente_id": 101,
        "producto": "Webcam Logitech C920 HD",
        "monto": 3200.0,
        "fecha": "2024-03-15",
        "estado": "entregado",
    },
    {
        "pedido_id": 7,
        "cliente_id": 104,
        "producto": "Disco SSD Samsung 1TB",
        "monto": 4000.0,
        "fecha": "2024-04-01",
        "estado": "en camino",
    },
]

# ──────────────────────────────────────────────
# Herramientas LangChain
# ──────────────────────────────────────────────


@tool
def buscar_cliente(nombre: str) -> str:
    """
    Busca clientes en la base de datos por nombre (total o parcial).

    Usa esta herramienta cuando el usuario proporcione el nombre de un cliente
    y necesites obtener su ID numérico u otros datos del perfil (email, ciudad).
    También útil cuando el usuario no conoce el ID numérico del cliente.

    La búsqueda es insensible a mayúsculas/minúsculas y acepta coincidencias
    parciales (ej: "Carlos" encuentra "Carlos Ruiz").

    Args:
        nombre: Nombre completo o parcial del cliente a buscar.

    Returns:
        JSON con lista de clientes encontrados (cliente_id, nombre, email, ciudad)
        y el total de coincidencias. Retorna error si no hay resultados.

    Ejemplos:
        buscar_cliente("Carlos") → encuentra al cliente con nombre "Carlos Ruiz"
        buscar_cliente("García") → encuentra al cliente con nombre "Ana García"
    """
    nombre_lower = nombre.lower().strip()
    resultados = [
        {"cliente_id": cid, **datos}
        for cid, datos in _CLIENTES.items()
        if nombre_lower in datos["nombre"].lower()
    ]

    if not resultados:
        return json.dumps(
            {
                "error": f"No se encontró ningún cliente con nombre '{nombre}'. "
                "Verifica la ortografía o usa un nombre más corto.",
                "clientes": [],
                "total_encontrados": 0,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {"clientes": resultados, "total_encontrados": len(resultados)},
        ensure_ascii=False,
    )


@tool
def buscar_pedidos(cliente_id: int) -> str:
    """
    Recupera todos los pedidos registrados para un cliente, identificado por su ID.

    Usa esta herramienta para responder preguntas como:
    "¿Cuántos pedidos hizo X?", "¿Cuánto gastó X en total?",
    "¿Qué productos compró X?", "¿Cuál es el historial de compras de X?".

    IMPORTANTE: Si el usuario proporcionó el nombre del cliente (no el ID),
    llama primero a buscar_cliente para obtener el cliente_id numérico.

    Args:
        cliente_id: Identificador numérico único del cliente (ej: 101, 102, 103).

    Returns:
        JSON con la lista completa de pedidos, el total de pedidos y el monto
        acumulado. Incluye pedido_id, producto, monto, fecha y estado de cada
        pedido. Retorna error si el cliente_id no existe.

    Ejemplo:
        buscar_pedidos(102) → retorna los 3 pedidos del cliente 102 con total $14.500
    """
    if cliente_id not in _CLIENTES:
        return json.dumps(
            {
                "error": f"No existe ningún cliente con ID {cliente_id}. "
                "Usa buscar_cliente para obtener el ID correcto.",
                "pedidos": [],
                "total_pedidos": 0,
                "monto_total": 0.0,
            },
            ensure_ascii=False,
        )

    pedidos_cliente = [p for p in _PEDIDOS if p["cliente_id"] == cliente_id]
    monto_total = sum(p["monto"] for p in pedidos_cliente)

    return json.dumps(
        {
            "cliente_id": cliente_id,
            "nombre_cliente": _CLIENTES[cliente_id]["nombre"],
            "pedidos": pedidos_cliente,
            "total_pedidos": len(pedidos_cliente),
            "monto_total": monto_total,
        },
        ensure_ascii=False,
    )


@tool
def obtener_detalle_pedido(pedido_id: int) -> str:
    """
    Obtiene información completa de un pedido específico por su ID único.

    Usa esta herramienta cuando el usuario pregunte por el detalle de un pedido
    particular: "¿cuál fue el último pedido?", "¿en qué estado está el pedido N?",
    "¿qué compró exactamente en ese pedido?".

    Para saber el ID del último pedido de un cliente, primero llama a
    buscar_pedidos para obtener la lista y luego usa el mayor pedido_id o
    la fecha más reciente.

    Args:
        pedido_id: Identificador numérico único del pedido (ej: 1, 2, 3, 4, 5...).

    Returns:
        JSON con todos los campos del pedido: pedido_id, cliente_id,
        nombre_cliente, producto, monto, fecha y estado actual.
        Retorna error si el pedido_id no existe.

    Ejemplos:
        obtener_detalle_pedido(5) → Teclado Mecánico de Carlos Ruiz, $3.000, pendiente
        obtener_detalle_pedido(1) → Auriculares de Ana García, $2.500, entregado
    """
    pedido = next((p for p in _PEDIDOS if p["pedido_id"] == pedido_id), None)

    if not pedido:
        return json.dumps(
            {
                "error": f"No existe ningún pedido con ID {pedido_id}. "
                "Usa buscar_pedidos para ver los IDs disponibles."
            },
            ensure_ascii=False,
        )

    cliente = _CLIENTES.get(pedido["cliente_id"], {})
    return json.dumps(
        {**pedido, "nombre_cliente": cliente.get("nombre", "Desconocido")},
        ensure_ascii=False,
    )


ALL_TOOLS = [buscar_cliente, buscar_pedidos, obtener_detalle_pedido]
