# test_agents.ipynb
from network_graph import network_graph

user_input = (
    "Recolecta publicaciones sobre quantum echoes en general, guarda los datos y preprocesa,"
    " no te limites en tiempo, no importa que sean publicaciones antiguas"
)

response = network_graph.invoke(
    {
        "messages": [{"role": "user", "content": user_input}],
        "context": {"max_turns": 10}
    },
    config={"configurable": {"thread_id": "network-run"}}
)

print("🟢 Salida del grafo completo:")
print(response["messages"][-1])
