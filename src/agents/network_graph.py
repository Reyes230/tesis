# src/agents/network_graph.py
import os
import sys

# Aseguramos que el sistema pueda encontrar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph.graph import StateGraph, END
from src.agents.state import AgentState

# --- 1. IMPORTACIÓN DE AGENTES (Inteligencia / LLMs) ---

# Agente A (Recolección - ReAct)
from src.agents.agent_a_graph import (
    AGENT_A_LLM_NODE, 
    AGENT_A_TOOLS_NODE, 
    AGENT_A_SHOULD_CONTINUE
)

# [NUEVO] Agente SR (Síntesis y Reporte - ReAct)
# Reemplaza al antiguo agent_e_node
from src.agents.sr.synthesis_node import (
    AGENT_SR_NODE,
    AGENT_SR_TOOLS_NODE,
    AGENT_SR_SHOULD_CONTINUE
)

# --- 2. IMPORTACIÓN DE PIPELINES (Procesos Deterministas) ---
from src.agents.nodes import cleaning_node, sentiment_node 
from src.agents.trends.trend_node import trend_node

# =============================================================================
# DEFINICIÓN DEL GRAFO (WORKFLOW HÍBRIDO)
# =============================================================================

workflow = StateGraph(AgentState)

# ---------------------------------------------------------
# 1. AÑADIR NODOS
# ---------------------------------------------------------

# --- FASE 1: RECOLECCIÓN (Agente A) ---
workflow.add_node("agent_a", AGENT_A_LLM_NODE)
workflow.add_node("tools_a", AGENT_A_TOOLS_NODE)

# --- FASE 2: PROCESAMIENTO DETERMINISTA (Pipeline) ---
workflow.add_node("cleaning_pipeline", cleaning_node)
workflow.add_node("sentiment_pipeline", sentiment_node) 
workflow.add_node("trend_pipeline", trend_node)         

# --- FASE 3: SÍNTESIS ESTRATÉGICA (Agente SR) [NUEVO] ---
workflow.add_node("agent_sr", AGENT_SR_NODE)
workflow.add_node("tools_sr", AGENT_SR_TOOLS_NODE)

# ---------------------------------------------------------
# 2. DEFINIR FLUJO (ARISTAS / EDGES)
# ---------------------------------------------------------

# --- INICIO ---
workflow.set_entry_point("agent_a")

# --- LÓGICA CÍCLICA AGENTE A ---
workflow.add_conditional_edges("agent_a", AGENT_A_SHOULD_CONTINUE, {
    "tools": "tools_a",           # Si necesita herramientas
    "agent_a": "agent_a",         # Loop de pensamiento
    "agent_b": "cleaning_pipeline", # Salida exitosa hacia limpieza
    END: END
})
workflow.add_edge("tools_a", "agent_a")

# --- SECUENCIA LINEAL (Pipeline de Datos) ---
# Limpieza -> Sentimiento -> Tendencias
workflow.add_edge("cleaning_pipeline", "sentiment_pipeline")
workflow.add_edge("sentiment_pipeline", "trend_pipeline")

# --- CONEXIÓN HACIA EL AGENTE FINAL ---
# Cuando terminan las tendencias, despertamos al Agente SR
workflow.add_edge("trend_pipeline", "agent_sr")

# --- LÓGICA CÍCLICA AGENTE SR [NUEVO] ---
# El Agente SR también necesita pensar y usar herramientas
workflow.add_conditional_edges("agent_sr", AGENT_SR_SHOULD_CONTINUE, {
    "sr_tools": "tools_sr",       # Si necesita leer datos o calcular gravedad
    "agent_sr": "agent_sr",       # Vuelve a pensar con los datos
    END: END                      # Si dice "LISTO_SR", termina el flujo
})
workflow.add_edge("tools_sr", "agent_sr")

# ---------------------------------------------------------
# 3. COMPILACIÓN
# ---------------------------------------------------------
network_graph = workflow.compile()

# Diagrama visual
try:
    with open("network_graph_final.mmd", "w", encoding="utf-8") as f:
        f.write(network_graph.get_graph().draw_mermaid())
    print("✅ Grafo compilado y diagrama guardado en 'network_graph_final.mmd'")
except Exception as e:
    print(f"⚠️ Grafo compilado, pero error al guardar diagrama: {e}")