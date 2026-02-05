# src/pipeline_controller.py
import time
from src.agents.network_graph import network_graph

def run_analysis_pipeline(topic: str, status_callback=None):
    """
    Ejecuta el flujo completo (Agents + Pipelines) para un tema dado.
    Usa 'stream' para reportar progreso en tiempo real a Streamlit.
    """
    
    # 1. Estado Inicial
    initial_state = {
        "messages": [],
        "research_topic": topic,
        "context": {}
    }

    if status_callback: 
        status_callback(f"🚀 Iniciando protocolos para: {topic}...")
    
    # 2. Ejecución Streaming (Paso a paso)
    # network_graph.stream() nos permite ver qué nodo se acaba de ejecutar
    try:
        for output in network_graph.stream(initial_state):
            for node_name, value in output.items():
                
                # Traducimos el nombre técnico del nodo a mensaje para humanos
                if node_name == "agent_a":
                    if status_callback: status_callback("🕵️ Agente A: Recolectando inteligencia en Reddit...")
                
                elif node_name == "cleaning_pipeline":
                    if status_callback: status_callback("🧹 Nodo de Limpieza: Eliminando ruido y URLs...")
                
                elif node_name == "sentiment_pipeline":
                    if status_callback: status_callback("🧠 Nodo Neural: Analizando sentimiento (RoBERTa)...")
                
                elif node_name == "trend_pipeline":
                    if status_callback: status_callback("🌌 Nodo de Clusters: Detectando comunidades (BERTopic)...")
                
                elif node_name == "agent_sr":
                    # El Agente SR tarda un poco más porque piensa e investiga
                    if status_callback: status_callback("🌐 Agente SR: Investigando en Internet y redactando informe estratégico...")
        
        # 3. Finalización
        if status_callback: status_callback("✅ Misión Cumplida. Generando visualizaciones...")
        time.sleep(1) # Pausa dramática para que el usuario vea el check verde
        return True

    except Exception as e:
        print(f"❌ Error crítico en el pipeline: {e}")
        if status_callback: status_callback(f"❌ Error del Sistema: {e}")
        return False