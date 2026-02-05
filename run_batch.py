# run_batch.py
import time
import sys
import os

# Añadir ruta raíz al path para importar módulos src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage
from src.agents.network_graph import network_graph

# =============================================================================
# ⚙️ CONFIGURACIÓN DE BATCH
# =============================================================================

# Lista de temas para la tesis
TOPICS = [
    "DeepSeek R1 vs OpenAI o1 reasoning models",
    "Argentina economic measures milei impact",
    "Lakers defense rating nba updates",
    "Taylor Swift Eras Tour impact",
    "Indoor gardening tips for beginners",
    "Palworld new update features"
]

# 🕒 TIEMPO DE ESPERA ENTRE TEMAS (Segundos)
# Recomendado Free Tier: 60 a 90 segundos para recargar tokens
COOLDOWN_SECONDS = 90 

# =============================================================================
# 🚀 EJECUCIÓN
# =============================================================================

def run_batch():
    print("🚀 INICIANDO BATCH TEST DE TESIS")
    print(f"📋 Total de temas a procesar: {len(TOPICS)}")
    print(f"⏱️ Tiempo de espera entre temas: {COOLDOWN_SECONDS} segundos")
    print("=================================================\n")

    for i, topic in enumerate(TOPICS, 1):
        print(f"\n🔸 [{i}/{len(TOPICS)}] PROCESANDO TEMA: '{topic}'")
        print("-" * 50)
        
        # 1. Crear estado inicial
        initial_state = {
            "messages": [HumanMessage(content=f"Investiga sobre: {topic}")],
            "research_topic": topic,
            "context": {} # Contexto vacío al inicio
        }

        # 2. Ejecutar el grafo (Workflow completo)
        try:
            # invoke ejecuta todo el pipeline (A -> B -> Sentimiento -> Tendencias -> E)
            final_state = network_graph.invoke(initial_state)
            
            # Verificación rápida del resultado
            ctx = final_state.get("context", {})
            report_path = ctx.get("last_trends_path", "").replace("_trends_report.json", ".md") # Aprox
            
            # Buscamos mensaje del Agente E
            last_msg = final_state["messages"][-1].content if final_state["messages"] else "Sin respuesta"
            print(f"✅ TEMA FINALIZADO: {last_msg}")
            
        except Exception as e:
            print(f"❌ ERROR CRÍTICO EN TEMA '{topic}': {e}")
            import traceback
            traceback.print_exc()

        # 3. Enfriamiento (Rate Limiting)
        if i < len(TOPICS): # No esperar después del último
            print(f"💤 Enfriando API por {COOLDOWN_SECONDS} segundos para evitar bloqueo (Error 429)...")
            # Barra de progreso simple
            for _ in range(COOLDOWN_SECONDS):
                time.sleep(1)
                print(".", end="", flush=True)
            print("\n✅ Listo para el siguiente.\n")

    print("\n=================================================")
    print("🏁 BATCH TEST FINALIZADO. Revisa la carpeta: REPORTES_TESIS")

if __name__ == "__main__":
    run_batch()