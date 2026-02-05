# ─────────────────────────────────────────────────────────────
# File: run.py (UBICACIÓN: En la raíz del proyecto 'tesis')
# ─────────────────────────────────────────────────────────────
import sys
import os
import traceback

print("🔧 Configurando entorno de ejecución...")

# 1. DEFINICIÓN DE RUTAS
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")

# Rutas específicas que dan problemas de importación
agents_dir = os.path.join(src_dir, "agents")
tools_dir = os.path.join(agents_dir, "tools") 
trends_dir = os.path.join(agents_dir, "trends") # <-- NUEVA RUTA AGREGADA

# 2. DIAGNÓSTICO DE ESTRUCTURA
# ---------------------------------------------------------
paths_to_check = {
    "src": src_dir,
    "agents": agents_dir,
    "tools": tools_dir,
    "trends": trends_dir
}

for name, path in paths_to_check.items():
    if not os.path.exists(path):
        print(f"❌ ERROR CRÍTICO: No existe la carpeta '{name}' en: {path}")
        if name == "tools": print("Confirma que esté en src/agents/tools")
        if name == "trends": print("Confirma que esté en src/agents/trends")
        sys.exit(1)

# Crear __init__.py en tools si falta
init_tools = os.path.join(tools_dir, "__init__.py")
if not os.path.exists(init_tools):
    try:
        with open(init_tools, "w") as f: f.write("# Tools pkg")
    except: pass

# Crear __init__.py en trends si falta
init_trends = os.path.join(trends_dir, "__init__.py")
if not os.path.exists(init_trends):
    try:
        with open(init_trends, "w") as f: f.write("# Trends pkg")
    except: pass

# 3. CONFIGURACIÓN DEL PATH (PRIORIDAD ALTA)
# ---------------------------------------------------------
# Agregamos las carpetas al inicio del sys.path para que Python encuentre los módulos
# sin importar desde dónde se llamen.

sys.path.insert(0, src_dir)     # Para 'from src...'
sys.path.insert(0, agents_dir)  # Para 'from tools...' (si tools está en agents)
sys.path.insert(0, trends_dir)  # Para 'import config', 'import topic_engine' dentro de trends

print(f"✅ Rutas configuradas. Agregado al path:\n   - {src_dir}\n   - {agents_dir}\n   - {trends_dir}")

# 4. IMPORTACIONES Y EJECUCIÓN
# ---------------------------------------------------------
try:
    from dotenv import load_dotenv
    # Ahora sí debería encontrar 'config' dentro de trends
    from src.agents.network_graph import network_graph
    
except ImportError as e:
    print("\n❌ ERROR DE IMPORTACIÓN:")
    print(f"Detalle: {e}")
    print("TIP: Si el error es 'No module named config', verifica que 'config.py' exista dentro de src/agents/trends/")
    sys.exit(1)

# Cargar variables de entorno (.env)
load_dotenv()

def run_test():
    print("🚀 INICIANDO PRUEBA DE FLUJO (GEMINI POWERED) 🚀")
    print("=================================================")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: No se encontró GOOGLE_API_KEY en .env")
        return

    # Input inicial de prueba
    initial_state = {
        "messages": [],
        "research_topic": "Inteligencia Artificial en Finanzas",
        "context": {"max_turns": 5}, 
        "current_file_path": None
    }

    print(f"📝 Tema: {initial_state['research_topic']}")
    print("⏳ Ejecutando grafo...")

    step_count = 0
    try:
        for event in network_graph.stream(initial_state):
            step_count += 1
            for node_name, node_output in event.items():
                print(f"\n--- [Paso {step_count}] Nodo Finalizado: {node_name} ---")
                
                if node_name == "adapter_b":
                    path = node_output.get('current_file_path', 'DESCONOCIDO')
                    print(f"   👀 [Adaptador] Ruta: {path}")
                
                if node_name == "sentiment_pipeline":
                    stats = node_output.get('processing_stats', {})
                    print(f"   ✅ [Sentimiento] OK. Stats: {stats}")
                    
                if node_name == "trend_pipeline":
                    print(f"   ✅ [Tendencias] OK.")
                    
                if node_name == "join_node":
                    print("   🏁 [Sincronización] Listo para reporte.")

    except Exception as e:
        print(f"\n❌ ERROR EN EJECUCIÓN: {e}")
        traceback.print_exc()

    print("\n=================================================")
    print("✅ FIN")

if __name__ == "__main__":
    run_test()