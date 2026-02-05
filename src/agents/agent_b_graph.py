# ─────────────────────────────────────────────────────────────
# File: src/agents/agent_b_graph.py
# ─────────────────────────────────────────────────────────────
from typing import Annotated, Dict, Any, List
from dotenv import load_dotenv
import json
import os  # <--- NUEVO: Necesario para manejar rutas de archivos

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from src.agents.state import AgentState

from src.agents.tools.preprocess_tool import preprocess_posts

load_dotenv()

# --- CONFIGURACIÓN MODELO ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    max_retries=2
)

tools = [preprocess_posts]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = (
    "Eres un agente experto en preprocesamiento de datos (ETL).\n"
    "Tu objetivo es limpiar los datos crudos recolectados.\n"
    "INSTRUCCIONES:\n"
    "1. Ejecuta la herramienta `preprocess_posts` con los parámetros indicados.\n"
    "2. Cuando recibas la confirmación de la herramienta, responde SOLO con: LISTO_B\n"
)

def agent_b(state: AgentState) -> Dict[str, Any]:
    print("--- 🧹 AGENTE B (Limpieza) Pensando... ---")
    
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    
    ctx = state.get("context", {})
    topic = state.get("research_topic") or "Investigación"
    
    # 1. RECUPERAR LA RUTA EXACTA DE A
    raw_input_path = ctx.get("last_collect_path")
    
    if not raw_input_path:
        # Fallback genérico (mejor si falla A)
        raw_input_path = "data/collect/reddit_24h.jsonl"
        print("   ⚠️ Usando ruta por defecto (A no dejó ruta).")
    else:
        print(f"   📂 Leyendo archivo generado por A: {raw_input_path}")

    # --- CORRECCIÓN AQUÍ: NOMBRE DINÁMICO ---
    # Extraemos el nombre base del archivo que nos dio A (ej: "bitcoin_2024-12-01.jsonl")
    base_filename = os.path.basename(raw_input_path)
    # Quitamos la extensión (ej: "bitcoin_2024-12-01")
    filename_no_ext, _ = os.path.splitext(base_filename)
    
    # Creamos el nuevo nombre único para la salida
    # Ej: data/preprocessed/bitcoin_2024-12-01_cleaned.jsonl
    output_filename = f"{filename_no_ext}_cleaned.jsonl"
    output_path = os.path.join("data/preprocessed", output_filename)
    
    # Aseguramos que el directorio exista (por seguridad)
    os.makedirs("data/preprocessed", exist_ok=True)
    # ----------------------------------------

    # 2. INSTRUCCIÓN AL LLM
    instruction = HumanMessage(
        content=f"Los datos crudos sobre '{topic}' están en: '{raw_input_path}'.\n"
        f"TAREA: Ejecuta 'preprocess_posts' con input_path='{raw_input_path}' y output_path='{output_path}'."
    )
    msgs.append(instruction)

    # 3. RECUPERACIÓN DE HISTORIAL
    global_history = state.get("messages", [])
    agent_b_history = []
    
    for msg in reversed(global_history):
        if isinstance(msg, AIMessage) and "LISTO_A" in str(msg.content):
            break 
        if isinstance(msg, (ToolMessage, AIMessage)):
            agent_b_history.insert(0, msg)
            
    if agent_b_history:
        print(f"   🧠 [Memoria] Recuperados {len(agent_b_history)} mensajes previos de B.")
        msgs.extend(agent_b_history)

    try:
        res = llm_with_tools.invoke(msgs)
    except Exception as e:
        print(f"❌ Error invocando Gemini (Agent B): {e}")
        return {"messages": [AIMessage(content="Error API en Agente B")]}

    # --- ACTUALIZACIÓN CRÍTICA DEL CONTEXTO ---
    # Devolvemos el mensaje del LLM Y actualizamos el contexto
    # para que el Agente C sepa dónde está el archivo limpio.
    new_context = ctx.copy()
    new_context["last_processed_path"] = output_path
    
    return {
        "messages": [res],
        "context": new_context  # <--- Esto guarda la ruta para el futuro
    }

def agent_b_should_continue(state: AgentState):
    if not state["messages"]: return END
    last = state["messages"][-1]

    if isinstance(last, AIMessage):
        if getattr(last, "tool_calls", None):
            return "tools"
        
        content = last.content
        if isinstance(content, list):
            try: content = " ".join([p if isinstance(p, str) else p.get("text", "") for p in content])
            except: content = str(content)
            
        if "LISTO_B" in (content or "").upper():
            return END
            
    return END

AGENT_B_LLM_NODE = agent_b
AGENT_B_SHOULD_CONTINUE = agent_b_should_continue
AGENT_B_TOOLS_NODE = ToolNode(tools)