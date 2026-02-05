# File: src/agents/agent_a_graph.py
from typing import Dict, Any, List
from dotenv import load_dotenv
import json

from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from src.agents.state import AgentState
# Usamos la herramienta "Supervitaminada" que acabamos de crear
from src.agents.tools.reddit_tool import reddit_collect

load_dotenv()

# --- CONFIGURACIÓN DEL MODELO ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    convert_system_message_to_human=True,
    max_retries=2
)

# Ahora reddit_collect: busca, etiqueta idioma y guarda archivo.
tools = [reddit_collect]
llm_with_tools = llm.bind_tools(tools)

# --- PROMPT CON INGENIERÍA DE IDIOMAS ---
SYSTEM_PROMPT = """Eres el AGENTE DE RECOLECCIÓN (Agente A).
Tu misión es obtener datos de Reddit para una investigación y decidir el mejor idioma para ello.

**PROTOCOLO DE RUTEO DE IDIOMA (CRÍTICO):**
Analiza el tema del usuario y decide el idioma de búsqueda:
1. **Español ('es'):** Temas locales de LatAm/España (Economía, Política, Social).
   - Ej: "Economía Ecuador", "Elecciones México", "Seguridad Bogotá".
2. **Inglés ('en'):** Temas de Tecnología Global, Ciencia, Hardware, Software.
   - Ej: "Apple M5", "NVIDIA AI", "Climate Change Global", "Python coding".

**INSTRUCCIONES DE EJECUCIÓN:**
1. Traduce la consulta mentalmente si es necesario (ej: "Apple M5 opiniones" -> "Apple M5 reviews" para 'en').
2. EJECUTA la herramienta `reddit_collect` con:
   - `query`: Tu consulta optimizada.
   - `search_lang`: 'es' o 'en'.
   - `subreddit`: 'all' (o uno específico si el tema lo requiere).
   - si lo requiere puedes ampliar el tiempo de publicaciones, pero las mas recientes son mas importantes, si hay pocas puedes tomar publicaciones mas antiguas. tu decides que tan antiguas tomas las publicaciones y basate en que si hay mucha información reciente toma publicaciones recientes, si hay poca información reciente amplias la búsqueda a publicaciones mas antiguas para tener datos
3. Cuando la herramienta confirme ("status": "success"), responde SOLO con: LISTO_A
4. No charles, actúa
"""

def agent_a(state: AgentState) -> Dict[str, Any]:
    print("--- 🕵️‍♂️ AGENTE A (Recolección Inteligente) Pensando... ---")
    
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    incoming_msgs = state.get("messages", [])
    topic = state.get("research_topic") or "Tema general"
    ctx = dict(state.get("context") or {})
    
    # Variables para actualizar el estado principal
    state_updates = {}

    # --- LÓGICA DE CAPTURA DE CONTEXTO E IDIOMA ---
    # Revisamos si el último mensaje fue una herramienta exitosa
    if incoming_msgs and isinstance(incoming_msgs[-1], ToolMessage):
        last_tool = incoming_msgs[-1]
        try:
            # La herramienta devuelve JSON: {"path": "...", "detected_lang": "es", ...}
            tool_output = json.loads(last_tool.content)
            
            if isinstance(tool_output, dict):
                # 1. Capturar Ruta
                if "path" in tool_output:
                    saved_path = tool_output["path"]
                    ctx["last_collect_path"] = saved_path
                    print(f"   💾 [Contexto] Ruta capturada: {saved_path}")
                
                # 2. Capturar Idioma
                if "detected_lang" in tool_output:
                    lang = tool_output["detected_lang"]
                    state_updates["language"] = lang # Actualizamos el State global
                    print(f"   🌍 [Contexto] Idioma fijado en State: {lang.upper()}")

        except Exception as e:
            print(f"   ⚠️ Warning leyendo output de herramienta: {e}")

    # --- TRIGGER PARA GEMINI ---
    trigger_message = HumanMessage(content=f"INVESTIGACIÓN: Busca información sobre '{topic}' y guarda los datos.")

    if not incoming_msgs:
        msgs.append(trigger_message)
    elif isinstance(incoming_msgs[0], AIMessage):
        # Fix si el historial quedó corrupto
        msgs.append(trigger_message)
        msgs.extend(incoming_msgs)
    else:
        msgs.extend(incoming_msgs)

    # Invocación
    try:
        res = llm_with_tools.invoke(msgs)
    except Exception as e:
        print(f"❌ Error invocando Gemini (Agent A): {e}")
        return {"messages": [AIMessage(content="Error API.")]}

    # Retornamos mensajes nuevos + contexto actualizado + idioma detectado
    return {
        "messages": [res], 
        "context": ctx, 
        **state_updates # Esto inyecta "language": "es/en" al estado principal
    }

def agent_a_should_continue(state: AgentState):
    if not state["messages"]: return END
    last = state["messages"][-1]

    # Debug visual
    content_preview = str(last.content)[:50].replace("\n", " ") if hasattr(last, "content") else ""
    is_tool_call = bool(getattr(last, "tool_calls", None))
    print(f"   🧐 [Decisión A] Msg: {type(last).__name__} | Call?: {is_tool_call} | Txt: {content_preview}...")

    # 1. Si el LLM quiere usar una herramienta -> TOOLS
    if is_tool_call:
        return "tools"
    
    # 2. Si es respuesta final ("LISTO_A") -> AGENT_B (Limpieza)
    content = getattr(last, "content", "") or ""
    if "LISTO_A" in str(content).upper():
        return "agent_b"

    # 3. Si no ha terminado, sigue pensando (loop)
    return "agent_a"

# Exportamos nodos y constantes
AGENT_A_LLM_NODE = agent_a
AGENT_A_SHOULD_CONTINUE = agent_a_should_continue
AGENT_A_TOOLS_NODE = ToolNode(tools)