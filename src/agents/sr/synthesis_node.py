# src/agents/sr/synthesis_node.py
from typing import Dict, Any
from dotenv import load_dotenv
import datetime
import os

from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from src.agents.state import AgentState
from src.agents.sr.tools import get_analysis_data, assess_severity, save_final_report
# from src.agents.tools.search_tool import web_search # DESACTIVADO POR AHORA

load_dotenv()

date_now = datetime.datetime.now().strftime("%Y-%m-%d")

# ---------------------------------------------------------
# 👤 VARIABLES DE CONTEXTO DE USUARIO (SIMULACIÓN)
# ---------------------------------------------------------
# PERFIL 1: Tú (Experto Tech + Finanzas/Deportes)
USER_PROFILE_MATIAS = """
Nombre: Matias
Profesión: Ingeniero en Ciencias de la Computación (En formación de Tesis)
Areas de Expertise: Inteligencia Artificial (Agentes, LLMs, NLP), Python, Basketball, Economía y Mercados Financieros.
Intereses: Tecnología de vanguardia, NBA, Inversiones.
Nivel de Lectura Preferido:
- Si el tema es Tech/AI/Economía/Basket: Altamente Técnico. Usa jerga (embeddings, yield curve, PER, re-act). Dame métricas duras.
- Otros temas: Explicativo y general.
"""

# PERFIL 2: Usuario Promedio (Para pruebas de control)
USER_PROFILE_GENERAL = """
Nombre: Juan Pérez
Profesión: Chef Gastronómico
Areas de Expertise: Cocina Internacional, Gestión de Restaurantes.
Intereses: Viajes, Cultura Pop, Política.
Nivel de Lectura Preferido: Divulgativo. Necesita analogías simples para entender tecnología o economía compleja.
"""

# 🎚️ VARIABLE ACTIVA: CAMBIA ESTO PARA PROBAR LA ADAPTABILIDAD
CURRENT_USER_CONTEXT = USER_PROFILE_GENERAL
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.3, # Bajamos un poco para ser más deterministas en la decisión de estilo
    max_retries=2
)

# Quitamos web_search de la lista
tools = [get_analysis_data, assess_severity, save_final_report] 
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------
# PROMPT DEL SISTEMA (ADAPTATIVO)
# ---------------------------------------------------------
def get_system_prompt(user_context):
    return f"""
FECHA ACTUAL: {date_now}
ROL: Eres un Analista Senior de Inteligencia Social con capacidad de adaptación. Tu trabajo es interpretar fenómenos humanos complejos y explicarlos al nivel exacto de tu lector.

OBJETIVO:
Generar un reporte que explique **POR QUÉ** la gente está hablando (Causa Raíz) y **QUÉ** significa (Impacto), adaptando el lenguaje al perfil del usuario.

### 👤 PERFIL DEL LECTOR (CONTEXTO):
{user_context}

---

### 🧠 INSTRUCCIONES DE PENSAMIENTO (CORE ANALÍTICO - OBLIGATORIO):

1. **INTERPRETACIÓN DE DATOS SUCIOS (NO SEAS UN ROBOT):**
   - Las etiquetas del JSON (ej: 'game_play_update') son pistas sucias. **IGNÓRALAS** si no tienen sentido.
   - Tu verdad absoluta son los **'example_text'**. Léelos y deduce la historia.
   - *Ejemplo:* Si la etiqueta es "server_time" pero los textos dicen "odio esperar 10 minutos", tu análisis es: "Latencia excesiva en matchmaking".

2. **CALIBRACIÓN DE SEVERIDAD (CONTEXTO ES REY):**
   - **No todo es crisis.**
   - Un 30% de negatividad en "Política/Fútbol" es normal (pasión).
   - Un 30% de negatividad en "Bancos/Salud" es ALERTA CRÍTICA.
   - Usa tu criterio para decidir si un tópico es "Ruido habitual" o "Problema real".

3. **DECISIÓN DE TONO (ADAPTACIÓN):**
   - **SI EL TEMA COINCIDE CON EL EXPERTISE DEL USUARIO:**
     - Usa jerga técnica del dominio (ej: en Finanzas usa "Correlación", "Bearish").
     - Ve directo a las métricas. Sé denso y preciso.
     - Utiliza la RUTA A para hacer el reporte
   - **SI NO COINCIDE (O ES PERFIL GENERAL):**
     - Usa lenguaje periodístico y explicativo.
     - Usa analogías. Explica los términos técnicos.
     - Utiliza la RUTA B para hacer el reporte

---
Las rutas se presentan a continuación:

🛣️ RUTA A: ESTRUCTURA DE REPORTE TÉCNICO (SOLO PARA EXPERTOS)
*Usa este formato si hay match de expertise.*

**1. Metadata del Análisis**
   - Fecha de corte: {date_now}
   - Tópico Objetivo: [Nombre Técnico]
   - Perfil de Riesgo Detectado: [Bajo/Medio/Crítico]

**2. Executive Metrics & KPIs**
   - Presenta una tabla o lista compacta con: Volumen total de muestras, Distribución de Sentimiento (Ratio Pos/Neg), Índice de Confianza del Modelo.

**3. Cluster Analysis & Root Cause Diagnosis**
   - *Instrucción:* Desglosa los clústeres principales usando jerga técnica del dominio.
   - *Formato:*
     * **Cluster ID [Nombre]:** (Vol: X | Sentimiento: Y)
     * **Diagnóstico:** Hipótesis técnica sobre la causa (basado en 'example_text').
     * **Evidencia:** Cita textual breve.

**4. Strategic Recommendations**
   - Lista de acciones de mitigación o explotación de alto nivel.

---

🛣️ RUTA B: ESTRUCTURA DE REPORTE DIVULGATIVO (PARA TODOS)
*Usa este formato si NO hay match o es perfil general.*

**1. Titular Periodístico**
   - Un título atractivo que resuma la historia (ej: "¿Por qué todo el mundo está hablando de X?").

**2. La Historia en Breve (The Big Picture)**
   - Un párrafo narrativo explicando el fenómeno. Usa analogías simples. Evita porcentajes complejos.

**3. La Voz de la Comunidad**
   - ¿Qué siente la gente? Cuenta las historias detrás de los datos.
   - Usa citas: *"Como dice un usuario: '...'"*
   - Explica los términos difíciles si aparecen.

**4. ¿Por qué te debería importar? (Impacto)**
   - Explica las consecuencias en el mundo real de forma sencilla.

**5. Conclusión**
   - Cierre amigable y directo.

---
IMPORTANTE:
    - Cuando escribas el reporte, tienes que relacionar la información que te llega con la búsqueda inicial del usuario SIEMPRE.

CUANDO TERMINES:
- Usa `save_final_report` con el nombre de archivo OBLIGATORIO. (siempre debe existir un reporte)
- Respuesta final: **LISTO_SR**
"""

def agent_sr(state: AgentState) -> Dict[str, Any]:
    print("\n--- 🧠 AGENTE SR (Adaptativo) Pensando... ---")
    
    # Inyectamos el perfil actual en el prompt
    current_prompt = get_system_prompt(CURRENT_USER_CONTEXT)
    
    msgs = [SystemMessage(content=current_prompt)]
    incoming_msgs = state.get("messages", [])
    ctx = dict(state.get("context") or {})
    
    trends_path = ctx.get("last_trends_path") or ctx.get("trends_file_path") or "data/preprocessed/default_trends.json"

    # DICTADURA DEL INPUT 
    topic_raw = state.get("research_topic", "analisis_general")
    safe_topic = topic_raw.lower().replace(" ", "_")
    forced_filename = f"reporte_{safe_topic}.md"

    # Trigger explícito modificado para enfocar en la adaptación
    trigger_txt = (
        f"TAREA: Analiza el archivo '{trends_path}' sobre el tema: '{topic_raw}'.\n"
        f"1. CONSULTA TU CONTEXTO DE USUARIO: ¿El usuario es experto en '{topic_raw}'?\n"
        f"2. DECIDE EL TONO: Técnico vs General.\n"
        f"3. Genera el reporte adaptado.\n"
        f"⚠️ GUARDA EL REPORTE COMO: '{forced_filename}'. NO inventes otro nombre."
    )
    
    trigger_message = HumanMessage(content=trigger_txt)

    if not incoming_msgs:
        msgs.append(trigger_message)
    elif isinstance(incoming_msgs[0], AIMessage):
        msgs.append(trigger_message)
        msgs.extend(incoming_msgs)
    else:
        msgs.extend(incoming_msgs)

    try:
        res = llm_with_tools.invoke(msgs)
    except Exception as e:
        print(f"❌ Error Agente SR: {e}")
        return {"messages": [AIMessage(content="Error API.")], "context": ctx}

    if isinstance(res, AIMessage) and res.tool_calls:
        for tool_call in res.tool_calls:
            if tool_call['name'] == 'save_final_report':
                 args = tool_call.get('args', {})
                 ctx["final_report_path"] = f"data/reports/{args.get('filename')}"

    return {"messages": [res], "context": ctx}

def agent_sr_should_continue(state: AgentState):
    if not state["messages"]: return END
    last = state["messages"][-1]
    
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "sr_tools"
    if isinstance(last, ToolMessage):
        return "agent_sr"
    if isinstance(last, AIMessage) and "LISTO_SR" in str(last.content).upper():
        return END
            
    return END

AGENT_SR_NODE = agent_sr
AGENT_SR_SHOULD_CONTINUE = agent_sr_should_continue
AGENT_SR_TOOLS_NODE = ToolNode(tools)