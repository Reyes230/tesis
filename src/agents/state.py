# ─────────────────────────────────────────────────────────────
# File: src/agents/state.py
# ─────────────────────────────────────────────────────────────
from typing import TypedDict, List, Annotated, Dict, Any, Optional
import operator
from langchain_core.messages import BaseMessage

def add_messages(left: list, right: list):
    return left + right

class AgentState(TypedDict):
    # --- CORE ---
    messages: Annotated[List[BaseMessage], add_messages] 
    context: Dict[str, Any]

    # --- INPUT DE USUARIO ---
    research_topic: str

    # --- CONTROL DE ARCHIVOS ---
    current_file_path: Optional[str]
    sentiment_file_path: Optional[str] 
    trends_file_path: Optional[str]
    
    # --- ESTADÍSTICAS ---
    processing_stats: Optional[Dict[str, Any]]

    # Aquí se guardará "es" o "en"
    language: Optional[str]