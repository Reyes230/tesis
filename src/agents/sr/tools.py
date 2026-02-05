# src/agents/sr/tools.py
import json
import os
import pandas as pd
import re
from langchain_core.tools import tool
from pathlib import Path

# Calculamos rutas absolutas del proyecto para no perdernos nunca
# tools.py -> sr -> agents -> src -> [ROOT]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True) # Aseguramos que la carpeta exista

@tool
def get_analysis_data(trends_path: str):
    """
    Lee el archivo de tendencias JSON. 
    Es capaz de encontrar el archivo aunque solo se le pase el nombre.
    Args:
        trends_path: Ruta completa O solo el nombre del archivo (ej: 'reporte.json').
    """
    print(f"   🔎 [Tool] Buscando archivo: {trends_path}")
    
    # 1. ESTRATEGIA DE BÚSQUEDA BLINDADA
    possible_paths = [
        trends_path,                                              # Tal cual lo mandó el agente
        os.path.join(REPORTS_DIR, os.path.basename(trends_path)), # En la carpeta reports
        os.path.abspath(trends_path)                              # Ruta absoluta del sistema
    ]
    
    final_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            final_path = p
            break
            
    if not final_path:
        # Fallback de emergencia: Buscar el JSON más reciente si el nombre falló
        json_files = [os.path.join(REPORTS_DIR, f) for f in os.listdir(REPORTS_DIR) if f.endswith('.json')]
        if json_files:
            final_path = max(json_files, key=os.path.getmtime)
            print(f"   ⚠️ [Tool] Nombre no encontrado. Usando el más reciente: {final_path}")
        else:
            return {
                "error": f"No se encontró el archivo. Se buscó en: {possible_paths}. Verifica que el paso anterior haya guardado el JSON."
            }

    print(f"   ✅ [Tool] Archivo encontrado en: {final_path}")

    try:
        with open(final_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not data: return {"error": "El archivo de datos está vacío"}

        df = pd.DataFrame(data)
        
        # Validación de columnas
        if 'volume' not in df.columns or 'sentiment_avg' not in df.columns:
            return {"error": "Estructura JSON incorrecta. Faltan columnas 'volume' o 'sentiment_avg'."}

        total_vol = df['volume'].sum()
        if total_vol == 0: return {"error": "El volumen total es 0, no hay datos para analizar."}

        # Cálculos de Negocio
        neg_cluster = df[df['sentiment_avg'] < -0.15]
        pos_cluster = df[df['sentiment_avg'] > 0.15]
        
        pct_neg = (neg_cluster['volume'].sum() / total_vol) * 100
        pct_pos = (pos_cluster['volume'].sum() / total_vol) * 100
        
        # Top Tema
        top_topic = data[0] if len(data) > 0 else {}
        
        return {
            "meta": {
                "file_analyzed": os.path.basename(final_path),
                "total_volume": int(total_vol)
            },
            "balance": {
                "negativity_pct": round(pct_neg, 1),
                "positivity_pct": round(pct_pos, 1)
            },
            "top_topic": {
                "label": top_topic.get('label', 'Desconocido'),
                "sentiment": top_topic.get('sentiment_avg', 0),
                "priority": top_topic.get('priority', 'MEDIA'),
                "share_of_voice": f"{(top_topic.get('volume',0)/total_vol)*100:.1f}%"
            },
            "raw_topics_sample": data[:5] # Pasamos los top 5 temas
        }
    except Exception as e:
        return {"error": f"Error interno procesando datos: {str(e)}"}

@tool
def assess_severity(negativity_pct: float, top_priority: str):
    """
    Evalúa la gravedad de la situación basándose en datos numéricos.
    Usa esta herramienta DESPUÉS de leer los datos.
    """
    severity = "LOW"
    action_plan = "Monitoreo estándar."
    tone = "Informativo"

    # Lógica ajustada para ser sensible
    if negativity_pct > 35 or top_priority == "CRÍTICA":
        severity = "CRITICAL"
        action_plan = "ALERTA: Contención de crisis requerida."
        tone = "Urgente y Directo"
    elif negativity_pct > 20 or top_priority == "ALTA":
        severity = "HIGH"
        action_plan = "PRIORIDAD: Investigar causas."
        tone = "Analítico y Preocupado"
    elif negativity_pct < 10 and top_priority != "MEDIA":
        severity = "POSITIVE_TREND"
        action_plan = "OPORTUNIDAD: Amplificar mensaje."
        tone = "Optimista"

    return {
        "severity_level": severity,
        "recommended_tone": tone,
        "strategic_advice": action_plan
    }

@tool
def save_final_report(content: str, filename: str):
    """
    Guarda el reporte final en Markdown en la carpeta 'data/reports'.
    NORMALIZA EL NOMBRE AUTOMÁTICAMENTE para evitar errores de búsqueda (ej: 'Greenland issue' -> 'greenland_issue.md').
    """
    try:
        # 1. Limpieza de nombre (Dictadura del Input)
        # Convertimos a minúsculas y reemplazamos espacios por guiones bajos
        clean_name = filename.lower().replace(" ", "_")
        
        # Eliminamos caracteres raros (solo letras, numeros, guion bajo y punto)
        clean_name = re.sub(r'[^a-z0-9_\.]', '', clean_name)
        
        # Aseguramos extensión .md
        if not clean_name.endswith(".md"): 
            clean_name += ".md"
        
        path = os.path.join(REPORTS_DIR, clean_name)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"REPORTE_GUARDADO_EXITOSAMENTE en {path}"
    except Exception as e:
        return f"ERROR guardando archivo: {str(e)}"