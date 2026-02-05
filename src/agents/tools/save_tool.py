# src/agents/tools/save_tool.py
from __future__ import annotations
from typing import Dict, Any, List
from langchain_core.tools import tool
from pathlib import Path
import json
import csv
import os

# 1. DEFINIMOS LA RUTA BASE SEGURA (Ancla al proyecto)
# Calculamos la raíz del proyecto basándonos en la ubicación de este archivo
# save_tool.py -> tools -> agents -> src -> [RAIZ]
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3] 
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Aseguramos que la carpeta exista antes de nada
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

@tool("save_posts")
def save_posts(posts: List[Dict[str, Any]], path: str) -> Dict[str, Any]:
    """
    Guarda la lista de posts en la carpeta 'data/raw' del proyecto.
    Args:
        posts: Lista de diccionarios con los datos.
        path: Nombre del archivo (ej: 'economia.jsonl'). Se ignoran rutas absolutas.
    """
    # 2. SANEAMIENTO DE RUTA (La Jaula)
    # Si el LLM manda "C:/User/Data/archivo.jsonl", nosotros tomamos solo "archivo.jsonl"
    filename = Path(path).name 
    
    # Construimos la ruta real forzada
    safe_path = RAW_DATA_DIR / filename
    
    print(f"   💾 [SaveTool] Guardando {len(posts)} items en: {safe_path}")

    # 3. LÓGICA DE GUARDADO (Igual que tenías, pero con safe_path)
    try:
        if safe_path.suffix.lower() == ".jsonl":
            with safe_path.open("w", encoding="utf-8") as f:
                for item in posts:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            # Retornamos la ruta como string para que el Agente la lea
            return {"ok": True, "saved": len(posts), "path": str(safe_path)}

        if safe_path.suffix.lower() == ".csv":
            cols = ["id", "created_at", "url", "len", "text_clean"]
            # Pre-filtrado para evitar errores si faltan columnas
            filtered_posts = []
            for p in posts:
                filtered_posts.append({k: p.get(k, "") for k in cols})

            with safe_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(filtered_posts)
                
            return {"ok": True, "saved": len(posts), "path": str(safe_path)}

        return {"ok": False, "error": f"Extensión no soportada: {safe_path.suffix}"}
        
    except Exception as e:
        return {"ok": False, "error": str(e)}