# src/agents/tools/reddit_tool.py
from __future__ import annotations
from typing import Dict, Any, Iterable, List, Union
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool
from dotenv import load_dotenv
import os, praw, re, hashlib, json

# --- FUNCIONES AUXILIARES (Tus funciones originales intactas) ---
def _normalize_spaces(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def _clean_text(t: str) -> str:
    return _normalize_spaces(t)

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _normalize_terms(terms: Iterable[str]) -> List[str]:
    return [re.sub(r"\s+", " ", (t or "").lower()).strip() for t in terms if t and t.strip()]

def _match_text(text: str, terms: List[str], mode: str = "ANY") -> bool:
    base = (text or "").lower()
    if not terms:
        return True
    if mode.upper() == "ALL":
        return all(t in base for t in terms)
    return any(t in base for t in terms)

# --- CONFIGURACIÓN DE RUTAS (Nuevo: Para guardar el archivo físicamente) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

@tool("reddit_collect")
def reddit_collect(
    query: str,
    search_lang: str = "es", # <--- NUEVO: Para el ruteo de idioma
    subreddit: str = "all",
    since_minutes: int = 43200, # (30 días por defecto) Ajustado para encontrar más cosas
    match: str = "ANY",
    limit: int = 100, # Ajustado para tesis
    max_fetch: int = 2000,
) -> str: # <--- CAMBIO: Retorna string JSON con la ruta, no el objeto gigante
    """
    Recolecta posts de Reddit, los ETIQUETA con el idioma y los GUARDA en disco.
    
    Args:
        query: Tema a buscar.
        search_lang: 'es' o 'en'. CRÍTICO para activar el modelo de IA correcto después.
        subreddit: Donde buscar (default 'all').
        since_minutes: Filtro de tiempo hacia atrás.
    """
    load_dotenv()
    
    # Normalización de idioma
    lang_code = search_lang.lower().strip()
    if lang_code not in ['es', 'en']: lang_code = 'es'

    print(f"   📡 [RedditTool] Buscando: '{query}' | Lang: {lang_code.upper()} | Sub: {subreddit}")

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID", ""),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
        user_agent=os.getenv("REDDIT_USER_AGENT", "tesis-agent/0.1"),
        check_for_async=False,
    )
    
    after_ts = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    terms = _normalize_terms(query.split())
    out: List[Dict[str, Any]] = []
    fetched = 0

    try:
        # Usamos subreddit("all") o el que pida el usuario
        target_sub = reddit.subreddit(subreddit) if subreddit else reddit.subreddit("all")
        
        for sub in target_sub.search(query=query, sort="relevance", limit=max_fetch):
            fetched += 1
            created = datetime.fromtimestamp(sub.created_utc, tz=timezone.utc)
            
            # Filtro de tiempo (Tu lógica original)
            if created < after_ts:
                continue
            
            # Limpieza y Matching (Tu lógica original)
            text_full = _clean_text(f"{sub.title or ''}\n\n{sub.selftext or ''}")
            if not _match_text(text_full, terms, mode=match):
                continue
            
            # Construcción del Item (Tu estructura + ETIQUETA DE IDIOMA)
            item = {
                "id": sub.id,
                "text": text_full,
                # Campos compatibles con tus scripts avanzados
                "text_raw": text_full, 
                "created_at": created.isoformat(),
                "created_utc": sub.created_utc, # Para compatibilidad con nodes.py
                "url": f"https://www.reddit.com{sub.permalink}",
                "metadata": {
                    "subreddit": str(sub.subreddit),
                    "author": str(sub.author) if sub.author else None,
                    "score": sub.score,
                    "num_comments": sub.num_comments,
                    "over_18": sub.over_18,
                },
                "hash": _sha1(f"{sub.id}:{text_full[:200]}"),
                
                # --- LA CLAVE DEL ÉXITO ---
                "lang": lang_code 
            }
            out.append(item)
            
            if len(out) >= limit:
                break
        
        if not out:
            return json.dumps({"status": "warning", "message": f"No se encontraron posts para '{query}'."})

        # --- GUARDADO EN DISCO (Para pasar al siguiente nodo) ---
        safe_name = "".join([c if c.isalnum() else "_" for c in query]).strip("_")[:50]
        filename = f"{safe_name}_{lang_code}.jsonl"
        filepath = os.path.join(RAW_DATA_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            for p in out:
                json.dump(p, f, ensure_ascii=False)
                f.write("\n")

        # Retornamos JSON con la ruta para el Agente
        return json.dumps({
            "status": "success",
            "count": len(out),
            "fetched": fetched,
            "path": filepath,         # <--- El Agente A leerá esto
            "detected_lang": lang_code
        })

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})