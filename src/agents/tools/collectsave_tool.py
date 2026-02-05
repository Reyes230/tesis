# src/agents/tools/collectsave_tool.py
from __future__ import annotations
from typing import Dict, Any
from langchain_core.tools import tool
import os

from .reddit_tool import reddit_collect
from .save_tool import save_posts

@tool("collect_and_save")
def collect_and_save(
    subreddit: str,
    query: str,
    since_minutes: int,
    match: str,
    limit: int,
    path: str
) -> Dict[str, Any]:
    """
    Recolecta publicaciones desde Reddit y las guarda en un archivo JSONL o CSV.
    Garantiza que 'path' sea absoluto y termine en .jsonl si no trae extensión.
    """
    # Normaliza path
    if not path:
        path = "data/collect/collected_posts.jsonl"
    root, ext = os.path.splitext(path)
    if not ext:  # sin extensión → .jsonl
        path = root + ".jsonl"
    abs_path = os.path.abspath(path)

    # 1) Recolección
    reddit_result = reddit_collect.invoke({
        "subreddit": subreddit,
        "query": query,
        "since_minutes": since_minutes,
        "match": match,
        "limit": limit
    })

    posts = reddit_result.get("posts", [])
    if not posts:
        return {"ok": False, "error": "No se recolectaron posts", "path": abs_path}

    # 2) Guardado
    save_result = save_posts.invoke({
        "posts": posts,
        "path": abs_path
    })

    return {
        "ok": save_result.get("ok", False),
        "saved": save_result.get("saved", 0),
        "path": abs_path,  # <-- SIEMPRE absoluta
        "message": "Recolección y guardado completados"
    }
