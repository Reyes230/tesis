# src/agents/tools/preprocess_tool.py
from langchain_core.tools import tool
import re, json, os, hashlib
from typing import List
from pydantic import BaseModel, Field
from datetime import datetime

class PreprocessInput(BaseModel):
    path: str = Field(..., description="Ruta del archivo JSON o JSONL con los datos a limpiar y normalizar")
    output_path: str = Field(default="data/processed/cleaned_reddit.jsonl", description="Ruta de salida para guardar los datos procesados")

def clean_text(text: str) -> str:
    t = text or ""
    # quita URLs
    t = re.sub(r"https?://\S+", "", t)     # http o https
    t = re.sub(r"\bwww\.\S+", "", t)       # www.*
    # preserva letras, números, espacios, apóstrofes, !?, emojis básicos
    t = re.sub(r"[^\w\s'!?🙂-🙏💔🔥❤️😂🤣😍🙃😏😉😊😭😡😱👍👎]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t



def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()

def _to_iso_z(ts):
    """Admite epoch (segundos) o ISO (string). Devuelve ISO-8601 con Z o None."""
    if ts is None:
        return None
    try:
        # epoch numérico
        if isinstance(ts, (int, float)):
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
        s = str(ts).strip()
        if not s:
            return None
        # ya viene con Z
        if s.endswith("Z"):
            return s
        # normaliza +00:00 / timezone-less
        try:
            dt = datetime.fromisoformat(s.replace("Z",""))
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except:
            return None
    except:
        return None

def _detect_lang_simple(text: str):
    """Heurística ligera ES/EN; si no se puede, devuelve None (para que salga null en JSON)."""
    t = (text or "").lower()
    if not t.strip():
        return None
    ES_SW = {" no ", " si ", " pero ", " muy ", " más ", " mas ", " también ", " tambien ", " porque ", " que ", " esto ", " esta ", " estos ", " estas ", " otro ", " otra "}
    EN_SW = {" the ", " and ", " but ", " very ", " more ", " also ", " because ", " that ", " this ", " these ", " those ", " again "}

    es = sum(1 for w in ES_SW if w in " " + t + " ")
    en = sum(1 for w in EN_SW if w in " " + t + " ")
    if es >= 2 and es > en: return "es"
    if en >= 2 and en > es: return "en"
    # pistas por acentos
    if re.search(r"[áéíóúñ¿¡]", t): return "es"
    return None

def _extract_hashtags(text):
    return re.findall(r"#(\w+)", text or "")

def _read_any(path: str):
    """Lee .jsonl (línea por línea) o .json (lista u objeto) y retorna lista de dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                try:
                    items.append(json.loads(line))
                except:
                    continue
        else:  # .json
            data = json.load(f)
            if isinstance(data, list):
                items = [x for x in data if isinstance(x, dict)]
            elif isinstance(data, dict):
                items = [data]
    return items

@tool(args_schema=PreprocessInput)
def preprocess_posts(path: str, output_path: str) -> str:
    """
    Preprocesa posts (JSON/JSONL) al esquema estándar para el agente de sentimiento.
    - Descarta registros vacíos y sin timestamp.
    - Normaliza timestamp a ISO-8601 Z.
    - lang = null si no se detecta.
    - Dedup por sha256(content), url y post_id.
    - Cuenta procesados/descartes/deduplicados.
    """
    # Validar entrada
    if not os.path.exists(path):
        return f"Error: archivo de entrada no encontrado: {path}"

    # Preparar salida
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Lectura flexible
    raw_items = _read_any(path)

    # Dedup sets
    seen_sha = set()
    seen_url = set()
    seen_pid = set()

    # Contadores
    total_read = 0
    processed = 0
    skipped_empty = 0
    skipped_no_timestamp = 0
    dedup_content = 0
    dedup_url = 0
    dedup_post = 0

    cleaned = []

    for data in raw_items:
        total_read += 1
        # Campos típicos Reddit; si vienes de otra fuente agrega más claves aquí
        post_id = data.get("id") or data.get("post_id") or ""
        title = data.get("title") or ""
        body = data.get("selftext") or data.get("text") or data.get("body") or ""
        raw_text = (f"{title} {body}").strip()
        text_norm = clean_text(raw_text)
        tokens_hint = max(1, len(text_norm.split()))

        # DESCARTAR: sin texto
        if not text_norm.strip():
            skipped_empty += 1
            continue

        # Timestamp (int epoch u otros)
        ts = data.get("created_utc") or data.get("created_at") or data.get("created")
        timestamp = _to_iso_z(ts)
        # DESCARTAR: sin timestamp
        if not timestamp:
            skipped_no_timestamp += 1
            continue

        # Dedup por post_id (si hay)
        if post_id:
            if post_id in seen_pid:
                dedup_post += 1
                continue
            seen_pid.add(post_id)

        # URL (si hay)
        url = data.get("url")
        if url:
            if url in seen_url:
                dedup_url += 1
                continue
            seen_url.add(url)

        # Dedup por contenido
        content_hash = _sha256(raw_text)
        if content_hash in seen_sha:
            dedup_content += 1
            continue
        seen_sha.add(content_hash)

        # Lang (null si no detecta)
        lang = _detect_lang_simple(raw_text)

        # Mapea engagement
        likes = data.get("score") or data.get("ups") or data.get("likes")
        comments = data.get("num_comments") or data.get("comments")

        # NSFW (reddit: over_18)
        nsfw = data.get("over_18") if "over_18" in data else data.get("nsfw")

        item = {
            "post_id": f"reddit:{post_id}" if post_id else "",
            "timestamp": timestamp,
            "channel": data.get("channel") or "reddit",
            "lang": lang,  # None -> null en JSON
            "text_norm": text_norm,
            "text_raw": raw_text,
            "text_raw_ref": None,
            "meta": {
                "author": data.get("author"),
                "author_id": data.get("author_id"),
                "url": url,
                "subreddit": data.get("subreddit"),
                "hashtags": _extract_hashtags(raw_text),
                "engagement": {
                    "likes": likes,
                    "rts": data.get("rts"),
                    "comments": comments
                },
                "nsfw": nsfw,
                "tokens_hint": tokens_hint 
            },
            "quality_flags": {
                "very_short": len(text_norm) < 30,
                "spam": False,
                "irony_hint": bool(re.search(r"(^|\s)/s(\s|$)|🙃|😏|😉", raw_text.lower()))
            },
            "context_pack": {
                "thread_window": [],
                "entity_lexicon_hits": [],
                "topic_hint": data.get("topic_hint") or []
            },
            "content_sha256": content_hash
        }

        cleaned.append(item)
        processed += 1

    # Escribir salida JSONL
    with open(output_path, "w", encoding="utf-8") as out:
        for item in cleaned:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Resumen
    summary = (
        f"✅ Preprocesamiento completo → {output_path}\n"
        f"Total leídos: {total_read} | Procesados: {processed}\n"
        f"Descartados → vacíos: {skipped_empty}, sin timestamp: {skipped_no_timestamp}\n"
        f"Deduplicados → contenido: {dedup_content}, url: {dedup_url}, post_id: {dedup_post}"
    )
    return summary
