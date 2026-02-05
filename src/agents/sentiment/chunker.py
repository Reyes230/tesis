# src/agents/sentiment/chunker.py
from __future__ import annotations
import os, json, re
from typing import Dict, List, Iterable, Tuple, Optional

DEFAULT_MAX_TOKENS = 320     # tamaño objetivo del chunk
DEFAULT_OVERLAP    = 40      # solapamiento entre chunks (tokens)
CHUNK_PREFIX       = "ck"    # prefijo para id de chunk

# Tokenización ligera (sin deps)
def _simple_tokenize(text: str) -> List[str]:
    """Tokenizer muy simple por espacios/puntuación neutralizada."""
    if not text:
        return []
    t = re.sub(r"\s+", " ", text.strip())
    return t.split(" ")

def _count_tokens(text: str) -> int:
    return len(_simple_tokenize(text))

# Split por oraciones con “fall-back”
# Divide después de . ! ? ¡ ¿ y antes de cualquier no-espacio
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?¡¿])\s+(?=\S)")

def _split_sentences(text: str) -> List[str]:
    """Split sencillo por oraciones; si no encuentra, devuelve el texto completo."""
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    if len(parts) <= 1:
        return [text.strip()]
    return [p.strip() for p in parts if p and p.strip()]

# Lógica de chunking
def _greedy_pack(sentences: List[str], max_tokens: int) -> Tuple[str, int]:
    """
    Empaca oraciones hasta llegar (o aproximarse) a max_tokens.
    Devuelve: texto_chunk, tokens_consumidos.
    """
    packed: List[str] = []
    cur_tokens = 0
    for s in sentences:
        s_tokens = _count_tokens(s)
        if cur_tokens == 0 and s_tokens >= max_tokens:
            # oraciones gigantes: recorta por tokens
            toks = _simple_tokenize(s)
            take = max_tokens
            text = " ".join(toks[:take])
            return text, take
        if cur_tokens + s_tokens <= max_tokens:
            packed.append(s)
            cur_tokens += s_tokens
        else:
            break
    return (" ".join(packed), cur_tokens)

def chunk_text(
    text: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int   = DEFAULT_OVERLAP
) -> List[Tuple[str, int, int]]:
    """
    Divide un texto en chunks aproximados por tokens.
    Devuelve lista de tuplas: (chunk_text, token_start, token_end)
    """
    if not text:
        return []

    # estrategia por oraciones con fallback a stream de tokens
    sentences = _split_sentences(text)

    # si no hay oraciones claras o es demasiado simple, usar ventana por tokens
    if len(sentences) <= 1:
        toks = _simple_tokenize(text)
        chunks = []
        i = 0
        while i < len(toks):
            j = min(i + max_tokens, len(toks))
            chunk = " ".join(toks[i:j])
            chunks.append((chunk, i, j))
            # avance con solapamiento
            next_i = j - overlap
            i = next_i if next_i > i else j
        return chunks

    # pack por oraciones
    chunks: List[Tuple[str, int, int]] = []
    consumed_global = 0
    sent_idx = 0

    while sent_idx < len(sentences):
        chunk_text_str, used_tokens = _greedy_pack(sentences[sent_idx:], max_tokens)
        if not chunk_text_str or used_tokens == 0:
            break

        start = consumed_global
        end   = consumed_global + used_tokens
        chunks.append((chunk_text_str, start, end))

        # mover sent_idx según tokens aproximados consumidos
        advanced = 0
        used = 0
        for k in range(sent_idx, len(sentences)):
            stoks = _count_tokens(sentences[k])
            if used + stoks <= used_tokens:
                used += stoks
                advanced += 1
            else:
                break
        if advanced == 0:
            # seguridad: evitar loop infinito
            advanced = 1
            used = _count_tokens(sentences[sent_idx])

        # aplicar solapamiento en términos de oraciones
        back_tokens = overlap
        back_sents = 0
        tmp = 0
        for k in range(advanced):
            idx = sent_idx + advanced - 1 - k
            if idx < sent_idx:
                break
            tmp += _count_tokens(sentences[idx])
            if tmp >= back_tokens:
                back_sents = k  # retroceder k oraciones
                break

        sent_idx = sent_idx + advanced - back_sents
        consumed_global = end - (tmp if back_sents > 0 else 0)

        # clamps
        if sent_idx <= 0:
            sent_idx = 0
        if sent_idx >= len(sentences):
            break

    return chunks

# Construcción de objetos-chunk
def build_chunk_records(
    item: Dict,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int   = DEFAULT_OVERLAP
) -> List[Dict]:
    """
    Convierte un item preprocesado (del JSONL limpiado) en N registros chunk.
    Mantiene metadatos relevantes y añade “chunk_info”.
    """
    text = item.get("text_norm") or ""
    parent_post_id = item.get("post_id") or ""
    parent_sha     = item.get("content_sha256") or ""
    lang           = item.get("lang")
    timestamp      = item.get("timestamp")
    meta           = dict(item.get("meta") or {})
    quality_flags  = dict(item.get("quality_flags") or {})
    context_pack   = dict(item.get("context_pack") or {})

    # heurística de tokens si no viene
    base_tokens_hint = int(meta.get("tokens_hint") or _count_tokens(text) or 1)

    # chunking
    chunks = chunk_text(text, max_tokens=max_tokens, overlap=overlap)
    out: List[Dict] = []

    # texto corto: crear un único chunk si no hubo
    if not chunks:
        ck_tokens = _count_tokens(text)
        out.append({
            "chunk_id": f"{CHUNK_PREFIX}:{parent_sha}:1",
            "parent_post_id": parent_post_id,
            "parent_sha256": parent_sha,

            "timestamp": timestamp,
            "channel": item.get("channel") or "reddit",
            "lang": lang,

            "text": text,
            "span_tokens": [0, ck_tokens],
            "chunk_index": 1,
            "chunk_count": 1,

            "meta": {
                **meta,
                "tokens_hint": ck_tokens,
                "source_url": meta.get("url"),
            },
            "quality_flags": quality_flags,
            "context_pack": context_pack,
            "routing": {
                "needs_chunking": False,
                "is_single_chunk": True,
                "base_tokens_hint": ck_tokens,
            },
        })
        return out

    total_chunks = len(chunks)
    for idx, (c_text, t_start, t_end) in enumerate(chunks, start=1):
        ck_tokens = _count_tokens(c_text)

        record = {
            "chunk_id": f"{CHUNK_PREFIX}:{parent_sha}:{idx}",
            "parent_post_id": parent_post_id,
            "parent_sha256": parent_sha,

            "timestamp": timestamp,
            "channel": item.get("channel") or "reddit",
            "lang": lang,

            "text": c_text,
            "span_tokens": [int(t_start), int(t_end)],
            "chunk_index": idx,
            "chunk_count": total_chunks,

            "meta": {
                **meta,
                "tokens_hint": ck_tokens,
                "source_url": meta.get("url"),
            },
            "quality_flags": quality_flags,
            "context_pack": context_pack,

            # hints para ruteo aguas arriba
            "routing": {
                "needs_chunking": total_chunks > 1,
                "is_single_chunk": total_chunks == 1,
                "base_tokens_hint": base_tokens_hint,
            },
        }
        out.append(record)

    return out

# Procesamiento de archivos
def _read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except:
                continue

def _write_jsonl(path: str, records: Iterable[Dict]) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def chunk_file(
    input_path: str,
    output_path: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int   = DEFAULT_OVERLAP
) -> Dict[str, int]:
    """
    Lee el JSONL preprocesado y escribe un JSONL de chunks.
    Devuelve contadores.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    total_items = 0
    total_chunks = 0

    def _gen():
        nonlocal total_items, total_chunks
        for item in _read_jsonl(input_path):
            total_items += 1
            chunks = build_chunk_records(item, max_tokens=max_tokens, overlap=overlap)
            total_chunks += len(chunks)
            for c in chunks:
                yield c

    _write_jsonl(output_path, _gen())
    return {"items": total_items, "chunks": total_chunks}

# CLI mínimo 
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Chunker para análisis de sentimiento")
    ap.add_argument("--in", dest="input_path", required=True, help="Ruta del JSONL preprocesado")
    ap.add_argument("--out", dest="output_path", required=True, help="Ruta del JSONL de chunks")
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = ap.parse_args()

    stats = chunk_file(
        args.input_path,
        args.output_path,
        max_tokens=args.max_tokens,
        overlap=args.overlap
    )
    print(f"✅ Chunking OK → {args.output_path} | items={stats['items']} chunks={stats['chunks']}")
