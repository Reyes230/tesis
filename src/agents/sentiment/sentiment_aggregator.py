# src/agents/sentiment/sentiment_aggregator.py
from __future__ import annotations
import argparse
import json
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional

LABELS = ("negative", "neutral", "positive")

def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except: continue

def _label_to_idx(label: str) -> int:
    if label in LABELS: return LABELS.index(label)
    return 1 # neutral fallback

def _idx_to_label(i: int) -> str:
    return LABELS[max(0, min(2, int(i)))]

def _length_weighted(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula el promedio ponderado por longitud del chunk."""
    totals = [0.0, 0.0, 0.0]
    total_w = 0.0
    
    # Para encontrar el chunk más representativo
    best_chunk_data = None
    max_score = -1.0

    for c in chunks:
        # Extraer datos normalizados
        res = c.get("sentiment") or {}
        lab = res.get("label", "neutral")
        conf = float(res.get("confidence", 0.0))
        
        # Peso por longitud (span)
        span = c.get("span_tokens")
        w = (span[1] - span[0]) if (span and len(span)==2) else 1
        w = max(1, w)

        idx = _label_to_idx(lab)
        totals[idx] += conf * w
        total_w += w
        
        # Guardar "decider" (chunk con mayor confianza * peso)
        if (conf * w) > max_score:
            max_score = conf * w
            best_chunk_data = {
                "chunk_index": c.get("chunk_index"),
                "text_snippet": c.get("text", "")[:50] + "...",
                "label": lab,
                "score": conf
            }

    if total_w <= 0:
        return {"label_final": "neutral", "score_final": 0.0}

    # Ganador
    winner_idx = max(range(3), key=lambda i: totals[i])
    final_score = totals[winner_idx] / total_w

    return {
        "label_final": _idx_to_label(winner_idx),
        "score_final": final_score,
        "decider_chunk": best_chunk_data
    }

def aggregate_post(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunks: return {}
    
    # 1. Reconstruir Texto Completo (Ordenando por índice de chunk)
    chunks_sorted = sorted(chunks, key=lambda x: x.get("chunk_index", 0))
    full_text = " ".join([c.get("text", "").strip() for c in chunks_sorted])
    
    # 2. Metadatos base (del primer chunk)
    base = chunks[0]
    
    # 3. Calcular Sentimiento Agregado
    agg_res = _length_weighted(chunks)
    
    # 4. Estadísticas de Idioma y Rutas
    langs = [c.get("lang") or "unknown" for c in chunks]
    sources = [(c.get("sentiment") or {}).get("source", "unknown") for c in chunks]

    return {
        "post_id": base.get("parent_post_id") or base.get("post_id"),
        "timestamp": base.get("timestamp"),
        "channel": base.get("channel"),
        "text_full": full_text,
        
        # Resultados
        "label_final": agg_res["label_final"],
        "score_final": agg_res["score_final"],
        "decider_chunk": agg_res["decider_chunk"],
        
        # Stats
        "total_chunks": len(chunks),
        "lang_counts": dict(Counter(langs)),
        "route_counts": dict(Counter(sources))
    }

def run_aggregator(input_path: str, output_path: str):
    buckets = defaultdict(list)
    # Agrupar chunks por Post ID
    for row in _read_jsonl(input_path):
        pid = row.get("parent_post_id") or row.get("post_id")
        if pid: buckets[str(pid)].append(row)

    # Procesar y escribir
    with open(output_path, "w", encoding="utf-8") as out:
        for pid, rows in buckets.items():
            record = aggregate_post(rows)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True)
    ap.add_argument("--out", dest="output_path", required=True)
    args = ap.parse_args()
    
    run_aggregator(args.input_path, args.output_path)
    print(f"✅ Agregación completada. Resultados en: {args.output_path}")